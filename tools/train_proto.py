import argparse
import glob
import math
import os
import shutil
import time

import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

from mdistiller.dataset import get_dataset_strong
from mdistiller.distillers import distiller_dict
from mdistiller.distillers._base import Distiller
from mdistiller.engine import trainer_dict
from mdistiller.engine.cfg import CFG as cfg
from mdistiller.engine.cfg import show_cfg
from mdistiller.engine.utils import load_checkpoint, log_msg
from mdistiller.models import cifar_model_dict

cudnn.benchmark = True


# ==============================================================================
# 1. Checkpoint cleanup
# ==============================================================================
def cleanup_useless_checkpoints(exp_dir):
    print(log_msg("Cleaning up extra checkpoints...", "INFO"))
    garbage_patterns = ['epoch_*', 'student_[0-9]*', 'latest*', 'student_latest*', 'best*']
    deleted_count = 0
    for pattern in garbage_patterns:
        if 'student_best' in pattern:
            continue
        for filepath in glob.glob(os.path.join(exp_dir, pattern)):
            if 'student_best' in os.path.basename(filepath):
                continue
            try:
                if os.path.isfile(filepath):
                    os.remove(filepath)
                elif os.path.isdir(filepath):
                    shutil.rmtree(filepath)
                deleted_count += 1
            except Exception as e:
                pass
    print(log_msg(f"Cleanup finished. Removed {deleted_count} files.", "INFO"))


# ==============================================================================
# 2. Feature extraction hook
# ==============================================================================
class FeatureHook:
    def __init__(self, module):
        self.hook = module.register_forward_hook(self.hook_fn)
        self.feature = None

    def hook_fn(self, module, input, output):
        self.feature = output.view(output.size(0), -1)

    def clear(self):
        self.feature = None

    def close(self):
        self.hook.remove()


def register_avgpool_hook(model):
    hook = None
    for name, module in model.named_modules():
        if "avg_pool" in name or "global_pool" in name or "avgpool" in name:
            hook = FeatureHook(module)
            break
    if hook is None:
        modules = list(model.children())
        for i, module in enumerate(modules):
            if isinstance(module, nn.Linear) and i > 0:
                hook = FeatureHook(modules[i - 1])
                break
    return hook


# ==============================================================================
# 3. Multi-center GLVQ prototypes: K-Means warm start + smooth sigmoid GLVQ
# ==============================================================================
def precompute_multicenter_glvq_prototypes(teacher, train_loader, num_classes, K=3, device='cuda',
                                           feature_dim=128, tau_w=0.8):
    teacher.eval()
    hook = register_avgpool_hook(teacher)
    max_entropy = math.log(num_classes)

    print(log_msg(f"Phase 1: Mining high-purity features (tau_w={tau_w})...", "INFO"))
    class_features = {c: [] for c in range(num_classes)}

    with torch.no_grad():
        for batch in tqdm(train_loader, desc="Feature Mining"):
            image_weak = batch[0][0] if isinstance(batch[0], (list, tuple)) else batch[0]
            target = batch[1].to(device)
            image_weak = image_weak.to(device)

            logits = teacher(image_weak)
            if isinstance(logits, (list, tuple)): logits = logits[0]

            features = hook.feature.detach()
            # Normalize features before LVQ/GLVQ.
            features_norm = F.normalize(features, dim=1)
            hook.clear()

            probs = F.softmax(logits, dim=1)
            entropy = -torch.sum(probs * torch.log(probs + 1e-8), dim=1)
            conf_w = 1.0 - (entropy / max_entropy)
            preds = torch.argmax(probs, dim=1)

            for i in range(image_weak.size(0)):
                c = target[i].item()
                if conf_w[i].item() >= tau_w and preds[i].item() == c:
                    class_features[c].append(features_norm[i])
    hook.close()

    print(log_msg(f"Phase 2: K-Means initialization (K={K}) for each class...", "INFO"))
    prototypes_nk = torch.zeros(num_classes, K, feature_dim, device=device)
    all_feats_list = []
    all_targets_list = []

    for c in range(num_classes):
        feats_c = class_features[c]
        if len(feats_c) == 0:
            prototypes_nk[c] = F.normalize(torch.randn(K, feature_dim, device=device), dim=1)
            continue

        X = torch.stack(feats_c)
        N_c = X.size(0)
        all_feats_list.append(X)
        all_targets_list.append(torch.full((N_c,), c, dtype=torch.long, device=device))

        # Use K-Means centers as the initial sub-prototypes for this class.
        if N_c < K:
            indices = torch.randint(0, N_c, (K,), device=device)
            centers = X[indices]
        else:
            indices = torch.randperm(N_c, device=device)[:K]
            centers = X[indices].clone()
            for _ in range(10):
                sim_matrix = torch.matmul(X, centers.t())
                cluster_assignments = torch.argmax(sim_matrix, dim=1)
                new_centers = torch.zeros_like(centers)
                for k in range(K):
                    mask = (cluster_assignments == k)
                    if mask.sum() > 0:
                        new_centers[k] = F.normalize(X[mask].mean(dim=0), dim=0)
                    else:
                        new_centers[k] = X[torch.randint(0, N_c, (1,))].squeeze(0)
                centers = new_centers
        prototypes_nk[c] = centers

    prototypes_flat = prototypes_nk.view(-1, feature_dim).clone()
    warmup_prototypes_flat = prototypes_flat.clone().detach()

    if len(all_feats_list) == 0:
        return prototypes_nk

    print(log_msg("Phase 3: Smooth GLVQ optimization via autograd...", "INFO"))
    all_feats = torch.cat(all_feats_list, dim=0)
    all_targets = torch.cat(all_targets_list, dim=0)

    glvq_dataset = TensorDataset(all_feats, all_targets)
    glvq_loader = DataLoader(glvq_dataset, batch_size=512, shuffle=True)
    proto_labels = torch.arange(num_classes, device=device).unsqueeze(1).repeat(1, K).view(-1)

    # Full-data GLVQ iterations with cosine learning-rate decay.
    glvq_epochs = 100
    base_lr = 0.01
    min_lr = 0.001
    gamma = 10.0

    with torch.enable_grad():
        prototypes_param = nn.Parameter(prototypes_flat, requires_grad=True)
        # SGD is stable for this small prototype-only optimization.
        optimizer = torch.optim.SGD([prototypes_param], lr=base_lr, momentum=0.9)

        for epoch in range(glvq_epochs):
            # Keep base LR for the first 70%, then decay to min_lr.
            if epoch < int(0.7 * glvq_epochs):
                lr = base_lr
            else:
                progress = (epoch - 0.7 * glvq_epochs) / (0.3 * glvq_epochs)
                lr = min_lr + 0.5 * (base_lr - min_lr) * (1 + math.cos(math.pi * progress))

            for param_group in optimizer.param_groups:
                param_group['lr'] = lr

            for f_batch, y_batch in glvq_loader:
                optimizer.zero_grad()
                p_norm = F.normalize(prototypes_param, dim=1)

                sim = torch.matmul(f_batch, p_norm.t())
                dist = 1.0 - sim

                pos_mask = (proto_labels.unsqueeze(0) == y_batch.unsqueeze(1))
                neg_mask = ~pos_mask

                # Soft-min distances to positive and negative prototypes.
                d_pos_matrix = dist.masked_fill(neg_mask, 10.0)
                d_neg_matrix = dist.masked_fill(pos_mask, 10.0)

                d_pos = - (1.0 / gamma) * torch.logsumexp(-gamma * d_pos_matrix, dim=1)
                d_neg = - (1.0 / gamma) * torch.logsumexp(-gamma * d_neg_matrix, dim=1)

                # GLVQ relative-distance objective with a sigmoid loss.
                mu = (d_pos - d_neg) / (d_pos + d_neg + 1e-8)
                loss = torch.sigmoid(10.0 * mu).mean()

                loss.backward()
                optimizer.step()

                # Re-normalize prototypes after every update.
                with torch.no_grad():
                    prototypes_param.copy_(F.normalize(prototypes_param, dim=1))

    prototypes_nk_final = prototypes_param.detach().view(num_classes, K, feature_dim)
    print(log_msg("Smooth GLVQ optimization finished.", "INFO"))

    # Track how far GLVQ moved prototypes from their K-Means warm start.
    with torch.no_grad():
        final_flat = prototypes_param.detach()
        cos_sim = (warmup_prototypes_flat * final_flat).sum(dim=1)
        mean_shift = cos_sim.mean().item()
        min_shift = cos_sim.min().item()

    print(log_msg("=" * 60, "INFO"))
    print(log_msg("[Manifold Analysis] Cosine similarity (warmup vs GLVQ):", "INFO"))
    print(log_msg(f"   => Average Similarity across {num_classes * K} sub-centers: {mean_shift:.4f}", "INFO"))
    print(log_msg(f"   => Minimum Similarity (Most Shifted Sub-center): {min_shift:.4f}", "INFO"))
    print(log_msg("=" * 60, "INFO"))

    return prototypes_nk_final


# ==============================================================================
# 4. Evaluate N*K multi-center prototypes
# ==============================================================================
def evaluate_prototypes_nk(teacher, val_loader, prototypes_nk, K, device='cuda'):
    teacher.eval()
    hook = register_avgpool_hook(teacher)
    correct, total = 0, 0

    num_classes, k_num, feat_dim = prototypes_nk.shape
    prototypes_norm = F.normalize(prototypes_nk.view(-1, feat_dim), dim=1)

    print(log_msg(f"Evaluating GLVQ prototypes (K={K})...", "INFO"))

    with torch.no_grad():
        for batch in tqdm(val_loader, desc="GLVQ NPC"):
            images = batch[0][0] if isinstance(batch[0], (list, tuple)) else batch[0]
            targets = batch[1].to(device)
            images = images.to(device)

            _ = teacher(images)
            # Test features use the same L2 normalization as prototype learning.
            features_norm = F.normalize(hook.feature.detach(), dim=1)
            hook.clear()

            sim_logits = torch.matmul(features_norm, prototypes_norm.t())
            best_subcenter_indices = torch.argmax(sim_logits, dim=1)

            # Map the winning sub-prototype back to its class label.
            preds = best_subcenter_indices // K

            correct += (preds == targets).sum().item()
            total += targets.size(0)

    hook.close()
    proto_acc = 100. * correct / total
    print(log_msg("=" * 60, "INFO"))
    print(log_msg(f"GLVQ prototype (K={K}) Top-1 Accuracy: {proto_acc:.2f}%", "INFO"))
    print(log_msg("=" * 60, "INFO"))

    return proto_acc


# ==============================================================================
# 5. CRLD_Proto: prototype prior fusion
# ==============================================================================
class CRLD_Proto(Distiller):
    def __init__(self, student, teacher, cfg, precomputed_prototypes=None):
        super(CRLD_Proto, self).__init__(student, teacher)

        self.temperature = getattr(cfg.CRLD, 'TEMPERATURE', 4.0)
        self.proto_temp = getattr(cfg.CRLD, 'PROTO_TEMP', 0.1)

        loss_cfg = getattr(cfg.CRLD, 'LOSS', None)
        self.ce_loss_weight = getattr(loss_cfg, 'CE_WEIGHT', getattr(cfg.CRLD, 'CE_WEIGHT', 1.0))
        self.kd_loss_weight = getattr(loss_cfg, 'KD_WEIGHT', 1.0)
        self.tau_w = getattr(cfg.CRLD, 'TAU_W', 0.9)
        self.proto_max_weight = getattr(cfg.CRLD, 'PROTO_MAX_WEIGHT', 1.0)
        self.proto_min_weight = getattr(cfg.CRLD, 'PROTO_MIN_WEIGHT', 0.0)
        self.proto_label_adjust = getattr(cfg.CRLD, 'PROTO_LABEL_ADJUST', False)
        self.proto_label_margin = getattr(cfg.CRLD, 'PROTO_LABEL_MARGIN', 0.0)
        self.proto_wrong_suppress = getattr(cfg.CRLD, 'PROTO_WRONG_SUPPRESS', 1.0)
        self.proto_true_boost = getattr(cfg.CRLD, 'PROTO_TRUE_BOOST', 1.0)
        self.proto_reliable_use_psim = getattr(cfg.CRLD, 'PROTO_RELIABLE_USE_PSIM', True)
        self.logit_swap = getattr(cfg.CRLD, 'LOGIT_SWAP', False)
        self.logit_swap_alpha = getattr(cfg.CRLD, 'LOGIT_SWAP_ALPHA', 1.0)
        self.logit_swap_margin = getattr(cfg.CRLD, 'LOGIT_SWAP_MARGIN', 0.0)
        self.logit_swap_disable_proto = getattr(cfg.CRLD, 'LOGIT_SWAP_DISABLE_PROTO', True)

        self.student.cuda()
        self.teacher.cuda()
        self.teacher_hook = register_avgpool_hook(self.teacher)
        self.student_hook = register_avgpool_hook(self.student)

        self.num_classes = 100 if "100" in cfg.DATASET.TYPE else 10
        self.max_entropy = math.log(self.num_classes)

        # Frozen global prototypes.
        self.register_buffer("prototypes", precomputed_prototypes.clone())
        self.prototypes = F.normalize(self.prototypes, dim=-1)

    def get_learnable_parameters(self):
        return list(self.student.parameters())

    def get_extra_parameters(self):
        return 0

    def _prototype_logits(self, features):
        features = F.normalize(features, dim=1)
        if self.prototypes.dim() == 3:
            num_classes, k_num, feat_dim = self.prototypes.shape
            proto_flat = self.prototypes.view(num_classes * k_num, feat_dim)
            sim = torch.matmul(features, proto_flat.t()).view(-1, num_classes, k_num)
            scaled_sim = sim / self.proto_temp
            logits = torch.logsumexp(scaled_sim, dim=2) - math.log(k_num)
        else:
            logits = torch.matmul(features, self.prototypes.t()) / self.proto_temp
        return logits

    def _soft_target_kd(self, logits_student, target_prob):
        log_prob = F.log_softmax(logits_student / self.temperature, dim=1)
        return F.kl_div(log_prob, target_prob, reduction='none').sum(1) * (self.temperature ** 2)

    def _label_adjust_proto_logits(self, proto_logits, target):
        """Use labels to suppress wrong prototype attractions before softmax."""
        if not self.proto_label_adjust:
            reliable = torch.ones(
                proto_logits.size(0), 1, dtype=torch.bool, device=proto_logits.device
            )
            return proto_logits, reliable

        target = target.to(proto_logits.device)
        target_index = target.view(-1, 1)
        true_logits = proto_logits.gather(1, target_index)

        target_mask = torch.zeros_like(proto_logits, dtype=torch.bool)
        target_mask.scatter_(1, target_index, True)

        excess = (proto_logits - true_logits + self.proto_label_margin).clamp_min(0.0)
        excess = excess.masked_fill(target_mask, 0.0)
        max_excess = excess.max(dim=1, keepdim=True).values

        adjusted_logits = proto_logits - self.proto_wrong_suppress * excess
        adjusted_logits = adjusted_logits.scatter_add(
            1, target_index, self.proto_true_boost * max_excess
        )

        reliable = max_excess.le(1e-12)
        return adjusted_logits, reliable

    def _swap_teacher_logits(self, logits, target):
        """Swap/shift true and strongest wrong logits when strong-view teacher is wrong."""
        if not self.logit_swap:
            return logits

        target = target.to(logits.device)
        target_index = target.view(-1, 1)
        true_logits = logits.gather(1, target_index)

        target_mask = torch.zeros_like(logits, dtype=torch.bool)
        target_mask.scatter_(1, target_index, True)
        wrong_logits = logits.masked_fill(target_mask, float('-inf'))
        wrong_index = wrong_logits.argmax(dim=1, keepdim=True)
        strongest_wrong = logits.gather(1, wrong_index)

        gap = (strongest_wrong - true_logits + self.logit_swap_margin).clamp_min(0.0)
        delta = self.logit_swap_alpha * gap

        adjusted_logits = logits.clone()
        adjusted_logits = adjusted_logits.scatter_add(1, target_index, delta)
        adjusted_logits = adjusted_logits.scatter_add(1, wrong_index, -delta)
        return adjusted_logits

    def forward_train(self, image, target, epoch=0, **kwargs):
        image_weak, image_strong = image

        # Teacher inference for weak and strong views.
        with torch.no_grad():
            logits_t_w = self.teacher(image_weak)
            if isinstance(logits_t_w, (list, tuple)):
                logits_t_w = logits_t_w[0]
            self.teacher_hook.clear()

            logits_t_s = self.teacher(image_strong)
            if isinstance(logits_t_s, (list, tuple)):
                logits_t_s = logits_t_s[0]
            f_s_t = self.teacher_hook.feature.clone()
            self.teacher_hook.clear()

            probs_t_s = F.softmax(logits_t_s.detach(), dim=1)
            entropy_t_s = -torch.sum(probs_t_s * torch.log(probs_t_s + 1e-8), dim=1)

        probs_t_w = F.softmax(logits_t_w.detach(), dim=1)
        conf_w = probs_t_w.max(dim=1).values
        mask_w = conf_w.ge(self.tau_w).float()

        # Prototype fusion weight: strong-view uncertainty.
        uncertainty_t_s = entropy_t_s / self.max_entropy
        lambda_i = (
            self.proto_min_weight
            + uncertainty_t_s.clamp(max=self.proto_max_weight - self.proto_min_weight)
        ).view(-1, 1)
        lambda_i = lambda_i.clamp(0.0, self.proto_max_weight)

        image_all = torch.cat([image_weak, image_strong], dim=0)
        logits_s_all = self.student(image_all)
        if isinstance(logits_s_all, (list, tuple)):
            logits_s_all = logits_s_all[0]
        self.student_hook.clear()

        logits_s_w, logits_s_s = logits_s_all.chunk(2)

        with torch.no_grad():
            logits_t_s_target = self._swap_teacher_logits(logits_t_s.detach(), target)
            prob_t_s_soft = F.softmax(logits_t_s_target / self.temperature, dim=1)
            prob_t_w_soft = F.softmax(logits_t_w.detach() / self.temperature, dim=1)
            if self.logit_swap and self.logit_swap_disable_proto:
                prob_t_s_prior = prob_t_s_soft.detach()
            else:
                proto_logits_s = self._prototype_logits(f_s_t)
                proto_logits_s, proto_reliable = self._label_adjust_proto_logits(proto_logits_s, target)
                prob_sim = F.softmax(proto_logits_s, dim=1)
                lambda_eff = lambda_i
                if self.proto_label_adjust and not self.proto_reliable_use_psim:
                    lambda_eff = lambda_i * (~proto_reliable).float()
                prob_t_s_prior = (
                    (1.0 - lambda_eff) * prob_t_s_soft
                    + lambda_eff * prob_sim
                ).detach()

        loss_ce = self.ce_loss_weight * (
            F.cross_entropy(logits_s_w, target) + F.cross_entropy(logits_s_s, target)
        )
        kd_w_to_weak = self._soft_target_kd(logits_s_w, prob_t_w_soft)
        kd_s_to_prior = self._soft_target_kd(logits_s_s, prob_t_s_prior)

        loss_kd_wv = self.kd_loss_weight * (
            (kd_w_to_weak + kd_s_to_prior)
            * mask_w
        ).mean()

        return logits_s_w, {
            "loss_ce": loss_ce,
            "loss_kd_wv": loss_kd_wv,
        }

    def forward(self, *args, **kwargs):
        if 'image_w' in kwargs and 'image_s' in kwargs and 'target' in kwargs:
            image = (kwargs['image_w'], kwargs['image_s'])
            target = kwargs['target']
            filtered_kwargs = {k: v for k, v in kwargs.items() if k not in ['image_w', 'image_s', 'target']}
            return self.forward_train(image, target, **filtered_kwargs)
        elif 'image' in kwargs:
            with torch.no_grad():
                logits = self.student(kwargs['image'])
                return logits[0] if isinstance(logits, (list, tuple)) else logits
        raise ValueError("Missing required inputs.")

distiller_dict["CRLD_Proto"] = CRLD_Proto


# ==============================================================================
# 6. Main entry
# ==============================================================================
def main(cfg, resume, opts):
    experiment_name = cfg.EXPERIMENT.NAME or cfg.EXPERIMENT.TAG
    tags = cfg.EXPERIMENT.TAG.split(",")
    if opts:
        additional_tags = ["{}:{}".format(k, v) for k, v in zip(opts[::2], opts[1::2])]
        tags += additional_tags
        experiment_name += ",".join(additional_tags)
    experiment_name = os.path.join(cfg.EXPERIMENT.PROJECT, experiment_name)

    cfg.defrost()
    cfg.LOG.SAVE_CHECKPOINT_FREQ = 9999
    cfg.freeze()

    show_cfg(cfg)
    train_loader, val_loader, num_data, num_classes = get_dataset_strong(cfg)

    print(log_msg("Loading models...", "INFO"))
    net, pretrain_model_path = cifar_model_dict[cfg.DISTILLER.TEACHER]
    model_teacher = net(num_classes=num_classes)
    model_teacher.load_state_dict(load_checkpoint(pretrain_model_path)["model"])
    model_student = cifar_model_dict[cfg.DISTILLER.STUDENT][0](num_classes=num_classes)

    model_teacher = model_teacher.cuda()
    model_student = model_student.cuda()

    dummy_size = 64 if cfg.DATASET.TYPE.startswith("tiny") else 32
    with torch.no_grad():
        hook = register_avgpool_hook(model_teacher)
        _ = model_teacher(torch.randn(1, 3, dummy_size, dummy_size).cuda())
        feat_dim = hook.feature.shape[1]
        hook.close()

        # Prototype construction parameters.
        tau_w = getattr(cfg.CRLD, 'PROTO_TAU_W', 0.80)
        K_subcenters = 3

        # 1. Build high-purity multi-center prototypes.
        prototypes_nk = precompute_multicenter_glvq_prototypes(
            teacher=model_teacher, train_loader=train_loader,
            num_classes=num_classes, K=K_subcenters, device='cuda',
            feature_dim=feat_dim, tau_w=tau_w
        )

        # 2. Evaluate prototype nearest-center accuracy.
        evaluate_prototypes_nk(model_teacher, val_loader, prototypes_nk, K=K_subcenters, device='cuda')

        # 3. Pass prototypes to the distiller.
        distiller = distiller_dict[cfg.DISTILLER.TYPE](
            model_student, model_teacher, cfg, precomputed_prototypes=prototypes_nk
        )
        distiller = torch.nn.DataParallel(distiller.cuda())

    trainer = trainer_dict[cfg.SOLVER.TRAINER](
        experiment_name, distiller, train_loader, val_loader, cfg
    )

    print(log_msg("Training started...", "INFO"))
    start_time = time.time()

    trainer.train(resume=resume)

    end_time = time.time()
    total_seconds = end_time - start_time
    hours = int(total_seconds // 3600)
    minutes = int((total_seconds % 3600) // 60)
    seconds = int(total_seconds % 60)
    print(log_msg(f"Total time: {hours}h {minutes}m {seconds}s", "INFO"))

    cleanup_useless_checkpoints(experiment_name)


if __name__ == "__main__":
    import multiprocessing

    multiprocessing.freeze_support()
    parser = argparse.ArgumentParser("CRLD-Proto training")
    parser.add_argument("--cfg", type=str, default="")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("opts", default=None, nargs=argparse.REMAINDER)
    args = parser.parse_args()

    cfg.defrost()
    cfg.set_new_allowed(True)
    if args.cfg:
        with open(args.cfg, 'r', encoding='utf-8') as f:
            cfg.merge_from_other_cfg(cfg.load_cfg(f))
    cfg.merge_from_list(args.opts)
    cfg.freeze()
    main(cfg, args.resume, args.opts)
