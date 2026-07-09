# CRLD-Proto 实验记录

更新时间：2026-07-05

## 统一实验流程

1. 使用 CIFAR-100 双视图训练流程，弱视图 `x_w` 和强视图 `x_s` 同时输入学生网络。
2. 使用预训练教师模型提取弱视图教师特征 `f_w^T`。
3. 构建原型时，仅使用满足以下条件的可靠样本：

$$
\mathrm{Mask}_i
=
\mathbb{I}\left(\mathrm{Conf}_{w,i}^{T}\ge \tau_w\right)
\cdot
\mathbb{I}\left(\hat{y}_{w,i}^{T}=y_i\right)
$$

LaTeX 源码：

```latex
\mathrm{Mask}_i
=
\mathbb{I}\left(\mathrm{Conf}_{w,i}^{T}\ge \tau_w\right)
\cdot
\mathbb{I}\left(\hat{y}_{w,i}^{T}=y_i\right)
```

4. 对每个类别的可靠弱视图教师特征执行 K-Means，得到每类 `K=3` 个子原型。
5. 使用 GLVQ 对多中心原型进行判别式微调，并对原型做 L2 归一化。
6. 训练学生时使用联合 CE 和 KD：
   - 弱视图学生对齐弱视图教师软标签。
   - 强视图学生对齐原型先验修正后的强视图教师软标签。
7. 当前主线默认使用概率级融合：

$$
\hat{p}_{s,i}^{T}
=
(1-\lambda_i)\widetilde{p}_{s,i}^{T}
+
\lambda_i p_{\mathrm{sim},i}
$$

LaTeX 源码：

```latex
\hat{p}_{s,i}^{T}
=
(1-\lambda_i)\widetilde{p}_{s,i}^{T}
+
\lambda_i p_{\mathrm{sim},i}
```

## 参数说明

| 参数                 | 含义                                                        |
| -------------------- | ----------------------------------------------------------- |
| `TEMPERATURE`      | KD 蒸馏温度`T`，控制教师软标签平滑程度                    |
| `TAU_W`            | 训练 KD 时弱视图教师置信度阈值                              |
| `PROTO_TEMP`       | 原型温度`tau_p`，控制原型相似度分布尖锐程度               |
| `PROTO_AGG_MODE`   | 多原型聚合方式，`logsumexp` 为软聚合，`max` 为 hard max |
| `PROTO_MAX_WEIGHT` | 原型先验融合权重`lambda_i` 的最大值                       |
| `PROTO_MIN_WEIGHT` | 原型先验融合权重`lambda_i` 的最小值                       |
| `PROTO_TAU_W`      | 构建原型时的弱视图教师置信度阈值                            |
| `CE_WEIGHT`        | 交叉熵损失权重                                              |
| `KD_WEIGHT`        | KD 损失权重                                                 |

## ResNet32x4 -> ResNet8x4 主线实验

| 实验                            | 配置/Tag                                                | 核心改动                                                                          | 关键参数                                                                                     |      Best Acc | 记录来源                        | 结论                                                             |
| ------------------------------- | ------------------------------------------------------- | --------------------------------------------------------------------------------- | -------------------------------------------------------------------------------------------- | ------------: | ------------------------------- | ---------------------------------------------------------------- |
| Base 历史最好                   | `crld_proto,res32x4,res8x4`                           | 多中心原型 + log-sum-exp 聚合 + 概率级融合                                        | `T=4.0`, `TAU_W=0.8`, `PROTO_TEMP=0.5`, `PROTO_MAX_WEIGHT=1.0`, `PROTO_TAU_W=0.80` |         76.95 | `worklog.txt` 历史记录        | 当前日志中最高结果                                               |
| Base 稳定版本                   | `crld_proto,res32x4,res8x4`                           | 同上                                                                              | 同上                                                                                         |         76.76 | `worklog.txt` 历史记录        | 后续多数消融对照的 base                                          |
| 原型一致性门控                  | `crld_proto,res32x4,res8x4,ablate_consistency`        | 仅当原型预测与弱视图教师一致时使用原型先验                                        | base +`PROTO_CONSISTENCY_GATE=True`                                                        |         76.10 | 口头/消融记录                   | 低于 base，硬门控削弱有效修正                                    |
| 一致性 + margin 门控            | `crld_proto,res32x4,res8x4,ablate_consistency_margin` | 增加原型 top-1/top-2 margin 过滤                                                  | base + consistency + margin                                                                  |         76.55 | 口头/消融记录                   | 仍低于 base                                                      |
| 一致性 + margin + 自适应 lambda | `crld_proto,res32x4,res8x4,ablate_full`               | 再按原型置信度缩放`lambda_i`                                                    | base + consistency + margin + adaptive lambda                                                |         76.40 | 口头/消融记录                   | 仍低于 base                                                      |
| Hard max 聚合                   | `crld_proto,res32x4,res8x4,hardmax`                   | 用最大子原型相似度替换 log-sum-exp 聚合                                           | `PROTO_AGG_MODE=max`, `PROTO_TEMP=0.5`                                                   |         76.20 | 口头记录                        | 低于 log-sum-exp，说明软聚合更稳                                 |
| Soft refine 多样本加权          | `softrefine`                                          | 错误/低可信样本按教师类别概率软加权加入原型                                       | `PROTO_REFINE_CONF_POWER` 调参                                                             | 76.67 / 76.48 | 口头记录                        | 覆盖率提高但污染原型，效果下降                                   |
| Logit-prior 消融 1              | `crld_proto,res32x4,res8x4,logit_beta05_lmax05`       | 原型先验加到教师 logits，而非概率直接相加                                         | `PROTO_LOGIT_SCALE=0.5`, `PROTO_MAX_WEIGHT=0.5`                                          |        待记录 | 输出目录存在，暂未读到 best_acc | 待训练完成后补充                                                 |
| Raw-logit hardmax               | `crld_proto,res32x4,res8x4,rawlogit_hardmax_tp005`    | 去掉两项 softmax，使用`(1-lambda) z_s^T/T + lambda max(cos/tau_p)` 后再 softmax | `PROTO_AGG_MODE=max`, `PROTO_FUSION_MODE=raw_logit`, `PROTO_TEMP=0.05`                 |        待记录 | 新配置                          | `tau_p=0.5` 时原型项约为教师项 1/10，因此按量级对齐调小到 0.05 |
| Z-Score logit 对齐              | `crld_proto,res32x4,res8x4,rawlogit_hardmax_tp005`    | 去掉原型温度，将 hard-max cosine 按样本级 Z-Score 对齐到教师 logits/T 空间        | `PROTO_AGG_MODE=max`, `PROTO_FUSION_MODE=zscore_logit`, 无 `PROTO_TEMP`                |         76.53 | `worklog.txt`                 | 低于 raw-logit hardmax，对低方差样本加保护                       |
| Z-Score logit 对齐 + 低方差保护 | `crld_proto,res32x4,res8x4,rawlogit_hardmax_tp005`    | 当原型相似度分布过平时按`sigma_s/rho` 缩小有效 `lambda`                       | `PROTO_ZSCORE_MIN_STD=0.05`                                                                |        待记录 | 新配置                          | 当前待跑                                                         |

Raw-logit hardmax 训练结束后，会额外生成：

```text
output/cifar100_baselines/crld_proto,res32x4,res8x4,rawlogit_hardmax_tp005/raw_logit_diagnostics.md
```

该文件记录四组 logit 分数统计：

```text
teacher_raw = z_s^T / T
proto_raw = max_k cos(f_s^T, p_c,k) / tau_p
teacher_weighted = (1-lambda) * z_s^T / T
proto_weighted = lambda * max_k cos(f_s^T, p_c,k) / tau_p
```

## 其他模型组合实验

| 配置/Tag                         | Teacher -> Student    | 关键参数                                                                                    | 当前记录 Acc | 对应基线 Acc | 记录来源        | 结论                 |
| -------------------------------- | --------------------- | ------------------------------------------------------------------------------------------- | -----------: | -----------: | --------------- | -------------------- |
| `crld_proto,res110,res32`      | ResNet110 -> ResNet32 | `T=4.0`, `TAU_W=0.8`, `PROTO_TEMP=0.5`, `PROTO_MAX_WEIGHT=1.0`, `KD_WEIGHT=1.0`   |        74.52 |        74.42 | `worklog.txt` | 略高于基线           |
| `crld_proto,wrn_40_2,wrn_16_2` | WRN-40-2 -> WRN-16-2  | `T=4.0`, `TAU_W=0.85`, `PROTO_TEMP=0.45`, `PROTO_MAX_WEIGHT=1.0`, `KD_WEIGHT=1.2` |        76.03 |        76.45 | `worklog.txt` | 仍低于基线           |
| `crld_proto,wrn_40_2,wrn_40_1` | WRN-40-2 -> WRN-40-1  | `T=4.0`, `TAU_W=0.8`, `PROTO_TEMP=0.6`, `PROTO_MAX_WEIGHT=0.7`, `KD_WEIGHT=1.0`   |        75.34 |        75.58 | `worklog.txt` | 仍低于基线           |
| `crld_proto,vgg13,vgg8`        | VGG13 -> VGG8         | `T=4.0`, `TAU_W=0.8`, `PROTO_TEMP=0.7`, `PROTO_MAX_WEIGHT=0.6`, `KD_WEIGHT=1.0`   |        74.98 |        75.27 | `worklog.txt` | 仍低于基线           |
| `crld_proto,res56,res20`       | ResNet56 -> ResNet20  | `T=3.0`, `TAU_W=0.8`, `PROTO_TEMP=0.5`, `PROTO_MAX_WEIGHT=1.0`, `KD_WEIGHT=1.0`   |        72.15 |        72.10 | `worklog.txt` | 当前最好记录为 72.15 |

## 当前 hard max 配置

文件：`configs/cifar100/crld_proto/res32x4_res8x4.yaml`

```yaml
EXPERIMENT:
  TAG: "crld_proto,res32x4,res8x4,hardmax"

CRLD:
  TEMPERATURE: 4.0
  TAU_W: 0.8
  PROTO_TEMP: 0.5
  PROTO_AGG_MODE: "max"
  PROTO_MAX_WEIGHT: 1.0
  PROTO_MIN_WEIGHT: 0.0
  PROTO_TAU_W: 0.80
  LOSS:
    CE_WEIGHT: 1.0
    KD_WEIGHT: 1.0
```

运行命令：

```powershell
python tools/train_proto.py --cfg configs/cifar100/crld_proto/res32x4_res8x4.yaml
```

## 后续记录模板

| 日期       | 配置/Tag | Teacher -> Student | 实验改动 | 关键参数 | Best Acc | 结论 |
| ---------- | -------- | ------------------ | -------- | -------- | -------: | ---- |
| 2026-07-05 |          |                    |          |          |          |      |
