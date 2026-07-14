# CRLD-Proto 实验记录

更新时间：2026-07-14

## 76.80 版改进

### 改进动机

原始 CRLD-Proto 使用强视图教师特征与类别原型计算相似度分布 `p_sim`，再修正强视图教师软标签。如果强增强已经使教师特征偏向错误类别，那么 `p_sim` 也可能把较高概率分配给错误类，继续融合会放大错误。因此，76.80 版不再根据强视图特征生成的原型分布修正软标签，而是直接对教师强视图 logits 做最小幅度的标签感知校正。

该版本只关心最终用于蒸馏的 logits 排序是否合理，不改变教师网络参数，也不反向更新教师内部特征。

### Soft logit correction

对于第 $i$ 个训练样本，教师在强视图下输出 logits：

$$
z_{s,i}^{T}\in\mathbb{R}^{C},
$$

真实标签为 $y_i$。首先找到除真实类别之外 logit 最大的类别：

$$
c_i^{-}
=
\arg\max_{c\neq y_i}z_{s,i,c}^{T}.
$$

定义真实类相对最强错误类尚缺少的 margin：

$$
d_i
=
\left[
z_{s,i,c_i^{-}}^{T}
-
z_{s,i,y_i}^{T}
+
m
\right]_{+},
$$

其中 $[x]_{+}=\max(0,x)$，当前设置 $m=0.05$。随后仅调整真实类和最强错误类两个维度：

$$
\bar{z}_{s,i,y_i}^{T}
=
z_{s,i,y_i}^{T}+\alpha d_i,
$$

$$
\bar{z}_{s,i,c_i^{-}}^{T}
=
z_{s,i,c_i^{-}}^{T}-\alpha d_i,
$$

$$
\bar{z}_{s,i,c}^{T}=z_{s,i,c}^{T},
\qquad
\forall c\neq y_i,\quad c\neq c_i^{-}.
$$

当前设置 $\alpha=0.5$。因此，当校正被触发时：

$$
\bar{z}_{s,i,y_i}^{T}
-
\bar{z}_{s,i,c_i^{-}}^{T}
=m=0.05.
$$

同时有：

$$
\bar{z}_{s,i,y_i}^{T}
+
\bar{z}_{s,i,c_i^{-}}^{T}
=
z_{s,i,y_i}^{T}
+
z_{s,i,c_i^{-}}^{T},
$$

即两个类别的 logit 总量保持不变，只对它们的相对排序做最小对称修正。其余 $C-2$ 个类别 logits 完全不变，从而尽量保留教师分布中的暗知识。

校正触发条件为：

$$
z_{s,i,y_i}^{T}
-
z_{s,i,c_i^{-}}^{T}
<m.
$$

因此，该版本不仅修正教师强视图 top-1 预测错误的样本，也会轻微修正真实类已经为 top-1、但领先最强错误类不足 `0.05` 的边界样本。如果真实类原本已经至少领先 `0.05`，则 $d_i=0$，强视图 logits 保持不变。

### 蒸馏目标

校正后再除以蒸馏温度 $T=4.0$ 并执行 Softmax：

$$
\bar{p}_{s,i}^{T}
=
\operatorname{Softmax}
\left(
\frac{\bar{z}_{s,i}^{T}}{T}
\right).
$$

强视图学生使用该分布作为 KD 目标：

$$
\mathcal{L}_{KD}^{s}
=
T^{2}
\operatorname{KL}
\left(
\bar{p}_{s,i}^{T}
\;\|\;
\operatorname{Softmax}
\left(
\frac{z_{s,i}^{S}}{T}
\right)
\right).
$$

`LOGIT_SWAP_DISABLE_PROTO=True` 表示这条强视图分支不再融合 `p_sim`，原型融合权重 $\lambda_i$ 在该分支中不生效。弱视图 KD、强弱视图 CE、弱视图置信度 Mask、训练计划和优化器设置均保持不变。当前训练入口仍会预计算并评估原型，但这些原型不参与 76.80 版的强视图 KD target。

### 与 hard swap 的区别及结果

hard swap 使用 $\alpha=1.0$，会直接交换真实类和最强错误类的 logits，校正幅度较大；soft correction 使用 $\alpha=0.5$，只把二者投影到真实类领先 `0.05` 的最小 margin 状态。

| 方法            | $\alpha$ | $m$ | 原型融合 |        Best Acc |
| --------------- | ---------: | ----: | -------- | --------------: |
| Hard swap       |        1.0 |  0.00 | 关闭     |           76.54 |
| Soft correction |        0.5 |  0.05 | 关闭     | **76.80** |
| Base 历史最好   |          - |     - | 开启     | **76.95** |

Soft correction 相比 hard swap 提升 `0.26` 个百分点，说明保持教师 logits 结构并进行小幅排序校正比直接交换更稳定；但仍比历史最好低 `0.15` 个百分点。该次训练的峰值出现在第 235 轮，Best Acc 为 `76.80%`，第 240 轮准确率为 `76.62%`。

## Base 统一实验流程

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
6. Base 训练学生时使用联合 CE 和 KD：
   - 弱视图学生对齐弱视图教师软标签。
   - 强视图学生对齐原型先验修正后的强视图教师软标签。
7. Base 主线使用概率级融合：

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

| 实验                            | 配置/Tag                                                 | 核心改动                                                                   | 关键参数                                                                                     | Best Acc | 记录来源                 | 结论                                       |
| ------------------------------- | -------------------------------------------------------- | -------------------------------------------------------------------------- | -------------------------------------------------------------------------------------------- | -------: | ------------------------ | ------------------------------------------ |
| Base 历史最好                   | `crld_proto,res32x4,res8x4`                            | 多中心原型 + log-sum-exp 聚合 + 概率级融合                                 | `T=4.0`, `TAU_W=0.8`, `PROTO_TEMP=0.5`, `PROTO_MAX_WEIGHT=1.0`, `PROTO_TAU_W=0.80` |    76.95 | `worklog.txt` 历史记录 | 当前确认的历史最好结果                     |
| Logit hard swap                 | `crld_proto,res32x4,res8x4,logit_swap`                 | 强视图教师 logits 中交换真实类与最强错误类，关闭原型融合                   | `LOGIT_SWAP_ALPHA=1.0`, `LOGIT_SWAP_MARGIN=0.0`                                          |    76.54 | 实验记录                 | 过度校正，低于 base                        |
| Logit soft correction           | `crld_proto,res32x4,res8x4,logit_soft_a05_m005`        | 按错误类与真实类的 logit 间隔进行软校正，关闭原型融合                      | `LOGIT_SWAP_ALPHA=0.5`, `LOGIT_SWAP_MARGIN=0.05`                                         |    76.80 | `worklog.txt`          | 高于 hard swap，但仍低于历史最好           |
| Logit soft + prototype gate     | `crld_proto,res32x4,res8x4,logit_soft_proto_gate_l025` | 软 logits 校正，并以原型可靠性门控小权重原型先验                           | `PROTO_MAX_WEIGHT=0.25`                                                                    |    76.76 | `worklog.txt`          | 原型小辅助未带来额外提升                   |
| 原型一致性门控                  | `crld_proto,res32x4,res8x4,ablate_consistency`         | 仅当原型预测与弱视图教师一致时使用原型先验                                 | base +`PROTO_CONSISTENCY_GATE=True`                                                        |    76.10 | 口头/消融记录            | 低于 base，硬门控削弱有效修正              |
| 一致性 + margin 门控            | `crld_proto,res32x4,res8x4,ablate_consistency_margin`  | 增加原型 top-1/top-2 margin 过滤                                           | base + consistency + margin                                                                  |    76.55 | 口头/消融记录            | 仍低于 base                                |
| 一致性 + margin + 自适应 lambda | `crld_proto,res32x4,res8x4,ablate_full`                | 再按原型置信度缩放`lambda_i`                                             | base + consistency + margin + adaptive lambda                                                |    76.40 | 口头/消融记录            | 仍低于 base                                |
| Hard max 聚合                   | `crld_proto,res32x4,res8x4,hardmax`                    | 用最大子原型相似度替换 log-sum-exp 聚合                                    | `PROTO_AGG_MODE=max`, `PROTO_TEMP=0.5`                                                   |    76.20 | 口头记录                 | 低于 log-sum-exp，说明软聚合更稳           |
| Soft refine 多样本加权          | `softrefine`                                           | 错误/低可信样本按教师类别概率软加权加入原型                                | `PROTO_REFINE_CONF_POWER` 调参                                                             |    76.67 | 口头记录                 | 覆盖率提高但污染原型，效果下降             |
| Z-Score logit 对齐              | `crld_proto,res32x4,res8x4,rawlogit_hardmax_tp005`     | 去掉原型温度，将 hard-max cosine 按样本级 Z-Score 对齐到教师 logits/T 空间 | `PROTO_AGG_MODE=max`, `PROTO_FUSION_MODE=zscore_logit`, 无 `PROTO_TEMP`                |    76.53 | `worklog.txt`          | 低于 raw-logit hardmax，对低方差样本加保护 |

## 其他模型组合实验

| 配置/Tag                         | Teacher -> Student    | 关键参数                                                                                    | 当前记录 Acc | 对应基线 Acc | 记录来源        | 结论                 |
| -------------------------------- | --------------------- | ------------------------------------------------------------------------------------------- | -----------: | -----------: | --------------- | -------------------- |
| `crld_proto,res110,res32`      | ResNet110 -> ResNet32 | `T=4.0`, `TAU_W=0.8`, `PROTO_TEMP=0.5`, `PROTO_MAX_WEIGHT=1.0`, `KD_WEIGHT=1.0`   |        74.52 |        74.42 | `worklog.txt` | 略高于基线           |
| `crld_proto,wrn_40_2,wrn_16_2` | WRN-40-2 -> WRN-16-2  | `T=4.0`, `TAU_W=0.85`, `PROTO_TEMP=0.45`, `PROTO_MAX_WEIGHT=1.0`, `KD_WEIGHT=1.2` |        76.03 |        76.45 | `worklog.txt` | 仍低于基线           |
| `crld_proto,wrn_40_2,wrn_40_1` | WRN-40-2 -> WRN-40-1  | `T=4.0`, `TAU_W=0.8`, `PROTO_TEMP=0.6`, `PROTO_MAX_WEIGHT=0.7`, `KD_WEIGHT=1.0`   |        75.34 |        75.58 | `worklog.txt` | 仍低于基线           |
| `crld_proto,vgg13,vgg8`        | VGG13 -> VGG8         | `T=4.0`, `TAU_W=0.8`, `PROTO_TEMP=0.7`, `PROTO_MAX_WEIGHT=0.6`, `KD_WEIGHT=1.0`   |        74.98 |        75.27 | `worklog.txt` | 仍低于基线           |
| `crld_proto,res56,res20`       | ResNet56 -> ResNet20  | `T=3.0`, `TAU_W=0.8`, `PROTO_TEMP=0.5`, `PROTO_MAX_WEIGHT=1.0`, `KD_WEIGHT=1.0`   |        72.15 |        72.10 | `worklog.txt` | 当前最好记录为 72.15 |
