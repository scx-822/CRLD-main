# CRLD-Proto: 基于多中心原型先验的强增强鲁棒知识蒸馏

本项目在 CRLD（Cross-View Consistency Regularisation for Knowledge Distillation）的基础上，引入 **GLVQ 多中心类别原型**，用于修正教师模型在强增强视图下产生的不可靠软标签。

核心目标是解决以下问题：

> 在知识蒸馏中，教师模型面对强增强数据时会产生不可靠预测，严重影响一致性正则化的有效性。弱增强视图通常保持较高预测可靠性，因此可以利用弱视图构建纯净类别先验，对强视图教师目标进行动态修正。

## 方法概述

当前实验方案为：

```text
K-Means 多中心暖启动
+ GLVQ 原型判别式微调
+ log-sum-exp 多原型相似度融合
+ 强视图不确定性动态权重
+ 概率级原型先验融合
```

### 1. 原型构建

使用教师模型提取弱增强视图特征，仅保留满足以下条件的样本参与原型构建：

```text
1. 教师弱视图预测置信度 >= PROTO_TAU_W
2. 教师弱视图预测类别 = 真实类别
```

每个类别构建 `K=3` 个子原型：

```text
K-Means 初始化 -> GLVQ 离线微调 -> L2 归一化
```

### 2. 强视图原型先验

训练阶段，使用教师强视图特征与每类多个子原型计算余弦相似度，并通过 log-sum-exp 得到类别级原型 logit：

```text
g_{i,c} = log(1/K * sum_k exp(cos(f_s^T, p_{c,k}) / tau_p))
```

随后通过 Softmax 得到类别级原型先验分布：

```text
q_i = Softmax(g_i)
```

### 3. 动态融合权重

基础版本使用教师强视图预测熵计算动态原型融合权重：

```text
lambda_i = H(p_s^T) / log(C)
```

当前 `res32x4_res8x4.yaml` 额外加入弱强视图预测差异项：

```text
lambda_i = H(p_s^T) / log(C) + eta * JS(p_w^T, p_s^T)
```

其中 `eta` 对应：

```yaml
PROTO_JS_WEIGHT: 0.5
```

最终强视图教师目标分布为：

```text
p_bar_s^T = (1 - lambda_i) * p_s^T + lambda_i * q_i
```

也就是说，强视图越不可靠、弱强预测差异越大，原型先验的修正权重越高。

## 主要代码

```text
tools/train_proto.py
```

关键模块：

| 函数 / 类 | 作用 |
| --- | --- |
| `precompute_multicenter_glvq_prototypes` | 训练前构建 K-Means + GLVQ 多中心原型 |
| `evaluate_prototypes_nk` | 评估多中心原型最近中心分类准确率 |
| `CRLD_Proto._prototype_logits` | 计算强视图教师特征与原型的类别级 logit |
| `CRLD_Proto.forward_train` | 执行原型先验融合并计算 CE / KD 损失 |

## 实验配置

### ResNet32x4 -> ResNet8x4

当前默认实验配置：

```text
configs/cifar100/crld_proto/res32x4_res8x4.yaml
```

关键参数：

| 参数 | 当前值 | 说明 |
| --- | ---: | --- |
| `TEACHER` | `resnet32x4` | 教师模型 |
| `STUDENT` | `resnet8x4` | 学生模型 |
| `TEMPERATURE` | 4.0 | 蒸馏温度 |
| `TAU_W` | 0.8 | 训练阶段弱视图 KD mask 阈值 |
| `PROTO_TAU_W` | 0.80 | 原型构建阶段弱视图筛选阈值 |
| `PROTO_TEMP` | 0.5 | 原型相似度温度 |
| `PROTO_MIN_WEIGHT` | 0.0 | 原型融合权重下限 |
| `PROTO_MAX_WEIGHT` | 1.0 | 原型融合权重上限 |
| `PROTO_JS_WEIGHT` | 0.5 | 弱强视图 JS 差异权重 |

输出目录：

```text
output/cifar100_baselines/crld_proto,res32x4,res8x4,jslambda05
```

### ResNet56 -> ResNet20

稳定对照配置：

```text
configs/cifar100/crld_proto/res56_res20.yaml
```

该配置未启用 `PROTO_JS_WEIGHT`，对应基础 CRLD-Proto：

```text
GLVQ 多中心原型 + log-sum-exp + 概率级原型先验融合
```

历史最好记录：

```text
best_acc = 72.15
```

输出目录：

```text
output/cifar100_baselines/crld_proto,res56,res20
```

## 运行方法

### 1. 环境安装

```powershell
pip install -r requirements.txt
pip install -e .
```

### 2. 准备数据和教师模型

由于 GitHub 不适合存放大文件，本仓库不会上传以下目录：

```text
data/
download_ckpts/
output/
```

请在本地准备 CIFAR-100 数据和教师模型权重。目录结构可参考原 mdistiller / CRLD 工程。

### 3. 训练当前实验

```powershell
python tools/train_proto.py --cfg configs/cifar100/crld_proto/res32x4_res8x4.yaml
```

训练 ResNet56 -> ResNet20：

```powershell
python tools/train_proto.py --cfg configs/cifar100/crld_proto/res56_res20.yaml
```

## 输出文件

训练结果默认保存在：

```text
output/{PROJECT}/{TAG}
```

例如：

```text
output/cifar100_baselines/crld_proto,res32x4,res8x4,jslambda05
```

常见输出：

| 文件 | 说明 |
| --- | --- |
| `worklog.txt` | 训练日志，包含每个 epoch 的准确率和 loss |
| `best` / `student_best` | 最佳模型权重 |
| `latest` / `student_latest` | 最新模型权重 |
| `train.events/` | TensorBoard 日志 |

这些训练产物已经被 `.gitignore` 忽略，不会上传到 GitHub。

## GitHub 上传说明

仓库中已忽略大文件目录：

```text
data/
download_ckpts/
output/
```

如果后续需要分享模型权重，建议使用：

```text
1. GitHub Releases
2. Git LFS
3. 网盘链接
```

不要直接把 checkpoint、数据集、TensorBoard 日志提交到 Git。

## 致谢

本项目基于以下工作和代码框架修改：

- CRLD: Cross-View Consistency Regularisation for Knowledge Distillation
- mdistiller
- NormKD
- MLLD

原始 CRLD 引用：

```bibtex
@inproceedings{crld,
author = {Weijia Zhang and Dongnan Liu and Weidong Cai and Chao Ma},
title = {Cross-View Consistency Regularisation for Knowledge Distillation},
booktitle = {ACM MM},
year = {2024}
}
```

