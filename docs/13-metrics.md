---
title: 评估指标
nav_order: 13
---

# 评估指标

## 模块结构

```
tzrec/metrics/
├── __init__.py
├── grouped_auc.py                 # GroupedAUC
├── xauc.py                        # XAUC
├── grouped_xauc.py                # GroupedXAUC
├── decay_auc.py                   # DecayAUC
├── recall_at_k.py                 # Recall@K
├── normalized_entropy.py          # NormalizedEntropy
├── unique_ratio.py                # UniqueRatio（语义 ID 多样性）
└── train_metric_wrapper.py        # TrainMetricWrapper
```

## GroupedAUC

[`torcheasyrec/tzrec/metrics/grouped_auc.py:22`](../torcheasyrec/tzrec/metrics/grouped_auc.py#L22)

按 group 分组计算 AUC 后取平均：

```
GAUC = Σ(AUC(group_i)) / N_groups
```

特性：

- 支持分布式（多 GPU）规约
- 按 `group_id` 分组，各组独立计算 AUC 后 mean
- 支持在线增量更新

## XAUC

[`torcheasyrec/tzrec/metrics/xauc.py:74`](../torcheasyrec/tzrec/metrics/xauc.py#L74)

交叉 AUC，专为短视频推荐设计的非个性化指标。

**原理**：从正负样本中随机采样配对，计算正样本得分高于负样本的比例。

```
XAUC = P(score_pos > score_neg)
```

复杂度 `O(n²)`，必须采样：

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| sample_ratio | float | — | 采样比例 |
| max_pairs | int | — | 最大对数 |
| in_batch | bool | false | 是否只在本 batch 内计算 |

## GroupedXAUC

[`torcheasyrec/tzrec/metrics/grouped_xauc.py`](../torcheasyrec/tzrec/metrics/grouped_xauc.py)

GroupedAUC + XAUC 组合：先按 group 分组，在组内计算 XAUC，再取平均。

## DecayAUC

[`torcheasyrec/tzrec/metrics/decay_auc.py`](../torcheasyrec/tzrec/metrics/decay_auc.py)

训练过程中用于监控的衰减 AUC，使用指数移动平均：

```python
class DecayAUC(Metric):
    """Decay AUC with exponential moving average."""
```

适用于训练期间不稳定环境的平滑指标追踪。

## Recall@K

[`torcheasyrec/tzrec/metrics/recall_at_k.py`](../torcheasyrec/tzrec/metrics/recall_at_k.py)

Top-K 召回率：

```
Recall@K = (# of correct in top K) / (# of total positives)
```

```python
def recall_at_k(preds: torch.Tensor, targets: torch.Tensor, k: int) -> torch.Tensor:
    ...
```

## NormalizedEntropy

[`torcheasyrec/tzrec/metrics/normalized_entropy.py:19`](../torcheasyrec/tzrec/metrics/normalized_entropy.py#L19)

归一化熵，模型交叉熵除以基线（按全局平均标签率的常数预测器）的交叉熵：

```python
class NormalizedEntropy(Metric):
    def update(self, preds, target):
        ce = F.binary_cross_entropy(preds, labels, reduction="none")
        self.cross_entropy_sum += ce.sum()

    def compute(self):
        mean_label = (self.sum_labels / self.num_samples).clamp(eta, 1 - eta)
        ce_norm = -(sum_labels * log(mean_label) + (num_samples - sum_labels) * log(1 - mean_label))
        return cross_entropy_sum / ce_norm
```

NE < 1 表示模型优于基线常数预测器。常用于分类模型对比。

## UniqueRatio

[`torcheasyrec/tzrec/metrics/unique_ratio.py:16`](../torcheasyrec/tzrec/metrics/unique_ratio.py#L16)

每 batch 不同行比例的均值，衡量语义 ID 多样性：

```python
class UniqueRatio(Metric):
    def update(self, codes):
        unique = torch.unique(codes, dim=0).shape[0]
        self.ratio_sum += unique / batch_size
```

注意：这是**多样性的轻量代理**（仅计算每个 batch 内的 distinct 比例），不是全局 codebook 覆盖率。

## TrainMetricWrapper

[`torcheasyrec/tzrec/metrics/train_metric_wrapper.py`](../torcheasyrec/tzrec/metrics/train_metric_wrapper.py)

用于训练期间指标收集的包装器：

```python
class TrainMetricWrapper(Metric):
    def __init__(self, metric, decay_rate: float, decay_step: int):
        # decay_rate: EMA 衰减率
        # decay_step: EMA 衰减步数
```

包装标准 metric，叠加指数移动平均以平滑训练曲线。

## 标准指标（torchmetrics）

框架还通过 torchmetrics 使用以下标准指标（见 [05-训练流程](05-training-flow)）：

| 指标 | torchmetrics 类 | 场景 |
|------|-----------------|------|
| AUC | `AUROC` | 通用排序 |
| MAE | `MeanAbsoluteError` | 回归 |
| MSE | `MeanSquaredError` | 回归 |
| Accuracy | `Accuracy` | 分类 |

## 场景推荐

| 场景 | 推荐指标 |
|------|----------|
| 通用排序 | AUC, GAUC |
| 短视频/信息流 | XAUC |
| 召回任务 | Recall@K |
| 训练监控 | DecayAUC |
| 多任务学习 | 各任务独立 AUC |
| 模型对比 | NormalizedEntropy |
| 语义 ID 多样性 | UniqueRatio |

## 配置示例

```protobuf
metrics {
    auc {}
}

metrics {
    grouped_auc {}
}

metrics {
    xauc {
        sample_ratio: 0.001
        max_pairs: 1000000
        in_batch: false
    }
}

metrics {
    recall_at_k {
        k: 10
    }
}
```

训练指标包装：

```protobuf
train_metrics {
    auc {}
    decay_rate: 0.9
    decay_step: 1000
}
```

## 关键文件

| 文件 | 行数 | 功能 |
|------|------|------|
| `metrics/grouped_auc.py` | 125 | 分组 AUC |
| `metrics/xauc.py` | ~173 | 交叉 AUC |
| `metrics/grouped_xauc.py` | ~180 | 分组交叉 AUC |
| `metrics/decay_auc.py` | ~80 | 衰减 AUC |
| `metrics/recall_at_k.py` | ~60 | Recall@K |
| `metrics/normalized_entropy.py` | 72 | 归一化熵 |
| `metrics/unique_ratio.py` | 50 | 唯一比率（语义 ID） |
| `metrics/train_metric_wrapper.py` | ~80 | 训练指标包装 |

依赖：`torch`、`torchmetrics`、`torch.distributed`。
