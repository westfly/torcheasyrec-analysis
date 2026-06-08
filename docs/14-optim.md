---
title: 优化器与学习率调度器
nav_order: 14
---

# 优化器与学习率调度器

## 模块结构

```
tzrec/optim/
├── __init__.py
├── optimizer.py              # TZRecOptimizer 封装
├── optimizer_builder.py      # 优化器构建器
└── lr_scheduler.py           # 学习率调度器
```

## 优化器类型

### 稀疏优化器（Embedding）

[`torcheasyrec/tzrec/optim/optimizer_builder.py:30-97`](../torcheasyrec/tzrec/optim/optimizer_builder.py#L30-L97)

用于 embedding 参数的优化器，通过 `apply_optimizer_in_backward` 在反向传播中原地更新：

| proto 类型 | TorchRec 类 | 备注 |
|------------|-------------|------|
| `sgd_optimizer` | `optimizers.SGD` | 随机梯度下降 |
| `adagrad_optimizer` | `optimizers.Adagrad` | Adagrad，通过 FBGEMM_MOMENTUM1_STATE_INIT_VALUE 支持 initial_accumulator_value |
| `adam_optimizer` | `optimizers.Adam` | Adam |
| `lars_sgd_optimizer` | `optimizers.LarsSGD` | LARS |
| `lamb_optimizer` | `optimizers.LAMB` | LAMB |
| `partial_rowwise_lamb_optimizer` | `optimizers.PartialRowWiseLAMB` | 行级 LAMB |
| `partial_rowwise_adam_optimizer` | `optimizers.PartialRowWiseAdam` | 行级 Adam |
| `rowwise_adagrad_optimizer` | `rowwise_adagrad.RowWiseAdagrad` | 行级 Adagrad |
| `adadelta_optimizer` | `optimizers.AdaDelta` | 需内部维护版 torchrec |
| `rmsprop_optimizer` | `optimizers.RMSProp` | 需内部维护版 torchrec |

Adagrad 的 `initial_accumulator_value` tensorflow 兼容性补丁见 [`torcheasyrec/tzrec/optim/optimizer.py:75-237`](../torcheasyrec/tzrec/optim/optimizer.py#L75-L237) — `apply_split_helper` monkey-patch 用 `FBGEMM_MOMENTUM1_STATE_INIT_VALUE` 环境变量初始化 momentum1 状态。

### 稠密优化器（MLP）

[`torcheasyrec/tzrec/optim/optimizer_builder.py:100-136`](../torcheasyrec/tzrec/optim/optimizer_builder.py#L100-L136)

用于 MLP 等稠密参数的优化器，标准 `torch.optim`：

| proto 类型 | PyTorch 类 |
|------------|------------|
| `sgd_optimizer` | `torch.optim.SGD` |
| `adagrad_optimizer` | `torch.optim.Adagrad` |
| `adam_optimizer` | `torch.optim.Adam` |
| `adamw_optimizer` | `torch.optim.AdamW` |
| `adadelta_optimizer` | `torch.optim.Adadelta` |
| `rmsprop_optimizer` | `torch.optim.RMSprop` |

### 参数分组优化器

支持为不同参数组指定不同优化器（通过 `part_optimizers` 和正则匹配），见 [`torcheasyrec/tzrec/optim/optimizer_builder.py:139-260`](../torcheasyrec/tzrec/optim/optimizer_builder.py#L139-L260)。

## TZRecOptimizer

[`torcheasyrec/tzrec/optim/optimizer.py:26`](../torcheasyrec/tzrec/optim/optimizer.py#L26)

优化器顶层封装，处理梯度累积和混合精度：

```python
class TZRecOptimizer(OptimizerWrapper):
    def step(self, closure=None):
        if self._grad_scaler is not None:
            self._grad_scaler.step(self._optimizer)
            self._grad_scaler.update()
        else:
            self._optimizer.step()
```

完整的优化器层级和构建流程见 [05-训练流程 ⇒ Optimizer 构建](05-training-flow#optimizer-构建)。

## 学习率调度器

[`torcheasyrec/tzrec/optim/lr_scheduler.py`](../torcheasyrec/tzrec/optim/lr_scheduler.py)

所有调度器继承自 `BaseLR`（[`lr_scheduler.py:26`](../torcheasyrec/tzrec/optim/lr_scheduler.py#L26)），支持 `by_epoch` 属性（按 epoch 或按 step 调度）。

### ConstantLR

[`lr_scheduler.py:53`](../torcheasyrec/tzrec/optim/lr_scheduler.py#L53)

固定学习率，等效于无调度器。

### ExponentialDecayLR

[`lr_scheduler.py:64`](../torcheasyrec/tzrec/optim/lr_scheduler.py#L64)

指数衰减，支持预热：

```python
def _get_lr(self):
    if step < warmup_size:
        # 线性预热
        lr = base_lr * (step / warmup_size) + warmup_lr * (1 - step / warmup_size)
    else:
        p = (step - warmup_size) / decay_size
        if staircase: p = floor(p)
        lr = max(base_lr * decay_factor^p, min_lr)
```

| 参数 | 说明 |
|------|------|
| decay_size | 衰减间隔 |
| decay_factor | 衰减率 |
| staircase | 是否阶梯衰减 |
| warmup_learning_rate | 预热初始 LR |
| warmup_size | 预热步数 |
| min_learning_rate | 下限 |

### ManualStepLR

[`lr_scheduler.py:119`](../torcheasyrec/tzrec/optim/lr_scheduler.py#L119)

在指定 step 阶梯切换学习率：

```python
def _get_lr(self):
    idx = bisect_left(schedule_sizes, step_count)
    if idx > 0:
        return learning_rates[idx - 1]
    elif warmup:
        return base_lr + (learning_rates[0] - base_lr) * step / schedule_sizes[0]
```

### CosineAnnealingLR

[`lr_scheduler.py:162`](../torcheasyrec/tzrec/optim/lr_scheduler.py#L162)

余弦退火衰减：

```python
cos_scale = 0.5 * (1 + cos(π * t / T_max))
lr = min_lr + (base_lr - min_lr) * cos_scale
```

支持可选预热。

### CosineAnnealingWarmRestartsLR

[`lr_scheduler.py:212`](../torcheasyrec/tzrec/optim/lr_scheduler.py#L212)

带热重启的余弦退火：

```
T_cur ← 当前周期内步数
cos_scale = 0.5 * (1 + cos(π * T_cur / T_i))
lr = min_lr + (base_lr - min_lr) * cos_scale
```

参数：

| 参数 | 说明 |
|------|------|
| T_0 | 首个周期长度 |
| T_mult | 每个周期后的长度倍数 |

## 混合精度与梯度

标准训练配置示例（完整优化器架构见 [05-训练流程](05-training-flow)）：

```protobuf
train_config {
    num_steps: 100000

    optimizer {
        adam_optimizer {
            learning_rate: 0.001
        }
    }

    lr_scheduler {
        exponential_decay_lr {
            decay_size: 10000
            decay_factor: 0.9
            staircase: true
            warmup_learning_rate: 0.0001
            warmup_size: 1000
            min_learning_rate: 1e-6
        }
    }

    mixed_precision: "FP16"
    gradient_accumulation_steps: 4
    max_grad_norm: 5.0
}
```

## 关键文件

| 文件 | 行数 | 功能 |
|------|------|------|
| `optim/optimizer.py` | 237 | TZRecOptimizer + FBGEMM Adagrad patch |
| `optim/optimizer_builder.py` | 260 | 稀疏/稠密/分组优化器构建 |
| `optim/lr_scheduler.py` | 274 | 5 种 LR 调度器 |

依赖：`torch`、`torch.optim`、`torchrec`、`fbgemm_gpu`。
