---
title: 损失函数
nav_order: 12
---

# 损失函数

## 模块结构

```
tzrec/loss/
├── __init__.py
├── focal_loss.py                 # BinaryFocalLoss
├── jrc_loss.py                   # JRCLoss
└── pe_mtl_loss.py                # ParetoEfficientMultiTaskLoss
```

## BinaryFocalLoss

[`torcheasyrec/tzrec/loss/focal_loss.py:18`](../torcheasyrec/tzrec/loss/focal_loss.py#L18)

二分类 Focal Loss，处理类别不平衡。通过调制因子 `(1-p)^γ` 降低易分类样本的权重：

```
FL(p) = -α * (1-p)^γ * log(p)          (正样本)
FL(p) = -(1-α) * p^γ * log(1-p)        (负样本)
```

参数：

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| gamma | float | 2.0 | 聚焦参数，越大难样本权重越高 |
| alpha | float | 0.5 | 正负样本平衡系数，取值 (0, 1) |
| reduction | str | "mean" | 聚合方式：'none' \| 'mean' \| 'sum' |

[`torcheasyrec/tzrec/loss/focal_loss.py:46-72`](../torcheasyrec/tzrec/loss/focal_loss.py#L46-L72)：

```python
def forward(self, preds, labels):
    p = F.sigmoid(preds)
    weight = self._alpha * labels * torch.pow(1 - p, self._gamma) + \
             (1 - self._alpha) * (1 - labels) * torch.pow(p, self._gamma)
    loss = F.binary_cross_entropy_with_logits(
        preds, labels, weight=weight.detach(), reduction=self._reduction
    )
    return loss
```

权重在送入 `BCEWithLogitsLoss` 前 `detach()`，不参与反向传播的梯度计算。

## JRCLoss

[`torcheasyrec/tzrec/loss/jrc_loss.py:29`](../torcheasyrec/tzrec/loss/jrc_loss.py#L29)

会话内样本竞争损失（论文 [https://arxiv.org/abs/2208.06164](https://arxiv.org/abs/2208.06164)）：

```
Loss = α * CE_Loss + (1-α) * GE_Loss
```

- **CE_Loss**：标准交叉熵（全局视角）
- **GE_Loss**：会话内竞争损失，分为正样本竞争（组内正样本互相压制）和负样本竞争（组内负样本互相压制）

[`torcheasyrec/tzrec/loss/jrc_loss.py:51-117`](../torcheasyrec/tzrec/loss/jrc_loss.py#L51-L117)：

```python
def forward(self, logits, labels, session_ids):
    ce_loss = self._ce_loss(logits, labels)

    batch_size = labels.shape[0]
    mask = torch.eq(session_ids.unsqueeze(1), session_ids.unsqueeze(0)).float()

    # 正样本竞争：同 session 内的正样本之间做 CE
    pos_mask = torch.where(labels == 1.0)[0]
    logits_pos = logits[:, 1].unsqueeze(0).tile([pos_num, 1])
    pos_session_mask = torch.index_select(mask, 0, pos_mask)
    # mask 同 session 且非对角的正样本 logits
    logits_pos = logits_pos + ((1 - pos_session_mask) + ...) * -1e9
    loss_pos = self._ce_loss(logits_pos, pos_diag_label)

    # 负样本竞争：同 session 内的负样本之间做 CE
    # ...对称逻辑...

    loss = self._alpha * ce_loss + (1 - self._alpha) * ge_loss
    return loss
```

关键设计：

- `session_ids` 是额外输入张量，标识每个样本所属的会话
- 正样本和负样本**分开计算**竞争，互不干扰
- 用 `-1e9` mask 屏蔽非同 session 和对角线样本

## ParetoEfficientMultiTaskLoss

[`torcheasyrec/tzrec/loss/pe_mtl_loss.py:18`](../torcheasyrec/tzrec/loss/pe_mtl_loss.py#L18)

多任务 Pareto 高效动态权重。自动计算各任务损失的最优加权，无需手动调权。

[`torcheasyrec/tzrec/loss/pe_mtl_loss.py:21-28`](../torcheasyrec/tzrec/loss/pe_mtl_loss.py#L21-L28)：

```python
class ParetoEfficientMultiTaskLoss(torch.nn.Module):
    def __init__(self, min_c: list[float]):
        # min_c: 各任务的最小权重占比，总和 < 1.0
```

### 算法流程

1. **计算梯度矩阵 G**：对每个任务损失 `loss_i` 求导，拼成矩阵 `(K, M)`（K 任务数 × M 参数量）
2. **求解 Pareto 权重**：通过 [Pareto 论文](http://ofey.me/papers/Pareto.pdf) 的方法，用 `scipy.optimize.nnls` + `SLSQP` 求 Pareto 前沿最优权重
3. **加权聚合**：用新权重对各任务损失加权求和

[`torcheasyrec/tzrec/loss/pe_mtl_loss.py:77-110`](../torcheasyrec/tzrec/loss/pe_mtl_loss.py#L77-L110)：

```python
def forward(self, losses, model):
    grads = []
    for loss in losses.values():
        gradients = torch.autograd.grad(
            torch.sum(loss, dim=0), trainable_params,
            retain_graph=True, allow_unused=True
        )
        grads.append(torch.cat([g.view(-1) for g in gradients]))
    grads = torch.stack(grads)  # (K, M)

    init_weight = np.array([1/len(losses)] * len(losses))
    new_w = self._pareto_step(init_weight, self._c, grads)
    return torch.stack([loss * new_w[i] for i, loss in enumerate(losses.values())]).sum()
```

注意 `retain_graph=True`：每个任务的梯度计算是独立的，必须保留计算图供后续任务复用。

## 标准损失

框架还直接使用 PyTorch 内置的标准损失（见 [05-训练流程](05-training-flow)）：

| 损失 | PyTorch 类 | 场景 |
|------|------------|------|
| 二分类交叉熵 | `nn.BCEWithLogitsLoss` | CTR 预估 |
| 多分类交叉熵 | `nn.CrossEntropyLoss` | 多分类/匹配 |
| L2 损失 | `nn.MSELoss` | 回归任务 |

## 场景推荐

| 场景 | 推荐损失 |
|------|----------|
| CTR 预估（类别平衡） | BCEWithLogitsLoss |
| CTR 预估（类别不平衡） | BinaryFocalLoss |
| 会话级排序 | JRCLoss |
| 多任务学习 | ParetoEfficientMultiTaskLoss |
| 匹配模型 | CrossEntropyLoss |

## 配置示例

```protobuf
losses {
    binary_focal_loss {
        gamma: 2.0
        alpha: 0.5
    }
}
```

```protobuf
losses {
    jrc_loss {
        alpha: 0.5
    }
}
```

```protobuf
model_config {
    use_pareto_loss_weight: true
    pareto_init_weight_cs: [0.4, 0.3, 0.3]
}
```

## 关键文件

| 文件 | 行数 | 功能 |
|------|------|------|
| `loss/focal_loss.py` | 72 | BinaryFocalLoss |
| `loss/jrc_loss.py` | 117 | JRCLoss（session 内竞争） |
| `loss/pe_mtl_loss.py` | 110 | Pareto 多任务动态权重 |

依赖：`torch`、`torch.nn`、`scipy.optimize`（Pareto）、`numpy`。
