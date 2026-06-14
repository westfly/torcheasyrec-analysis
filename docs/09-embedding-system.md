---
title: 嵌入系统
nav_order: 6
has_children: true
parent: 训练篇
---

# 嵌入系统

## 概览

嵌入系统是解析后的特征与模型计算之间的桥梁。它处理：

1. **稀疏特征嵌入** — 类别型 ID → 可训练嵌入向量（通过 TorchRec `EmbeddingBagCollection`）
2. **稠密特征嵌入** — 数值 → 学习到的表示（通过 `DenseEmbeddingCollection`）
3. **序列嵌入** — 序列特征 → query + sequence 张量
4. **托管碰撞（ZCH）** — 带淘汰策略的动态嵌入表
5. **分布式分片** — TorchRec 的 table-wise/row-wise/column-wise 分片

## EmbeddingGroup

[`torcheasyrec/tzrec/modules/embedding.py`](../torcheasyrec/tzrec/modules/embedding.py#L139-L519)

`EmbeddingGroup` 是顶层编排器。它：

- 按 data group（base、neg）将特征组织为**实现键**
- 为非序列组创建 `EmbeddingGroupImpl`
- 为序列组创建 `SequenceEmbeddingGroupImpl`
- 创建序列编码器（LSTM、Pooling 等）
- 提供查询组维度的方法（`group_dims()`、`group_total_dim()`）

```python
class EmbeddingGroup(nn.Module):
    def __init__(self, features, feature_groups, ...):
        # 对每个特征组:
        #   1. 检查组类型 (DEEP/WIDE/SEQUENCE)
        #   2. 分配实现键 (data group)
        #   3. 创建 EmbeddingGroupImpl 或 SequenceEmbeddingGroupImpl
        #   4. 如果已配置则创建序列编码器

    def forward(self, batch):
        result_dicts = []
        for key, emb_impl in self.emb_impls.items():
            result_dicts.append(emb_impl(sparse_kjt, dense_kt, ...))
        for key, seq_emb_impl in self.seq_emb_impls.items():
            result_dicts.append(seq_emb_impl(sparse_kjt, dense_kt, ...))
        # 应用序列编码器
        result = merge(result_dicts)
        return result
```

## EmbeddingGroupImpl

[`torcheasyrec/tzrec/modules/embedding.py`](../torcheasyrec/tzrec/modules/embedding.py#L623-L916)

处理非序列（标量）特征。它创建：

1. **`EmbeddingBagCollection` (EBC)**：用于无 zero-collision hash 的稀疏 ID 特征
2. **`ManagedCollisionEmbeddingBagCollection` (MC-EBC)**：用于带 ZCH 的稀疏特征
3. **`DenseEmbeddingCollection`**：用于带 AutoDis/MLP 嵌入的稠密特征

```
Input: KeyedJaggedTensor (sparse) + KeyedTensor (dense)
    │
    ├─▶ EBC → KeyedTensor (sparse embedding)
    ├─▶ MC-EBC → KeyedTensor (ZCH embedding)
    ├─▶ ebc_user → tile → KeyedTensor (input-tiled user embedding)
    └─▶ DenseEmbeddingCollection → KeyedTensor (dense embedding)
    │
    ▼
KeyedTensor.regroup_as_dict() → {group_name: tensor}
```

### EmbeddingBagCollection

TorchRec 的 `EmbeddingBagCollection` 管理多个嵌入表：

```python
self.ebc = EmbeddingBagCollection(
    [EmbeddingBagConfig(
        num_embeddings=hash_bucket_size,  # 或 vocab size
        embedding_dim=16,                  # 用户定义
        name="user_id_emb",
        feature_names=["user_id"],
        pooling=PoolingType.SUM,
    ), ...],
    device=device,
)
```

每个 `BaseFeature` 通过 `emb_bag_config` 属性生成其 `emb_bag_config`。

### Managed Collision（ZCH）

[`torcheasyrec/tzrec/features/feature.py`](../torcheasyrec/tzrec/features/feature.py#L693-L726)

Zero-Collision Hash（ZCH）支持动态嵌入表：

```python
feature.mc_module(device) → MCHManagedCollisionModule
    ├── zch_size: 最大表大小
    ├── eviction_interval: 淘汰频率
    └── eviction_policy: LFU、LRU 或 DistanceLFU
```

如果特征有 `zch` 配置，其 `mc_module()` 返回 `MCHManagedCollisionModule`。`ManagedCollisionEmbeddingBagCollection` 包装了 EBC 与 collision module。

> **进一步阅读**: [ZCH 零碰撞哈希](09-01-zch) — 完整的三阶段链路、淘汰策略详解、导出限制与在线推理方案

### Input Tile Embedding

对于 input-tile 模式（用于生成式推荐），user 侧特征有独立的嵌入集合：

```python
if need_input_tile_emb and feature.is_user_feat:
    # user 特征进入 ebc_user
    _add_embedding_bag_config(emb_bag_configs_user, emb_bag_config)
else:
    # item 特征进入主 ebc
    _add_embedding_bag_config(emb_bag_configs, emb_bag_config)
```

在前向过程中，user 嵌入被平铺（tiled）：`keyed_tensor_user.values().tile(tile_size, 1)`。

## SequenceEmbeddingGroupImpl

[`torcheasyrec/tzrec/modules/embedding.py`](../torcheasyrec/tzrec/modules/embedding.py#L931-1199+)

处理序列特征。将每个特征分割为：
- **Query**：标量部分（如最后一个 item）
- **Sequence**：列表部分（如点击历史）
- **Sequence length**：填充前的实际长度

对序列特征嵌入使用 TorchRec `EmbeddingCollection`（非 `EmbeddingBagCollection`），它返回 per-position 嵌入而非池化的 bag 嵌入。

## DenseEmbeddingCollection

[`torcheasyrec/tzrec/modules/dense_embedding_collection.py`](../torcheasyrec/tzrec/modules/dense_embedding_collection.py)

处理需要学习嵌入的稠密特征：

| 类型 | 说明 |
|------|------|
| `AutoDisEmbedding` | 自动离散化（软分桶 + 嵌入） |
| `MLPDenseEmbedding` | MLP 投影到嵌入空间 |

在特征 proto 中配置：

```protobuf
raw_feature {
    feature_name: "age"
    autodis { num_channels: 8 temperature: 0.1 }
    embedding_dim: 16
}
```

## 分布式分片

TorchEasyRec 使用 TorchRec 的 `DistributedModelParallel` 进行分布式训练：

[`torcheasyrec/tzrec/utils/dist_util.py`](../torcheasyrec/tzrec/utils/dist_util.py)

### 参数约束

特征可指定分片约束：

```protobuf
id_feature {
    feature_name: "large_user_id"
    hash_bucket_size: 100000000
    embedding_constraints {
        sharding_types: ["table_wise", "row_wise"]
        compute_kernels: ["fused"]
    }
}
```

### 计划创建

[`torcheasyrec/tzrec/utils/plan_util.py`](../torcheasyrec/tzrec/utils/plan_util.py)

`create_planner()` 函数从所有特征收集 `ParameterConstraints` 并创建 TorchRec 分片计划。可用的分片类型：

| 类型 | 说明 |
|------|------|
| `data_parallel` | 每个 GPU 上的完整表 |
| `table_wise` | 整表在单个 GPU |
| `row_wise` | 行跨 GPU 分割 |
| `column_wise` | 列跨 GPU 分割 |
| `table_row_wise` | Table-wise + row-wise 混合 |
| `grid_shard` | 2D 网格分片 |

### Optimizer 集成

使用 TorchRec 的 `KeyedOptimizerWrapper` 与 `apply_optimizer_in_backward`：

- **稀疏参数**（嵌入）：为效率在 backward 中优化
- **稠密参数**（MLP 权重）：标准 optimizer（SGD、Adam）

`CombinedOptimizer` 协调两者：

```python
optimizer = CombinedOptimizer([
    KeyedOptimizerWrapper(dense_params, Adam(..., lr=0.001)),
    KeyedOptimizerWrapper(sparse_params_in_backward, SGD(..., lr=0.01)),
])
```

## 序列编码器

嵌入查找之后，序列特征可通过序列编码器：

[`torcheasyrec/tzrec/modules/sequence.py`](../torcheasyrec/tzrec/modules/sequence.py)

在特征组中配置：

```protobuf
feature_groups {
    group_name: "deep"
    sequence_groups { group_name: "click_seq" feature_names: ["item_id_seq"] }
    sequence_encoders {
        lstm { input: "click_seq" hidden_size: 64 }
    }
}
```

可用编码器：LSTM、GRU、Pooling（sum/mean）、Attention、Transformer。

## EmbeddingGroupImpl + DenseEmbeddingCollection 前向流

```
forward(sparse_feature, dense_feature, ...):
    │
    ├── if has_sparse:
    │       kt = ebc(sparse_feature)           # 来自 EmbeddingBagCollection 的 KeyedTensor
    │
    ├── if has_mc_sparse:
    │       kt = mc_ebc(sparse_feature)[0]     # 来自 ManagedCollisionEBC 的 KeyedTensor
    │
    ├── if has_sparse_user:
    │       kt_user = ebc_user(sparse_user)
    │       kt_user_values = tile(kt_user)      # 平铺 user 嵌入
    │
    ├── if has_dense:
    │       if has_dense_embedding:
    │           kt = dense_ec(dense_feature)   # AutoDis/MLP 嵌入
    │       else:
    │           kt = dense_feature              # 直通
    │
    └── group_tensors = KeyedTensor.regroup_as_dict(
            kts, feature_names, group_names)    # 按组张量
```

## DynamicEmb 透传路径

`embedding.py` 自身**不直接处理** DynamicEmb——`grep "dynamicemb"` 在该文件零命中。但 DynamicEmb 特征确实进入了模型：是通过**元数据透传**让下游 planner / DMP 自动切换到 dynamicemb 路径。`embedding.py` 是统一的"上游配置打包器"，对 dynamiceb 完全透明、零侵入。

### 透传链 5 步

```
1) feature.emb_bag_config.use_dynamicemb = True           # 标记（feature.py:631/657）
2) feature.parameter_constraints()                          # 返回 DynamicEmbParameterConstraints（feature.py:828-833）
        ↓
3) EmbeddingGroupImpl._emb_bag_constraints[name] = const   # 一视同仁收集（embedding.py:729/764）
        ↓
4) plan_util._emit_dynamicemb_variants()                   # dynamicemb 走 2模式×10比率 展开（plan_util.py:887-916）
        ↓
5) DMP 按 plan 替换 compute_kernel                          # CUSTOMIZED_KERNEL → BatchedDynamicEmbeddingTablesV2
```

### 关键事实

| 路径 | 行为 |
|------|------|
| `feature.emb_bag_config` | 普通 `EmbeddingBagConfig` + `use_dynamicemb=True` 标签（[feature.py:631](../torcheasyrec/tzrec/features/feature.py#L631)） |
| `feature.parameter_constraints` | DynamicEmb 特征返回 `DynamicEmbParameterConstraints`（[feature.py:828-833](../torcheasyrec/tzrec/features/feature.py#L828-L833)），非 DynamicEmb 走标准 `ParameterConstraints` 或 `None` |
| `EmbeddingGroupImpl.__init__` | **无差别**收集到 `self._emb_bag_constraints`（[embedding.py:729/764](../torcheasyrec/tzrec/modules/embedding.py#L729)）— 它不知道也不需要知道这是 DynamicEmb |
| `self.ebc = EmbeddingBagCollection(...)` | 同样会建 EBC — 但这是 **planner 占位**，DMP 替换时会被丢弃 |
| Planner 派发 | DynamicEmb 走 `DynamicEmbParameterSharding`（`compute_kernel=CUSTOMIZED_KERNEL`、`customized_compute_kernel=DynamicEmb`），普通走标准 `ParameterSharding` |
| DMP 实例化 | DynamicEmb 走 `BatchedDynamicEmbeddingTablesV2`（C++/CUDA），普通走 `ShardedEmbeddingBagCollection`（FBGEMM） |

### 集成边界

`embedding.py` 只在 **planner 链路（静态规划）** 上"被动"提供 constraints 数据；**DMP 替换链路（动态执行）** 完全不被它管。完整分流决策在两条独立路径上发生：

- **A) Planner 链路（静态）**：`feature.parameter_constraints()` → `EmbeddingGroupImpl._emb_bag_constraints` → `plan_util.create_planner()` → `_emit_dynamicemb_variants()` → `DynamicEmbParameterSharding`
- **B) DMP 替换链路（动态）**：`DMP(model, plan)` 按每个 table 的 `compute_kernel` 字段分流：
  - `standard` → `ShardedEmbeddingBagCollection`（FBGEMM `dense_lookups`）
  - `CUSTOMIZED_KERNEL / DynamicEmb` → `BatchedDynamicEmbeddingTablesV2`（CUDA `DynamicEmbStorage / HybridStorage / DynamicEmbCache`）

### 设计含义

1. **零侵入**：DynamicEmb 集成对 `embedding.py` 透明——只要 feature 层打了 `use_dynamicemb` 标记，下游自动切换路径，`embedding.py` 一行代码都不用改。
2. **占位 EBC 的浪费**：构造时所有特征（含 DynamicEmb）都建 EBC；DMP 替换时丢弃 DynamicEmb 部分。短期内存会双份（TorchRec 标准行为）。
3. **动态约束**：DynamicEmb 的所有特殊处理（4 个 monkey-patch、variant emission、storage estimator、sharding plan 改写）都在 [`dynamicemb_util.py`](../torcheasyrec/tzrec/utils/dynamicemb_util.py) 与 [`plan_util.py`](../torcheasyrec/tzrec/utils/plan_util.py) 中，`embedding.py` 完全不参与。
4. **运行时假象**：`embedding.py` 看到的"EBC 里有这个 table"是规划视角的假象；运行时该 table 实际由 `BatchedDynamicEmbeddingTablesV2` 接管（通过 `DynamicEmbeddingBagCollectionSharder` 的 `ModuleSharder` 流程），不存在标准 EBC。

详细的双视角深度解析（NVIDIA 上游 + TorchEasyRec 集成）见 [10-dynamicemb-integration](10-dynamicemb)。

## 关键文件

| 文件 | 用途 |
|------|------|
| [`torcheasyrec/tzrec/modules/embedding.py`](../torcheasyrec/tzrec/modules/embedding.py) | `EmbeddingGroup`、`EmbeddingGroupImpl`、`SequenceEmbeddingGroupImpl` |
| [`torcheasyrec/tzrec/modules/dense_embedding_collection.py`](../torcheasyrec/tzrec/modules/dense_embedding_collection.py) | `DenseEmbeddingCollection`、AutoDis/MLP 配置 |
| [`torcheasyrec/tzrec/modules/sequence.py`](../torcheasyrec/tzrec/modules/sequence.py) | 序列编码器（LSTM、Pooling、Attention） |
| [`torcheasyrec/tzrec/features/feature.py`](../torcheasyrec/tzrec/features/feature.py) | `emb_bag_config`、`emb_config`、`mc_module()` on BaseFeature |
| [`torcheasyrec/tzrec/utils/plan_util.py`](../torcheasyrec/tzrec/utils/plan_util.py) | TorchRec 分片规划器 + `_emit_dynamicemb_variants`（[L887-916](../torcheasyrec/tzrec/utils/plan_util.py#L887-L916)） |
| [`torcheasyrec/tzrec/utils/dist_util.py`](../torcheasyrec/tzrec/utils/dist_util.py) | `DistributedModelParallel` |
| [`torcheasyrec/tzrec/utils/dynamicemb_util.py`](../torcheasyrec/tzrec/utils/dynamicemb_util.py) | DynamicEmb 约束构建 + 4 个 planner monkey-patch |
