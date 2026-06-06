---
title: Embedding System
nav_order: 9
---

# Embedding System

## Overview

The embedding system is the bridge between parsed features and model computation. It handles:

1. **Sparse feature embedding** — categorical IDs → trainable embedding vectors (via TorchRec `EmbeddingBagCollection`)
2. **Dense feature embedding** — numerical values → learned representations (via `DenseEmbeddingCollection`)
3. **Sequence embedding** — sequential features → query + sequence tensors
4. **Managed collision (ZCH)** — dynamic embedding tables with eviction policies
5. **Distributed sharding** — TorchRec's table-wise/row-wise/column-wise sharding

## EmbeddingGroup

[`torcheasyrec/tzrec/modules/embedding.py`](../torcheasyrec/tzrec/modules/embedding.py#L139-L519)

`EmbeddingGroup` is the top-level orchestrator. It:

- Organizes features into **implementation keys** by data group (base, neg)
- Creates `EmbeddingGroupImpl` for non-sequence groups
- Creates `SequenceEmbeddingGroupImpl` for sequence groups
- Creates sequence encoders (LSTM, Pooling, etc.)
- Provides methods to query group dimensions (`group_dims()`, `group_total_dim()`)

```python
class EmbeddingGroup(nn.Module):
    def __init__(self, features, feature_groups, ...):
        # For each feature group:
        #   1. Inspect group type (DEEP/WIDE/SEQUENCE)
        #   2. Assign implementation key (data group)
        #   3. Create EmbeddingGroupImpl or SequenceEmbeddingGroupImpl
        #   4. Create sequence encoders if configured

    def forward(self, batch):
        result_dicts = []
        for key, emb_impl in self.emb_impls.items():
            result_dicts.append(emb_impl(sparse_kjt, dense_kt, ...))
        for key, seq_emb_impl in self.seq_emb_impls.items():
            result_dicts.append(seq_emb_impl(sparse_kjt, dense_kt, ...))
        # Apply sequence encoders
        result = merge(result_dicts)
        return result
```

## EmbeddingGroupImpl

[`torcheasyrec/tzrec/modules/embedding.py`](../torcheasyrec/tzrec/modules/embedding.py#L623-L916)

Handles non-sequence (scalar) features. It creates:

1. **`EmbeddingBagCollection` (EBC)**: for sparse ID features without zero-collision hash
2. **`ManagedCollisionEmbeddingBagCollection` (MC-EBC)**: for sparse features with ZCH
3. **`DenseEmbeddingCollection`**: for dense features with AutoDis/MLP embedding

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

TorchRec's `EmbeddingBagCollection` manages multiple embedding tables:

```python
self.ebc = EmbeddingBagCollection(
    [EmbeddingBagConfig(
        num_embeddings=hash_bucket_size,  # or vocab size
        embedding_dim=16,                  # user-defined
        name="user_id_emb",
        feature_names=["user_id"],
        pooling=PoolingType.SUM,
    ), ...],
    device=device,
)
```

Each `BaseFeature` produces its `emb_bag_config` via the `emb_bag_config` property.

### ManagedCollision (ZCH)

[`torcheasyrec/tzrec/features/feature.py`](../torcheasyrec/tzrec/features/feature.py#L693-L726)

Zero-Collision Hash (ZCH) enables dynamic embedding tables:

```python
feature.mc_module(device) → MCHManagedCollisionModule
    ├── zch_size: max table size
    ├── eviction_interval: how often to evict
    └── eviction_policy: LFU, LRU, or DistanceLFU
```

If a feature has a `zch` config, its `mc_module()` returns a `MCHManagedCollisionModule`. The `ManagedCollisionEmbeddingBagCollection` wraps both the EBC and the collision module.

### Input Tile Embedding

For input-tile mode (used in generative recommendation), user-side features have separate embedding collections:

```python
if need_input_tile_emb and feature.is_user_feat:
    # User features go to ebc_user
    _add_embedding_bag_config(emb_bag_configs_user, emb_bag_config)
else:
    # Item features go to main ebc
    _add_embedding_bag_config(emb_bag_configs, emb_bag_config)
```

During forward, user embeddings are tiled: `keyed_tensor_user.values().tile(tile_size, 1)`.

## SequenceEmbeddingGroupImpl

[`torcheasyrec/tzrec/modules/embedding.py`](../torcheasyrec/tzrec/modules/embedding.py#L931-1199+)

Handles sequence features. Splits each feature into:
- **Query**: scalar part (e.g., last item)
- **Sequence**: list part (e.g., click history)
- **Sequence length**: actual length before padding

Uses TorchRec `EmbeddingCollection` (not `EmbeddingBagCollection`) for sequence feature embedding, which returns per-position embeddings rather than pooled bag embeddings.

## DenseEmbeddingCollection

[`torcheasyrec/tzrec/modules/dense_embedding_collection.py`](../torcheasyrec/tzrec/modules/dense_embedding_collection.py)

Handles dense features that need learned embeddings:

| Type | Description |
|------|-------------|
| `AutoDisEmbedding` | Automatic discretization (soft binning + embedding) |
| `MLPDenseEmbedding` | MLP projection to embedding space |

These are configured in the feature proto:

```protobuf
raw_feature {
    feature_name: "age"
    autodis { num_channels: 8 temperature: 0.1 }
    embedding_dim: 16
}
```

## Distributed Sharding

TorchEasyRec uses TorchRec's `DistributedModelParallel` for distributed training:

[`torcheasyrec/tzrec/utils/dist_util.py`](../torcheasyrec/tzrec/utils/dist_util.py)

### Parameter Constraints

Features can specify sharding constraints:

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

### Plan Creation

[`torcheasyrec/tzrec/utils/plan_util.py`](../torcheasyrec/tzrec/utils/plan_util.py)

The `create_planner()` function collects `ParameterConstraints` from all features and creates a TorchRec sharding plan. Available sharding types:

| Type | Description |
|------|-------------|
| `data_parallel` | Full table on each GPU |
| `table_wise` | Whole table on one GPU |
| `row_wise` | Rows split across GPUs |
| `column_wise` | Columns split across GPUs |
| `table_row_wise` | Table-wise + row-wise hybrid |
| `grid_shard` | 2D grid sharding |

### Optimizer Integration

TorchRec's `KeyedOptimizerWrapper` and `apply_optimizer_in_backward` are used:

- **Sparse parameters** (embeddings): optimized in-backward for efficiency
- **Dense parameters** (MLP weights): standard optimizers (SGD, Adam)

The `CombinedOptimizer` orchestrates both:

```python
optimizer = CombinedOptimizer([
    KeyedOptimizerWrapper(dense_params, Adam(..., lr=0.001)),
    KeyedOptimizerWrapper(sparse_params_in_backward, SGD(..., lr=0.01)),
])
```

## Sequence Encoders

After embedding lookup, sequence features can pass through sequence encoders:

[`torcheasyrec/tzrec/modules/sequence.py`](../torcheasyrec/tzrec/modules/sequence.py)

Configured in feature groups:

```protobuf
feature_groups {
    group_name: "deep"
    sequence_groups { group_name: "click_seq" feature_names: ["item_id_seq"] }
    sequence_encoders {
        lstm { input: "click_seq" hidden_size: 64 }
    }
}
```

Available encoders: LSTM, GRU, Pooling (sum/mean), Attention, Transformer.

## EmbeddingGroupImpl + DenseEmbeddingCollection Forward Flow

```
forward(sparse_feature, dense_feature, ...):
    │
    ├── if has_sparse:
    │       kt = ebc(sparse_feature)           # KeyedTensor from EmbeddingBagCollection
    │
    ├── if has_mc_sparse:
    │       kt = mc_ebc(sparse_feature)[0]     # KeyedTensor from ManagedCollisionEBC
    │
    ├── if has_sparse_user:
    │       kt_user = ebc_user(sparse_user)
    │       kt_user_values = tile(kt_user)      # Tile user embeddings
    │
    ├── if has_dense:
    │       if has_dense_embedding:
    │           kt = dense_ec(dense_feature)   # AutoDis/MLP embedding
    │       else:
    │           kt = dense_feature              # Pass through
    │
    └── group_tensors = KeyedTensor.regroup_as_dict(
            kts, feature_names, group_names)    # Per-group tensors
```

## Key Files

| File | Purpose |
|------|---------|
| [`torcheasyrec/tzrec/modules/embedding.py`](../torcheasyrec/tzrec/modules/embedding.py) | `EmbeddingGroup`, `EmbeddingGroupImpl`, `SequenceEmbeddingGroupImpl` |
| [`torcheasyrec/tzrec/modules/dense_embedding_collection.py`](../torcheasyrec/tzrec/modules/dense_embedding_collection.py) | `DenseEmbeddingCollection`, AutoDis/MLP configs |
| [`torcheasyrec/tzrec/modules/sequence.py`](../torcheasyrec/tzrec/modules/sequence.py) | Sequence encoders (LSTM, Pooling, Attention) |
| [`torcheasyrec/tzrec/features/feature.py`](../torcheasyrec/tzrec/features/feature.py) | `emb_bag_config`, `emb_config`, `mc_module()` on BaseFeature |
| [`torcheasyrec/tzrec/utils/plan_util.py`](../torcheasyrec/tzrec/utils/plan_util.py) | TorchRec sharding planner |
| [`torcheasyrec/tzrec/utils/dist_util.py`](../torcheasyrec/tzrec/utils/dist_util.py) | `DistributedModelParallel` |
