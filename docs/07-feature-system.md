---
title: Feature System
nav_order: 8
---

# Feature System

## Overview

TorchEasyRec's feature system handles the complete pipeline from raw input data → parsed tensors → embedding lookup. It supports 10+ feature types through a common `BaseFeature` interface.

```
Raw Input (pyarrow table)
        │
        ▼
   Feature.parse()  ─── FG_NONE: direct parsing
        │              ─── FG_NORMAL: pyfg handler
        │              ─── FG_DAG: DAG-based FG
        ▼
   ParsedData (SparseData / DenseData / SequenceSparseData / SequenceDenseData)
        │
        ▼
   DataParser.to_batch() → Batch(sparse_features=KJT, dense_features=KT)
        │
        ▼
   EmbeddingGroup → EmbeddingBagCollection / DenseEmbeddingCollection
        │
        ▼
   Grouped feature tensors (per feature group)
```

## BaseFeature

[`torcheasyrec/tzrec/features/feature.py`](../torcheasyrec/tzrec/features/feature.py#L380-L1149)

`BaseFeature` is the abstract base for all feature types:

```python
class BaseFeature(object, metaclass=_meta_cls):
    def __init__(self, feature_config, fg_mode, fg_encoded_multival_sep,
                 is_sequence, sequence_name, ...):
        self.fg_mode = fg_mode      # FG_NONE, FG_NORMAL, FG_DAG, FG_BUCKETIZE
        self._is_sparse = None      # True = categorical, False = numerical
        self._is_sequence = False   # True = sequential feature
        self._is_weighted = False   # True = weighted ID feature
        self._data_group = BASE_DATA_GROUP  # or NEG_DATA_GROUP

    def parse(self, input_data, is_training=False) -> ParsedData:
        # FG-specific parsing logic
    def fg_json(self) -> List[Dict[str, Any]]:
        # Feature generation JSON config
    def _fg_json(self) -> List[Dict[str, Any]]:
        # Subclass-specific FG config
```

## Feature Types

[`torcheasyrec/tzrec/protos/feature.proto`](../torcheasyrec/tzrec/protos/feature.proto)

Each feature type has its own proto config and Python class:

| Feature Type | Proto Config | Python Class | Description |
|-------------|-------------|--------------|-------------|
| IdFeature | `IdFeatureConfig` | `IdFeature` | Categorical (sparse) features with embedding |
| RawFeature | `RawFeatureConfig` | `RawFeature` | Numerical (dense) features |
| ComboFeature | `ComboFeatureConfig` | `ComboFeature` | Cross of multiple categorical features |
| SequenceFeature | `SequenceFeature` | (container) | Groups features into sequences |
| LookupFeature | `LookupFeatureConfig` | `LookupFeature` | Key-value lookup features |
| ExprFeature | `ExprFeatureConfig` | `ExprFeature` | Expression-based features |
| TokenizeFeature | `TokenizeFeatureConfig` | `TokenizeFeature` | Text tokenization |
| MatchFeature | `MatchFeatureConfig` | `MatchFeature` | Retrieval match features |
| CombineFeature | `CombineFeatureConfig` | `CombineFeature` | Feature combination |
| BoolMaskFeature | `BoolMaskFeatureConfig` | `BoolMaskFeature` | Boolean masking |
| OverlapFeature | `OverlapFeatureConfig` | `OverlapFeature` | Set overlap features |
| KV Dot Product | (in config) | `KVDotProduct` | Key-value dot product |

### IdFeature (Categorical)

The most common feature type. Each unique ID maps to an embedding vector.

```python
# Config:
feature_config {
    id_feature {
        feature_name: "user_id"
        embedding_dim: 16
        hash_bucket_size: 100000
    }
}
```

Key properties:
- `num_embeddings`: from `hash_bucket_size`, `vocab_list`, or `vocab_dict`
- `embedding_dim`: output embedding dimension
- `pooling`: SUM or MEAN (for multi-valued features)
- Optional ZCH (Zero-Collision Hash) for dynamic embedding tables

### RawFeature (Numerical)

For dense numerical values:

```python
# Config:
feature_config {
    raw_feature {
        feature_name: "price"
        value_dim: 1      # scalar, or >1 for vector
        boundaries: [0, 10, 100, 1000]  # optional bucketization
    }
}
```

RawFeature can be:
- Used directly as dense input
- Bucketized into categorical IDs via `boundaries`
- Embedded via `AutoDisEmbedding` (automatic discretization) or `MLPEmbedding`

### SequenceFeature

Groups features into sequences:

```python
# Config:
feature_config {
    sequence_feature {
        sequence_name: "click_history"
        sequence_delim: ";"
        sequence_length: 50
        features: [
            { id_feature { feature_name: "item_id" ... } },
            { id_feature { feature_name: "category_id" ... } }
        ]
    }
}
```

Sequence features are automatically split into query (scalar) and sequence (list) parts by the `SequenceEmbeddingGroupImpl`.

## Feature Generation (FG) Modes

TorchEasyRec supports four FG modes:

### FG_NONE

Input data is **pre-encoded**. Features are parsed directly from pyarrow arrays:
- Sparse: `"1,2,3"` → `SparseData(values=[1,2,3], lengths=[3])`
- Dense: `"0.5,1.2"` → `DenseData(values=[[0.5, 1.2]])`

### FG_NORMAL

Uses `pyfg` (Alibaba's feature generation library) to process features. Each feature's `_fg_json()` returns its FG config, and `pyfg.FgArrowHandler` processes the raw data.

### FG_DAG

DAG-based feature generation. Features can reference intermediate features via `"feature:..."` side inputs, forming a computation DAG. The DAG is detected in `create_features()` and a single `FgArrowHandler` is created for all DAG features.

### FG_BUCKETIZE

Special mode for raw feature bucketization.

## Feature Grouping

Features are organized into **feature groups** in the model config:

```protobuf
feature_groups {
    group_name: "wide"
    feature_names: ["user_id", "item_id"]
    group_type: WIDE
}
feature_groups {
    group_name: "deep"
    feature_names: ["price", "category"]
    group_type: DEEP
}
```

The `EmbeddingGroup` creates separate embedding lookups per group. Groups can be:
- **DEEP**: standard embedding + MLP
- **WIDE**: wide (linear) model
- **SEQUENCE**: sequential features with query/sequence split
- **JAGGED_SEQUENCE**: variable-length sequences

## Data Types

[`torcheasyrec/tzrec/datasets/utils.py`](../torcheasyrec/tzrec/datasets/utils.py)

| Python Class | Description | Used For |
|-------------|-------------|----------|
| `SparseData` | Multi-valued categorical | IdFeature |
| `DenseData` | Fixed-dim numerical | RawFeature |
| `SequenceSparseData` | Sequence of multi-valued categorical | SequenceFeature (sparse) |
| `SequenceDenseData` | Sequence of fixed-dim numerical | SequenceFeature (dense) |
| `Batch` | Complete batch container | All model input |
| `KeyedJaggedTensor` | TorchRec sparse format | Embedding lookup |
| `KeyedTensor` | TorchRec dense format | Dense feature passing |

## Dense Embedding Types

For non-sparse features, TorchEasyRec provides specialized dense embedding:

[`torcheasyrec/tzrec/modules/dense_embedding_collection.py`](../torcheasyrec/tzrec/modules/dense_embedding_collection.py)

- **AutoDisEmbedding**: Automatic discretization + soft embedding
- **MLPDenseEmbedding**: MLP-based embedding for dense features

## Feature Creation Pipeline

[`torcheasyrec/tzrec/features/feature.py`](../torcheasyrec/tzrec/features/feature.py#L1151-L1230)

```python
def create_features(feature_configs, fg_mode, neg_fields, ...):
    features = []
    for feat_config in feature_configs:
        feat_type = feat_config.WhichOneof("feature")
        if feat_type == "sequence_feature":
            # Expand sequence feature into sub-features
            for sub_feature in sequence_feature.features:
                features.append(BaseFeature.create_sub_feature(...))
        else:
            features.append(BaseFeature.create(feat_type, feat_config, ...))

    # Detect DAG features and determine user/item side
    if has_dag:
        fg_handler = pyfg.FgArrowHandler(fg_json, 1)
        user_feats = fg_handler.user_features()
        for feature in features:
            feature.is_user_feat = feature.name in user_feats

    return features
```

## Key Files

| File | Purpose |
|------|---------|
| [`torcheasyrec/tzrec/features/feature.py`](../torcheasyrec/tzrec/features/feature.py) | `BaseFeature`, `create_features()`, parsing functions |
| [`torcheasyrec/tzrec/features/id_feature.py`](../torcheasyrec/tzrec/features/id_feature.py) | IdFeature implementation |
| [`torcheasyrec/tzrec/features/raw_feature.py`](../torcheasyrec/tzrec/features/raw_feature.py) | RawFeature implementation |
| [`torcheasyrec/tzrec/protos/feature.proto`](../torcheasyrec/tzrec/protos/feature.proto) | All feature proto definitions |
| [`torcheasyrec/tzrec/datasets/utils.py`](../torcheasyrec/tzrec/datasets/utils.py) | Data types (SparseData, Batch, etc.) |
| [`torcheasyrec/tzrec/datasets/data_parser.py`](../torcheasyrec/tzrec/datasets/data_parser.py) | DataParser — raw → Batch |
| [`torcheasyrec/tzrec/modules/dense_embedding_collection.py`](../torcheasyrec/tzrec/modules/dense_embedding_collection.py) | AutoDis, MLP dense embedding |

## Appendix: Pure-Python FG Plan (Migration from pyfg)

**Context:** The standard `FG_NORMAL` / `FG_DAG` / `FG_BUCKETIZE` modes
depend on **pyfg**, Alibaba's closed-source `libfg.so` (833 MB binary).
This dependency is problematic for:

- Online serving (can't ship `libfg.so` everywhere)
- Open-source deployment (need a permissive-license FG)
- Some advanced features that pyfg doesn't support

The planned solution is a **pure-Python FG** implementation that
covers ~80% of use cases without the binary dependency.

### Coverage Plan

| Tier | Feature Type | Core Logic | Priority |
|------|--------------|------------|----------|
| Core (P0) | `id_feature` | hash bucketize / vocab lookup | P0 |
| Core (P0) | `raw_feature` | normalizer (log10 / zscore / minmax) | P0 |
| Core (P0) | `combo_feature` | cartesian product | P0 |
| Core (P1) | `lookup_feature` | KV map lookup | P1 |
| Core (P1) | `expr_feature` | expression parse (`eval`) | P1 |
| Advanced (P2) | `match_feature` | nested dict match | P2 |
| Advanced (P2) | `overlap_feature` | string similarity | P2 |
| Advanced (P2) | `sequence_feature` | sequence parse + combine | P2 |
| Advanced (P2) | `custom_feature` | user-defined op | P2 |

### Architecture

```
+-------------------------------------------------------------+
|              PurePythonFG (纯 Python FG)                     |
+-------------------------------------------------------------+
|                                                              |
|  +---------------------+    +---------------------+            |
|  |  FG Config          -->|  PyArrowHandler     |            |
|  |  (JSON/Proto)       |    |                    |            |
|  +---------------------+    +---------+-----------+            |
|                                      |                    |
|  +-----------------------------------+--------------------+  |
|  |              Feature Processors       |                    |
|  +-----------------------------------+--------------------+  |
|  | IdFeatureProcessor  | RawFeatureProcessor       |  |
|  | ComboFeatureProc |  LookupFeatureProcessor   |  |
|  | ExprFeatureProc  |  MatchFeatureProcessor   |  |
|  | ...                                            |  |
|  +-----------------------------------+----------------+  |
|                                      |                    |
|                                      v                    |
|  +-------------------------------------------------+        |
|  |         ParsedData (输出)                            |        |
|  |  SparseData / DenseData / SequenceSparseData    |        |
|  +-------------------------------------------------+        |
+-------------------------------------------------------------+
```

### New FG Mode

```protobuf
// tzrec/protos/data.proto
enum FgMode {
    FG_NONE = 0;
    FG_NORMAL = 1;
    FG_DAG = 2;
    FG_BUCKETIZE = 3;
    FG_PURE_PYTHON = 4;  // 新增
}
```

### DataParser Integration

```python
# tzrec/datasets/data_parser.py
if self._fg_mode == FgMode.FG_PURE_PYTHON:
    from tzrec.features.pure_python_fg import PurePythonFGHandler
    self._fg_handler = PurePythonFGHandler(fg_json)
```

### Risks and Mitigations

| Risk | Impact | Mitigation |
|------|--------|-----------|
| Expression safety | `expr_feature` may execute arbitrary code | Sandbox `eval` |
| Performance | Python may be slower than C++ | PyArrow vectorization + threading |
| Compatibility | May differ from pyfg behavior | Regression test against pyfg |

### Success Criteria

- 80% coverage of real-world scenarios
- 100% behavior parity with `FG_NONE` (no FG applied)
- < 20% performance regression
- Zero pyfg dependency
