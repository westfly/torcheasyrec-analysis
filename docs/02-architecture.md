---
title: Architecture Overview
nav_order: 3
---

# Architecture Overview

## Layers

TorchEasyRec's architecture is organized into six layers, each with a clear responsibility:

```
 Pipeline Config (protobuf)
        │
        ▼
 ┌─────────────┐    ┌──────────────────┐
 │  Data Layer  │───▶│  Feature Layer   │
 │  (Dataset)   │    │  (Parse + FG)    │
 └─────────────┘    └────────┬─────────┘
                             │
                             ▼
 ┌─────────────────────────────────────┐
 │         Embedding Layer             │
 │  (EmbeddingGroup → EBC/EC + ZCH)    │
 └──────────────┬──────────────────────┘
                │
                ▼
 ┌─────────────────────────────────────┐
 │         Model Layer                 │
 │  BaseModel → RankModel/MatchModel   │
 │  → DeepFM, DSSM, MMoE, HSTU...     │
 └──────────────┬──────────────────────┘
                │
                ▼
 ┌─────────────────────────────────────┐
 │    Loss / Metric / Export Layer     │
 └─────────────────────────────────────┘
```

## High-Level System View

```
┌────────────────────────────────────────────────────────────────────────────┐
│                              TorchEasyRec                                    │
├────────────────────────────────────────────────────────────────────────────┤
│                                                                            │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────┐                   │
│  │  Pipeline    │───▶│   Feature    │───▶│    Model     │                   │
│  │   Config     │    │   System     │    │   System     │                   │
│  │  (Protobuf)  │    │              │    │              │                   │
│  └──────────────┘    └──────────────┘    └──────────────┘                   │
│         │                   │                   │                            │
│         ▼                   ▼                   ▼                            │
│  ┌─────────────────────────────────────────────────────────────┐           │
│  │                     Data Pipeline                             │           │
│  │  CSV │ Parquet │ ODPS │ Kafka  ──▶  PyArrow  ──▶  Batch     │           │
│  └─────────────────────────────────────────────────────────────┘           │
│                              │                                              │
│                              ▼                                              │
│  ┌─────────────────────────────────────────────────────────────┐           │
│  │              Training / Evaluation / Export / Predict          │           │
│  └─────────────────────────────────────────────────────────────┘           │
│                                                                            │
└────────────────────────────────────────────────────────────────────────────┘
```

The three top-level subsystems (Pipeline Config, Feature System, Model
System) are independent protobuf-driven modules that share a common
`Batch` data structure. The Data Pipeline joins them at runtime, and the
Training/Evaluation/Export/Predict layer drives the model through its
lifecycle.

## Data Flow (Training)

A single training step follows this pipeline:

```
1. Dataset.__iter__() → raw RecordBatch (pyarrow)
        │
        ▼
2. DataParser.to_batch() → Batch(sparse_features=KJT,
        │                     dense_features=KT,
        │                     labels=dict, sample_weights=dict)
        │
        ▼
3. TrainWrapper.forward(batch)
        │
        ├─▶ model.predict(batch)
        │       │
        │       ├─▶ EmbeddingGroup(batch) → grouped_features dict
        │       │       │
        │       │       ├─ EmbeddingBagCollection (sparse)
        │       │       ├─ ManagedCollisionEmbeddingBagCollection (ZCH)
        │       │       ├─ DenseEmbeddingCollection (dense)
        │       │       └─ SequenceEmbeddingGroupImpl (sequence)
        │       │
        │       └─▶ model-specific forward (DeepFM, DSSM, etc.) → predictions
        │
        ├─▶ model.loss(predictions, batch) → dict of losses
        │
        └─▶ total_loss = sum(losses) → backward
```

## Pipeline Config (`EasyRecConfig`)

The entire pipeline is configured via a single protobuf:

[`torcheasyrec/tzrec/protos/pipeline.proto`](../torcheasyrec/tzrec/protos/pipeline.proto#L11-L29):

```protobuf
message EasyRecConfig {
    required string train_input_path = 1;
    required string eval_input_path = 2;
    required string model_dir = 3;
    optional TrainConfig train_config = 4;
    optional EvalConfig eval_config = 5;
    optional ExportConfig export_config = 6;
    optional DataConfig data_config = 7;
    repeated FeatureConfig feature_configs = 8;
    optional ModelConfig model_config = 9;
}
```

## Model Config (`ModelConfig`)

[`torcheasyrec/tzrec/protos/model.proto`](../torcheasyrec/tzrec/protos/model.proto#L45-L96) defines the model as a oneof:

```protobuf
message ModelConfig {
    repeated FeatureGroupConfig feature_groups = 1;
    oneof model {
        DLRM dlrm = 100;
        DeepFM deepfm = 101;
        MultiTower multi_tower = 102;
        ...
        DSSM dssm = 301;
        TDM tdm = 400;
    }
    optional uint32 num_class = 2;
    repeated LossConfig losses = 3;
    repeated MetricConfig metrics = 4;
}
```

## Key Architecture Decisions

### 1. Configuration-Driven

Models, features, data sources, training, and export are all defined in protobuf configs. This enables:
- Zero-code model experimentation
- Easy parameter sweeps
- Consistent config between training and serving

### 2. Feature Generation (FG) Encapsulation

Each `BaseFeature` subclass owns its FG logic (`_fg_json()`). Features can be:
- **FG_NONE**: pre-encoded (IDs already hashed, values already normalized)
- **FG_NORMAL**: feature config parsed by pyfg
- **FG_DAG**: DAG-based FG with intermediate feature dependencies

### 3. Embedding Separation

Sparse features (categorical) use `EmbeddingBagCollection` from TorchRec. Dense features with custom embedding (AutoDis, MLP) use `DenseEmbeddingCollection`. The `EmbeddingGroup` class orchestrates both and groups features by data group (base, neg, user).

### 4. Model Agnostic Training Loop

The training loop in [`main.py`](../torcheasyrec/tzrec/main.py#L317-L530) (`_train_and_evaluate()`) is model-agnostic. All model-specific logic is in:
- `model.predict(batch)` → predictions dict
- `model.loss(predictions, batch)` → losses dict
- `model.update_metric(predictions, batch, losses)` → metric state

### 5. Wrapper Pattern

Three wrappers decouple the model from pipeline concerns:
- [`TrainWrapper`](../torcheasyrec/tzrec/models/model.py#L235-L288): adds autocast, loss aggregation, Pareto MTL
- [`PredictWrapper`](../torcheasyrec/tzrec/models/model.py#L291-L340): adds output column filtering, CPU offload
- [`ScriptWrapper`](../torcheasyrec/tzrec/models/model.py#L343-L393): adds data parsing for JIT/torch.export

## Inference Architecture

For production serving, TorchEasyRec supports multiple export paths:

```
Training Checkpoint
        │
        ├─▶ JIT Script (TorchScript) → TensorRT
        ├─▶ torch.export → AOTInductor
        └─▶ Combined: JIT sparse + AOTI dense
```

The [`CombinedModelWrapper`](../torcheasyrec/tzrec/models/model.py#L453-L512) splits the model into sparse (embedding, scripted) and dense (MLP, AOTInductor) parts for optimized inference.

## References

| File | Purpose |
|------|---------|
| [`torcheasyrec/tzrec/main.py`](../torcheasyrec/tzrec/main.py) | `train_and_evaluate()`, `_train_and_evaluate()`, `_evaluate()` |
| [`torcheasyrec/tzrec/models/model.py`](../torcheasyrec/tzrec/models/model.py) | `BaseModel`, `TrainWrapper`, `PredictWrapper`, `ScriptWrapper` |
| [`torcheasyrec/tzrec/protos/pipeline.proto`](../torcheasyrec/tzrec/protos/pipeline.proto) | `EasyRecConfig` |
| [`torcheasyrec/tzrec/protos/model.proto`](../torcheasyrec/tzrec/protos/model.proto) | `ModelConfig`, `FeatureGroupConfig` |
| [`torcheasyrec/tzrec/utils/config_util.py`](../torcheasyrec/tzrec/utils/config_util.py) | Config loading and editing utilities |
