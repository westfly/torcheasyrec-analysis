---
title: Project Overview & Background
nav_order: 2
---

# Project Overview & Background

## What is TorchEasyRec?

TorchEasyRec is a **PyTorch-based recommendation system framework** developed by Alibaba's PAI team. It provides a complete workflow for building, training, and deploying large-scale deep learning recommendation models.

Unlike general-purpose DL frameworks, TorchEasyRec is purpose-built for recommendation tasks:

- **Matching (Candidate Generation)**: DSSM, MIND, TDM, DAT
- **Scoring (Ranking)**: DeepFM, DIN, DLRM, DCN, xDeepFM, WuKong
- **Multi-Task Learning**: MMoE, PLE, DBMTL, PEPNet
- **Generative Recommendation**: DLRM-HSTU, ULTRA-HSTU, HSTU-Match

## Problem It Solves

Building production recommendation systems requires solving many hard problems:

1. **Feature engineering** — handling categorical features (millions of IDs), raw numerical features, sequential behavior features, cross features
2. **Large-scale embeddings** — embedding tables can be terabytes; need distributed sharding
3. **Training efficiency** — hybrid data/model parallelism, mixed precision, gradient clipping
4. **Serving** — TensorRT / AOTInductor acceleration, EAS deployment on Alibaba Cloud
5. **Feature generation consistency** — same feature transforms for training and serving

TorchEasyRec addresses all of these with a **configuration-driven** approach: users define data schema, feature configs, and model architecture in protobuf configs, and the framework builds the corresponding PyTorch model.

## Key Differentiators

| Feature | TorchEasyRec | EasyRec (TF) | TorchRec | XGBoost-based |
|---------|-------------|-------------|----------|---------------|
| Backend | PyTorch | TensorFlow | PyTorch | N/A |
| Model Count | 20+ | 20+ | 1 (DLRM) | ~5 |
| Distributed Training | TorchRec sharding | TF distribution | Native | No |
| Feature Generation | pyfg (DAG-based) | Feature configs | Manual | Manual |
| Custom Models | Plugin-based | Plugin-based | Write from scratch | N/A |
| Cloud Integration | MaxCompute, PAI, EAS | Similar | N/A | N/A |
| Inference Acceleration | TensorRT, AOTInductor | TF-TRT | AOTInductor | N/A |

## When to Use TorchEasyRec

- You need a **production-ready** recommendation system on Alibaba Cloud
- You want to quickly experiment with 20+ SOTA models without writing code
- You need **distributed training** for large-scale embeddings
- You require **consistent feature generation** between training and serving
- You're migrating from EasyRec (TF) to PyTorch

## High-Level Architecture

```
┌─────────────────────────────────────────────────────────┐
│                   Pipeline Config                        │
│  (EasyRecConfig: data + features + model + train/eval)   │
└─────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────┐
│               Data Layer (datasets/)                     │
│  CSV  │  Parquet  │  ODPS  │  Kafka  │  ...               │
└─────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────┐
│             Feature Layer (features/)                     │
│  IdFeature │ RawFeature │ SequenceFeature │ ComboFeature  │
│  + pyfg (Feature Generation DAG)                         │
└─────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────┐
│           Embedding Layer (modules/embedding.py)          │
│  EmbeddingBagCollection │ EmbeddingCollection            │
│  + ManagedCollision (ZCH) │ + TorchRec sharding          │
└─────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────┐
│              Model Layer (models/)                        │
│  BaseModel → RankModel / MatchModel / MultiTaskRank       │
│  → DeepFM │ DSSM │ MMoE │ HSTU │ ...                     │
└─────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────┐
│           Loss / Metric / Export Layer                    │
│  BCE │ Focal │ JRC │ Softmax │ AUC │ Recall@K │ TRT/AOTI │
└─────────────────────────────────────────────────────────┘
```

## References

- [Official Docs](https://torcheasyrec.readthedocs.io/)
- [GitHub Repository](../torcheasyrec/)
- [EasyRec (TensorFlow predecessor)](https://github.com/alibaba/EasyRec)

## Key Files

| File | Purpose |
|------|---------|
| [`torcheasyrec/tzrec/main.py`](../torcheasyrec/tzrec/main.py) | Core training, evaluation, export pipeline |
| [`torcheasyrec/tzrec/models/model.py`](../torcheasyrec/tzrec/models/model.py) | BaseModel and wrappers (TrainWrapper, PredictWrapper) |
| [`torcheasyrec/tzrec/models/rank_model.py`](../torcheasyrec/tzrec/models/rank_model.py) | RankModel base — loss, metric, prediction logic |
| [`torcheasyrec/tzrec/models/match_model.py`](../torcheasyrec/tzrec/models/match_model.py) | MatchModel base — embedding similarity, in-batch neg |
| [`torcheasyrec/tzrec/features/feature.py`](../torcheasyrec/tzrec/features/feature.py) | BaseFeature and create_features() factory |
| [`torcheasyrec/tzrec/modules/embedding.py`](../torcheasyrec/tzrec/modules/embedding.py) | EmbeddingGroup, EmbeddingGroupImpl, SequenceEmbeddingGroupImpl |
| [`torcheasyrec/tzrec/protos/pipeline.proto`](../torcheasyrec/tzrec/protos/pipeline.proto) | Pipeline config (EasyRecConfig) definition |
| [`torcheasyrec/tzrec/protos/model.proto`](../torcheasyrec/tzrec/protos/model.proto) | Model config definition (all model types) |
