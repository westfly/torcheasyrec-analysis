---
title: Home
nav_order: 1
---

# TorchEasyRec Source Analysis

TorchEasyRec is an **Alibaba open-source, PyTorch-based recommendation system framework** for production-ready deep learning models. It implements 20+ state-of-the-art models for candidate generation (matching), scoring (ranking), multi-task learning, and generative recommendation.

## What's Inside

| Doc | Description |
|-----|-------------|
| [Project Overview](01-project-overview) | What TorchEasyRec is, why it exists, key features |
| [Architecture Overview](02-architecture) | Layers, data flow, and how components connect |
| [Code Structure](03-code-structure) | Directory layout, build system, proto configs |
| [Initialization Flow](04-initialization-flow) | From CLI to ready-to-train |
| [Training Flow](05-training-flow) | Training pipeline: data → feature → model → loss |
| [Model System](06-model-system) | BaseModel → RankModel → 20+ concrete models |
| [Feature System](07-feature-system) | 10+ feature types, FG modes, parsing |
| [Embedding System](08-embedding-system) | EmbeddingGroup, TorchRec, distributed sharding |
| [Export Pipeline](09-export-pipeline) | JIT/TRT/AOTI/RTP export, FX graph surgery, sparse/dense split |
| [DynamicEmb Integration](10-dynamicemb-integration) | NVIDIA GPU hash-table embedding backend, integration shim |

## Quick Stats

- **457+ commits**, **392 stars**, **74 forks**
- **295 Python files** in `tzrec/`
- **20+ models**: DSSM, TDM, DeepFM, DIN, MMoE, PLE, PEPNet, DLRM-HSTU
- **10+ feature types**: IdFeature, RawFeature, SequenceFeature, etc.
- **Built on**: PyTorch, TorchRec, pyfg (feature generation)
