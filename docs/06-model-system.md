---
title: Model System
nav_order: 7
---

# Model System

## Class Hierarchy

```
BaseModule (modules/utils.py)
    └── BaseModel (models/model.py)  [abstract]
            ├── RankModel (models/rank_model.py)     ─── ranking models
            ├── MatchModel (models/match_model.py)   ─── matching models
            ├── MultiTaskRank (models/multi_task_rank.py) ─── multi-task
            ├── TDM (models/tdm.py)                  ─── tree-based
            └── RocketLaunching (models/rocket_launching.py)
```

## BaseModel

[`torcheasyrec/tzrec/models/model.py`](../torcheasyrec/tzrec/models/model.py#L41-L228)

Defines the contract every model must implement:

```python
class BaseModel(BaseModule, metaclass=_meta_cls):
    def __init__(self, model_config, features, labels, sample_weights):
        self._model_config = model_config
        self._model_type = model_config.WhichOneof("model")
        self._features = features
        self._feature_groups = list(model_config.feature_groups)
        self._labels = labels
        self._metric_modules = nn.ModuleDict()
        self._loss_modules = nn.ModuleDict()

    def predict(self, batch) -> Dict[str, torch.Tensor]:  # Abstract
    def init_loss(self) -> None:                           # Abstract
    def loss(self, predictions, batch) -> Dict[str, torch.Tensor]:  # Abstract
    def init_metric(self) -> None:                         # Abstract
    def update_metric(self, predictions, batch, losses):   # Abstract
    def compute_metric(self) -> Dict[str, torch.Tensor]:   # Concrete
```

Models are **auto-registered** via `get_register_class_meta()` which maintains `_MODEL_CLASS_MAP`. The `create_class()` classmethod looks up models by name, enabling the config-driven instantiation in `_create_model()`.

### Wrappers

BaseModel is used through several wrappers:

| Wrapper | Purpose | Key Behavior |
|---------|---------|-------------|
| [`TrainWrapper`](../torcheasyrec/tzrec/models/model.py#L235-L288) | Training | autocast, loss aggregation, Pareto MTL |
| [`PredictWrapper`](../torcheasyrec/tzrec/models/model.py#L291-L340) | Evaluation | output column filtering, GPU→CPU |
| [`ScriptWrapper`](../torcheasyrec/tzrec/models/model.py#L343-L393) | JIT export | adds DataParser for tensor→Batch |
| [`CombinedModelWrapper`](../torcheasyrec/tzrec/models/model.py#L453-L512) | TRT export | splits sparse (scripted) + dense (AOTI) |
| [`UnifiedAOTIModelWrapper`](../torcheasyrec/tzrec/models/model.py#L515-L554) | AOTI export | single AOTInductor model, thread-safe |

## RankModel

[`torcheasyrec/tzrec/models/rank_model.py`](../torcheasyrec/tzrec/models/rank_model.py#L57-L523)

Base class for ranking (scoring) models. Provides:

- **`init_input()`**: Creates `EmbeddingGroup` + optional variational dropout
- **`build_input(batch)`**: Runs embedding lookup, applies variational dropout
- **`_output_to_prediction()`**: Converts logits to predictions with sigmoid/softmax
- **`_init_loss_impl()`**: Creates loss modules (BCE, Focal, SoftmaxCE, JRC, MSE)
- **`_init_metric_impl()`**: Creates metric modules (AUC, GroupedAUC, XAUC, etc.)
- **`_init_train_metric_impl()`**: Creates training metrics (DecayAUC, etc.)

### Ranking Models

| Model | File | Key Innovation |
|-------|------|---------------|
| DeepFM | [`deepfm.py`](../torcheasyrec/tzrec/models/deepfm.py) | FM + MLP parallel |
| MultiTower | [`multi_tower.py`](../torcheasyrec/tzrec/models/multi_tower.py) | Multiple input towers |
| MultiTowerDIN | [`multi_tower_din.py`](../torcheasyrec/tzrec/models/multi_tower_din.py) | DIN attention on sequences |
| WideAndDeep | [`wide_and_deep.py`](../torcheasyrec/tzrec/models/wide_and_deep.py) | Wide (memorization) + Deep (generalization) |
| DCN | [`dcn.py`](../torcheasyrec/tzrec/models/dcn.py) | Cross network for explicit feature interaction |
| DCN-V2 | [`dcn_v2.py`](../torcheasyrec/tzrec/models/dcn_v2.py) | Improved cross network |
| DLRM | [`dlrm.py`](../torcheasyrec/tzrec/models/dlrm.py) | Feature interaction + MLP |
| MaskNet | [`masknet.py`](../torcheasyrec/tzrec/models/masknet.py) | Instance-guided mask |
| xDeepFM | [`xdeepfm.py`](../torcheasyrec/tzrec/models/xdeepfm.py) | Compressed Interaction Network (CIN) |
| WuKong | [`wukong.py`](../torcheasyrec/tzrec/models/wukong.py) | Dense scaling + high-order interactions |
| RocketLaunching | [`rocket_launching.py`](../torcheasyrec/tzrec/models/rocket_launching.py) | Knowledge distillation |

## MatchModel

[`torcheasyrec/tzrec/models/match_model.py`](../torcheasyrec/tzrec/models/match_model.py#L246-L471)

Base class for matching (candidate generation) models. Key concepts:

- **Two-tower architecture**: user tower + item tower
- **Similarity computation**: dot product / cosine similarity
- **In-batch negative sampling**: optionally use other items in batch as negatives
- **Loss**: always `softmax_cross_entropy`

```python
class MatchModel(BaseModel):
    def sim(self, user_emb, item_emb, hard_neg_indices):
        if self._in_batch_negative:
            return torch.mm(user_emb, item_emb.T)
        else:
            return _sim_with_sampler(user_emb, item_emb, hard_neg_indices)
```

### Match Models

| Model | File | Key Innovation |
|-------|------|---------------|
| DSSM | [`dssm.py`](../torcheasyrec/tzrec/models/dssm.py) | Two-tower deep semantic matching |
| DSSM-V2 | [`dssm_v2.py`](../torcheasyrec/tzrec/models/dssm_v2.py) | Improved DSSM |
| DAT | [`dat.py`](../torcheasyrec/tzrec/models/dat.py) | Dual Augmented two-tower |
| MIND | [`mind.py`](../torcheasyrec/tzrec/models/mind.py) | Multi-interest with dynamic routing |
| TDM | [`tdm.py`](../torcheasyrec/tzrec/models/tdm.py) | Tree-based deep retrieval |

## MultiTask Models

[`torcheasyrec/tzrec/models/multi_task_rank.py`](../torcheasyrec/tzrec/models/multi_task_rank.py)

| Model | File | Key Innovation |
|-------|------|---------------|
| MMoE | [`mmoe.py`](../torcheasyrec/tzrec/models/mmoe.py) | Multi-gate Mixture-of-Experts |
| PLE | [`ple.py`](../torcheasyrec/tzrec/models/ple.py) | Progressive Layered Extraction |
| DBMTL | [`dbmtl.py`](../torcheasyrec/tzrec/models/dbmtl.py) | Deep Bayesian MTL |
| PEPNet | [`pepnet.py`](../torcheasyrec/tzrec/models/pepnet.py) | Personalized Embedding & Parameter Net |
| DC2VR | [`dc2vr.py`](../torcheasyrec/tzrec/models/dc2vr.py) | Deep Cross network for VR |

## Generative Recommendation Models

| Model | File | Key Innovation |
|-------|------|---------------|
| DLRM-HSTU | [`dlrm_hstu.py`](../torcheasyrec/tzrec/models/dlrm_hstu.py) | HSTU transducer for generative rec |
| ULTRA-HSTU | [`ultra_hstu.py`](../torcheasyrec/tzrec/models/ultra_hstu.py) | Semi-local attention, attention truncation, MoT |
| HSTU-Match | (in match_models) | HSTU-based two-tower retrieval |

## Model Implementation Pattern

Every model follows the same pattern. Here's DeepFM as an example:

[`torcheasyrec/tzrec/models/deepfm.py`](../torcheasyrec/tzrec/models/deepfm.py)

```python
class DeepFM(RankModel):
    def __init__(self, model_config, features, labels, sample_weights, **kwargs):
        super().__init__(model_config, features, labels, sample_weights, **kwargs)
        self.wide_embedding_dim = self._model_config.wide_embedding_dim
        self.init_input()  # Creates EmbeddingGroup
        self.fm = FactorizationMachine()
        # Get feature dims for FM and Deep groups
        self._fm_feature_dims = self.embedding_group.group_dims("fm")
        deep_feature_dim = self.embedding_group.group_total_dim("deep")
        self.deep_mlp = MLP(deep_feature_dim, ...)
        self.output_mlp = nn.Linear(final_dim, self._num_class)

    def predict(self, batch):
        grouped_features = self.build_input(batch)
        # Each named group (wide, deep, fm) → tensor
        y_wide = grouped_features["wide"].sum(dim=1, keepdim=True)
        y_deep = self.deep_mlp(grouped_features["deep"])
        y_fm = self.fm(grouped_features["fm"].reshape(...))
        y = y_wide + y_fm + self.output_mlp(y_deep)
        return self._output_to_prediction(y)
```

## Config → Model Mapping

The model config name (from proto oneof) maps to class via auto-registration:

```
"deepfm" → DeepFM (in deepfm.py)
"dssm"   → DSSM (in dssm.py)
"mmoe"   → MMoE (in mmoe.py)
...
```

Registration is automatic: the metaclass (`_meta_cls`) records all `BaseModel` subclasses. When `train_eval.py` imports `tzrec`, `auto_import()` scans and imports all modules, triggering all registrations.

## Key Files

| File | Models |
|------|--------|
| [`torcheasyrec/tzrec/models/model.py`](../torcheasyrec/tzrec/models/model.py) | BaseModel, TrainWrapper, PredictWrapper, wrappers |
| [`torcheasyrec/tzrec/models/rank_model.py`](../torcheasyrec/tzrec/models/rank_model.py) | RankModel base (loss, metric, prediction) |
| [`torcheasyrec/tzrec/models/match_model.py`](../torcheasyrec/tzrec/models/match_model.py) | MatchModel, MatchTower, similarity |
| [`torcheasyrec/tzrec/models/multi_task_rank.py`](../torcheasyrec/tzrec/models/multi_task_rank.py) | MultiTaskModel base |
| [`torcheasyrec/tzrec/models/deepfm.py`](../torcheasyrec/tzrec/models/deepfm.py) | DeepFM (reference implementation) |
| [`torcheasyrec/tzrec/protos/models/rank_model.proto`](../torcheasyrec/tzrec/protos/models/rank_model.proto) | Rank model protos |
| [`torcheasyrec/tzrec/protos/models/match_model.proto`](../torcheasyrec/tzrec/protos/models/match_model.proto) | Match model protos |

## Complete Model Catalog

### Ranking Models (Single-Task)

| Model | Sequence | Sparse Output | Feature Interaction |
|-------|---------|---------------|---------------------|
| DeepFM | ❌ | KeyedTensor | FM + MLP |
| WideAndDeep | ❌ | KeyedTensor | Linear + MLP |
| MultiTower | ❌ | KeyedTensor | Multi-tower concat |
| **MultiTowerDIN** | ✅ | Mixed (KT + Dict) | Multi-tower + DIN attention |
| DLRM | ❌ | KeyedTensor | Per-bit dot product |
| **DLRM-HSTU** | ✅ | Mixed (KT + Dict) | HSTU transducer, generative |
| DCN | ❌ | KeyedTensor | Cross + MLP |
| DCN V2 | ❌ | KeyedTensor | Improved cross |
| xDeepFM | ❌ | KeyedTensor | CIN + MLP |
| MaskNet | ❌ | KeyedTensor | Mask + MLP |
| WuKong | ❌ | KeyedTensor | WuKong network |
| RocketLaunching | ❌ | KeyedTensor | Dual MLP + distillation |

**Feature interaction patterns:**

| Interaction | Models | Mechanism |
|-------------|--------|-----------|
| **FM** | DeepFM | 2nd-order feature crossing |
| **Cross** | DCN, DCN V2 | Explicit cross network |
| **CIN** | xDeepFM | Compressed Interaction Network |
| **Dot** | DLRM | Per-bit dot product |
| **Attention** | MultiTowerDIN, DAT, MIND | Attention mechanism |
| **MoE** | MMoE, DBMTL | Mixture of Experts |
| **Mask** | MaskNet, DBMTL | Mask mechanism |
| **HSTU** | DLRM-HSTU, HSTU | Hierarchical Sequential |
| **Direct** | MultiTower, WideAndDeep | Direct concat |

### Multi-Task Ranking Models

| Model | Sequence | Sparse Output | Sharing |
|-------|---------|---------------|---------|
| MMoE | ❌ | KeyedTensor | Soft share (all experts) |
| PLE | ❌ | KeyedTensor | Hard share (dedicated + shared) |
| PEPNet | ❌ | KeyedTensor | Parameter-level personalization |
| **DBMTL** | ✅ | Mixed | Mask + MMoE + sequence |
| DC2VR | ❌ | KeyedTensor | DCN V2 + CGC |

### Matching Models

| Model | Sequence | Sparse Output | User/Item |
|-------|---------|---------------|-----------|
| DSSM | ❌ | KeyedTensor | Two-tower |
| DSSM V2 | ❌ | KeyedTensor | Two-tower |
| **DAT** | ✅ | Mixed | Two-tower |
| **MIND** | ✅ | Mixed | Two-tower |
| HSTU | ✅ | Mixed | Two-tower |
| **TDM** | ✅ | Mixed | Single-tower (tree index) |

**Sequence encoders:**

| Encoder | Used By | Mechanism |
|---------|---------|-----------|
| `DINEncoder` | MultiTowerDIN | Target-aware attention pooling |
| `MultiWindowDINEncoder` | TDM | Multi-window attention |
| `HSTUEncoder` | HSTU | A2A communication |
| `CapsuleLayer` | MIND | Multi-interest capsule |
| `DATTower` | DAT | Dual attention |

## Model Selection Decision Trees

### Single-Task Ranking

```
单任务排序模型选择:

是否有序列特征?
│
├── 否 → 特征交互方式?
│   ├── 通用推荐 → DeepFM
│   ├── 需要记忆 → WideAndDeep
│   ├── 稠密特征为主 → DLRM
│   ├── 显式高阶交叉 → DCN / DCN V2
│   ├── CIN 交叉 → xDeepFM
│   ├── 特征分组 → MultiTower
│   └── 高阶交互 → WuKong / MaskNet
│
└── 是 → 序列类型?
    ├── 短期兴趣 (DIN) → MultiTowerDIN
    └── 生成式序列 → DLRM-HSTU
```

### Multi-Task

```
多任务排序模型选择:

任务相关性如何?
│
├── 很低 (正交) → MMoE (Soft 共享)
├── 中等 → PLE (Hard 分离)
├── 需要参数个性化 → PEPNet
├── 有序列信号 → DBMTL (Mask + 序列)
└── 需要特征交叉 → DC2VR (DCN V2 + CGC)
```

### Matching

```
┌─────────────────────────────────────────────────────────────────┐
│                      匹配模型选择                                │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  规模多大?                                                     │
│  │                                                              │
│  ├── 百万级                                                    │
│  │   └── DSSM / DSSM V2 (简单高效)                            │
│  │                                                              │
│  ├── 千万级                                                    │
│  │   ├── 需要多兴趣 → MIND                                     │
│  │   ├── 序列信号强 → DAT / HSTU                              │
│  │   └── 通用 → DSSM V2                                       │
│  │                                                              │
│  └── 亿级+                                                     │
│      └── TDM (树索引加速)                                      │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

### Match → Rank Pipeline

```
┌─────────────────────────────────────────────────────────────────┐
│                    两阶段推荐系统                                 │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  ┌─────────┐     ┌─────────┐     ┌─────────┐                   │
│  │  召回    │ ──▶ │  粗排   │ ──▶ │  精排   │                   │
│  │ (Match)  │     │ (Rank)  │     │ (Rank)  │                   │
│  └─────────┘     └─────────┘     └─────────┘                   │
│                                                                  │
│  召回模型: DSSM / DAT / MIND / HSTU / TDM                      │
│  粗排模型: MultiTower / DeepFM                                 │
│  精排模型: MultiTowerDIN / DLRM-HSTU / DCN                     │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

## Export Compatibility Matrix

Every model in the catalog supports all four export backends:

| Category | Model | Default | AOT | TRT | RTP | INPUT_TILE |
|----------|-------|---------|-----|-----|-----|-----------|
| **Rank-Single** | DeepFM | ✅ | ✅ | ✅ | ✅ | ✅ |
| | WideAndDeep | ✅ | ✅ | ✅ | ✅ | ✅ |
| | MultiTower | ✅ | ✅ | ✅ | ✅ | ✅ |
| | **MultiTowerDIN** | ✅ | ✅ | ✅ | ✅ | ✅ |
| | DLRM | ✅ | ✅ | ✅ | ✅ | ✅ |
| | **DLRM-HSTU** | ✅ | ✅ | ✅ | ✅ | ✅ |
| | DCN | ✅ | ✅ | ✅ | ✅ | ✅ |
| | DCN V2 | ✅ | ✅ | ✅ | ✅ | ✅ |
| | xDeepFM | ✅ | ✅ | ✅ | ✅ | ✅ |
| | MaskNet | ✅ | ✅ | ✅ | ✅ | ✅ |
| | WuKong | ✅ | ✅ | ✅ | ✅ | ✅ |
| | RocketLaunching | ✅ | ✅ | ✅ | ✅ | ✅ |
| **Rank-Multi** | MMoE | ✅ | ✅ | ✅ | ✅ | ✅ |
| | PLE | ✅ | ✅ | ✅ | ✅ | ✅ |
| | PEPNet | ✅ | ✅ | ✅ | ✅ | ✅ |
| | **DBMTL** | ✅ | ✅ | ✅ | ✅ | ✅ |
| | DC2VR | ✅ | ✅ | ✅ | ✅ | ✅ |
| **Match** | DSSM | ✅ | ✅ | ✅ | ✅ | ✅ |
| | DSSM V2 | ✅ | ✅ | ✅ | ✅ | ✅ |
| | **DAT** | ✅ | ✅ | ✅ | ✅ | ✅ |
| | **MIND** | ✅ | ✅ | ✅ | ✅ | ✅ |
| | HSTU | ✅ | ✅ | ✅ | ✅ | ✅ |
| | **TDM** | ✅ | ✅ | ✅ | ✅ | ✅ |

### `WORLD_SIZE` Constraint

| Export | `WORLD_SIZE` | Notes |
|--------|-------------|-------|
| Default | 1 | Single-process export |
| AOT | 1 | Single-process export |
| TRT | 1 | Single-process export |
| RTP | N | Supports distributed export |

### Known Limitations

| Model | Limitation | Notes |
|-------|-----------|-------|
| DLRM-HSTU | Special handling | Uses `JaggedTensor`, needs `_fx_construct_payload` wrapper |
| RocketLaunching | Special handling | Includes distillation logic |

### Sparse Output Format

| Format | Models | Why |
|--------|--------|-----|
| `KeyedTensor` (pooled) | DeepFM, WideAndDeep, DCN, MMoE, … | All non-sequence features are pooled into fixed-size vectors per EBC |
| `Dict` (unpooled) | DLRM-HSTU, sequence models | Sequence outputs are per-step, can't be packed into a single `KeyedTensor` |
| Mixed | MultiTowerDIN, DBMTL, DAT, MIND, TDM | Some feature groups are pooled (KT) and some are sequential (Dict) |

The mixed output format is what forces the `fx_mark_*` sentinel design
described in [09-export-pipeline.md](09-export-pipeline): the boundary
between sparse and dense must be marked at multiple points in the
graph, and each mark carries a different type signature.
