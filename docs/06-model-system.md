---
title: 模型系统
nav_order: 3
has_children: true
parent: 训练篇
---

# 模型系统

## 类层级

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

定义每个模型必须实现的契约：

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

模型通过 `get_register_class_meta()` **自动注册**，它维护 `_MODEL_CLASS_MAP`。`create_class()` classmethod 按名称查找模型，从而支持 `_create_model()` 中的配置驱动实例化。

### Wrappers

BaseModel 通过几个 wrapper 使用：

| Wrapper | 用途 | 关键行为 |
|---------|------|---------|
| [`TrainWrapper`](../torcheasyrec/tzrec/models/model.py#L235-L288) | 训练 | autocast、损失聚合、Pareto MTL |
| [`PredictWrapper`](../torcheasyrec/tzrec/models/model.py#L291-L340) | 评估 | 输出列过滤、GPU→CPU |
| [`ScriptWrapper`](../torcheasyrec/tzrec/models/model.py#L343-L393) | JIT 导出 | 为 tensor→Batch 添加 DataParser |
| [`CombinedModelWrapper`](../torcheasyrec/tzrec/models/model.py#L453-L512) | TRT 导出 | 切分稀疏（已 script）+ 稠密（AOTI） |
| [`UnifiedAOTIModelWrapper`](../torcheasyrec/tzrec/models/model.py#L515-L554) | AOTI 导出 | 单一 AOTInductor 模型、线程安全 |

## RankModel

[`torcheasyrec/tzrec/models/rank_model.py`](../torcheasyrec/tzrec/models/rank_model.py#L57-L523)

排序（打分）模型的基类。提供：

- **`init_input()`**：创建 `EmbeddingGroup` + 可选的 variational dropout
- **`build_input(batch)`**：运行嵌入查找，应用 variational dropout
- **`_output_to_prediction()`**：将 logits 转换为带 sigmoid/softmax 的 predictions
- **`_init_loss_impl()`**：创建损失模块（BCE、Focal、SoftmaxCE、JRC、MSE）
- **`_init_metric_impl()`**：创建指标模块（AUC、GroupedAUC、XAUC 等）
- **`_init_train_metric_impl()`**：创建训练指标（DecayAUC 等）

### 排序模型

| 模型 | 文件 | 核心创新 |
|------|------|----------|
| DeepFM | [`deepfm.py`](../torcheasyrec/tzrec/models/deepfm.py) | FM + MLP 并行 |
| MultiTower | [`multi_tower.py`](../torcheasyrec/tzrec/models/multi_tower.py) | 多输入塔 |
| MultiTowerDIN | [`multi_tower_din.py`](../torcheasyrec/tzrec/models/multi_tower_din.py) | 序列上 DIN 注意力 |
| WideAndDeep | [`wide_and_deep.py`](../torcheasyrec/tzrec/models/wide_and_deep.py) | Wide（记忆）+ Deep（泛化） |
| DCN | [`dcn.py`](../torcheasyrec/tzrec/models/dcn.py) | 显式特征交叉的 Cross network |
| DCN-V2 | [`dcn_v2.py`](../torcheasyrec/tzrec/models/dcn_v2.py) | 改进的 Cross network |
| DLRM | [`dlrm.py`](../torcheasyrec/tzrec/models/dlrm.py) | 特征交互 + MLP |
| MaskNet | [`masknet.py`](../torcheasyrec/tzrec/models/masknet.py) | 实例引导的 mask |
| xDeepFM | [`xdeepfm.py`](../torcheasyrec/tzrec/models/xdeepfm.py) | 压缩交互网络（CIN） |
| WuKong | [`wukong.py`](../torcheasyrec/tzrec/models/wukong.py) | 稠密扩展 + 高阶交互 |
| RocketLaunching | [`rocket_launching.py`](../torcheasyrec/tzrec/models/rocket_launching.py) | 知识蒸馏 |

## MatchModel

[`torcheasyrec/tzrec/models/match_model.py`](../torcheasyrec/tzrec/models/match_model.py#L246-L471)

匹配（候选生成）模型的基类。核心概念：

- **双塔架构**：user 塔 + item 塔
- **相似度计算**：点积 / 余弦相似度
- **批内负采样**：可选地将批内其他 item 作为负样本
- **损失**：始终为 `softmax_cross_entropy`

```python
class MatchModel(BaseModel):
    def sim(self, user_emb, item_emb, hard_neg_indices):
        if self._in_batch_negative:
            return torch.mm(user_emb, item_emb.T)
        else:
            return _sim_with_sampler(user_emb, item_emb, hard_neg_indices)
```

### 匹配模型

| 模型 | 文件 | 核心创新 |
|------|------|----------|
| DSSM | [`dssm.py`](../torcheasyrec/tzrec/models/dssm.py) | 双塔深度语义匹配 |
| DSSM-V2 | [`dssm_v2.py`](../torcheasyrec/tzrec/models/dssm_v2.py) | 改进版 DSSM |
| DAT | [`dat.py`](../torcheasyrec/tzrec/models/dat.py) | 双增强双塔 |
| MIND | [`mind.py`](../torcheasyrec/tzrec/models/mind.py) | 动态路由多兴趣 |
| TDM | [`tdm.py`](../torcheasyrec/tzrec/models/tdm.py) | 基于树的深度检索 |

## 多任务模型

[`torcheasyrec/tzrec/models/multi_task_rank.py`](../torcheasyrec/tzrec/models/multi_task_rank.py)

| 模型 | 文件 | 核心创新 |
|------|------|----------|
| MMoE | [`mmoe.py`](../torcheasyrec/tzrec/models/mmoe.py) | 多门控 Mixture-of-Experts |
| PLE | [`ple.py`](../torcheasyrec/tzrec/models/ple.py) | 渐进式分层提取 |
| DBMTL | [`dbmtl.py`](../torcheasyrec/tzrec/models/dbmtl.py) | 深度贝叶斯多任务 |
| PEPNet | [`pepnet.py`](../torcheasyrec/tzrec/models/pepnet.py) | 个性化 Embedding & Parameter Net |
| DC2VR | [`dc2vr.py`](../torcheasyrec/tzrec/models/dc2vr.py) | DCN 网络用于 VR |

## 生成式推荐模型

| 模型 | 文件 | 核心创新 |
|------|------|----------|
| DLRM-HSTU | [`dlrm_hstu.py`](../torcheasyrec/tzrec/models/dlrm_hstu.py) | 用于生成式推荐的 HSTU transducer |
| ULTRA-HSTU | [`ultra_hstu.py`](../torcheasyrec/tzrec/models/ultra_hstu.py) | 半局部注意力、注意力截断、MoT |
| HSTU-Match | (在 match_models 中) | 基于 HSTU 的双塔检索 |

## 模型实现模式

每个模型都遵循相同的模式。以 DeepFM 为例：

[`torcheasyrec/tzrec/models/deepfm.py`](../torcheasyrec/tzrec/models/deepfm.py)

```python
class DeepFM(RankModel):
    def __init__(self, model_config, features, labels, sample_weights, **kwargs):
        super().__init__(model_config, features, labels, sample_weights, **kwargs)
        self.wide_embedding_dim = self._model_config.wide_embedding_dim
        self.init_input()  # 创建 EmbeddingGroup
        self.fm = FactorizationMachine()
        # 获取 FM 与 Deep 组的特征维度
        self._fm_feature_dims = self.embedding_group.group_dims("fm")
        deep_feature_dim = self.embedding_group.group_total_dim("deep")
        self.deep_mlp = MLP(deep_feature_dim, ...)
        self.output_mlp = nn.Linear(final_dim, self._num_class)

    def predict(self, batch):
        grouped_features = self.build_input(batch)
        # 每个命名组 (wide, deep, fm) → tensor
        y_wide = grouped_features["wide"].sum(dim=1, keepdim=True)
        y_deep = self.deep_mlp(grouped_features["deep"])
        y_fm = self.fm(grouped_features["fm"].reshape(...))
        y = y_wide + y_fm + self.output_mlp(y_deep)
        return self._output_to_prediction(y)
```

## Config → Model 映射

模型配置名（来自 proto oneof）通过自动注册映射到类：

```
"deepfm" → DeepFM (在 deepfm.py)
"dssm"   → DSSM (在 dssm.py)
"mmoe"   → MMoE (在 mmoe.py)
...
```

注册是自动的：metaclass（`_meta_cls`）记录所有 `BaseModel` 子类。当 `train_eval.py` 导入 `tzrec` 时，`auto_import()` 扫描并导入所有模块，触发所有注册。

## 关键文件

| 文件 | 模型 |
|------|------|
| [`torcheasyrec/tzrec/models/model.py`](../torcheasyrec/tzrec/models/model.py) | BaseModel、TrainWrapper、PredictWrapper、wrappers |
| [`torcheasyrec/tzrec/models/rank_model.py`](../torcheasyrec/tzrec/models/rank_model.py) | RankModel 基类（loss、metric、prediction） |
| [`torcheasyrec/tzrec/models/match_model.py`](../torcheasyrec/tzrec/models/match_model.py) | MatchModel、MatchTower、similarity |
| [`torcheasyrec/tzrec/models/multi_task_rank.py`](../torcheasyrec/tzrec/models/multi_task_rank.py) | MultiTaskModel 基类 |
| [`torcheasyrec/tzrec/models/deepfm.py`](../torcheasyrec/tzrec/models/deepfm.py) | DeepFM（参考实现） |
| [`torcheasyrec/tzrec/protos/models/rank_model.proto`](../torcheasyrec/tzrec/protos/models/rank_model.proto) | 排序模型 protos |
| [`torcheasyrec/tzrec/protos/models/match_model.proto`](../torcheasyrec/tzrec/protos/models/match_model.proto) | 匹配模型 protos |

## 完整模型目录

### 排序模型（单任务）

| 模型 | 序列 | 稀疏输出 | 特征交互 |
|------|------|---------|---------|
| DeepFM | ❌ | KeyedTensor | FM + MLP |
| WideAndDeep | ❌ | KeyedTensor | Linear + MLP |
| MultiTower | ❌ | KeyedTensor | 多塔 concat |
| **MultiTowerDIN** | ✅ | 混合（KT + Dict） | 多塔 + DIN 注意力 |
| DLRM | ❌ | KeyedTensor | 按位点积 |
| **DLRM-HSTU** | ✅ | 混合（KT + Dict） | HSTU transducer，生成式 |
| DCN | ❌ | KeyedTensor | Cross + MLP |
| DCN V2 | ❌ | KeyedTensor | 改进版 cross |
| xDeepFM | ❌ | KeyedTensor | CIN + MLP |
| MaskNet | ❌ | KeyedTensor | Mask + MLP |
| WuKong | ❌ | KeyedTensor | WuKong 网络 |
| RocketLaunching | ❌ | KeyedTensor | 双 MLP + 蒸馏 |

**特征交互模式：**

| 交互 | 模型 | 机制 |
|------|------|------|
| **FM** | DeepFM | 二阶特征交叉 |
| **Cross** | DCN, DCN V2 | 显式 Cross network |
| **CIN** | xDeepFM | 压缩交互网络 |
| **Dot** | DLRM | 按位点积 |
| **Attention** | MultiTowerDIN, DAT, MIND | 注意力机制 |
| **MoE** | MMoE, DBMTL | 混合专家 |
| **Mask** | MaskNet, DBMTL | Mask 机制 |
| **HSTU** | DLRM-HSTU, HSTU | 分层序列 |
| **Direct** | MultiTower, WideAndDeep | 直接 concat |

### 多任务排序模型

| 模型 | 序列 | 稀疏输出 | 共享 |
|------|------|---------|------|
| MMoE | ❌ | KeyedTensor | 软共享（所有专家） |
| PLE | ❌ | KeyedTensor | 硬共享（专用 + 共享） |
| PEPNet | ❌ | KeyedTensor | 参数级个性化 |
| **DBMTL** | ✅ | 混合 | Mask + MMoE + 序列 |
| DC2VR | ❌ | KeyedTensor | DCN V2 + CGC |

### 匹配模型

| 模型 | 序列 | 稀疏输出 | User/Item |
|------|------|---------|----------|
| DSSM | ❌ | KeyedTensor | 双塔 |
| DSSM V2 | ❌ | KeyedTensor | 双塔 |
| **DAT** | ✅ | 混合 | 双塔 |
| **MIND** | ✅ | 混合 | 双塔 |
| HSTU | ✅ | 混合 | 双塔 |
| **TDM** | ✅ | 混合 | 单塔（树索引） |

**序列编码器：**

| 编码器 | 使用者 | 机制 |
|--------|--------|------|
| `DINEncoder` | MultiTowerDIN | 目标感知注意力池化 |
| `MultiWindowDINEncoder` | TDM | 多窗口注意力 |
| `HSTUEncoder` | HSTU | A2A 通信 |
| `CapsuleLayer` | MIND | 多兴趣 capsule |
| `DATTower` | DAT | 双注意力 |

## 模型选择决策树

### 单任务排序

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

### 多任务

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

### 匹配

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

## 导出兼容矩阵

目录中的每个模型都支持全部四种导出后端：

| 类别 | 模型 | Default | AOT | TRT | RTP | INPUT_TILE |
|------|------|---------|-----|-----|-----|-----------|
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

### `WORLD_SIZE` 约束

| 导出 | `WORLD_SIZE` | 说明 |
|------|-------------|------|
| Default | 1 | 单进程导出 |
| AOT | 1 | 单进程导出 |
| TRT | 1 | 单进程导出 |
| RTP | N | 支持分布式导出 |

### 已知限制

| 模型 | 限制 | 说明 |
|------|------|------|
| DLRM-HSTU | 特殊处理 | 使用 `JaggedTensor`，需要 `_fx_construct_payload` 包装器 |
| RocketLaunching | 特殊处理 | 包含蒸馏逻辑 |

### 稀疏输出格式

| 格式 | 模型 | 原因 |
|------|------|------|
| `KeyedTensor`（池化） | DeepFM、WideAndDeep、DCN、MMoE、… | 所有非序列特征按 EBC 池化为固定大小向量 |
| `Dict`（未池化） | DLRM-HSTU、序列模型 | 序列输出是 per-step 的，无法打包到单个 `KeyedTensor` |
| 混合 | MultiTowerDIN、DBMTL、DAT、MIND、TDM | 部分特征组是池化的（KT），部分是序列的（Dict） |

混合输出格式是 [09-export-pipeline.md](13-export-pipeline) 中描述的 `fx_mark_*` 哨兵设计所必需的：稀疏与稠密之间的边界必须在图中的多个点标记，每个标记都带有不同的类型签名。
