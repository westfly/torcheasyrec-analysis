---
title: Modules 模块与模型拓扑
parent: 模型系统
nav_order: 1
---

# Modules 模块与模型拓扑

## Modules 概览

`tzrec/modules/` 是模型搭建层，核心目标是把 `Batch` 里的特征组装成可复用的网络组件。

```
modules/
├── embedding.py                  # EmbeddingGroup（稀疏/序列入口）
├── dense_embedding_collection.py # 稠密特征 embedding (MLP/AutoDis)
├── sequence.py                   # DIN/Attention/MultiWindowDIN/HSTUEncoder
├── hstu.py                       # HSTU 注意力核心
├── mlp.py                        # Perceptron/MLP
├── interaction.py                # Cross/CrossV2/CIN/WuKongLayer
├── fm.py                         # FM 二阶交互
├── mmoe.py                       # Multi-gate Mixture of Experts
├── extraction_net.py             # PLE 抽取网络
├── task_tower.py                 # 单任务/融合塔
├── masknet.py                    # MaskNet block
├── personalized_net.py           # PPNet/EPNet
├── capsule.py                    # MIND 胶囊网络
├── variational_dropout.py        # 特征级可学习 dropout
├── intervention.py               # 干预网络（如 DC2VR）
├── activation.py                 # Dice/动态激活构建
├── norm.py                       # LN/BN/SwishLayerNorm
└── utils.py                      # BaseModule/div_no_nan
```

## 稀疏网络 vs 稠密网络 vs 序列网络

### 稀疏网络主干

核心入口 `EmbeddingGroup`（[`embedding.py:139`](../torcheasyrec/tzrec/modules/embedding.py#L139)）按 data_group 分流：

- `EmbeddingGroupImpl`（普通 group，[`embedding.py:585`](../torcheasyrec/tzrec/modules/embedding.py#L585)）
- `SequenceEmbeddingGroupImpl`（序列 group，[`embedding.py:885`](../torcheasyrec/tzrec/modules/embedding.py#L885)）

`is_sparse=True` 的特征进入 `EmbeddingBagCollection` / `ManagedCollisionEmbeddingBagCollection`，否则走 dense 侧。

### 稠密网络主干

- 通用基元：`MLP`（`mlp.py:86`）与 `Perceptron`（`mlp.py:21`）
- 交互模块：`FM`、`Cross/CrossV2`、`CIN`、`InteractionArch`、`WuKongLayer`
- 多任务模块：`MMoE`、`ExtractionNet`、`TaskTower`、`FusionMTLTower`

### 序列网络主干

编码器基类 `SequenceEncoder` + 工厂 `create_seq_encoder`（`sequence.py:54`、`sequence.py:580`），当前注册的编码器：

- `DINEncoder`（`sequence.py:70`）
- `SimpleAttention`（`sequence.py:136`）
- `PoolingEncoder`（sum/mean，`sequence.py:179`）
- `SelfAttentionEncoder`（`sequence.py:226`）
- `MultiWindowDINEncoder`（`sequence.py:293`）
- `HSTUEncoder`（`sequence.py:375`）

## 模型→模块反向索引

| 模型文件 | 直接依赖模块（`tzrec.modules.*`） |
|---|---|
| `wide_and_deep.py` | `mlp.MLP` |
| `dcn.py` | `interaction.Cross`, `mlp.MLP` |
| `dcn_v2.py` | `interaction.CrossV2`, `mlp.MLP` |
| `deepfm.py` | `fm.FactorizationMachine`, `mlp.MLP` |
| `xdeepfm.py` | `interaction.CIN`, `mlp.MLP` |
| `dlrm.py` | `interaction.InteractionArch`, `mlp.MLP` |
| `wukong.py` | `interaction.WuKongLayer`, `mlp.MLP` |
| `masknet.py` | `masknet.MaskNetModule` |
| `multi_tower.py` | `mlp.MLP` |
| `multi_tower_din.py` | `mlp.MLP`, `sequence.DINEncoder` |
| `tdm.py` | `embedding.EmbeddingGroup`, `mlp.MLP`, `sequence.MultiWindowDINEncoder` |
| `dssm.py` | `mlp.MLP` |
| `dssm_v2.py` | `embedding.EmbeddingGroup`, `mlp.MLP`, `variational_dropout.VariationalDropout` |
| `mind.py` | `capsule.CapsuleLayer`, `mlp.MLP` |
| `dat.py` | `mlp.MLP`, `utils.div_no_nan` |
| `hstu.py` | `embedding.EmbeddingGroup`, `sequence.HSTUEncoder` |
| `mmoe.py` | `mmoe.MMoE`, `task_tower.TaskTower` |
| `ple.py` | `extraction_net.ExtractionNet`, `task_tower.TaskTower` |
| `dbmtl.py` | `masknet.MaskNetModule`, `mlp.MLP`, `mmoe.MMoE` |
| `dc2vr.py` | `intervention.Intervention`, `mlp.MLP`, `mmoe.MMoE` |
| `pepnet.py` | `personalized_net.EPNet`, `personalized_net.PPNet`, `task_tower.TaskTower` |
| `rocket_launching.py` | `mlp.MLP`, `utils.div_no_nan` |
| `dlrm_hstu.py` | `gr.hstu_transducer.HSTUTransducer`, `norm.*`, `task_tower.FusionMTLTower`, `utils.*` |

该表列出的是 **直接 import 依赖**，不包含通过基类间接继承的模块（如 `RankModel` / `MatchModel` 基类统一依赖 `embedding.EmbeddingGroup`）。

## 模型 ASCII 拓扑速查

### Rank 系列

```
wide_and_deep:   EmbeddingGroup -> MLP
dcn:             EmbeddingGroup -> MLP + Cross
dcn_v2:          EmbeddingGroup -> MLP + CrossV2
deepfm:          EmbeddingGroup -> FM + DeepMLP
xdeepfm:         EmbeddingGroup -> CIN + DeepMLP
dlrm:            EmbeddingGroup -> InteractionArch + TopMLP
wukong:          EmbeddingGroup -> WuKongLayer + MLP
masknet:         EmbeddingGroup -> MaskNetModule
multi_tower:     EmbeddingGroup -> User/Item MLP Towers -> FinalMLP
multi_tower_din: EmbeddingGroup + DINEncoder -> Towers -> FinalMLP
rocket_launching:EmbeddingGroup -> Teacher/Student MLP
```

### Match 系列

```
dssm:            EmbeddingGroup -> User/Item MLP Towers -> Similarity
dssm_v2:         EmbeddingGroup -> VariationalDropout -> Towers -> Similarity
mind:            EmbeddingGroup -> Capsule + MLP Towers -> Matching
dat:             EmbeddingGroup -> Dual Towers -> Similarity
hstu:            EmbeddingGroup -> HSTUEncoder -> Match Towers
```

### Multi-task 系列

```
mmoe:            EmbeddingGroup -> MMoE -> TaskTower x N
ple:             EmbeddingGroup -> ExtractionNet -> TaskTower x N
dbmtl:           EmbeddingGroup -> MMoE + MaskNet + MLP -> Multi-task Heads
dc2vr:           EmbeddingGroup -> MMoE -> Intervention -> Heads
pepnet:          EmbeddingGroup -> EPNet/PPNet -> TaskTower x N
```

### Sequential / Hybrid 系列

```
tdm:             EmbeddingGroup -> MultiWindowDINEncoder -> MLP
dlrm_hstu:       EmbeddingGroup + HSTUTransducer -> Norm -> FusionMTLTower
```

## EmbeddingGroup 关键约束

- 同组 feature 不能跨多个 `data_group`（`embedding.py:190-194`）
- DEEP 组的 `sequence_groups` 与 `sequence_encoders` 必须配对（`embedding.py:281-336`）
- WIDE 组禁止 dense feature（`embedding.py:725-729`）
- 默认 `meta` device 初始化（`embedding.py:159-160`），调试时注意延迟 materialize

## 关键文件

| 文件 | 功能 |
|------|------|
| `modules/embedding.py` | EmbeddingGroup（稀疏入口 + 序列编码器衔接） |
| `modules/sequence.py` | 6 种序列编码器 |
| `modules/mlp.py` | MLP / Perceptron |
| `modules/interaction.py` | Cross / CIN / InteractionArch |
| `modules/mmoe.py` | MMoE（多门控专家混合） |
| `modules/extraction_net.py` | PLE 抽取网络 |
| `modules/task_tower.py` | TaskTower / FusionMTLTower |
