---
title: 项目概览与背景
nav_order: 2
---

# 项目概览与背景

## 什么是 TorchEasyRec？

TorchEasyRec 是由阿里巴巴 PAI 团队开发的**基于 PyTorch 的推荐系统框架**，提供从模型构建、训练到部署的完整工作流，可处理大规模深度学习推荐模型。

与通用深度学习框架不同，TorchEasyRec 是为推荐任务专门设计的：

- **匹配（召回，Matching / Candidate Generation）**：DSSM、MIND、TDM、DAT
- **打分（排序，Scoring / Ranking）**：DeepFM、DIN、DLRM、DCN、xDeepFM、WuKong
- **多任务学习（Multi-Task Learning）**：MMoE、PLE、DBMTL、PEPNet
- **生成式推荐（Generative Recommendation）**：DLRM-HSTU、ULTRA-HSTU、HSTU-Match

## 它要解决的问题

构建生产级推荐系统需要应对一系列棘手难题：

1. **特征工程（Feature engineering）** — 处理类别特征（数百万 ID）、原始数值特征、序列行为特征、交叉特征
2. **大规模嵌入（Large-scale embeddings）** — 嵌入表可达 TB 级，需要分布式分片
3. **训练效率（Training efficiency）** — 数据 / 模型混合并行、混合精度、梯度裁剪
4. **在线服务（Serving）** — TensorRT / AOTInductor 加速、阿里云 EAS 部署
5. **特征生成一致性（Feature generation consistency）** — 训练与推理使用相同的特征变换

TorchEasyRec 以**配置驱动（configuration-driven）** 的方式解决上述所有问题：用户在 protobuf 配置中定义数据 schema、特征配置、模型架构，框架据此构建对应的 PyTorch 模型。

## 关键差异化优势

| 维度 | TorchEasyRec | EasyRec (TF) | TorchRec | XGBoost-based |
|------|-------------|-------------|----------|---------------|
| 后端 | PyTorch | TensorFlow | PyTorch | 不适用 |
| 模型数量 | 20+ | 20+ | 1（DLRM） | ~5 |
| 分布式训练 | TorchRec 分片 | TF 分布式 | 原生 | 不支持 |
| 特征生成 | pyfg（基于 DAG） | 特征配置 | 手动 | 手动 |
| 自定义模型 | 插件化 | 插件化 | 从零编写 | 不适用 |
| 云集成 | MaxCompute、PAI、EAS | 类似 | 不适用 | 不适用 |
| 推理加速 | TensorRT、AOTInductor | TF-TRT | AOTInductor | 不适用 |

## 何时选用 TorchEasyRec

- 你在阿里云上需要一个**生产就绪**的推荐系统
- 你希望快速实验 20+ SOTA 模型，而无需编写代码
- 你需要**分布式训练**以应对大规模嵌入
- 你要求训练与推理之间的**特征生成严格一致**
- 你正从 EasyRec（TF 版本）迁移到 PyTorch

## 整体架构

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

## 参考资料

- [官方文档](https://torcheasyrec.readthedocs.io/)
- [GitHub 仓库](../torcheasyrec/)
- [EasyRec（TensorFlow 前身）](https://github.com/alibaba/EasyRec)

## 关键文件

| 文件 | 用途 |
|------|------|
| [`torcheasyrec/tzrec/main.py`](../torcheasyrec/tzrec/main.py) | 训练、评估、导出核心管线 |
| [`torcheasyrec/tzrec/models/model.py`](../torcheasyrec/tzrec/models/model.py) | BaseModel 及包装器（TrainWrapper、PredictWrapper） |
| [`torcheasyrec/tzrec/models/rank_model.py`](../torcheasyrec/tzrec/models/rank_model.py) | RankModel 基类 — 损失、指标、预测逻辑 |
| [`torcheasyrec/tzrec/models/match_model.py`](../torcheasyrec/tzrec/models/match_model.py) | MatchModel 基类 — 嵌入相似度、批内负采样 |
| [`torcheasyrec/tzrec/features/feature.py`](../torcheasyrec/tzrec/features/feature.py) | BaseFeature 与 create_features() 工厂 |
| [`torcheasyrec/tzrec/modules/embedding.py`](../torcheasyrec/tzrec/modules/embedding.py) | EmbeddingGroup、EmbeddingGroupImpl、SequenceEmbeddingGroupImpl |
| [`torcheasyrec/tzrec/protos/pipeline.proto`](../torcheasyrec/tzrec/protos/pipeline.proto) | 管线配置（EasyRecConfig）定义 |
| [`torcheasyrec/tzrec/protos/model.proto`](../torcheasyrec/tzrec/protos/model.proto) | 模型配置定义（全部模型类型）|
