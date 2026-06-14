---
title: 训练篇
nav_order: 4
has_children: true
---

# 训练篇

TorchEasyRec 训练阶段的完整流程，从初始化到模型构建、特征处理、嵌入管理、扩展开发、数据管线与采样器，以及训练附属组件（损失函数、评估指标、优化器）。

| 章节 | 标题 | 简介 |
|------|------|------|
| [04](04-training-init) | 初始化流程 | 从 CLI 到可训练模型的完整序列 |
| [05](05-training-flow) | 训练流程 | 单步数据流、两阶段初始化、混合精度、Pipeline、optimizer 层级 |
| [06](06-training-deployment) | 训练部署 | Docker 镜像、torchrun 启动、PAI-DLC 集成、分布式运行时、拓扑约束 |
| [07](07-model-system) | 模型系统 | `BaseModel` 层级、20+ 模型目录、模型选择决策树、导出兼容矩阵 |
| [08](08-feature-system) | 特征系统 | 12 种特征类型、4 种 FG 模式、纯 Python FG 迁移方案 |
| [09](09-embedding-system) | 嵌入系统 | `EmbeddingGroup`、TorchRec 分片、ZCH、DenseEmbeddingCollection |
| [10](10-dynamicemb) | DynamicEmb 集成 | NVIDIA GPU 哈希表嵌入后端的双视角深度解析 |
| [11](11-loss) | 损失函数 | BinaryFocalLoss、JRCLoss、ParetoEfficientMultiTaskLoss |
| [12](12-metrics) | 评估指标 | GroupedAUC、XAUC、DecayAUC、Recall@K、NormalizedEntropy、UniqueRatio |
| [13](13-optimizer) | 优化器与学习率调度器 | 稀疏/稠密优化器类型、5 种 LR 调度器、混合精度与梯度配置 |
| [19](19-extending) | 扩展机制与自定义模型 | 元类注册机制 vs if-elif 硬编码、完整扩展步骤、Proto 生成流程 |
| [20](20-data-pipeline) | 数据管线：Dataset 体系 | BaseDataset 基类、5 种 Dataset 类型、DataLoader 配置、Batch 数据结构 |
| [21](21-sampler) | Sampler 采样体系 | 5 种采样器、graphlearn 框架、分布式协同、Batch 注入 |
| [22](22-feature-group) | Feature Group 与 Sequence Encoder | 三种 group_type、5 种 Sequence Encoder、embedding_name_suffix |
