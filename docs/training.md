---
title: 训练篇
nav_order: 4
has_children: true
---

# 训练篇

TorchEasyRec 训练阶段的完整流程，从初始化到模型构建、特征处理、嵌入管理，以及训练附属组件（损失函数、评估指标、优化器）。

| 章节 | 标题 | 简介 |
|------|------|------|
| [04](04-training-init) | 初始化流程 | 从 CLI 到可训练模型的完整序列 |
| [05](05-training-flow) | 训练流程 | 单步数据流、两阶段初始化、混合精度、Pipeline、optimizer 层级 |
| [06](06-model-system) | 模型系统 | `BaseModel` 层级、20+ 模型目录、模型选择决策树、导出兼容矩阵 |
| [07](07-feature-system) | 特征系统 | 12 种特征类型、4 种 FG 模式、纯 Python FG 迁移方案 |
| [08](08-embedding-system) | 嵌入系统 | `EmbeddingGroup`、TorchRec 分片、ZCH、DenseEmbeddingCollection |
| [09](09-dynamicemb) | DynamicEmb 集成 | NVIDIA GPU 哈希表嵌入后端的双视角深度解析 |
| [10](10-loss) | 损失函数 | BinaryFocalLoss、JRCLoss、ParetoEfficientMultiTaskLoss |
| [11](11-metrics) | 评估指标 | GroupedAUC、XAUC、DecayAUC、Recall@K、NormalizedEntropy、UniqueRatio |
| [12](12-optimizer) | 优化器与学习率调度器 | 稀疏/稠密优化器类型、5 种 LR 调度器、混合精度与梯度配置 |
