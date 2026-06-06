---
title: 首页
nav_order: 12
---

# TorchEasyRec 源码分析

[TorchEasyRec](https://github.com/alibaba/TorchEasyRec) 是阿里巴巴开源的**基于 PyTorch 的生产级推荐系统框架**，包含 20+ 候选生成（召回）、打分（排序）、多任务学习、生成式推荐领域的 SOTA 模型。

## 站点说明

本站点是对 TorchEasyRec 源码的逐文件分析文档，覆盖训练、评估、分布式、导出、DynamicEmb 集成等核心子系统。

**推荐入口：[中文导览](zh-nav)** — 包含完整的文档目录、仓库结构、引用格式与许可信息。

## 文档列表

| 文档 | 简介 |
|------|------|
| [中文导览](zh-nav) | 文档目录、仓库结构、引用与许可（推荐入口） |
| [项目概览与背景](01-project-overview) | TorchEasyRec 是什么、为何存在、关键特性 |
| [架构总览](02-architecture) | 分层、数据流、组件协同 |
| [代码结构](03-code-structure) | 目录布局、构建系统、proto 配置 |
| [初始化流程](04-initialization-flow) | 从 CLI 到可训练模型 |
| [训练流程](05-training-flow) | 训练管线：数据 → 特征 → 模型 → 损失 |
| [模型系统](06-model-system) | BaseModel → RankModel → 20+ 具体模型 |
| [特征系统](07-feature-system) | 12 种特征类型、FG 模式、解析 |
| [嵌入系统](08-embedding-system) | EmbeddingGroup、TorchRec、分布式分片 |
| [导出管线](09-export-pipeline) | JIT/TRT/AOTI/RTP 导出、FX 切图、稀疏/稠密分离 |
| [DynamicEmb 集成](10-dynamicemb-integration) | NVIDIA GPU 哈希表嵌入后端与集成 |

## 速览

- **457+ 提交 / 392 star / 74 fork**
- **`tzrec/` 中 295 个 Python 文件**
- **20+ 模型**：DSSM、TDM、DeepFM、DIN、MMoE、PLE、PEPNet、DLRM-HSTU
- **12 种特征类型**：IdFeature、RawFeature、SequenceFeature 等
- **技术栈**：PyTorch、TorchRec、pyfg（特征生成）

---

**在线地址**：<https://westfly.github.io/torcheasyrec-analysis/>  
**源码仓库**：<https://github.com/westfly/torcheasyrec-analysis>
