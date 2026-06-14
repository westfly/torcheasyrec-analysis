---
title: 首页
nav_order: 0
---

# TorchEasyRec 源码分析 · 中文导览

本仓库是对 [TorchEasyRec](https://github.com/alibaba/TorchEasyRec)（阿里巴巴开源的 PyTorch 推荐系统框架）的逐文件源码分析，覆盖训练、评估、分布式、导出、DynamicEmb 集成等核心子系统。每篇文档都对齐到具体源文件的精确行号（链接指向 GitHub），方便读者在阅读分析与跳转源码之间无缝切换。

## 在线阅读

完整 GitHub Pages 站点：<https://westfly.github.io/torcheasyrec-analysis/>

## 文档目录

| 章节 | 标题 | 简介 |
|------|------|------|
| [01](01-introduction) | 项目概览与背景 | 框架定位、解决的痛点、关键差异化、整体架构图 |
| [02](02-architecture) | 架构总览 | 六层架构、High-Level System View、配置驱动设计 |
| [03](03-code-structure) | 代码结构 | 顶层目录、`tzrec/` 子包划分、proto 配置体系 |
| | **训练篇** | |
| [04](04-training-init) | 初始化流程 | 从 CLI 到可训练模型的完整序列 |
| [05](05-training-flow) | 训练流程 | 单步数据流、两阶段初始化、混合精度、Pipeline、optimizer 层级 |
| [06](06-model-system) | 模型系统 | `BaseModel` 层级、20+ 模型目录、模型选择决策树、导出兼容矩阵 |
| [06-01](06-01-model-modules) | └ Modules 模块 | 稀疏/稠密/序列网络、模型→模块反向索引、ASCII 拓扑速查 |
| [07](07-feature-system) | 特征系统 | 12 种特征类型、4 种 FG 模式、纯 Python FG 迁移方案 |
| [07-01](07-01-feature-dataparser) | └ DataParser | 四模式对比、parse→to_batch 流程、SparseData→KJT 映射 |
| [08](08-embedding-system) | 嵌入系统 | `EmbeddingGroup`、TorchRec 分片、ZCH、DenseEmbeddingCollection |
| [08-01](08-01-zch) | └ ZCH 零碰撞哈希 | proto 定义、MCH 三阶段链路、淘汰策略、vs DynamicEmb、导出断裂、在线方案 |
| [09](09-dynamicemb) | DynamicEmb 集成 | NVIDIA GPU 哈希表嵌入后端的双视角深度解析 |
| [10](10-loss) | 损失函数 | BinaryFocalLoss、JRCLoss、ParetoEfficientMultiTaskLoss |
| [11](11-metrics) | 评估指标 | GroupedAUC、XAUC、DecayAUC、Recall@K、NormalizedEntropy、UniqueRatio |
| [12](12-optimizer) | 优化器与学习率调度器 | 稀疏/稠密优化器类型、5 种 LR 调度器、混合精度与梯度配置 |
| | **推理篇** | |
| [13](13-export-pipeline) | 导出管线 | JIT / TRT / AOTI / RTP 四后端对比、FX 切图、INPUT_TILE |
| [13-01](13-01-export-aot) | └ AOT 编译 | .so/.pt2 生成、meta.json 字段、检查调试工具 |
| [13-02](13-02-export-safetensors) | └ Checkpoint/Export 产物 | 目录结构、safetensors 格式、Default vs RTP 对比 |
| [13-03](13-03-export-sequence-online) | └ Sequence Embedding 在线推理 | FX marker 三段式链路、sparse padding、dense slicing、INPUT_TILE 交互 |
| [13-04](13-04-export-sparse-reconstruct) | └ RTP Sparse Model 重建 | fg.json + safetensors + metadata 三步骤、两种 lookup 模式对比 |
| [14](14-fsspec) | USE_FSSPEC 透传 | 外部文件系统抽象、`fsspec` 协议解析、10 个 IO 函数 monkeypatch、C++ IO 绕行 |
| [15](15-env) | 环境变量 | 分布式/推理/特征/导出/日志/数据源/测试全表 |
| | **参考** | |
| [16](16-troubleshooting) | 故障排查 | 导出/MCH/FSSPEC/INPUT_TILE/NCCL 常见问题 |

## 速览

- **仓库规模**：457+ 提交 / 392 star / 74 fork（截至 2025）
- **代码量**：`tzrec/` 295 个 Python 文件
- **模型数**：20+ 排序、召回、多任务、生成式推荐模型
- **特征类型**：12 种（`IdFeature` / `RawFeature` / `SequenceFeature` 等）
- **技术栈**：PyTorch + TorchRec + pyfg（特征生成）

## 仓库结构

```
torcheasyrec-analysis/
├── docs/                       # 本分析文档（16 篇 + 7 子页 + 4 section header + 首页）
├── torcheasyrec/               # TorchEasyRec 源码子模块（pin 7dc1c188）
├── external/recsys-examples/   # NVIDIA recsys-examples 子模块（pin 2091502，仅用于 DynamicEmb）
├── .github/workflows/pages.yml # GitHub Pages CI
├── scripts/update_submodule.sh # 子模块更新脚本
├── README.md                   # 中文 README
└── LICENSE                     # MIT
```

## 引用方式

```bibtex
@misc{torcheasyrec-analysis-2025,
  title  = {TorchEasyRec Source Analysis},
  author = {westfly},
  year   = {2025},
  url    = {https://github.com/westfly/torcheasyrec-analysis},
  note   = {Pin: TorchEasyRec@7dc1c188, recsys-examples@2091502}
}
```

## 许可

- 文档与脚本采用 [MIT](https://opensource.org/licenses/MIT) 许可
- 引用的 TorchEasyRec、recsys-examples 源码子模块保留各自上游的 Apache-2.0 许可
