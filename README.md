<div align="center">

# TorchEasyRec 源码分析

![Pages](https://img.shields.io/github/actions/workflow/status/westfly/torcheasyrec-analysis/pages.yml?branch=main&label=pages&style=flat-square)

阿里巴巴 [TorchEasyRec](https://github.com/alibaba/TorchEasyRec) 深度源码分析文档集

[🌐 在线阅读](https://westfly.github.io/torcheasyrec-analysis/) · [📚 文档目录](#-文档目录) · [🛠️ 本地构建](#-本地构建)

</div>

---

## 📖 简介

TorchEasyRec 是阿里巴巴开源的、基于 PyTorch 的工业级推荐系统框架，内置 20+ 个 SOTA 模型，覆盖召回（Matching）、排序（Ranking）、多任务学习（MTL）与生成式推荐四大方向。本仓库对该项目进行系统化的源码分析，覆盖架构、训练、模型、特征、Embedding、导出、DynamicEmb 集成等核心子系统。

> 本仓库**仅包含分析文档**，所有源码通过 git 子模块引用，不在本仓库内修改。

## 🌟 特色

- **4 个导出后端** — JIT / TRT / AOTI / RTP，逐路径分析稀疏/稠密分割、FX 图重写、环境变量开关
- **20+ 个模型** — DSSM / TDM / DeepFM / DIN / MMoE / PLE / PEPNet / DLRM-HSTU，附完整目录与导出兼容性矩阵
- **DynamicEmb 集成** — NVIDIA 开源 GPU 哈希表 Embedding 后端，覆盖上游 + TorchEasyRec 集成双视角
- **源码链接永久可点** — 通过 git 子模块锁定版本，行号引用不会因上游变动而失效
- **GitHub Pages 在线阅读** — Jekyll + just-the-docs 主题，push 即自动构建

## 🗂️ 文档目录

| #  | 文档 | 简介 |
|----|------|------|
| 01 | [项目概览](docs/01-project-overview.md) | TorchEasyRec 是什么、为什么存在、关键能力 |
| 02 | [架构总览](docs/02-architecture.md) | 6 层结构、数据流、关键架构决策 |
| 03 | [代码结构](docs/03-code-structure.md) | 目录树、构建系统、protobuf 配置 |
| 04 | [初始化流程](docs/04-initialization-flow.md) | 从 CLI 到可训练状态 |
| 05 | [训练流程](docs/05-training-flow.md) | 数据 → 特征 → 模型 → Loss，含两阶段初始化图 |
| 06 | [模型系统](docs/06-model-system.md) | BaseModel → RankModel → 20+ 模型的完整目录 + 兼容性矩阵 |
| 07 | [特征系统](docs/07-feature-system.md) | 10+ 特征类型、FG 模式、解析流程 |
| 08 | [Embedding 系统](docs/08-embedding-system.md) | EmbeddingGroup、TorchRec 分布式分片 |
| 09 | [导出流水线](docs/09-export-pipeline.md) | 4 后端对比、INPUT_TILE 模式、RTP 重写 |
| 10 | [DynamicEmb 集成](docs/10-dynamicemb-integration.md) | NVIDIA 上游 + TorchEasyRec 集成双视角 |

> 💡 建议阅读路径：**02 → 04 → 05 → 06 → 09 → 10**
>
> 📐 文档主体为英文；如需中文摘要，可结合 `git log` 的提交信息与本 README。

## 🚀 在线阅读

👉 **<https://westfly.github.io/torcheasyrec-analysis/>**

由 GitHub Pages 自动部署，每次 push 到 `main` 触发 Jekyll 构建。

## 🔧 仓库结构

```
torcheasyrec-analysis/
├── README.md                 # 本文件（中文）
├── LICENSE                   # MIT
├── docs/                     # 11 篇分析文档（Jekyll 站点源）
│   ├── _config.yml
│   ├── index.md              # Pages 首页
│   └── 01..10-*.md
├── torcheasyrec/             # 子模块：alibaba/TorchEasyRec
├── external/
│   └── recsys-examples/      # 子模块：NVIDIA/recsys-examples（仅取 dynamicemb）
├── scripts/
│   └── update_submodule.sh   # 子模块更新脚本
└── .github/workflows/
    └── pages.yml             # Pages CI（push 到 main 自动构建）
```

## 🛠️ 本地构建

```bash
# 1. 克隆（含子模块）
git clone --recurse-submodules https://github.com/westfly/torcheasyrec-analysis.git
cd torcheasyrec-analysis

# 2. 安装依赖（Ruby ≥ 3.0）
gem install bundler jekyll

# 3. 安装 Jekyll 依赖并启动本地预览
cd docs
bundle install
bundle exec jekyll serve
# 访问 http://localhost:4000/torcheasyrec-analysis/
```

> 如果不需要本地构建，可以直接访问 [在线站点](https://westfly.github.io/torcheasyrec-analysis/)。

## 📦 子模块管理

仓库锁定两个子模块的特定 commit，以保持文档中的源码行号引用始终有效：

```bash
# 初始化 / 拉取子模块
./scripts/update_submodule.sh
# 等价于：
git submodule update --init --recursive

# 升级到上游最新（会修改 .gitmodules 的 commit 引用）
./scripts/update_submodule.sh --latest
# 等价于：
cd torcheasyrec && git pull origin main && cd ..
cd external/recsys-examples && git pull origin main && cd ..
git add torcheasyrec external/recsys-examples
git commit -m "chore: bump submodules"
```

升级后需要重新核对文档中受影响章节的行号。

## 📝 引用

- [alibaba/TorchEasyRec](https://github.com/alibaba/TorchEasyRec) — 主项目（Apache-2.0）
- [NVIDIA/recsys-examples](https://github.com/NVIDIA/recsys-examples) — DynamicEmb 上游（Apache-2.0）
- [ai-code-analysis](https://github.com/anomalyco/ai-code-analysis) — 生成本仓库的方法论

## 📄 许可

- 本仓库**文档与脚本**采用 [MIT](LICENSE) 许可
- 引用的 TorchEasyRec 与 NVIDIA recsys-examples 代码遵循各自的 Apache-2.0 许可（仅通过 git 子模块引用，不在本仓库内修改）
