---
title: 训练部署
parent: 训练篇
nav_order: 3
---

# 训练部署

## 概述

TorchEasyRec 的训练部署栈基于纯 PyTorch 分布式生态，**没有** K8s 原生 CRD、没有 Horovod、没有 Parameter Server 架构。核心组件：

- **Launcher**: `torchrun`（唯一启动方式）
- **分布式运行时**: `torch.distributed` + TorchRec `DistributedModelParallel`
- **容器镜像**: `docker/Dockerfile`（CUDA 12.6 / CUDA 12.9 / CPU 三变体）
- **托管平台**: PAI-DLC（阿里云深度学习容器，基于 K8s 编排）
- **模型导出**: 导出需与训练**完全相同的拓扑**（same world_size、same GPU 数）

## Docker 镜像构建

`docker/Dockerfile` 使用 `DEVICE` 构建参数控制三种变体：

| DEVICE | PyTorch | CUDA Toolkit | FBGEMM | TorchRec | 适用场景 |
|--------|---------|-------------|--------|----------|---------|
| `cu126` | 2.11.0 | 12.6 | 1.6.0 | 1.6.0 | 通用 GPU 训练 |
| `cu129` | 2.11.0 | 12.9 | 1.6.0 | 1.6.0 | 最新 GPU 架构 |
| `cpu` | 2.11.0 | — | 1.6.0 | 1.6.0 | CPU 训练/测试 |

镜像关键特性：

- **PAI-DLC 集成**: 安装 `/home/pai/bin/prepare_dlc_environment` 脚本，用于 DLC 环境准备（`docker/Dockerfile:90-94`）
- **RDMA 网卡**: cu126/cu129 镜像安装 Mellanox RDMA 用户态库，支持跨节点高速通信（`docker/Dockerfile:107-116`）
- **OpenSSH**: 安装 ssh server 以支持 `torchrun` 跨节点通信（`docker/Dockerfile:93`）
- **CUDA 兼容库**: 安装 `cuda-compat-12-6` / `cuda-compat-12-9` 确保运行时兼容性
- **Conda 环境**: 基于 Miniforge3，Python 3.11

构建脚本 `scripts/build_docker.sh` 生成 `tzrec-test:1.2-{device}` 标签并推送到阿里云容器镜像仓库 `mybigpai-public-registry.cn-beijing.cr.aliyuncs.com/easyrec`。CI 通过后 `scripts/promote_docker.sh` 将 `tzrec-test` 提升为 `tzrec-devel`。

## 分布式启动方式

### torchrun（标准方式）

单机多卡（`torcheasyrec/docs/source/usage/train.md:6-8`）：

```bash
torchrun --master_addr=localhost --master_port=32555 \
  --nnodes=1 --nproc-per-node=2 --node_rank=0 \
  -m tzrec.train_eval \
  --pipeline_config_path example/multi_tower_taobao.config \
  --train_input_path data/train/*.parquet \
  --eval_input_path data/eval/*.parquet \
  --model_dir experiments/multi_tower_taobao
```

多机多卡（`torcheasyrec/docs/source/quick_start/dlc_tutorial.md:57-63`）：

```bash
torchrun --master_addr=$MASTER_ADDR --master_port=$MASTER_PORT \
  --nnodes=$WORLD_SIZE --nproc-per-node=$NPROC_PER_NODE \
  --node_rank=$RANK \
  -m tzrec.train_eval \
  ...
```

`torchrun` 自动设置的分布式环境变量：

| 变量 | 含义 |
|------|------|
| `RANK` | 全局 rank（所有节点 + GPU 全局编号） |
| `LOCAL_RANK` | 节点内 GPU 编号 |
| `WORLD_SIZE` | 全局进程总数 |
| `LOCAL_WORLD_SIZE` | 节点内进程数 |
| `MASTER_ADDR` | 主节点地址 |
| `MASTER_PORT` | 主节点端口 |

### TorchX（推荐替代）

外部 Fireworks 部署文档使用 TorchX 的 `dist.ddp` 调度器：

```bash
torchx run -s local_cwd dist.ddp -j "2x8" --rdzv_port 29500 \
  -m tzrec.train_eval -- \
  --pipeline_config_path example/multi_tower_taobao.config \
  --train_input_path data/taobao_data_train/*.parquet \
  --model_dir experiments/multi_tower_taobao
```

`-j "2x8"` 表示 2 节点 × 8 GPU。TorchX 自动处理节点发现和 rendezvous。

## 分布式运行时架构

### 核心模块

| 模块位置 | 职责 |
|---------|------|
| `tzrec/utils/dist_util.py:57-75` | `init_process_group()` — 初始化 `torch.distributed` |
| `tzrec/utils/dist_util.py:164-195` | `DistributedModelParallel()` — TorchRec DMP 封装 |
| `tzrec/utils/dist_util.py:221-302` | `TrainPipelineSparseDist` — 分布式训练 Pipeline |
| `tzrec/utils/plan_util.py:91-115` | `create_planner()` — EmbeddingShardingPlanner 创建 |
| `tzrec/main.py:572-681` | 训练入口分布式初始化流程 |

### 初始化流程

1. **`init_process_group()`** — 读取 `LOCAL_RANK`，选择 NCCL（GPU）或 GLOO（CPU），设置可选的 `PROCESS_GROUP_TIMEOUT_SECONDS`（`dist_util.py:57-75`）
2. **`create_planner()`** — 根据 `get_local_size()` 和 `dist.get_world_size()` 构造 `Topology`，创建 `EmbeddingShardingPlanner`（`plan_util.py:91-115`）
3. **`planner.collective_plan()`** — 所有 rank 共同规划 Embedding 分片策略（`main.py:672`）
4. **`DistributedModelParallel()`** — 包装模型，应用分片计划（`main.py:676-681`）

### 分布式 Pipeline

`TrainPipelineSparseDist`（`dist_util.py:221-302`）是 TorchRec 的 `TrainPipelineSparseDist` 的 TorchEasyRec 定制版：

- **数据同步**: `_next_batch()` 使用 `dist.all_reduce(has_batch, ReduceOp.AVG)` 确保所有 worker 数据一致性（`dist_util.py:290`）。当配置 `batch_cost_size` 时启用 `check_all_workers_data_status`。
- **NCCL 死锁回避**: cc < 7.0 GPU 上，`progress()` 入口处执行 `dist.barrier()` 避免 NCCL 死锁（`dist_util.py:254-271`）
- **梯度累积**: 支持 `gradient_accumulation_steps > 1`，反向传播时自动除以累积步数（`dist_util.py:198-211`）
- **GradScaler**: FP16 混合精度可选，在 pipeline backward 中处理（`dist_util.py:208-209`）

`TrainPipelineBase`（`dist_util.py:214-218`）用于无稀疏参数的模型（纯 MLP），无嵌入分片。

### 全局梯度裁剪

`main.py:714-731` 支持分布式全局梯度裁剪：

```protobuf
grad_clipping {
  clipping_type: "GLOBAL_NORM"
  max_gradient: 1.0
  enable_global_grad_clip: true
}
```

`enable_global_grad_clip=true` 时执行跨 rank 的 all-reduce 梯度范数。

## 环境变量调优

| 变量 | 默认值 | 作用 |
|------|--------|------|
| `INTRA_NODE_BANDWIDTH` | 自动 | 节点内带宽（Gbps），影响 sharding 规划器 |
| `CROSS_NODE_BANDWIDTH` | 自动 | 跨节点带宽（Gbps），影响 sharding 规划器 |
| `STORAGE_RESERVE_PERCENT` | 0.15 | GPU 内存保留比例 |
| `PROCESS_GROUP_TIMEOUT_SECONDS` | 1800 | 分布式进程组超时秒数 |
| `NCCL_DEBUG` | — | NCCL 调试日志（标准 NCCL 环境变量） |
| `NCCL_SOCKET_IFNAME` | — | NCCL 网络接口选择（标准 NCCL 变量） |

带宽控制在 `plan_util.py` 中被引用，用于 `Topology` 构造中的 `pipeline_io_cost` 计算。

## 分布式 Checkpoint

基于 `torch.distributed.checkpoint`（DCP），每个 rank 存储自己的分片：

- **模型参数**: `model-{rank:06d}-of-{world_size:06d}.pt`
- **优化器状态**: `optimizer-{rank:06d}-of-{world_size:06d}.pt`
- **元数据**: `metadata.json`（含模型总体信息）
- **分片计划**: `plan/`（ShardingPlan，用于恢复时重建分片）

关键约束（`checkpoint_util.py`）：

| 机制 | 说明 |
|------|------|
| `_ckpt_world_size()` | 从 checkpoint 目录读取保存时的 world_size |
| `_needs_mch_redistribution()` | 当恢复时的 world_size 与保存时不同时，自动触发参数重新分布 |
| `flatten_sharded_tensors` | 2D 并行模式下展平分片张量（`checkpoint_util.py:54`） |
| Dataloader 状态同步 | `dist.all_gather_object()` 用于 dataloader state 跨 rank 同步（`checkpoint_util.py:789`） |

## PAI-DLC 集成

PAI-DLC（Deep Learning Container）是阿里云基于 K8s 的深度学习训练平台。TorchEasyRec 的 DLC 集成方式：

- **Docker 镜像**: 镜像内预装 `/home/pai/bin/prepare_dlc_environment`，由 DLC 平台调用进行环境准备（网络、存储挂载等）
- **环境变量**: DLC 自动注入 `MASTER_ADDR`、`MASTER_PORT`、`WORLD_SIZE`、`NPROC_PER_NODE`、`RANK` 等标准分布式变量
- **数据源**: 支持直接读取 OSS（阿里云对象存储）、ODPS（MaxCompute 表）、Parquet 本地文件
- **训练命令**: 通过 DLC 控制台或 API 提交 `torchrun` 命令，DLC 自动分配节点和 GPU

**示例 DLC 作业 JSON**（参考 `export.md:85` 中的 PAI-EAS 格式，但 DLC 类似）：

```json
{
  "name": "tzrec_train",
  "worker_count": 2,
  "worker_gpu": 8,
  "image": "mybigpai-public-registry.cn-beijing.cr.aliyuncs.com/easyrec/tzrec-devel:1.2",
  "command": "torchrun --master_addr=$MASTER_ADDR --master_port=$MASTER_PORT --nnodes=$WORLD_SIZE --nproc-per-node=$NPROC_PER_NODE --node_rank=$RANK -m tzrec.train_eval --pipeline_config_path=... --model_dir=..."
}
```

## 拓扑约束

**核心知识：导出拓扑必须等于训练拓扑。**

`tzrec/main.py:164-168` 在导出模式下强制 `WORLD_SIZE=1`，但仅适用于单卡训练模型。对于多卡分布式训练后的模型：

- 导出时使用的 **world_size、GPU 拓扑必须与训练完全一致**
- 因为 checkpoint 是 per-rank 分片存储的（`ShardedTensor`、`DTensor`）
- 导出管线中的 `export_util.py:716-718` 按 `RANK`、`LOCAL_RANK`、`WORLD_SIZE` 生成 `model-{rank:06d}-of-{world_size:06d}` 文件
- 试图在更少 GPU 上导出会导致 `_needs_mch_redistribution()` 触发重分布，可能失败

## 与主流方案的对比

| 能力 | TorchEasyRec | TensorFlow PS | PyTorch DDP | Horovod |
|------|-------------|---------------|-------------|---------|
| 启动方式 | `torchrun` | `tf.distribute.MultiWorkerMirroredStrategy` | `torchrun` | `horovodrun` |
| 分布式策略 | TorchRec DMP（Data + Model Parallelism） | PS/Worker 分离 | DDP（Data Parallelism 仅） | Allreduce 仅 |
| 嵌入分片 | 自动（EmbeddingShardingPlanner） | 手动 PS 分配 | 不支持 | 不支持 |
| K8s 原生编排 | 无 YAML/CRD（通过 PAI-DLC 间接） | TFJob (Kubeflow) | PyTorchJob (Kubeflow) | 无 |
| 导出约束 | 拓扑一致（同 training） | 单进程 export | 单 GPU export | 单 GPU export |
| 跨节点通信 | NCCL + RDMA | gRPC + RDMA | NCCL | NCCL/MPI |

TorchEasyRec 选择了纯 PyTorch 分布式栈，利用 TorchRec 自动嵌入分片能力，避免了 PS 架构的运维复杂性，但也意味着它**不提供** K8s 原生 CRD 或 Helm Chart，训练编排依赖于 PAI-DLC 或自定义 `torchrun` 脚本。
