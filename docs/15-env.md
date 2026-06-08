---
title: 环境变量
nav_order: 15
---

# 环境变量

基于源码分析整理，按功能分类。帮助快速理解框架的配置入口和各环境变量的作用域。

### 分布式训练

| 环境变量 | 类型 | 默认值 | 说明 | 引用位置 |
|----------|------|--------|------|----------|
| `RANK` | int | 0 | 全球进程排名 | dist_util.py |
| `LOCAL_RANK` | int | 0 | 节点内进程排名 | dist_util.py |
| `WORLD_SIZE` | int | 1 | 总进程数 | dist_util.py |
| `LOCAL_WORLD_SIZE` | int | 1 | 节点内进程数 | dist_util.py |
| `GROUP_RANK` | int | 0 | 进程组排名 | dist_util.py |

### 推理加速

| 环境变量 | 类型 | 默认值 | 说明 | 引用位置 |
|----------|------|--------|------|----------|
| `ENABLE_AOT` | string | — | AOT 编译推理，取值 "1" 启用 | export_util.py |
| `ENABLE_TRT` | string | — | TensorRT 推理，取值 "1" 启用 | export_util.py |
| `DEBUG_TRT` | string | — | TensorRT 调试模式 | export_util.py |
| `DISABLE_TRT_DYNAMO` | string | "0" | 禁用 TRT Dynamo | export_util.py |
| `TRT_MAX_SEQ_LEN` | int | 100 | TensorRT 最大序列长度 | config_util.py |
| `TRT_MAX_BATCH_SIZE` | int | — | TensorRT 最大 batch | config_util.py |
| `MAX_EXPORT_BATCH_SIZE` | int | 512 | 导出时最大 batch | export_util.py |
| `ENABLE_TMA` | string | "0" | TMA 加速（CUDA 9.0+ + Triton 3.5.0+），取值 "1" 启用 | env_util.py:35 |
| `INPUT_TILE` | string | — | Input Tile 模式，取值 "2"（tile user side）或 "3"（tile + embedding split）| export_util.py |
| `INPUT_TILE_3_ONLINE` | int | 0 | 在线推理时使用 INPUT_TILE=3 | export_util.py |
| `QUANT_EMB` | string | — | Embedding 量化，取值 "INT8" | config_util.py |
| `QUANT_EC_EMB` | string | "0" | Embedding Cache 量化 | config_util.py |

### 特征处理

| 环境变量 | 类型 | 默认值 | 说明 | 引用位置 |
|----------|------|--------|------|----------|
| `USE_RTP` | string | "0" | RTP 在线推理模式，取值 "1" 启用 | env_util.py:24 |
| `USE_FARM_HASH_TO_BUCKETIZE` | string | "false" | 使用 FarmHash 分桶（RTP 下必须） | env_util.py:27 |
| `USE_HASH_NODE_ID` | string | "0" | 使用 Hash 节点 ID | env_util.py:19 |

### 模型导出

| 环境变量 | 类型 | 默认值 | 说明 | 引用位置 |
|----------|------|--------|------|----------|
| `FORCE_LOAD_SHARDING_PLAN` | string | "0" | 强制从 checkpoint 加载分片计划 | env_util.py:55 |
| `LOCAL_CACHE_DIR` | string | — | 本地缓存目录（C++ IO 绕行） | main.py:1081、export_util.py:97 |

### 日志与调试

| 环境变量 | 类型 | 默认值 | 说明 | 引用位置 |
|----------|------|--------|------|----------|
| `LOG_LEVEL` | string | — | 日志级别（DEBUG, INFO, WARNING, ERROR）| __init__.py:56 |
| `GLOG_logtostderr` | string | "1" | glog 重定向到 stderr | __init__.py:43 |
| `USE_DETERMINISTIC_ALGORITHMS` | string | "0" | 确定性算法（可复现结果）| __init__.py:70 |

### 数据源

| 环境变量 | 类型 | 默认值 | 说明 | 引用位置 |
|----------|------|--------|------|----------|
| `ODPS_CONFIG_FILE_PATH` | string | — | ODPS 配置文件路径 | odps_dataset.py |
| `ODPS_DATA_QUOTA_NAME` | string | — | ODPS 数据配额名称 | odps_dataset.py |
| `TUNNEL_READ_TIMEOUT` | int | 120 | ODPS Tunnel 读取超时 | odps_dataset.py |
| `USE_FSSPEC` | int | 0 | 使用 fsspec 文件系统操作 | filesystem_util.py |
| `ALIBABA_CLOUD_ECS_METADATA` | string | — | ECS 元数据地址（不设置则禁用） | __init__.py:20 |
| `ALIBABA_CLOUD_ECS_METADATA_DISABLED` | string | "true" | 禁用 ECS 元数据（自动设置） | __init__.py:21 |

### 测试相关

| 环境变量 | 类型 | 默认值 | 说明 | 引用位置 |
|----------|------|--------|------|----------|
| `CI` | string | "false" | CI 环境 | conftest.py |
| `CI_HYPOTHESIS` | string | "false" | 启用 Hypothesis 测试 | conftest.py |
| `TEST_NPROC_PER_NODE` | int | 2 | 测试时每节点进程数 | conftest.py |
| `CI_ODPS_PROJECT_NAME` | string | — | CI ODPS 项目名 | conftest.py |
| `CI_ODPS_SCHEMA_PROJECT_NAME` | string | — | CI ODPS Schema 项目名 | conftest.py |

### 其他

| 环境变量 | 类型 | 默认值 | 说明 | 引用位置 |
|----------|------|--------|------|----------|
| `OMP_NUM_THREADS` | int | 1 | OpenMP 线程数（自动设置） | __init__.py:33 |
| `TORCH_MANUAL_SEED` | int | — | PyTorch 随机种子 | __init__.py:64 |
| `NUMPY_MANUAL_SEED` | int | — | NumPy 随机种子 | __init__.py:67 |
| `FBGEMM_MOMENTUM1_STATE_INIT_VALUE` | float | — | FBGEMM Adagrad 动量初始值 | optimizer.py:94 |
| `PREDICT_QUEUE_TIMEOUT` | int | 600 | 预测队列超时（秒）| predict.py |

## 辅助函数

[`torcheasyrec/tzrec/utils/env_util.py`](../torcheasyrec/tzrec/utils/env_util.py)

```python
from tzrec.utils.env_util import use_rtp, use_hash_node_id, enable_tma, force_load_sharding_plan

if use_rtp():
    # RTP 模式

if enable_tma():
    # TMA 加速（自动检查 CUDA 版本和 triton 版本）
```

## 使用示例

```bash
# 分布式训练
export WORLD_SIZE=8
torchrun --nproc_per_node=8 train.py

# TensorRT 推理
export ENABLE_TRT=1
export TRT_MAX_BATCH_SIZE=256

# RTP 在线推理
export USE_RTP=1
export USE_FARM_HASH_TO_BUCKETIZE=true

# 非阿里云 fsspec
export USE_FSSPEC=1

# 确定性训练
export USE_DETERMINISTIC_ALGORITHMS=1
export TORCH_MANUAL_SEED=42
export NUMPY_MANUAL_SEED=42
```
