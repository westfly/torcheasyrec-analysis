---
title: Sampler 采样体系
parent: 训练篇
nav_order: 13
---

# Sampler 采样体系

## 概述

负采样（Negative Sampling）是推荐系统匹配模型训练的标配组件。TorchEasyRec 的 sampler 模块（`datasets/sampler.py:1108` 行）提供了 5 种采样器，全部基于 **graphlearn**（阿里巴巴开源的图采样框架）。

## 依赖：graphlearn 框架

```python
# tzrec/datasets/sampler.py:19-25
import graphlearn as gl
from graphlearn.python.data.values import Values
from graphlearn.python.nn.pytorch.data.utils import launch_server
```

每个 Sampler 内部以 **graphlearn 图服务**的形式运行：

1. **Server 端**: 在每个 rank 上启动 graphlearn 服务器，维护一个键值图（key → attributes）
2. **Client 端**: 通过 gRPC 向服务器发送采样请求，获取负样本

```python
# 启动采样集群（dataset.py:200-246）
sampler.init_cluster(num_client_per_rank, client_id_bias, cluster)
if cluster is None:
    sampler.launch_server()  # 第一个 rank 启动 server
```

## Sampler 注册

```python
# tzrec/datasets/sampler.py:129-130
_SAMPLER_CLASS_MAP = {}
_meta_cls = get_register_class_meta(_SAMPLER_CLASS_MAP)

class BaseSampler(object, metaclass=_meta_cls):
```

通过 `BaseSampler.create_class(name)` 按注册名称创建。

## 五种采样器

### 1. NegativeSampler（`sampler.py:131`）

- **原理**: 随机负采样，从全量 item 池中均匀抽取
- **配置**: `sampler.proto` → `NegativeSampler` 消息
- **参数**: `num_sample`（负样本数量）、`item_id_field`、`attr_fields`
- **适用**: 通用召回模型（DSSM）

### 2. NegativeSamplerV2（`sampler.py:252`）

- **原理**: 在 NegativeSampler 基础上增加 **频次加权**（popularity bias），高频 item 被采为负样本的概率更高
- **改进**: 引入 item 出现频率作为分布权重，提升训练稳定性
- **适用**: 长尾分布明显的推荐场景

### 3. HardNegativeSampler（`sampler.py:406`）

- **原理**: **硬负采样** — 不仅随机采样，还选择与正样本"相似"的 item 作为负样本（模型目前难以区分的边界样本）
- **实现**: 通过 item embedding 之间的相似度计算，选择 top-K 难负样本
- **适用**: 需要更精细决策边界的召回模型

### 4. HardNegativeSamplerV2（`sampler.py:570`）

- **原理**: HardNegativeSampler 的改进版，支持**混合采样**（随机负样本 + 硬负样本按比例混合）
- **参数**: `hard_sample_ratio`（硬负比例）
- **适用**: 兼顾多样性和难度的召回训练

### 5. TDMSampler（`sampler.py:744`）

- **原理**: **树形采样** — 基于 TDM（Tree-based Deep Model）的层次采样
- **实现**: 从 TDM 树的各层中分别采样，构造层次化的正负样本对
- **依赖**: 需要预构建的 TDM 树结构
- **适用**: TDM 层次检索模型

## 采样器流程

```
数据流:
1. DataLoader 读取 RecordBatch
2. BaseDataset.__iter__() 产出 batch
   └── Sampler 介入时机:
       a. 从 batch 中提取 item_id_field（正样本 ID）
       b. 构造 graphlearn 采样请求
       c. gRPC 调用 sampler server
       d. 获取负样本 ID + 属性
       e. 将负样本注入 batch（neg_sparse / neg_dense）
3. Batch 进入模型前向计算（正 + 负通道）

分布式场景:
- 每个 rank 启动一个 sampler client
- 多个 rank 共享 sampler server（通过 cluster 协调）
- server 端需要加载完整的 item 特征表
```

## Proto 配置

```protobuf
# protos/sampler.proto
message NegativeSampler {
    optional int32 num_sample = 1 [default = 1];
    optional string item_id_field = 2;
    optional string user_id_field = 3;
    repeated string attr_fields = 4;
}

message HardNegativeSampler {
    optional int32 num_sample = 1 [default = 1];
    optional float hard_sample_ratio = 2 [default = 0.0];
    optional string item_id_field = 3;
    optional string user_id_field = 4;
    repeated string attr_fields = 5;
}

message TDMSampler {
    optional int32 layer_num_sample = 1;
    optional float remain_ratio = 2;
    optional string probability_type = 3 [default = "mix"];
}
```

在 `data_config` 中通过 `oneof sampler` 选择：

```protobuf
data_config {
    negative_sampler {
        num_sample: 5
        item_id_field: "item_id"
        attr_fields: ["category", "price"]
    }
    ...
}
```

## 分布式协同

Sampler 在分布式训练中的关键约束：

| 问题 | 处理方式 |
|------|---------|
| **Server 启动** | Rank 0 启动 graphlearn server，其他 rank 作为 client 连接（`dataset.py:244-246`） |
| **Server 发现** | 通过 `launch_sampler_cluster()` 的 cluster 参数传递 server 地址 |
| **数据同步** | 使用 `dist.barrier()` 确保所有 rank 在 sampling 完成后再进入训练 |
| **重复采样** | 每个 rank 独立采样，可能采到重复的负样本（不影响训练收敛） |

## 与 Batch 数据结构的交互

Sampler 产出的负样本直接注入 `Batch` 数据组的 `NEG_DATA_GROUP`：

```python
# datasets/utils.py:32
NEG_DATA_GROUP = "__NEG__"

# Batch 中的负样本字段
batch.neg_sparse      # 负样本稀疏特征
batch.neg_dense       # 负样本稠密特征
batch.neg_sequence_sparse  # 负样本序列特征
```

模型的前向计算中，正负样本通过不同的特征通道分别计算相似度，最终合并到 loss 计算。
