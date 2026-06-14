---
title: Feature Group 与 Sequence Encoder
parent: 训练篇
nav_order: 14
---

# Feature Group 与 Sequence Encoder

## Feature Group 配置

`FeatureGroupConfig`（`protos/model.proto`）定义特征分组，控制特征如何组合、处理和送入模型：

```protobuf
message FeatureGroupConfig {
    optional string group_name = 1;
    optional FeatureGroupType group_type = 2;
    repeated string feature_names = 3;
    repeated SequenceGroupConfig sequence_groups = 4;
    repeated SeqEncoderConfig sequence_encoders = 5;
    optional string embedding_name_suffix = 6;
}
```

## 三种 Group Type

| group_type | 行为 | 适用模型 |
|-----------|------|---------|
| `DEEP` | 标准 MLP 输入。所有特征 Embedding concat 后通过 MLP | MultiTower、DeepFM、DCN、DLRM 等排序模型 |
| `WIDE` | Wide 部分。固定 `embedding_dim=4`，直接拼接后输出（过 MLP）（可选） | WideAndDeep、DeepFM |
| `SEQUENCE` | 序列特征。先拼接 group 内所有 sequence feature，再通过 `SequenceEncoder` | DIN、DIEN、SIM |

### DEEP 分组

```protobuf
feature_groups: {
    group_name: "user"
    group_type: DEEP
    feature_names: ["user_id", "age", "gender"]
}
```

所有特征 embedding 拼接成统一向量，过 MLP tower（由模型配置中的 tower 定义）。

### SEQUENCE 分组

```protobuf
feature_groups: {
    group_name: "hist"
    group_type: SEQUENCE
    feature_names: ["click_history"]
    sequence_encoders: {
        din_encoder: {
            attn_mlp: [128, 64]
            attn_mlp_activation: "prelu"
        }
    }
}
```

SEQUENCE 分组内部有**查询-序列**二分结构：group 中的 `feature_names` 分为：
- **查询特征（query）**：当前候选 item 的特征（如 candidate_id）
- **序列特征（sequence）**：用户历史行为序列特征（如 click_history）

Encoder 对序列应用注意力机制，以查询特征为 query，序列特征为 key/value。

### 嵌套 Sequence Groups

对于更复杂的序列结构，可以在 DEEP group 内嵌套 `sequence_groups`：

```protobuf
feature_groups: {
    group_name: "deep_tower"
    group_type: DEEP
    feature_names: ["user_id", "age"]
    sequence_groups: [{
        group_name: "hist_seq"
        feature_names: ["click_history"]
    }]
    sequence_encoders: [{
        pooling_encoder: {
            pooling_type: "mean"
        }
    }]
}
```

嵌套后，序列 encoder 的输出会拼接到 DEEP 分组的 output embedding 中，再一起过 MLP。

## Sequence Encoder 注册与类型

所有 Sequence Encoder 基于元类注册（`modules/sequence.py:29-30`）：

```python
_SEQ_ENCODER_CLASS_MAP = {}
_meta_cls = get_register_class_meta(_SEQ_ENCODER_CLASS_MAP)

class SequenceEncoder(nn.Module, metaclass=_meta_cls):
```

| Encoder 类型 | Proto 消息 | 原理 | 适用场景 |
|-------------|-----------|------|---------|
| `DINEncoder` | `din_encoder` | Target Attention: query 对序列各元素的加权求和 | DIEN、SIM 等 DIN 系列 |
| `SimpleAttentionEncoder` | `simple_attention` | 自注意力 + GRU，query 与序列的双向交互 | DIEN |
| `PoolingEncoder` | `pooling_encoder` | 简单池化（mean / sum / max） | 无注意力需求的序列 |
| `SelfAttentionEncoder` | `self_attention` | Multi-head Self-Attention | Transformer 类模型 |
| `MultiWindowDINEncoder` | `multi_window_din` | 多窗口 DIN，每个窗口独立执行 DIN 后拼接 | TDM |

### DINEncoder

```python
# modules/sequence.py:62
class DINEncoder(SequenceEncoder):
    def __init__(self, ...):
        self.attn_mlp = MLP(attn_mlp, ...)
```

核心计算：`attention_weight = MLP(concat(query, key, query-key, query*key))`，然后用 attention weight 对序列元素加权求和。

### SelfAttentionEncoder

使用标准的 Transformer multi-head attention，计算序列内部各元素的交互关系。

### PoolingEncoder

```protobuf
pooling_encoder: {
    pooling_type: "mean"
}
```

最简单的 encoder — 对序列 embedding 沿 seq 维度做 mean / sum / max 池化。

## embedding_name_suffix

```protobuf
feature_groups: {
    group_name: "user_tower"
    group_type: DEEP
    feature_names: ["user_id", "age"]
    embedding_name_suffix: "_user"
}

feature_groups: {
    group_name: "item_tower"
    group_type: DEEP
    feature_names: ["item_id"]
    embedding_name_suffix: "_item"
}
```

`embedding_name_suffix` 用于在双塔模型（如 DSSM）中隔离 user 侧和 item 侧的同名特征，使它们各自维护独立的 embedding 表。

## 在模型中的调用

`BaseModel._parse_features()` 方法负责根据 FeatureGroupConfig 解析 batch 中的特征：

```
batch (SparseData / DenseData)
  └── FeatureGroup（由 FeatureGroupConfig 定义）
        ├── 提取 group_name 对应的特征张量
        ├── 如果 group_type == SEQUENCE → 过 SequenceEncoder
        ├── 如果 group_type == DEEP → 拼接 + 过 MLP tower
        └── 如果 group_type == WIDE → 拼接（dim=4）+ 可选 MLP
```

解析后在 `model.predict()` 中以 `group_name` 为 key 的字典形式使用。
