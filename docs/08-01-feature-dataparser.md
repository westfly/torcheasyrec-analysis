---
title: DataParser 深度解析
parent: 特征系统
nav_order: 1
---

# DataParser 深度解析

## DataParser 在链路中的位置

```
BaseReader.to_batches()
  → Dict[str, pa.Array]
  → DataParser.parse()
  → Dict[str, torch.Tensor]
  → DataParser.to_batch()
  → Batch
```

DataParser 位于 Reader 与 Model 之间，负责把原始 pyarrow 数据转为模型可消费的 `Batch` 容器。核心入口：[`torcheasyrec/tzrec/datasets/data_parser.py:56`](../torcheasyrec/tzrec/datasets/data_parser.py#L56)。

## 四模式对比

| 模式 | FG 执行位置 | 输入形态 | 典型场景 |
|------|------------|----------|----------|
| `FG_NONE` | 不执行 pyfg（直接 parse） | 已 FG 编码列 | 上游已完成特征工程 |
| `FG_NORMAL` | 每个 feature 各自 `process_arrow` | 原始列 | 兼容旧链路 / 单特征调试 |
| `FG_DAG` | DataParser 全局一次 `process_arrow` | 原始列 | 推荐离线训练与推理 |
| `FG_BUCKETIZE` | DataParser 全局 handler（`bucketize_only=True`） | 原始列 | 只需要分桶场景 |

四种模式的接受端统一：[`torcheasyrec/tzrec/datasets/data_parser.py:178`](../torcheasyrec/tzrec/datasets/data_parser.py#L178) `DataParser.parse()`。

### FG_NONE

输入已经是 FG 编码后的列，直接按字段类型解析：

```
"1,2,3" → SparseData(values=[1,2,3], lengths=[3])
"0.5,1.2" → DenseData(values=[[0.5, 1.2]])
```

### FG_NORMAL

每个特征内部调用自己的 `self._fg_op.process_arrow(input_data)`——按特征各自跑 FG。不支持 `stub_type=True` 特征（`data_parser.py:137-142`）。

### FG_DAG

DataParser 级别一次性调用 `self._fg_handler.process_arrow(...)`，再按 feature 拆输出。DAG 一次处理，减少重复计算。输入列集合来自 `FgArrowHandler.user_inputs / item_inputs / context_inputs`（`data_parser.py:128-136`）。

### FG_BUCKETIZE

与 FG_DAG 共用 `_parse_feature_fg_handler()` 分支，初始化 handler 时传 `bucketize_only=True`（`data_parser.py:172-176`）。

## parse 阶段：Arrow → Tensor

[`torcheasyrec/tzrec/datasets/data_parser.py:178`](../torcheasyrec/tzrec/datasets/data_parser.py#L178)

1. 根据 FG 模式调用 `_parse_feature_fg_handler` 或 `_parse_feature_normal`
2. 解析标签列：支持 `int/float/list<int>/list<float>`
3. 解析 sample weights（要求 float）
4. 注入 hard negative 索引（若存在）

输出格式示例：

```
# 稀疏: f.values, f.lengths, 可选 f.weights
# 多值序列稀疏: 额外 f.key_lengths
# 序列稠密: f.values, f.lengths
# 标签: label 或 label.values + label.lengths
```

## to_batch 阶段：Tensor → Batch

[`torcheasyrec/tzrec/datasets/data_parser.py:385`](../torcheasyrec/tzrec/datasets/data_parser.py#L385)

### 主分支

```
dense  → _to_dense_features  → KeyedTensor
sparse → _to_sparse_features → KeyedJaggedTensor
```

`INPUT_TILE=2/3` 模式分别走 `user1_itemb` 与 `user1tile_itemb` 分支（`data_parser.py:397-431`）。

### sample_weights 流转

在 `data_config.sample_weight_fields` 配置权重列：

```protobuf
data_config {
    label_fields: "clk"
    sample_weight_fields: "sample_weight"
}
```

- parse 阶段：逐列读取权重并强制 `float32`（`data_parser.py:245-254`）
- to_batch 阶段：写入 `Batch.sample_weights` 字典（`data_parser.py:452-470`）
- 单任务 rank 模型默认用第一个权重列：`sample_weights[0]`

## 特征 → Batch 容器映射

决定容器落位的是两个运行时属性：`is_sparse` × `is_sequence` × `data_group`。

| `is_sparse` | `is_sequence` | Batch 容器 |
|---|---|---|
| `False` | `False` | `dense_features: KeyedTensor` |
| `True` | `False` | `sparse_features: KeyedJaggedTensor` |
| `True` | `True` | `sparse_features: KeyedJaggedTensor`（序列稀疏）|
| `False` | `True` | `sequence_dense_features: JaggedTensor` |

`data_group` 决定写到哪个分组 key（`__BASE__` / `__NEG__` / `__CNEG__`）。

### 常见配置示例

```protobuf
# id_feature → is_sparse=True → sparse_features (KJT)
id_feature { feature_name: "user_id" hash_bucket_size: 1000000 }

# raw_feature 无 boundaries → is_sparse=False → dense_features (KT)
raw_feature { feature_name: "price" value_dim: 1 }

# raw_feature 有 boundaries → is_sparse=True → sparse_features (KJT)
raw_feature { feature_name: "price_bucket" boundaries: [10, 20, 50, 100] }
```

## SparseData → KJT 详细链路

parse 阶段的中间态：

- `SparseData(name, values, lengths, weights?)`：普通稀疏特征
- `SequenceSparseData(name, values, key_lengths, seq_lengths)`：序列稀疏特征

`_to_sparse_features()` 按 `sparse_keys[dg]` 逐个 key 收集：读 values → lengths → weights，跨 key `torch.cat()` 后构造 `KeyedJaggedTensor`。

多值序列特殊处理（`value_dim != 1`）：

```python
length = segment_reduce(key_lengths, lengths=seq_lengths)
```

得到主 KJT 需要的 `lengths`（每条样本总 token 数）。

## 关键文件

| 文件 | 位置 | 功能 |
|------|------|------|
| `datasets/data_parser.py` | L56 | DataParser 主体 |
| `datasets/data_parser.py` | L178 | `parse()` 入口 |
| `datasets/data_parser.py` | L385 | `to_batch()` 入口 |
| `datasets/utils.py` | L234 | SparseData、Batch 数据类型 |
| `features/feature.py` | L75 | FG 编码解析函数 |
| `models/rank_model.py` | L77 | sample_weights 使用 |
