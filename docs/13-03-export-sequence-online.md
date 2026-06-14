---
title: Sequence Embedding 在线推理
parent: 导出与 RTP 管线
nav_order: 3
---

# Sequence Embedding 在线推理

## 背景

Sequence 特征（如用户点击历史 `[item_1, item_2, ..., item_N]`）需要**不做池化**地保持每个 ID 的 embedding，保留序列维度送入后续的 seq_encoder（DIN/DIEN/SIM 等）。

在 RTP 在线推理中，这通过 **FX marker → 稀疏模型 padding → 稠密模型 slicing** 的三段式链路实现。

## Pooled vs Sequence Embedding

| | Pooled (EmbeddingBag) | Sequence (EmbeddingCollection) |
|---|---|---|
| **lookup 模块** | `EmbeddingBagCollection` (sum/mean pooling) | `EmbeddingCollection` (不池化, 保持 JaggedTensor) |
| **输出** | `[B, emb_dim]` 2D | `[sum(seq_len), emb_dim]` Jagged |
| **数据通路** | 直接传接 | 分 query + sequence 两路 |
| **sparse→dense 桥接** | 直接 KeyedTensor.values() (2D) | `_rtp_pad_to_max_seq_len` 填充 (3D) → `_rtp_slice_with_seq_len` 切片 |

## 三段式链路

```
训练图 (embedding.py)
 │
 │  SequenceEmbeddingGroupImpl.forward()
 │   ├─ fx_mark_tensor("click__query", query_t)         ← query 侧
 │   ├─ fx_mark_seq_len("click", seq_len)               ← 序列长度 marker
 │   └─ fx_mark_seq_tensor("click", padded_seq, ...)    ← 序列 embedding marker
 │
 ▼
导出图变换 (export_util.py)  ← 编译期
 │
 │  稀疏模型提取 (sparse extraction):
 │   ├─ fx_mark_seq_tensor  → outputs["click_sequence"]   ← _rtp_pad_to_max_seq_len
 │   └─ fx_mark_seq_len     → outputs["click_sequence_length"]  ← unsqueeze
 │
 │  稠密模型重建 (dense reconstruction):
 │   ├─ fx_mark_seq_len  → 替换为 getitem(sparse_output, "click_sequence_length") + squeeze
 │   └─ fx_mark_seq_tensor → 替换为 getitem(sparse_output, "click_sequence") + _rtp_slice_with_seq_len
 │
 ▼
RTP 在线推理
 │
 │  FG 层: 按 fg.json 做 embedding lookup → 输出 JaggedTensor
 │  稀疏模型: EC lookup → padding → 输出 [B, max_seq_len, D] + [B, 1]
 │  稠密模型: slice → seq_encoder → 预测
```

## 1. 训练图 Marker

`SequenceEmbeddingGroupImpl.forward()` (`embedding.py:1344-1391`) 在 FX 图中插入 3 个 marker：

```python
# embedding.py:1367 — 标记序列长度
fx_mark_seq_len(f"{group_name}", sequence_length)

# embedding.py:1384-1390 — 标记序列 embedding（已 padding 到 max_seq_len）
fx_mark_seq_tensor(
    f"{group_name}",          # "click"
    seq_cat_t,                # [B, max_seq_len, emb_dim_total]
    keys=seq_t_keys,          # sub-feature 嵌入名列表
    max_seq_len=self._group_to_sequence_length[group_name],  # 预配置的 max_seq_len
    is_jagged_seq=self._group_to_is_jagged[group_name],
)
```

关键行为：

| 条件 | Padding 方式 | 代码 (embedding.py) |
|------|-------------|-------------------|
| 普通 SEQUENCE | `jt.to_padded_dense(max_seq_len)` → `[B, max_seq_len, D]` | L1375 |
| JAGGED_SEQUENCE | `jt.values()` → 平坦 `[total_values, D]` | L1373 |
| INPUT_TILE_3_ONLINE + user | 同 JAGGED_SEQUENCE，`jt.values()` | L1371-1372 |

## 2. 稀疏模型提取

`export_util.py:828-856` 遍历 FX 图，将 marker 节点替换为稀疏模型的输出：

```python
# export_util.py:828-847 — 序列 embedding
if node.target == fx_mark_seq_tensor:
    seq_name = node.args[0]                            # "click"
    name = _seq_feat_name(seq_name)                    # → "click_sequence"
    seq_node = node.args[1]

    if node.kwargs["is_jagged_seq"]:
        seq_node = unsqueeze(seq_node, 0)               # RTP 需要 batch 维度

    seq_node = _rtp_pad_to_max_seq_len(                 # 填充到配置的 max_seq_len
        seq_node, node.kwargs["max_seq_len"]
    )
    outputs["click_sequence"] = seq_node                # [B, max_seq_len, D]

# export_util.py:848-856 — 序列长度
if node.target == fx_mark_seq_len:
    seq_name = node.args[0]                            # "click"
    name = _seq_len_name(seq_name)                     # → "click_sequence_length"
    t = unsqueeze(node.args[1], 1)                     # [B] → [B, 1]
    outputs["click_sequence_length"] = t
```

稀疏模型输出 3 组数据：

| 输出名 | 形状 | 来源 |
|--------|------|------|
| `click_sequence` | `[B, max_seq_len, emb_dim_total]` | `fx_mark_seq_tensor` → padding |
| `click_sequence_length` | `[B, 1]` | `fx_mark_seq_len` → unsqueeze |
| `click__query` | `[B, query_dim]` | `fx_mark_tensor` (同组 query 侧) |

## 3. 稠密模型重建

`export_util.py:957-980` 将 marker 节点替换为从稀疏模型输出读取 + 切片：

```python
# export_util.py:957-980 — 序列 embedding
if node.target == fx_mark_seq_tensor:
    new_node = getitem(input_node, "click_sequence")           # 从 sparse 输出读
    new_node = _rtp_slice_with_seq_len(                        # 按真实长度切片
        new_node,
        seq_len_nodes["click"],     # 从 fx_mark_seq_len 替换得到的真实长度
        node.kwargs["max_seq_len"], # 配置的 max_seq_len
    )
    if node.kwargs["is_jagged_seq"]:
        new_node = squeeze(new_node, 0)
    node.replace_all_uses_with(new_node)

# export_util.py:905-927 — 序列长度
if node.target == fx_mark_seq_len:
    get_node = getitem(input_node, "click_sequence_length")   # 从 sparse 输出读
    new_node = squeeze(get_node, 1)                            # [B,1] → [B]
    seq_len_nodes["click"] = new_node
    node.replace_all_uses_with(new_node)

    # 同时将序列长度注册为额外特征
    additional_fg.append({
        "feature_name": "click_sequence_length",
        "feature_type": "raw_feature",
        "expression": "user:click_sequence_length",
    })
```

### `_rtp_slice_with_seq_len` 核心逻辑

```python
def _rtp_slice_with_seq_len(x, seq_len, max_seq_len):
    # x:   [B, max_seq_len, D]     <- 已 padding
    # seq_len:  [B]                 <- 真实长度
    real_len = max(seq_len).item()
    return x[:, :real_len, :]       # [B, real_len, D]
```

## 4. fg.json 中的 Sequence 特征结构

RTP 的 FG 配置中，sequence 特征是嵌套结构 (`export_util.py:617-635`)：

```json
{
  "sequence_name": "click",
  "sequence_length": 50,              // fg.json 中配置的 max_seq_len
  "sequence_delim": ";",
  "features": [
    {
      "feature_name": "item_id",
      "shared_name": "click_item_id",   // 复合名: {seq_name}_{feat_name}
      "gen_val_type": "lookup",
      "gen_key_type": "hash",
      "embedding_dimension": 16
    }
  ]
}
```

每个 sub-feature 使用复合名 `{seq_name}_{feat_name}` 关联到 embedding table，RTP FG 层按此名称做 lookup。

## 5. INPUT_TILE 交互

| 模式 | 对 Sequence 的影响 | 代码位置 |
|------|-------------------|---------|
| `INPUT_TILE=2` | user sparse tile 到 B 后再做 EC lookup | `data_parser.py` |
| `INPUT_TILE=3` | user 保持 batch=1，EC lookup 后在 embedding 层 tile | `embedding.py:1377-1378` |
| `INPUT_TILE_3_ONLINE=1` | user sequence 用 `jt.values()` 跳过 padding | `embedding.py:1371-1372` |

## 6. 参考源码

| 文件 | 关键内容 |
|------|---------|
| `tzrec/modules/embedding.py:1344-1391` | `SequenceEmbeddingGroupImpl.forward()` — marker 插入 |
| `tzrec/modules/embedding.py:1367` | `fx_mark_seq_len` 调用 |
| `tzrec/modules/embedding.py:1384-1390` | `fx_mark_seq_tensor` 调用 |
| `tzrec/utils/export_util.py:828-856` | 稀疏模型提取 — consume marker |
| `tzrec/utils/export_util.py:905-980` | 稠密模型重建 — 替换 marker |
| `tzrec/utils/export_util.py:617-635` | fg.json sequence 特征调整 |
| `tzrec/utils/export_util.py:1037-1065` | `_compute_seq_share_groups()` (非 RTP 路径) |
| `tzrec/utils/fx_util.py:108-126` | `fx_mark_seq_tensor` / `fx_mark_seq_len` 定义 |
