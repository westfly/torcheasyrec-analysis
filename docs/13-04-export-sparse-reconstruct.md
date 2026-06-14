---
title: RTP Sparse Model 重建
parent: 导出与 RTP 管线
nav_order: 4
---

# RTP Sparse Model 重建

## 背景

RTP 导出不再产生 `scripted_sparse_model.pt`，而是生成一套**元数据 + 权重文件**，由 RTP C++ 运行时在线重建稀疏模型（embedding lookup 层）。

核心产物：

```
export/
├── fg.json                             # 特征配置 → 关联 embedding table
├── model-000000-of-000001.safetensors  # 每个 rank 的权重张量
├── model-000000-of-000001.json         # 每个张量的元数据
└── fx_user_model/                      # 稠密模型 (torch.export 格式)
```

## 重建三步骤

```
fg.json                    model-*.json                   model-*.safetensors
    │                           │                               │
    │  feature.shared_name      │  tensor name 匹配              │  by tensor name
    │  gen_val_type="lookup"    │  → is_hashmap / shape / dtype  │
    ▼                           ▼                               ▼
[Step 1] 发现特征          [Step 2] 匹配 tensor            [Step 3] 加载权重
 哪些特征需要 lookup?         embedding 表在哪?               key-value / 稠密?
```

## Step 1: 从 fg.json 发现特征

```python
fg = load_json("export/fg.json")

def extract_embedding_features(fg_features):
    tables = []
    for feat in fg_features:
        if feat.get("gen_val_type") == "lookup":
            # 普通特征: shared_name = "user_id_emb"
            tables.append(feat)
        elif "features" in feat:
            # sequence 特征: sub-feature 使用复合名
            for sub_feat in feat["features"]:
                if sub_feat.get("gen_val_type") == "lookup":
                    tables.append(sub_feat)
    return tables
```

fg.json 关键字段：

| 字段 | 示例 | 用途 |
|------|------|------|
| `shared_name` | `"user_id_emb"` | 关联到 embedding config name |
| `gen_val_type` | `"lookup"` | `"lookup"`=需要做 embedding lookup |
| `gen_key_type` | `"hash"` / `"mod"` / `"boundary"` | 确定哈希策略 |
| `embedding_dimension` | `16` | embedding 维度 |
| `feature_name` | `"user_id"` | 特征名 |

## Step 2: 从 model-*.json 匹配 tensor

```python
meta = load_json("export/model-000000-of-000001.json")

for feat in features:
    shared_name = feat["shared_name"]
    for t_name, t_meta in meta.items():
        if shared_name in t_name:
            # 找到匹配的 tensor
            load_embedding(t_name, t_meta)
```

### model-*.json 字段完整参考

每个 tensor entry 的字段：

| 字段 | 类型 | 稠密模式 (ZCH/hash_bucket) | 稀疏模式 (DynamicEmb) |
|------|------|---------------------------|---------------------|
| `name` | string | `"...user_id_emb.values/part_0_1"` | 同上 |
| `dense` | bool | `false` | `false` |
| `dimension` | int | `16` | `16` |
| `dtype` | string | `"float32"` | `"float32"` |
| `memory` | int | 字节数 | 字节数 |
| `shape` | list[int] | `[1000000, 16]` | `[780000, 16]` |
| `is_hashmap` | bool | `false` | `true` |
| `hashmap_key` | str | — | `"...user_id_emb.keys/part_0_1"` |
| `hashmap_value` | str | — | 同 `name` |
| `hashmap_key_dtype` | str | — | `"int64"` |

### 两种模式判断

```python
if t_meta["is_hashmap"]:
    # DynamicEmb 模式 → 重建 key-value hashmap
    keys = safetensors[t_meta["hashmap_key"]]    # int64, [num_keys]
    values = safetensors[t_name]                  # float32, [num_keys, emb_dim]
    table = HashMapTable(keys, values)
else:
    # ZCH / hash_bucket 模式 → 稠密 weight 矩阵
    weight = safetensors[t_name]                  # float32, [vocab_size, emb_dim]
    table = DenseTable(weight)
```

### Shard 分发

tensor name 后缀 `part_{idx}_{num_shards}`：

```python
# model-000000-of-000002.safetensors → rank 0
# model-000001-of-000002.safetensors → rank 1

if num_shards > 1:
    # 每个 rank 只加载自己的 shard
    # RTP 需要跨 rank 组合或按 rank 分发请求
    local_table = load_tensor(f"model-{rank:06d}-of-{world_size:06d}.safetensors")
```

## Step 3: 在线推理 Lookup

```
RTP 接收请求:
  user_id = "u12345"

1. FG 层:
   读取 fg.json → user_id 是 id_feature, gen_val_type="lookup"
   → CityHash("u12345") → uint64 0x7F3A...B1C2

2. 查表:
   如果是 DenseTable:
     idx = uint64 % vocab_size
     vec = weight[idx]
   如果是 HashMapTable:
     vec = hashmap.get(uint64, default=zeros)

3. 输出给稠密模型:
   KeyedTensor(keys=["user_id"], values=vec)
```

## 全流程伪码

```python
# RTP 启动时
fg = load_json("export/fg.json")
meta = load_json("export/model-000000-of-000001.json")
weights = load_safetensors("export/model-000000-of-000001.safetensors")

tables = {}
for feat in extract_embedding_features(fg["features"]):
    shared_name = feat["shared_name"]
    emb_dim = feat["embedding_dimension"]
    for t_name, t_meta in meta.items():
        if shared_name in t_name and "values" in t_name:
            if t_meta["is_hashmap"]:
                keys = weights[t_meta["hashmap_key"]]
                values = weights[t_name]
                tables[shared_name] = HashMapTable(keys, values)
            else:
                weight = weights[t_name]
                tables[shared_name] = DenseTable(weight)

# 每次请求
def sparse_forward(batch):
    outputs = {}
    for feat in extract_embedding_features(fg["features"]):
        table = tables[feat["shared_name"]]
        raw_ids = batch[feat["feature_name"]]
        hashed = hash_fn(raw_ids, feat["gen_key_type"])

        if isinstance(table, DenseTable):
            idx = hashed % table.weight.shape[0]
            vec = table.weight[idx]
        else:
            vec = [table.hashmap.get(h, zeros) for h in hashed]

        if feat.get("gen_val_type") == "lookup":
            vec = pooling_fn(vec)           # sum/mean for multi-value
        outputs[feat["feature_name"]] = vec

    return KeyedTensor(outputs)
```

## Sequence 特征的特殊处理

对于 sequence 特征，sparse 模型多输出 `sequence_length`：

```
sparse_outputs = {
    "click_sequence":          [B, max_seq_len, D],
    "click_sequence_length":   [B, 1],
    "click__query":            [B, query_dim],
    "user_id":                 [B, 16],
    ...
}
```

稠密模型从 `sparse_outputs` 中按名取值，用 `_rtp_slice_with_seq_len` 切片。

详见 [Sequence Embedding 在线推理](13-03-export-sequence-online)。

## 参考源码

| 文件 | 关键内容 |
|------|---------|
| `tzrec/utils/export_util.py:463-570` | `_get_rtp_embedding_tensor()` — safetensors 元数据生成 |
| `tzrec/utils/export_util.py:576-614` | `_adjust_one_feature_for_rtp()` — fg.json 字段注入 |
| `tzrec/utils/export_util.py:617-635` | `_adjust_fg_json_for_rtp()` — 递归调整 FG 配置 |
| `tzrec/utils/export_util.py:806-872` | 稀疏模型提取 — marker → safetensors |
| `tzrec/utils/export_util.py:885-1002` | 稠密模型重建 — 替换 marker 为 sparse 输入 |
| `tzrec/features/feature.py:1308` | `create_fg_json()` — 基础 FG JSON 构建 |
