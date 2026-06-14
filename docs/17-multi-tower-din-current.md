---
title: MultiTowerDIN 当前现状
parent: 案例分析
nav_order: 1
---

# MultiTowerDIN：当前现状分析

## 1. 模型配置概览

### 配置文件

`multi_tower_din_taobao.config` 是一个完整的多任务推荐模型配置，使用 Taobao 广告数据集。

| 维度 | 值 |
|------|-----|
| **模型** | MultiTowerDIN（双塔 + DIN 注意力） |
| **特征类型** | 15 id_feature + 1 raw_feature(price with 100 boundaries) + 1 sequence_feature(click_50_seq, 3 sub-features) |
| **embedding_dim** | 16（全部一致） |
| **特征组** | `deep`(DEEP, 全量 16 个特征) + `seq`(SEQUENCE, 3 target + 3 seq sub) |
| **网络** | deep MLP: 512→256→128, DIN attn_mlp: 256→64, final: 64, output: 1 |
| **数据** | ODPS (MaxCompute), batch_size=8192, num_workers=8, fg_mode=FG_DAG |
| **优化器** | Sparse: AdaGrad(lr=1e-3), Dense: Adam(lr=1e-3) |

### 特征表

| Feature | num_buckets | 类型 |
|---------|-------------|------|
| user_id | 1,141,730 | user |
| adgroup_id | 846,812 | item |
| campaign_id | 423,438 | item |
| brand | 461,498 | item |
| customer | 255,877 | item |
| cate_id | 12,961 | item |
| cms_segid | 98 | user |
| cms_group_id | 14 | user |
| final_gender_code | 3 | user |
| age_level | 8 | user |
| pvalue_level | 5 | user |
| shopping_level | 5 | user |
| occupation | 3 | user |
| new_user_class_level | 6 | user |
| pid | 20 (hash_bucket_size) | context |
| price | raw_feature with 100 boundaries | item |

### 特征组逻辑

```
feature_groups {
    group_name: "deep"
    group_type: DEEP
    feature_names: 全量 16 个特征（id + raw + pid）
}
feature_groups {
    group_name: "seq"
    group_type: SEQUENCE
    feature_names: target(adgroup_id/cate_id/brand) + seq(click_50_seq__adgroup_id/...)
}
```

## 2. 训练数据流水线

### 完整链路

```
ODPS Table
  │  OdpsReader (MaxCompute Storage API)
  ▼
pyarrow.RecordBatch
  │  BaseReader._arrow_reader_iter() 积累到 batch_size, shuffle
  ▼
Dict[str, pa.Array] ← {"user_id": <Int64Array>, "click_50_seq": <ListArray>, ...}
  │
  ▼
DataParser.parse()
  │  FG_DAG 模式: pyfg.FgArrowHandler.process_arrow()
  │  → 对每个 feature 调用 _parse_feature_fg_handler()
  │  → id_feature: 产出 SparseData(values, lengths)
  │  → raw_feature with boundaries: 产出 SparseData（已分桶）
  │  → raw_feature without boundaries: 产出 DenseData(values)
  │  → sequence_feature: 按 delim="|" 拆分 + 按 multival_sep="\x03" 再拆分
  │    → 产出 SequenceSparseData(values, key_lengths, seq_lengths)
  ▼
DataParser.to_batch()
  │  → SparseData → KeyedJaggedTensor
  │  → DenseData → KeyedTensor
  │  → SequenceSparseData → KeyedJaggedTensor + sequence_mulval_lengths
  ▼
Batch { dense_features, sparse_features, labels, ... }
```

### 性能开销点

| 阶段 | 开销 | 原因 |
|------|------|------|
| **ODPS I/O** | I/O bound | 云端读取，8 workers 带宽上限 |
| **FG_DAG** | CPU | pyfg C++ handler, 但 sequence 分割在 Python 侧 |
| **Sequence 分割** | CPU | `\|` + `\x03` 两次拆分，8192 batch × 100 seq_len × 3 sub-features |
| **多值特征** | CPU | 15 个 id_feature 中部分多值，产生不固定长度的 values |

## 3. Embedding 层

### 显存分布

当前使用 `num_buckets` 方案：每张 id_feature → `EmbeddingBagConfig(num_embeddings=N, embedding_dim=16)` → 稠密矩阵 `[N, 16]`。

```
3,142,478  total vocab × 16 dim × 4 bytes = 192 MB (weights)
+ 192 MB (AdaGrad accumulator, 同等大小)
= 384 MB embedding memory (仅单精度)
```

### num_buckets 工作机制

```python
# 每个 id_feature 生成:
EmbeddingBagConfig(
    num_embeddings=feature.num_buckets,  # 固定大小
    embedding_dim=16,
    name="{feature_name}_emb",
)
# TorchRec 分配稠密矩阵 [num_buckets, 16]
# 前向: raw_id % num_buckets → 索引查表
# 碰撞: 不同 ID 落在同一 slot → 梯度共享
```

### 哈希碰撞分析

| Feature | num_buckets | 实际 ID 估计 | 碰撞风险 |
|---------|-------------|-------------|---------|
| user_id | 1,141,730 | 1M+ | 高 |
| adgroup_id | 846,812 | 800K+ | 高 |
| brand | 461,498 | 400K+ | 中高 |
| campaign_id | 423,438 | 400K+ | 中高 |
| customer | 255,877 | 200K+ | 中 |
| cate_id | 12,961 | 10K+ | 低 |
| 小表 | 3-98 | 有限 | 无 |

碰撞导致冷 ID 的 embedding 被热 ID 的梯度"拉偏"，对尾部推荐质量有负面影响。

## 4. 计算瓶颈分析

### 4.1 Sequence Padding 浪费（★ 最严重）

```python
# embedding.py 中 SEQUENCE group 的处理:
seq_t = jt.to_padded_dense(group_sequence_length)  # [B, max_seq_len=100, D]
```

配置中 `sequence_length: 100`，但淘宝广告场景平均用户行为序列长度 ~3-5。

**影响：**
- batch_size=8192, 3 seq sub-features, dim=16
- 实际计算量: 8192 × 5 × 3 × 16 = 1.97M 元素
- Padding 后计算量: 8192 × 100 × 3 × 16 = 39.3M 元素
- **无用计算占比: ~90-95%**

### 4.2 DIN Attention 4× 输入膨胀

```python
# DINEncoder.forward() (sequence.py:65-128)
attn_input = torch.cat(
    [queries, sequence, queries - sequence, queries * sequence], dim=-1
)
# dim: 16 → 64 (4×)
```

DIN 的标准做法：将 query 和 sequence 的 **差值** 和 **逐元素积** 拼接到一起，让 MLP 学习交叉特征。

```
在 max_seq_len=100 下:
  attn_input: [B, 100, 64]   (16→64, 4×)
  attn_mlp:   64 → 256 → 64 (训练参数 ~16K + 256 × 64)
  linear:     [B, 100, 64] → [B, 100, 1]
  mask + softmax → weighted sum → [B, 16]
```

**问题：4× 膨胀 + MLP 对所有 100 个位置计算，包括 95 个 padding 位。** softmax 的 mask 只阻止 padding 位参与加权，MLP 的 forward 已经算完了。

### 4.3 wide_embedding_dim workaround

```python
# embedding.py:719-724
# TODO(hongsheng.jhs): change to embedding_dim to 1
# when fbgemm support embedding_dim=1
wide_embedding_dim = 4  # hardcoded workaround
```

FBGEMM 不支持 `embedding_dim=1`，所以 wide 特征强制用 `dim=4`，浪费 4× 计算。本配置中无 wide 特征，不触发。

### 4.4 计算图汇总

```
batch → EmbeddingGroup
  │  384MB 表 lookup → all-to-all → [B, 272(deep) + 48(query) + 48×100(seq)]
  │
  ├─ deep MLP: [B, 272] → 512 → 256 → 128
  │   ≈ 2.8M params, ~2.1M MACs
  │
  ├─ DIN attention:
  │   ├─ pad+expand: [B,48] → [B,100,48]
  │   ├─ cat 4×:    [B,100,192]       ← 48→192
  │   ├─ attn_mlp:  [B,100,192] → 256 → 64
  │   │   ≈ 16K + 64K params, ~6.4M MACs (95% padding)
  │   ├─ linear+mask+softmax: [B,100,1]
  │   └─ weighted sum: [B,1,100] @ [B,100,48] → [B,48]
  │
  ├─ cat: [B,128] + [B,48] = [B,176]
  ├─ final MLP: 176 → 64 → 1
  └─ binary_cross_entropy + auc
```

## 5. 分布式通信

### 5.1 TorchRec All-to-All

TorchRec 的 `DistributedModelParallel` 在每个训练 step 中执行一次 **embedding all-to-all**：

```
前向:
  每 GPU 本地 lookup → KJT
  → all-to-all scatter: 将 KJT values 分发给需要它们的 GPU
  → 每 GPU 收到完整的 KJT 用于后续 forward

反向:
  grad all-to-all: 将梯度 scatter 回 embedding 的 owner GPU
```

### 5.2 ROW_WISE vs TABLE_WISE

本配置使用 `TABLE_WISE`（标准 TorchRec 默认分片）：

| 分片方式 | 每 GPU 持有 | 通信量 (8 GPU) | 通信模式 |
|---------|-------------|---------------|---------|
| TABLE_WISE | 2-3 张完整表 | ~210 KB/GPU | 8 路 all-to-all |
| ROW_WISE | 每张表的部分行 | ~210 KB/GPU | 8 路 all-to-all |

**通信量一致：** 不论怎么分片，总的 embedding 数据移动量是固定的。

```
batch_size=8192, 8 GPU, 15 features × 2 avg values × dim=16 × 4 bytes
= 8192/8 × 15 × 2 × 16 × 4 = 1.97 MB/GPU
远程占比: ~1.72 MB (7/8)
每条 all-to-all 消息: 1.72 MB / 7 ≈ 250 KB
```

### 5.3 延迟瓶颈

| 规模 | 网络 | all-to-all 延迟 | 占 step 时间比 |
|------|------|----------------|--------------|
| 8 GPU 单机 | NVLink | 50-200μs | <0.5% |
| 32 GPU 4 节点 | ROCE 100Gb | 1-5ms | 1-5% |
| 64 GPU 8 节点 | ROCE 100Gb | 2-8ms | 2-8% |

对于大部分生产配置，all-to-all 不是主要瓶颈。**计算（DIN attention + MLP）才是。**

## 6. 导出与推理

### 6.1 四路径

```
export_model() → use_rtp()? → yes → export_rtp_model()
                                no  → export_model_normal() → is_trt/aot? → branch
```

| 后端 | 产物 | 触发条件 | 本配置可用性 |
|------|------|---------|------------|
| **JIT** (默认) | `scripted_model.pt` | 默认 | ✅ |
| **TRT** | TensorRT engine | `ENABLE_TRT=1` | ✅ (需 GPU) |
| **AOTI** | `.so` / `.pt2` | `ENABLE_AOT=1` / `=2` | ✅ |
| **RTP** | safetensors + fx_user_model | `use_rtp()=true` | ✅ |

### 6.2 FX Marker 机制

导出时通过 `fx_mark_*` 函数在 FX 图中标记边界，将模型拆分为稀疏部分（embedding lookup）和稠密部分（MLP）：

```python
# embedding.py 中 SequenceEmbeddingGroupImpl.forward():
fx_mark_tensor("deep__query", query_cat_t)          # 非 sequence embedding
fx_mark_seq_len("seq", sequence_length)              # 序列长度
fx_mark_seq_tensor("seq", seq_cat_t, max_seq_len=100) # 序列 embedding

# export_util.py:
# 稀疏模型: 收集 fx_mark_* 输出 → safetensors + fg.json
# 稠密模型: 替换 fx_mark_* 为 getitem(input, name) → fx_user_model
```

### 6.3 RTP 导出（主要部署路径）

```
full FX graph
  ├─ copy → sparse graph
  │   ├─ 收集 fx_mark_* 输出
  │   ├─ 提取 embedding 权重 → safetensors (model-*.safetensors)
  │   ├─ 生成元数据 (model-*.json)
  │   └─ 生成 fg.json
  │
  └─ copy → dense graph
      ├─ 替换 fx_mark_* 为 getitem(input_node, name)
      ├─ sequence 额外做 _rtp_slice_with_seq_len()
      ├─ FBGEMM → torch native op mapping
      └─ ExportTorchFxTool → fx_user_model
```

### 6.4 推理时 Sequence 处理

```
稀疏模型输出:
  "seq__query":           [B, 48]
  "seq__sequence":        [B, 100, 48]  ← 已 padding
  "seq__sequence_length": [B, 1]

稠密模型输入:
  getitem(input, "seq__sequence")  → _rtp_slice_with_seq_len(seq_len)
  → slice[:, :real_len, :] → [B, real_len, 48]
```

**性能问题：** 稀疏模型已经 padding 到 100，稠密模型再 slice 回来。padding 的计算浪费在推理侧也存在。

## 7. 小结

| 方面 | 结论 |
|------|------|
| **最大瓶颈** | Sequence padding（90%+ 无效计算，训练+推理都有） |
| 次之 | DIN attention 4× 膨胀（padding 加剧此问题） |
| Embedding | num_buckets 碰撞影响尾部质量，384MB 显存固定开销 |
| 通信 | all-to-all 延迟在单机 8 卡可忽略，跨节点需关注 |
| 导出 | JIT/TRT/AOTI/RTP 四路径均可用 |
| 部署 | RTP 是生产首选（safetensors + fx_user_model） |
| 改进方向 | JAGGED_SEQUENCE / DynamicEmb / 动态长度 attention |

## 参考源码

| 文件 | 关键内容 |
|------|---------|
| `examples/multi_tower_din_taobao.config` | 完整配置 |
| `tzrec/models/multi_tower_din.py` | MultiTowerDIN forward |
| `tzrec/modules/sequence.py:65-128` | DINEncoder 注意力 |
| `tzrec/modules/embedding.py:1344-1391` | SequenceEmbeddingGroupImpl forward + marker |
| `tzrec/utils/export_util.py:697-1018` | RTP 导出全流程 |
| `tzrec/datasets/data_parser.py:184-402` | DataParser parse + to_batch |
| `tzrec/utils/export_util.py:828-856` | 稀疏模型 sequence marker 处理 |
| `tzrec/utils/export_util.py:905-980` | 稠密模型 sequence 重建 + slice |
