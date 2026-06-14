---
title: MultiTowerDIN 集成 DynamicEmb
parent: 案例分析
nav_order: 2
---

# MultiTowerDIN：集成 DynamicEmb

## 1. 迁移策略

### 核心理念：混用，不全换

DynamicEmb 不是 `num_buckets` 的银弹替换——对**大表**优势明显，对**小表**增加不必要开销。推荐分层策略：

| 层级 | vocab 范围 | 策略 | 本配置中 |
|------|-----------|------|---------|
| **大表** | > 100K | ✅ DynamicEmb | user_id(1.14M), adgroup_id(847K), campaign_id(423K), brand(461K), customer(256K) |
| **中表** | 1K-100K | ⚠️ 可选 DynamicEmb | cate_id(13K) |
| **小表** | < 1K | ❌ 保留 num_buckets | gender_code(3), age_level(8), pvalue_level(5), shopping_level(5), occupation(3), new_user_class_level(6), cms_segid(98), cms_group_id(14), pid(20) |

### 为什么小表不适合 DynamicEmb

```
num_buckets 方案:
  gender_code → 矩阵 [3, 16] = 192 bytes, O(1) 索引

DynamicEmb 方案:
  hash table 每行元数据: key(8B) + score(8B) + digest(1B) = 17B
  最小 init_capacity_per_rank = 128 → 128 × (17 + 16×4) = 10 KB
  + CUDA kernel launch overhead (微秒级 vs 纳秒级)
```

小表 DynamicEmb 的 **metadata 开销 > 数据本身**，且 hash lookup 比矩阵索引慢。因此小表保留 `num_buckets`，大表迁移 DynamicEmb。

### Table Fusion（FeatureGroup 级融合）

**这不是 `embedding_name` 共享**（会导致不同 feature 的 ID 碰撞），而是 DynamicEmb 内核在同一 FeatureGroup 内自动做的存储优化：

> 多个配置 DynamicEmb 的特征在同一 FeatureGroup 时，DynamicEmb 内核自动融合它们的存储，共享 cache 和 admission counter。

```
FeatureGroup "deep" 下的 5 张 DynamicEmb 表:
  user_id_emb  ─┐
  adgroup_id_emb ├─ DynamicEmb kernel → 融合存储
  campaign_id_emb│  (共享 HBM/DDR pool, 独立 key space)
  brand_emb     │
  customer_emb  ─┘
```

各自 `feature_name` 保持独立，ID 空间不碰撞，但底层存储和 cache 共享。

## 2. Config 改动

### 2.1 每条 id_feature 的 dynamicemb{} 配置

**当前 (num_buckets):**
```protobuf
id_feature {
    feature_name: "user_id"
    expression: "user:user_id"
    num_buckets: 1141730
    embedding_dim: 16
}
```

**改为 DynamicEmb:**
```protobuf
id_feature {
    feature_name: "user_id"
    expression: "user:user_id"
    embedding_dim: 16
    dynamicemb {
        max_capacity: 300000              # 每 GPU 300K 行
        initializer_args { mode: "NORMAL"; std_dev: 0.05 }
        score_strategy: "STEP"            # 淘汰最旧的
        init_capacity_per_rank: 1024      # 初始分配
        frequency_admission_strategy {
            threshold: 3                  # ID 出现 ≥3 次才入表
        }
    }
}
```

### 2.2 max_capacity 估算

`max_capacity` 是**每 GPU 最大行数**，不是全局 vocab 大小：

```
user_id（分布到 8 GPU）:
  总独特 ID ≈ 800K (训练集)
  per_rank ≈ 800K / 8 = 100K
  max_capacity = 100K × 2~3 (冗余) = 300K
```

### 2.3 小表保留 num_buckets

```protobuf
id_feature {
    feature_name: "final_gender_code"
    expression: "user:final_gender_code"
    num_buckets: 3
    embedding_dim: 16
    # 没有 dynamicemb{} — 走标准 TorchRec
}
```

小表 `num_buckets` 和 DynamicEmb 表可以共存于同一 `feature_groups`，由 TorchRec planner 自动处理（sharder 注册时同时注册标准 sharder 和 DynamicEmb sharder）。

### 2.4 FeatureGroup 结构不变

```protobuf
feature_groups {
    group_name: "deep"
    feature_names: "user_id"         # DynamicEmb
    feature_names: "adgroup_id"      # DynamicEmb
    feature_names: "final_gender_code" # num_buckets (小表)
    # ... 其余 13 个特征
    group_type: DEEP
}
feature_groups {
    group_name: "seq"
    feature_names: "adgroup_id"      # 使用同一 DynamicEmb 表
    feature_names: "click_50_seq__adgroup_id"  # sequence sub (共享表)
    group_type: SEQUENCE
}
```

序列特征 sub-feature（如 `click_50_seq__adgroup_id`）与顶层 `adgroup_id` 共享同一 `EmbeddingBagConfig`（相同的 `embedding_name`），因此它们自动共享同一张 DynamicEmb 表。

## 3. 训练变化

### 3.1 规划器变更

```python
# plan_util.py 中_emit_dynamicemb_variants():
# 每张 DynamicEmb 表产生 20 种 variant:
#   2 caching modes (HYBRID / CACHING)
# × 10 load_factors (0.1 ~ 1.0)
# = 20 variant per table
```

5 张大表 → 100 variant，DP proposer 搜索时间从秒级增加到数十秒。可优化：`cache_load_factor` 固定后 variant 降为 2。

### 3.2 前向变化

| 方面 | 当前 (num_buckets) | DynamicEmb |
|------|-------------------|-----------|
| lookup | O(1) 矩阵索引 | O(1) 开放寻址 hash 表 |
| 内存 | 384MB 固定 | HBM_ONLY: ~5 表 × 300K × (17+16×4) ≈ 107MB |
| 动态增长 | 不支持 | 训练过程中自动增删行 |
| 未见 ID | 映射到已有 slot | CONSTANT=0（经 admission 后可入表）|

### 3.3 Mixed Precision 注意事项

`BatchedDynamicEmbeddingTablesV2` CUDA kernel 对 fp16 梯度累加的支持取决于 NVIDIA release 版本。实践中建议：
- Embedding lookup: fp32（避免 hash table 精度损失）
- MLP forward: fp16（通过 `torch.amp.autocast`）
- DynamicEmb 内部梯度累加: fp32（保持累积精度）

## 4. DynamicEmb 特有考量

### 4.1 Admission Threshold

```protobuf
frequency_admission_strategy {
    threshold: 3        # 可调参数
}
```

ID 出现次数 < threshold 时不入表，给出 CONSTANT=0 embedding。影响：

| threshold | 效果 | 适用场景 |
|-----------|------|---------|
| 1 | 所有 ID 都入表 | 小数据量，无噪声 |
| 3-5 | 过滤低频噪声 | 推荐系统典型值 |
| 10+ | 只保留高频 ID | 超大规模，资源受限 |

### 4.2 Score Strategy（淘汰策略）

| 策略 | 含义 | 适用场景 |
|------|------|---------|
| `STEP` | 淘汰最早插入的 | 时效性敏感（新闻推荐） |
| `LFU` | 淘汰最少使用的 | 稳定分布（商品推荐） |
| `NO_EVICTION` | 不淘汰，满表后停止插入 | 严格不丢 ID |

淘宝场景建议 `LFU`：热门商品被高频点击，冷门商品被淘汰是合理的。

### 4.3 冷启动问题

推理时未见 ID → `CONSTANT=0`。对比 `num_buckets` 的碰撞方案：

```python
# num_buckets: 未见 ID 42 → 42 % num_buckets → slot[42] → 有 embedding（其他 ID 训练过的）
# DynamicEmb: 未见 ID 42 → hash("42") → 不在表中 → zero embedding
```

缓解：
- 调低 admission threshold 让更多 ID 入表
- 使用 `create_dynamicemb_init_ckpt.py` 从稠密 init 预热表
- 对冷 ID 使用 side information 做 backoff

### 4.4 Checkpoint 格式变化

| 方面 | 当前 | DynamicEmb |
|------|------|-----------|
| 状态字典 | state_dict 包含 `[N, D]` 权重 | state_dict 包含 dummy `[0, D]` 张量 |
| 实际数据 | 在 state_dict 中 | 在 `<checkpoint>/dynamicemb/` 目录下：`*.{keys,values,opt,counter,sizes}` |
| 恢复 | `torch.load(state_dict)` | `DynamicEmbLoad()` 读取二进制文件 |
| 跨 world_size | 任意 | 需相同 world_size（per-rank 文件） |
| 迁移工具 | — | `zch_to_dynamicemb_convert.py` |

## 5. 分布式通信分析

### 5.1 DynamicEmb ROW_WISE 强制

DynamicEmb 强制 `sharding_types=[ROW_WISE]`，即每 GPU 持有每张 DynamicEmb 表的 1/N 行。对比标准 TorchRec 可选的 TABLE_WISE：

```
TABLE_WISE (标准 TorchRec):
  GPU 0: user_id_emb (整张), final_gender_code_emb (整张)
  GPU 1: adgroup_id_emb (整张), age_level_emb (整张)
  ...
  → 每个 GPU 本地 lookup 部分特征，all-to-all 交换缺失分片

ROW_WISE (DynamicEmb):
  GPU 0: user_id_emb 行 0~37499, adgroup_id_emb 行 0~37499, ...
  GPU 1: user_id_emb 行 37500~74999, adgroup_id_emb 行 37500~74999, ...
  → 每个 GPU 做整个 batch 的 1/N lookup，all-to-all 汇总
```

### 5.2 通信量对比

以 8 GPU，batch_size=8192，5 张大表 + 10 张小表：

```
每 GPU local batch = 1024 samples
每 sample 平均 5 个 value（含多值）:
  1024 × 5 × 16 × 4 = 327,680 bytes = 0.31 MB/GPU

标准 TorchRec (TABLE_WISE):
  15 tables / 8 GPU ≈ 2 tables/GPU local
  远程: 13/15 × 0.31 = 0.27 MB/GPU
  消息数: 8 × 7 = 56 条, ~38 KB/条

DynamicEmb ROW_WISE:
  5 张大表全是远程 (1/8 local, 7/8 remote)
  10 张小表 TABLE_WISE (local if owned)
  远程: (5/15 × 7/8 + 10/15 × 13/15) × 0.31 ≈ 0.27 MB/GPU
  消息数: 8 × 7 = 56 条, ~38 KB/条
```

**通信量基本一致。** DynamicEmb 的 ROW_WISE 强制并没有带来额外通信开销。

### 5.3 跨节点延迟

| 规模 | 网络 | all-to-all 延迟 | 占 step 比 |
|------|------|----------------|-----------|
| 8 GPU 单机 | NVLink | 50-200μs | <0.5% |
| 32 GPU, 4 节点 | ROCE 100Gb | 1-5ms | 1-5% |
| 128 GPU, 16 节点 | ROCE 100Gb | 5-15ms | 5-15% |

### 5.4 缓解方案

| 手段 | 原理 | 效果 |
|------|------|------|
| **gradient accumulation** | N 步 local forward/backward → 1 步 all-to-all + optimizer | 通信频率降为 1/N |
| **local batch 翻倍** | batch=16384, step 数减半 | 总通信量减半 |
| **小表不移除** | 小表 TABLE_WISE 不走 all-to-all（当 local 时） | 减少消息数 |
| **table fusion** | DynamicEmb 融合后，单次 all-to-all 替代多次 | 减少 kernel launch |

## 6. 导出与推理

### 6.1 RTP-Only 导出

DynamicEmb 的 CUDA kernel（`BatchedDynamicEmbeddingTablesV2`）**不可被 `torch.jit.script`**，因此非 RTP 导出路径全部断裂：

| 后端 | DynamicEmb | 原因 |
|------|-----------|------|
| JIT | ❌ | CUDA kernel 不可 script |
| TRT | ❌ | 同上 + TensorRT 不识别 |
| AOTI | ❌ | 同上 |
| RTP | ✅ | safetensors + is_hashmap |

**必须部署在 RTP 平台上，这是前置条件。**

### 6.2 is_hashmap 格式

```json
// model-000000-of-000001.json 中的 DynamicEmb entry:
{
    "user_id_emb": {
        "name": "user_id_emb.values/part_0_1",
        "dense": false,
        "dimension": 16,
        "dtype": "float32",
        "shape": [298731, 16],
        "is_hashmap": true,
        "hashmap_key": "user_id_emb.keys/part_0_1",
        "hashmap_value": "user_id_emb.values/part_0_1",
        "hashmap_key_dtype": "int64"
    }
}
```

RTP 运行时加载时根据 `is_hashmap=true` 构建 hash table，而不是稠密矩阵。

### 6.3 推理时 Sequence 处理不变

DynamicEmb 不影响 FX marker 机制。Sequence 在线推理的 padding-slice 逻辑与 `num_buckets` 方案一致：

```
稀疏模型输出（针对 sequence）:
  "seq__sequence": [B, 100, 48]  ← DynamicEmb lookup + padding
  "seq__sequence_length": [B]
稠密模型输入:
  getitem → _rtp_slice_with_seq_len → [B, real_len, 48]
```

**唯一差异：** 稀疏模型的 embedding lookup 实现从矩阵索引换成了 DynamicEmb hash table。padding 到 100 的问题依然存在。

### 6.4 启动时间

DynamicEmb 服务启动时需从 safetensors 重建 hash table（插入 key-value 对），比直接 memcpy 稠密矩阵慢。

```
5 张大表: 每表 300K keys
总插入: 1.5M keys
hash table 构建时间: ~100-500ms（GPU 侧）
```

可接受范围。

## 7. 完整 config 示例

```protobuf
# 基于 multi_tower_din_taobao.config 的 DynamicEmb 混合版本

train_input_path: "odps://{PROJECT}/tables/taobao_multitask_sample_v1_train"
eval_input_path: "odps://{PROJECT}/tables/taobao_multitask_sample_v1/ds=20170513"
model_dir: "experiments/multi_tower_din_dynamicemb"

train_config {
    sparse_optimizer {
        adagrad_optimizer { lr: 0.001 }
        constant_learning_rate {}
    }
    dense_optimizer {
        adam_optimizer { lr: 0.001 }
        constant_learning_rate {}
    }
    num_epochs: 1
}

data_config {
    batch_size: 8192
    dataset_type: OdpsDataset
    fg_mode: FG_DAG
    label_fields: "clk"
    num_workers: 8
}

# ======== 大表: DynamicEmb ========

feature_configs {
    id_feature {
        feature_name: "user_id"
        expression: "user:user_id"
        embedding_dim: 16
        dynamicemb {
            max_capacity: 300000
            initializer_args { mode: "NORMAL"; std_dev: 0.05 }
            score_strategy: "LFU"
            init_capacity_per_rank: 1024
            frequency_admission_strategy { threshold: 3 }
        }
    }
}

feature_configs {
    id_feature {
        feature_name: "adgroup_id"
        expression: "item:adgroup_id"
        embedding_dim: 16
        dynamicemb {
            max_capacity: 250000
            initializer_args { mode: "NORMAL"; std_dev: 0.05 }
            score_strategy: "LFU"
            init_capacity_per_rank: 1024
            frequency_admission_strategy { threshold: 3 }
        }
    }
}

feature_configs {
    id_feature {
        feature_name: "brand"
        expression: "item:brand"
        embedding_dim: 16
        dynamicemb {
            max_capacity: 150000
            initializer_args { mode: "NORMAL"; std_dev: 0.05 }
            score_strategy: "LFU"
            init_capacity_per_rank: 1024
            frequency_admission_strategy { threshold: 3 }
        }
    }
}

feature_configs {
    id_feature {
        feature_name: "campaign_id"
        expression: "item:campaign_id"
        embedding_dim: 16
        dynamicemb {
            max_capacity: 150000
            initializer_args { mode: "NORMAL"; std_dev: 0.05 }
            score_strategy: "LFU"
            init_capacity_per_rank: 1024
            frequency_admission_strategy { threshold: 3 }
        }
    }
}

feature_configs {
    id_feature {
        feature_name: "customer"
        expression: "item:customer"
        embedding_dim: 16
        dynamicemb {
            max_capacity: 100000
            initializer_args { mode: "NORMAL"; std_dev: 0.05 }
            score_strategy: "LFU"
            init_capacity_per_rank: 1024
            frequency_admission_strategy { threshold: 3 }
        }
    }
}

# 中表（可选 DynamicEmb）
feature_configs {
    id_feature {
        feature_name: "cate_id"
        expression: "item:cate_id"
        embedding_dim: 16
        dynamicemb {
            max_capacity: 5000
            initializer_args { mode: "NORMAL"; std_dev: 0.05 }
            score_strategy: "LFU"
            init_capacity_per_rank: 1024
            frequency_admission_strategy { threshold: 3 }
        }
    }
}

# ======== 小表: 保留 num_buckets ========

feature_configs {
    id_feature {
        feature_name: "cms_segid"
        expression: "user:cms_segid"
        num_buckets: 98
        embedding_dim: 16
    }
}
feature_configs {
    id_feature {
        feature_name: "cms_group_id"
        expression: "user:cms_group_id"
        num_buckets: 14
        embedding_dim: 16
    }
}
feature_configs {
    id_feature {
        feature_name: "final_gender_code"
        expression: "user:final_gender_code"
        num_buckets: 3
        embedding_dim: 16
    }
}
feature_configs {
    id_feature {
        feature_name: "age_level"
        expression: "user:age_level"
        num_buckets: 8
        embedding_dim: 16
    }
}
feature_configs {
    id_feature {
        feature_name: "pvalue_level"
        expression: "user:pvalue_level"
        num_buckets: 5
        embedding_dim: 16
    }
}
feature_configs {
    id_feature {
        feature_name: "shopping_level"
        expression: "user:shopping_level"
        num_buckets: 5
        embedding_dim: 16
    }
}
feature_configs {
    id_feature {
        feature_name: "occupation"
        expression: "user:occupation"
        num_buckets: 3
        embedding_dim: 16
    }
}
feature_configs {
    id_feature {
        feature_name: "new_user_class_level"
        expression: "user:new_user_class_level"
        num_buckets: 6
        embedding_dim: 16
    }
}
feature_configs {
    id_feature {
        feature_name: "pid"
        expression: "context:pid"
        hash_bucket_size: 20
        embedding_dim: 16
    }
}

# ======== raw_feature（不受影响） ========
feature_configs {
    raw_feature {
        feature_name: "price"
        expression: "item:price"
        boundaries: [1.1, 2.2, 3.6, 5.2, 7.39, 9.5, 10.5, 12.9, 15,
                     17.37, 19, 20, 23.8, 25.8, 28, 29.8, 31.5, 34,
                     36, 38, 39, 40, 45, 48, 49, 51.6, 55.2, 58, 59,
                     63.8, 68, 69, 72, 78, 79, 85, 88, 90, 97.5, 98,
                     99, 100, 108, 115, 118, 124, 128, 129, 138, 139,
                     148, 155, 158, 164, 168, 171.8, 179, 188, 195,
                     198, 199, 216, 228, 238, 248, 258, 268, 278, 288,
                     298, 299, 316, 330, 352, 368, 388, 398, 399, 439,
                     478, 499, 536, 580, 599, 660, 699, 780, 859, 970,
                     1080, 1280, 1480, 1776, 2188, 2798, 3680, 5160, 8720]
        embedding_dim: 16
    }
}

# ======== sequence_feature（与顶层共享 DynamicEmb 表） ========
feature_configs {
    sequence_feature {
        sequence_name: "click_50_seq"
        sequence_length: 100
        sequence_delim: "|"
        features {
            id_feature {
                feature_name: "adgroup_id"
                expression: "item:adgroup_id"
                embedding_dim: 16
            }
        }
        features {
            id_feature {
                feature_name: "cate_id"
                expression: "item:cate_id"
                embedding_dim: 16
            }
        }
        features {
            id_feature {
                feature_name: "brand"
                expression: "item:brand"
                embedding_dim: 16
            }
        }
    }
}

# ======== 模型（与 num_buckets 版本完全一致） ========
model_config {
    feature_groups {
        group_name: "deep"
        feature_names: "user_id" "cms_segid" "cms_group_id"
        feature_names: "final_gender_code" "age_level"
        feature_names: "pvalue_level" "shopping_level"
        feature_names: "occupation" "new_user_class_level"
        feature_names: "adgroup_id" "cate_id"
        feature_names: "campaign_id" "customer" "brand"
        feature_names: "price" "pid"
        group_type: DEEP
    }
    feature_groups {
        group_name: "seq"
        feature_names: "adgroup_id" "cate_id" "brand"
        feature_names: "click_50_seq__adgroup_id"
        feature_names: "click_50_seq__cate_id"
        feature_names: "click_50_seq__brand"
        group_type: SEQUENCE
    }
    multi_tower_din {
        towers {
            input: 'deep'
            mlp { hidden_units: [512, 256, 128] }
        }
        din_towers {
            input: 'seq'
            attn_mlp { hidden_units: [256, 64] }
        }
        final { hidden_units: [64] }
    }
    metrics { auc {} }
    losses { binary_cross_entropy {} }
}
```

## 8. 迁移风险矩阵

| 风险 | 等级 | 缓解 |
|------|------|------|
| **JIT/TRT/AOTI 导出炸裂** | P0 blocker | 确认部署平台为 RTP |
| **冷启动质量下降** | P1 | 调 admission threshold + init_ckpt 预热 |
| **规划器变慢** | P2 | 固定 cache_load_factor 减少 variant |
| **checkpoint 恢复失败** | P1 | world_size 一致 + 备份 binary 文件 |
| **DynamicEmb CUDA kernel 兼容性** | P0 | 确认 NVIDIA driver 版本支持 |
| **小表 overhead** | 已规避 | 保留 num_buckets 混用 |

## 参考源码

| 文件 | 关键内容 |
|------|---------|
| `tzrec/utils/dynamicemb_util.py` | 7 个 monkey-patch、constraint builder、storage estimator |
| `tzrec/utils/plan_util.py:887-916` | `_emit_dynamicemb_variants()` — 20 variant 枚举 |
| `tzrec/utils/checkpoint_util.py:705-751` | DynamicEmbDump / DynamicEmbLoad |
| `tzrec/utils/export_util.py:497-570` | DynamicEmb is_hashmap 导出 |
| `tzrec/protos/feature.proto:90-114` | DynamicEmbedding proto 定义 |
| `tzrec/tests/configs/multi_tower_din_fg_dynamicemb_mock.config` | 测试用 DynamicEmb 配置 |
| `tzrec/modules/embedding.py:720` | TODO wide_embedding_dim 1→4 |
| `external/recsys-examples/corelib/dynamicemb/` | NVIDIA 上游 DynamicEmb kernel |
