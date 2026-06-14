---
title: DynamicEmb 集成
nav_order: 10
---

# DynamicEmb 集成

## 概览

TorchEasyRec 集成了 [DynamicEmb](https://github.com/NVIDIA/recsys-examples/tree/main/corelib/dynamicemb)——NVIDIA 开源的 GPU 哈希表嵌入库——作为 TorchRec 由 FBGEMM 支持的 `EmbeddingBagCollection` (EBC) / `EmbeddingCollection` (EC) 之外的替代稀疏嵌入后端。集成是**可选且累加**的——使用 `dynamicemb { ... }` 配置的特征走 DynamicEmb 路径；其他特征保持走标准 EBC/EC 路径。

DynamicEmb 是 TorchRec 分片框架中的**定制 compute kernel**，而非 EBC 的替代品。它通过 monkey-patch 的 hook 和一个小型集成 shim 接入现有的 planner / sharder / checkpoint 机制。

本文档涵盖两个视角：

1. **NVIDIA 上游** — DynamicEmb 作为库提供了什么
2. **TorchEasyRec 集成** — 集成在上游基础上增加了什么

```
┌────────────────────────────────────────────────────────────────────────────┐
│                    TWO PERSPECTIVES IN THIS DOCUMENT                        │
├────────────────────────────────────────────────────────────────────────────┤
│                                                                            │
│   NVIDIA upstream                          TorchEasyRec integration        │
│   ───────────────                          ──────────────────────          │
│   BatchedDynamicEmbeddingTablesV2          tzrec/utils/dynamicemb_util.py  │
│   DynamicEmbeddingShardingPlanner           tzrec/utils/plan_util.py         │
│   DynamicEmbeddingBagCollectionSharder      tzrec/utils/checkpoint_util.py  │
│   DynamicEmbDump / DynamicEmbLoad           tzrec/tools/dynamicemb/*         │
│   DynamicEmbTableOptions                   tzrec/features/feature.py        │
│                                                                            │
│                  ▲                                        ▲                 │
│                  │                                        │                 │
│                  └────── installed as `dynamicemb` ───────┘                 │
│                       Python package (Apache-2.0)                          │
│                                                                            │
└────────────────────────────────────────────────────────────────────────────┘
```

上游代码作为子模块存在于本分析仓库：
[`external/recsys-examples/corelib/dynamicemb/`](../external/recsys-examples/corelib/dynamicemb/)（commit `2091502`，Apache-2.0，NVIDIA 2025 版权所有）。

TorchEasyRec 集成位于：
[`torcheasyrec/tzrec/utils/dynamicemb_util.py`](../torcheasyrec/tzrec/utils/dynamicemb_util.py)（794 行，2024 Alibaba，Apache-2.0）。

## 为什么选择 DynamicEmb？

标准 FBGEMM 支持的嵌入表是**预分配的稠密数组**，由整数 ID 索引。当 ID 空间小且有界（例如 ≤ 10M）时效果良好，但在以下场景会崩溃：

- **巨大基数** — 长尾特征具有数十亿唯一 ID（可能永远不会出现在训练中的冷启动 ID）
- **高度偏斜的分布** — 大多数行是稀疏的 / 永远不被触及，浪费 HBM
- **动态工作负载** — 生产流量模式随时间变化，预分配很难

DynamicEmb 用**GPU 哈希表**解决此问题，其特性：

- **按需增长** — 新 ID 到达时添加桶
- **淘汰冷 ID** — LRU / LFU / 基于 epoch / 定制评分策略
- **溢出到主机内存** — 值可以驻留在主机（DDR）上，带 HBM 缓存，由 `cache_load_factor` 控制
- **不使用 FBGEMM 行** — kernel（`DynamicEmbStorage` / `HybridStorage` / `DynamicEmbCache`，C++/CUDA）是 `dynamicemb_extensions` 中的定制实现

权衡：DynamicEmb 的每次查找延迟比稠密 FBGEMM 高（哈希探测 + 可能的主机溢出），但消除了预分配的内存浪费。

## 存储模式（HBM / HYBRID / CACHING）

运行时有三种存储模式，由 planner 基于 `cache_load_factor` 和 `caching` 标志选择：

```
┌────────────────────────────────────────────────────────────────────────────┐
│                       STORAGE MODES (runtime dispatch)                      │
├────────────────────────────────────────────────────────────────────────────┤
│                                                                            │
│  cache_load_factor=1.0  ─────────▶  HBM_ONLY                                │
│                                     (DynamicEmbStorage kernel)              │
│                                     HBM holds 100% of values                │
│                                     host tier dropped                        │
│                                                                            │
│  cache_load_factor<1.0, caching=False  ─▶  HYBRID                           │
│                                          (HybridStorage kernel)              │
│                                          HBM holds  ratio * values          │
│                                          Host holds (1-ratio) * values      │
│                                          hash-partitioned across tiers      │
│                                                                            │
│  cache_load_factor<1.0, caching=True   ─▶  CACHING                          │
│                                          (DynamicEmbCache kernel)            │
│                                          Host holds 100% backing store       │
│                                          HBM is hot-row cache of size        │
│                                          ratio * total_value_memory          │
│                                                                            │
└────────────────────────────────────────────────────────────────────────────┘
```

该模式是**隐式**的，从 `(cache_load_factor, caching)` 派生：

| `cache_load_factor` | `caching` | 模式 | 说明 |
|---|---|---|---|
| `1.0` | 任一 | `HBM_ONLY` | 主机层被丢弃；`dynamicemb` 运行时切换到 `DynamicEmbStorage` kernel |
| `< 1.0` | `False` | `HYBRID` | 值跨 HBM / 主机哈希分区 |
| `< 1.0` | `True` | `CACHING` | HBM 是主机后备存储的热行缓存 |

`_dynamicemb_effective_cache_ratio()` 在 `cache_load_factor=1.0` 处的 8x 跳变（[`dynamicemb_util.py:94-95`](../torcheasyrec/tzrec/utils/dynamicemb_util.py#L94-L95)）反映了这种模式切换——运行时 kernel 改变，而不仅仅是参数。

## 淘汰策略

[`dynamicemb/dynamicemb_config.py:104-110`](../external/recsys-examples/corelib/dynamicemb/dynamicemb/dynamicemb_config.py#L104-L110)：

```python
class DynamicEmbEvictStrategy(enum.Enum):
    LRU = EvictStrategy.KLru           # 标准 LRU
    LFU = EvictStrategy.KLfu           # 标准 LFU
    EPOCH_LRU = EvictStrategy.KEpochLru  # 训练 epoch 内的 LRU
    EPOCH_LFU = EvictStrategy.KEpochLfu  # 训练 epoch 内的 LFU
    CUSTOMIZED = EvictStrategy.KCustomized  # 用户提供的 per-row score
```

与 `DynamicEmbScoreStrategy` 配对（[`types.py:113`](../external/recsys-examples/corelib/dynamicemb/dynamicemb/types.py#L113)）：
- `STEP` — score 按步骤单调递增（老化）
- `TIMESTAMP` — score 即步索引
- `CUSTOMIZED` — 用户为每行提供的 score

## Per-Feature 配置（protobuf）

通过在其 `FeatureConfig` 中设置 `dynamicemb { ... }` 来选择使用 DynamicEmb（[`feature.proto`](../torcheasyrec/tzrec/protos/feature.proto)）：

```protobuf
message FeatureConfig {
    ...
    oneof embedding {
        RegularEmbedding regular_embedding = 4;
        ZchEmbedding zch_embedding = 5;
        DynamicEmbedding dynamic_embedding = 6;  // <- 选择使用 dynamicemb
    }
    ...
}

message DynamicEmbedding {
    int64 max_capacity = 1;                       // per-rank 行上限
    optional int64 init_capacity_per_rank = 2;    // 初始分配
    optional int64 bucket_capacity = 3;            // 桶大小（默认 128）
    optional float cache_load_factor = 4;         // HYBRID 中的 HBM 份额 / CACHING 中的缓存大小
    optional DynamicEmbInitializerArgs initializer_args = 5;
    optional DynamicEmbInitializerArgs eval_initializer_args = 6;
    optional string score_strategy = 7;           // STEP / TIMESTAMP / CUSTOMIZED
    optional int32 evict_strategy = 8;            // LRU / LFU / ...
    oneof admission_strategy {
        FrequencyAdmissionStrategy frequency_admission_strategy = 9;
    }
    optional bool caching = 10;                   // false = HYBRID, true = CACHING
}

message FrequencyAdmissionStrategy {
    int64 threshold = 1;
    optional int64 counter_capacity = 2;
    optional int64 counter_bucket_capacity = 3;
    optional DynamicEmbInitializerArgs initializer_args = 4;
}
```

关键字段：
- `max_capacity` — per-rank 行数的硬上限。与 FBGEMM 不同，这是**存储限制**，不是 key 范围。
- `cache_load_factor` — 见上表中的存储模式。
- `frequency_admission_strategy` — 在 key 被看到 ≥ `threshold` 次之前跳过插入表中（为一次性访客节省内存）。

## NVIDIA 上游：构建块

### 包布局

```
external/recsys-examples/corelib/dynamicemb/
├── setup.py                                # CUDAExtension 构建 (Bazel/CMake)
├── dynamicemb/
│   ├── __init__.py                         # public API
│   ├── dynamicemb_config.py                # DynamicEmbTableOptions, enums
│   ├── types.py                            # Storage / Cache ABC, InitializerArgs
│   ├── shard/
│   │   ├── embeddingbag.py                  # DynamicEmbeddingBagCollectionSharder
│   │   └── embedding.py                     # DynamicEmbeddingCollectionSharder
│   ├── planner/
│   │   ├── planner.py                       # DynamicEmbeddingShardingPlanner
│   │   ├── enumerators.py                   # shard enumerator overrides
│   │   └── rw_sharding.py                   # row-wise sharding helper
│   ├── batched_dynamicemb_tables.py         # BatchedDynamicEmbeddingTablesV2
│   ├── batched_dynamicemb_compute_kernel.py # the CUDA kernel launcher
│   ├── batched_dynamicemb_function.py       # autograd Function
│   ├── input_dist.py                        # jagged input distribution
│   ├── optimizer.py                         # sparse optim state mgmt
│   ├── initializer.py                       # NORMAL/UNIFORM/CONSTANT/DEBUG
│   ├── embedding_admission.py               # FrequencyAdmissionStrategy, KVCounter
│   ├── dump_load.py                         # DynamicEmbDump / DynamicEmbLoad
│   ├── incremental_dump.py                  # append-only dumps
│   ├── key_value_table.py                   # KV table wrapper
│   ├── extendable_tensor.py                 # growing tensor
│   ├── lookup_meta.py                       # lookup metadata
│   ├── index_range_meta.py                  # index range metadata
│   ├── exportable_tables.py                 # export hook
│   ├── scored_hashtable.py                  # scored hash table
│   ├── construct_twin_module.py             # EBC <-> DynamicEmb twin
│   ├── get_planner.py                       # planner factory
│   └── utils.py                             # torch_to_dyn_emb
```

### 核心类型

**`DynamicEmbTableOptions`**（[`dynamicemb_config.py`](../external/recsys-examples/corelib/dynamicemb/dynamicemb/dynamicemb_config.py)）— per-table 配置 dataclass，包含：

| 字段 | 用途 |
|---|---|
| `max_capacity` | per-rank 行上限 |
| `init_capacity` | 初始分配（被 clamp 到 `max_capacity`） |
| `bucket_capacity` | 哈希表桶大小（默认 128，按 16 对齐） |
| `global_hbm_for_values` | values 的 HBM 预算，除以 world_size → `local_hbm_for_values` |
| `local_hbm_for_values` | per-rank HBM 预算（由 planner 填充） |
| `initializer_args` | 训练时初始化（UNIFORM/NORMAL/CONSTANT/...） |
| `eval_initializer_args` | 评估时初始化（通常为 CONSTANT=0） |
| `score_strategy` | STEP / TIMESTAMP / CUSTOMIZED |
| `evict_strategy` | LRU / LFU / EPOCH_LRU / EPOCH_LFU / CUSTOMIZED |
| `admit_strategy` | 可选 `FrequencyAdmissionStrategy` |
| `admission_counter` | 用于 admission 的可选 `KVCounter` |
| `caching` | `False` = HYBRID，`True` = CACHING |
| `dist_type` | `"roundrobin"`（目前唯一支持的） |
| `index_type` | `torch.int64`（默认） |
| `embedding_dtype` | `torch.float32`（默认） |
| `training` | 由 planner 设置；存在时触发 optimizer 状态 |
| `dim` | 嵌入维度（由 planner 填充） |
| `check_mode` | `DynamicEmbCheckMode.ERROR / WARNING / IGNORE` 用于插入失败 |

**`Storage` 和 `Cache`**（[`types.py:150`](../external/recsys-examples/corelib/dynamicemb/dynamicemb/types.py#L150)）— 三种存储模式的抽象基类：

```python
class Storage(abc.ABC):
    def find(self, unique_keys, table_ids, copy_mode, ...): ...
    def insert(self, keys, table_ids, values, scores, preserve_existing): ...
    def dump(self, table_id, meta_file_path, emb_key_path, ...): ...
    def load(self, table_id, ...): ...
    def embedding_dtype(self) -> torch.dtype: ...
    def embedding_dim(self, table_id) -> int: ...
    def value_dim(self, table_id) -> int: ...
    def export_keys_values(self, device, batch_size, table_id): ...
```

两个具体后端（`DynamicEmbStorage`、`HybridStorage`、`DynamicEmbCache`）位于 C++/CUDA 扩展 `dynamicemb_extensions` 中，通过 `setup.py` 构建。

### Sharders

[`dynamicemb/shard/embeddingbag.py`](../external/recsys-examples/corelib/dynamicemb/dynamicemb/shard/embeddingbag.py) 和 `embedding.py` 提供 `DynamicEmbeddingBagCollectionSharder` / `DynamicEmbeddingCollectionSharder`。它们将 `DynamicEmbKernel`（值 `"DynamicEmb"`）注册为 `EmbeddingComputeKernel.CUSTOMIZED_KERNEL` 的 `customized_compute_kernel`。

### Planner

[`dynamicemb/planner/planner.py:213`](../external/recsys-examples/corelib/dynamicemb/dynamicemb/planner/planner.py#L213) — `DynamicEmbeddingShardingPlanner`：

```python
class DynamicEmbeddingShardingPlanner:
    def __init__(self, eb_configs, topology=None, batch_size=None, ...,
                 constraints=None, debug=True):
        _prepare_dynemb_table_options(constraints, eb_configs)
        # 拆分为 dyn_emb + torchrec 约束
        # 构建 self._torchrec_planner = EmbeddingShardingPlanner(...)
        # 构建 self._dyn_emb_plan: name -> DynamicEmbParameterSharding
        #   (每个：ROW_WISE 分片，跨 rank 的 EnumerableShardingSpec，
        #    compute_kernel=CUSTOMIZED_KERNEL, customized_compute_kernel=DynamicEmb)

    def collective_plan(self, module, sharders, pg=...):
        torchrec_plan = self._torchrec_planner.collective_plan(module, sharders, pg)
        # 将 dyn_emb 计划覆盖到匹配的表名上
        return torchrec_plan
```

`DynamicEmbParameterSharding`（[`planner.py:81`](../external/recsys-examples/corelib/dynamicemb/dynamicemb/planner/planner.py#L81)）扩展了 `ParameterSharding`，增加了 `compute_kernel=CUSTOMIZED_KERNEL`、`customized_compute_kernel=DynamicEmb`、`dist_type="roundrobin"`，以及 `dynamicemb_options` 实例。

`pop_additional_fused_params()`（[`planner.py:108`](../external/recsys-examples/corelib/dynamicemb/dynamicemb/planner/planner.py#L108)）在 fused params 到达 `BatchedDynamicEmbeddingTablesV2` 之前，从中剥离 DynamicEmb 专有的 key。

### Checkpoint Dump/Load

[`dynamicemb/dump_load.py`](../external/recsys-examples/corelib/dynamicemb/dynamicemb/dump_load.py)：

- `find_sharded_modules(model)` — DFS 查找 `ShardedEmbeddingCollection` / `ShardedEmbeddingBagCollection`
- `get_dynamic_emb_module(model)` — DFS 查找 `BatchedDynamicEmbeddingTablesV2`（遍历 `nn.Module.children()` 不会发现的私有 `_lookups` / `_emb_modules` / `_emb_module` 属性）
- `DynamicEmbDump(path, model, table_names, optim, counter, pg, allow_overwrite)` — per-rank dump 到 `path/<collection_path>/<table_name>.{keys,values,opt,counter,sizes}` 二进制文件
- `DynamicEmbLoad(path, model, table_names, optim, counter, pg)` — 反向

目前仅支持 **row-wise 分片**。`dump_load.py:97-102` 和 `:205-210` 处的 TODO 注释标记了这一点和全有或全无的 optimizer dump。

## TorchEasyRec 集成

集成规模很小——大部分繁重的工作在上游 `dynamicemb` 中。TorchEasyRec 添加了：

1. **可插拔安装** — `has_dynamicemb` 标志（`dynamicemb_util.py:134-157`）
2. **从 protobuf 构建约束** — `build_dynamicemb_constraints()`（`dynamicemb_util.py:216-307`）
3. **Planner monkey-patch** — `to_sharding_plan` 和 `HardwarePerfConfig.get_device_bw` 覆盖（`dynamicemb_util.py:395-531`）
4. **变体发射** — `_emit_dynamicemb_variants()` 将每个选项扩展为 HYBRID + CACHING × load_factors（`plan_util.py:887-916`）
5. **存储估算器** — `dynamicemb_calculate_shard_storages()`（`dynamicemb_util.py:637-775`）
6. **Checkpoint** — 接入 `restore_model()` / `save_model()`（`checkpoint_util.py:705-722, 743-751`）
7. **范围检查绕过** — `_validate_feature_range_with_dynamicemb` 跳过 TorchRec 的 key 范围检查（`dynamicemb_util.py:777-793`）
8. **工具** — `zch_to_dynamicemb_convert.py`（从基于 ZCH 的表迁移）、`create_dynamicemb_init_ckpt.py`（从稠密 init 冷启动）

### 安装检测

[`dynamicemb_util.py:134-157`](../torcheasyrec/tzrec/utils/dynamicemb_util.py#L134-L157)：

```python
has_dynamicemb = False
try:
    import dynamicemb
    from dynamicemb import (DynamicEmbInitializerArgs, ...)
    from dynamicemb.planner import (DynamicEmbParameterConstraints, ...)
    from dynamicemb.shard import (DynamicEmbeddingBagCollectionSharder, ...)
    has_dynamicemb = True
except Exception:
    pass
```

所有后续 patch 都受 `if has_dynamicemb:` 保护。

### Per-Feature `use_dynamicemb` 标志

[`features/feature.py:631`](../torcheasyrec/tzrec/features/feature.py#L631) 和 `:657` — 当特征的 `embedding_config` 是 `DynamicEmbedding` 时，生成的 `BaseEmbeddingConfig` / `EmbeddingBagConfig` 被标记：

```python
emb_bag_config.use_dynamicemb = hasattr(self.embedding_config, "dynamicemb_options")
# 或
emb_config.use_dynamicemb = hasattr(self.embedding_config, "dynamicemb_options")
```

此标志流经 constraints → planner → sharder。

### 从 Protobuf 构建约束

[`dynamicemb_util.py:216-307`](../torcheasyrec/tzrec/utils/dynamicemb_util.py#L216-L307) — `build_dynamicemb_constraints()`：

```python
def build_dynamicemb_constraints(
    dynamicemb_cfg: feature_pb2.DynamicEmbedding,
    emb_config: BaseEmbeddingConfig,
) -> ParameterConstraints:
    score_strategy = DynamicEmbScoreStrategy[dynamicemb_cfg.score_strategy]
    init_capacity = align_to_table_size(dynamicemb_cfg.init_capacity_per_rank) or None

    # admission counter
    if dynamicemb_cfg.WhichOneof("admission_strategy") == "frequency_admission_strategy":
        admission_counter = KVCounter(
            capacity=align_to_table_size(int(counter_capacity / world_size)),
            bucket_capacity=admission_strategy_cfg.counter_bucket_capacity,
        )
        admit_strategy = FrequencyAdmissionStrategy(threshold=..., ...)

    dynamicemb_options = dynamicemb.DynamicEmbTableOptions(
        max_capacity=dynamicemb_cfg.max_capacity,
        init_capacity=init_capacity,
        initializer_args=_build_dynamicemb_initializer(...),
        eval_initializer_args=_build_dynamicemb_initializer(..., is_eval=True),
        score_strategy=score_strategy,
        admit_strategy=admit_strategy,
        admission_counter=admission_counter,
        bucket_capacity=dynamicemb_cfg.bucket_capacity,  # optional
    )
    if dynamicemb_cfg.HasField("cache_load_factor"):
        constraints_kwargs["cache_params"] = CacheParams(load_factor=...)

    return DynamicEmbParameterConstraints(
        use_dynamicemb=True,
        sharding_types=[ShardingType.ROW_WISE.value],   # 仅支持 row-wise
        compute_kernels=[EmbeddingComputeKernel.CUSTOMIZED_KERNEL.value],
        dynamicemb_options=dynamicemb_options,
        **constraints_kwargs,
    )
```

### Planner Monkey-Patch

四个 patch 使 TorchRec planner 对 DynamicEmb 感知。所有 patch 在导入时应用一次（受 `if has_dynamicemb:` 保护）。

#### 1. Compute Kernel 枚举（EBC + EC）

[`dynamicemb_util.py:313-340`](../torcheasyrec/tzrec/utils/dynamicemb_util.py#L313-L340)：

```python
enumerators.GUARDED_COMPUTE_KERNELS.add(EmbeddingComputeKernel.CUSTOMIZED_KERNEL)

def _ebc_compute_kernels(self, sharding_type, compute_device_type):
    compute_kernels = super().compute_kernels(sharding_type, compute_device_type)
    if compute_device_type == "cuda":
        compute_kernels += [EmbeddingComputeKernel.CUSTOMIZED_KERNEL.value]
    return compute_kernels

DynamicEmbeddingBagCollectionSharder.compute_kernels = _ebc_compute_kernels
DynamicEmbeddingCollectionSharder.compute_kernels = _ec_compute_kernels
```

没有这个，TorchRec enumerator 永远不会提供 `CUSTOMIZED_KERNEL` 作为候选 compute kernel。

#### 2. 定制 Sharding Plan 发射

[`dynamicemb_util.py:395-500`](../torcheasyrec/tzrec/utils/dynamicemb_util.py#L395-L500) — 覆盖 `planners.to_sharding_plan`：

```python
def _to_sharding_plan(sharding_options, topology):
    plan = {}
    for sharding_option in sharding_options:
        shards = sharding_option.shards
        # ... 从 shards 构建 EnumerableShardingSpec
        if sharding_option.use_dynamicemb:
            # 从实际 shard 布局计算 local_hbm_for_values
            dynamicemb_options.local_hbm_for_values = (
                _calculate_dynamicemb_table_storage_specific_size(
                    shards[0].size, tensor.element_size(),
                    optimizer_multipler,
                    sharding_option.cache_load_factor,
                    is_hbm=True, only_values=True,
                    bucket_capacity=dynamicemb_options.bucket_capacity,
                )
            )
            # 填充 dim / max_capacity / embedding_dtype / index_type
            # (在 NVIDIA recsys-examples PR #343 之后这些不再由上游自动填充；
            # 我们在这里设置)
            dynamicemb_options.dim = shards[0].size[1]
            dynamicemb_options.max_capacity = shards[0].size[0]
            if dynamicemb_options.embedding_dtype is None:
                dynamicemb_options.embedding_dtype = tensor.dtype
            if dynamicemb_options.index_type is None:
                dynamicemb_options.index_type = torch.int64

            module_plan[name] = DynamicEmbParameterSharding(
                sharding_spec=sharding_spec,
                sharding_type=ShardingType.ROW_WISE.value,
                ranks=list(range(world_size)),
                compute_kernel=EmbeddingComputeKernel.CUSTOMIZED_KERNEL.value,
                customized_compute_kernel=DynamicEmbKernel,
                dist_type="roundrobin",
                dynamicemb_options=dynamicemb_options,
            )
            _log_dynamicemb_table_plan(...)
        else:
            # 默认路径：标准 ParameterSharding
            module_plan[name] = ParameterSharding(...)
    return ShardingPlan(plan)

planners.to_sharding_plan = _to_sharding_plan
```

#### 3. 带宽性能模型覆盖

[`dynamicemb_util.py:502-531`](../torcheasyrec/tzrec/utils/dynamicemb_util.py#L502-L531)：

```python
def _customized_kernel_aware_get_device_bw(
    self, compute_device, compute_kernel,
    hbm_mem_bw, ddr_mem_bw, ssd_mem_bw, hbm_to_ddr_mem_bw,
    caching_ratio=None, prefetch_pipeline=False,
):
    if compute_kernel == EmbeddingComputeKernel.CUSTOMIZED_KERNEL.value:
        cr = caching_ratio if caching_ratio is not None else 0.0
        # (cached 比例 * hbm + spill 比例 * hbm->ddr) / 10
        # (/10 是校准因子——参见源代码中的注释)
        return (cr * hbm_mem_bw + (1 - cr) * hbm_to_ddr_mem_bw) / 10
    return _orig_hw_perf_config_get_device_bw(...)

HardwarePerfConfig.get_device_bw = _customized_kernel_aware_get_device_bw
```

这使得 TorchRec 性能模型为 DynamicEmb 收取正确的带宽——cached 部分用 HBM，spill 部分用 HBM→DDR。

#### 4. Perf Context Builder

[`dynamicemb_util.py:533-591`](../torcheasyrec/tzrec/utils/dynamicemb_util.py#L533-L591)：

```python
def _dynamicemb_aware_build_shard_perf_contexts(cls, config, shard_sizes, sharding_option,
                                                topology, constraints, sharder, *args, **kwargs):
    """将经验 x_eff 注入到两种模式的性能估算器中。

    临时用经验拟合的 x_eff 的克隆替换 sharding_option.cache_params 的 load_factor，
    适配 (mode, cache_load_factor) 组合。返回前恢复，以便存储估算器仍能看到未提升的比率。
    """
    dynamicemb_options = getattr(sharding_option, "dynamicemb_options", None)
    original_cache_params = sharding_option.cache_params
    if dynamicemb_options is not None:
        caching = bool(getattr(dynamicemb_options, "caching", False))
        stats = original_cache_params.stats if original_cache_params else None
        x_eff = _dynamicemb_effective_cache_ratio(
            sharding_option.cache_load_factor, caching=caching, stats=stats
        )
        sharding_option.cache_params = (
            dataclasses.replace(original_cache_params, load_factor=x_eff)
            if original_cache_params is not None else CacheParams(load_factor=x_eff)
        )
    try:
        result = _orig_build_shard_perf_contexts(...)
    finally:
        sharding_option.cache_params = original_cache_params
    return result

ShardPerfContext.build_shard_perf_contexts = classmethod(_dynamicemb_aware_build_shard_perf_contexts)
```

### 存储估算

`_calculate_dynamicemb_table_storage_specific_size`（[`dynamicemb_util.py:342-393`](../torcheasyrec/tzrec/utils/dynamicemb_util.py#L342-L393)）— per-shard HBM/DDR 字节预算：

```
value_bytes_per_row = round_up16(dim * (1 + opt_mult) * element)
total_value_memory  = align(rows) * value_bytes_per_row
num_buckets         = align(rows) / bucket_capacity

HBM budget = cache_ratio * total_value_memory                 # values
           + align(rows) * (key<8B> + score<8B> + digest<1B>) # per-row
           + num_buckets * bucket_header<4B>                  # per-bucket

DDR budget = HYBRID  (caching=False): (1 - cache_ratio) * total_value_memory
             CACHING (caching=True):  total_value_memory      # 完整后备
```

`dynamicemb_calculate_shard_storages`（[`dynamicemb_util.py:637-775`](../torcheasyrec/tzrec/utils/dynamicemb_util.py#L637-L775)）是 TorchRec `shard_estimators.calculate_shard_storages` 的直接替代品。它计算上述 HBM / DDR 分解，然后加上 pipeline I/O 成本（`shard_estimators.calculate_pipeline_io_cost(...)`）。

### 变体发射

[`plan_util.py:887-916`](../torcheasyrec/tzrec/utils/plan_util.py#L887-L916) — `_emit_dynamicemb_variants`：

```python
def _emit_dynamicemb_variants(base_option: ShardingOption) -> List[ShardingOption]:
    """将 dynamicemb ShardingOption 扩展为 HYBRID + CACHING 变体。

    扫描两种放置模式 (caching=False 和 caching=True)，并且
    当 base_option.cache_params 未设置时，扫描十个 cache_load_factor 值
    (0.1, 0.2, ..., 1.0)。下游 2D DP proposer 为每个表选择
    同时满足 HBM 和主机拓扑预算的 (mode, ratio)。
    """
    if base_option.cache_params is None:
        load_factors = [(i + 1) / 10 for i in range(10)]
        stats = None
    else:
        load_factors = [base_option.cache_params.load_factor]
        stats = base_option.cache_params.stats
    variants = []
    for caching_mode in (False, True):
        for load_factor in load_factors:
            opt = copy.deepcopy(base_option)
            opt.cache_params = CacheParams(load_factor=load_factor, stats=stats)
            opt.dynamicemb_options.caching = caching_mode
            variants.append(opt)
    return variants
```

当用户显式指定 `cache_load_factor` 时，扫描坍缩为单点。否则 proposer 看到每个 dynamicemb 表多达 20 个变体（2 模式 × 10 比率）并选择最便宜的适合变体。

### Checkpoint 集成

[`checkpoint_util.py:705-722, 743-751`](../torcheasyrec/tzrec/utils/checkpoint_util.py#L705-L722)：

```python
# restore_model(): 标准 load_state_dict 之后
if has_dynamicemb:
    from dynamicemb.dump_load import DynamicEmbLoad
    dynamicemb_path = os.path.join(checkpoint_dir, "dynamicemb")
    if os.path.exists(dynamicemb_path):
        DynamicEmbLoad(
            dynamicemb_path, model,
            table_names=meta.get("dynamicemb_load_table_names", None),
            optim=meta.get("dynamicemb_load_optim", optimizer is not None),
            counter=True,
        )

# save_model(): 标准 state_dict 保存之后
if has_dynamicemb:
    from dynamicemb.dump_load import DynamicEmbDump
    DynamicEmbDump(
        os.path.join(checkpoint_dir, "dynamicemb"),
        model,
        optim=optimizer is not None,
        counter=True,
    )
```

DynamicEmb checkpoint 单独存储在 `checkpoint_dir/dynamicemb/<collection_path>/<table_name>.{keys,values,opt,counter,sizes}`，并由上游 `dump_load.find_sharded_modules()` 中的 per-rank DFS 遍历重新加载。

### 范围检查绕过

[`dynamicemb_util.py:777-793`](../torcheasyrec/tzrec/utils/dynamicemb_util.py#L777-L793)：

```python
def _validate_feature_range_with_dynamicemb(kjt, configs):
    """跳过 dynamicemb 特征的范围检查。

    DynamicEmb 使用接受任意 uint64 key 的哈希表。
    max_capacity 是存储限制，不是有效 key 范围。
    """
    filtered_configs = [c for c in configs if not getattr(c, "use_dynamicemb", False)]
    if not filtered_configs:
        return True
    return _orig_validate_feature_range(kjt, filtered_configs)

_jtv._validate_feature_range = _validate_feature_range_with_dynamicemb
```

TorchRec 通常拒绝 key ID ≥ `num_embeddings` 的 KJT 输入（假定为 key 范围）。DynamicEmb 没有这样的约束——key 被哈希到无界大小的桶中，最大为 `max_capacity`。

## 迁移工具

### `tools/dynamicemb/zch_to_dynamicemb_convert.py`

将 checkpoint 从基于 ZCH（`ManagedCollisionEmbeddingBagCollection`）的表迁移到 DynamicEmb 表。读取 ZCH checkpoint，将原始 ID 哈希到目标 `max_capacity` 布局，并写入 `DynamicEmbLoad` 可以摄取的 `dynamicemb/` 目录。

> **背景阅读**: [ZCH 零碰撞哈希](07-02-zch) — ZCH 的完整机制、MCH 管理缓冲区结构及导出限制

为什么需要这个：ZCH 使用固定 `max_id` 碰撞模块；DynamicEmb 使用开放哈希表。不同的 ID 空间 → 不同的 checkpoint。

### `tools/dynamicemb/create_dynamicemb_init_ckpt.py`

通过以 `DynamicEmbLoad` 期望的磁盘格式写入初始 keys + values，从稠密 FBGEMM 初始化（例如预训练的 EBC）冷启动 DynamicEmb checkpoint。用于在不完全重新训练的情况下预热新部署。

## 何时使用 DynamicEmb（对比 FBGEMM）

| 标准 | FBGEMM (EBC/EC) | DynamicEmb |
|------|----------------|-----------|
| 基数 | 有界（通常 ≤ 10M） | 无界 |
| 内存浪费 | 稀疏时高 | 只为热数据付费 |
| 冷启动成本 | 无 | 一次性访客在阈值处被接纳 |
| 延迟（热路径） | 最低（稠密索引） | 较高（哈希探测 + 可能溢出） |
| 可导出到 RTP / TensorRT | 是（JIT script） | 有限（哈希 kernel 是定制的） |
| Checkpoint 格式 | 一个大 `state_dict` | per-rank 二进制 + `state_dict` |
| 操作模式 | ROW_WISE / TABLE_WISE / COLUMN_WISE / TP / DP | 仅 ROW_WISE |

推荐：
- **频繁、热的 ID 在有界空间中** → FBGEMM
- **长尾、冷启动、无界空间** → DynamicEmb
- **混合** → per-feature 选择，甚至在同一模型内

## 已知限制

- **仅 Row-wise 分片** — 上游 TODO 位于 `dump_load.py:97`、`:205`（以及 planner `planner.py:333` 中的注释：`TODO:0 is we don't have column-wise sharding now`）
- **全有或全无的 optimizer dump** — `DynamicEmbDump(optim=True)` 为该表 dump 所有 opt 状态（无 per-param opt 控制）
- **无 FBGEMM 风格的 `bounds_check_mode`** — DynamicEmb 使用自己的 `DynamicEmbCheckMode` 枚举（ERROR/WARNING/IGNORE）处理插入失败，而非 TorchRec 的
- **Planner 中无跨表融合** — 每个表独立枚举，即使它们可以共享存储（`KeyedJaggedTensor` / `FeatureGroup` 中的 table_fusion 逻辑未扩展到 DynamicEmb）
- **导出兼容性有限** — `BatchedDynamicEmbeddingTablesV2` CUDA kernel 不能直接 JIT script。对于 RTP/TRT 导出，带 `dynamicemb { }` 的特征通常需要在导出时回退到 FBGEMM，或使用自定义 RTP hook 导出（后者是进行中的工作，不是默认路径）

## 参考资料

### NVIDIA 上游（子模块）

| 文件 | 用途 |
|---|---|
| [`external/recsys-examples/corelib/dynamiceb/dynamicemb/__init__.py`](../external/recsys-examples/corelib/dynamicemb/dynamicemb/__init__.py) | 公共 API |
| [`external/recsys-examples/corelib/dynamicemb/dynamicemb/dynamicemb_config.py`](../external/recsys-examples/corelib/dynamicemb/dynamicemb/dynamicemb_config.py) | `DynamicEmbTableOptions`、枚举 |
| [`external/recsys-examples/corelib/dynamicemb/dynamicemb/types.py`](../external/recsys-examples/corelib/dynamicemb/dynamicemb/types.py) | `Storage`、`Cache` ABC |
| [`external/recsys-examples/corelib/dynamicemb/dynamicemb/planner/planner.py`](../external/recsys-examples/corelib/dynamicemb/dynamicemb/planner/planner.py) | `DynamicEmbeddingShardingPlanner` |
| [`external/recsys-examples/corelib/dynamicemb/dynamicemb/shard/embeddingbag.py`](../external/recsys-examples/corelib/dynamicemb/dynamicemb/shard/embeddingbag.py) | `DynamicEmbeddingBagCollectionSharder` |
| [`external/recsys-examples/corelib/dynamicemb/dynamicemb/shard/embedding.py`](../external/recsys-examples/corelib/dynamicemb/dynamicemb/shard/embedding.py) | `DynamicEmbeddingCollectionSharder` |
| [`external/recsys-examples/corelib/dynamicemb/dynamicemb/dump_load.py`](../external/recsys-examples/corelib/dynamicemb/dynamicemb/dump_load.py) | `DynamicEmbDump` / `DynamicEmbLoad` |
| [`external/recsys-examples/corelib/dynamicemb/dynamicemb/embedding_admission.py`](../external/recsys-examples/corelib/dynamicemb/dynamicemb/embedding_admission.py) | `FrequencyAdmissionStrategy`、`KVCounter` |
| [`external/recsys-examples/corelib/dynamicemb/dynamicemb/optimizer.py`](../external/recsys-examples/corelib/dynamicemb/dynamicemb/optimizer.py) | 稀疏 optim 状态 |

### TorchEasyRec 集成

| 文件 | 用途 |
|---|---|
| [`torcheasyrec/tzrec/utils/dynamicemb_util.py`](../torcheasyrec/tzrec/utils/dynamicemb_util.py) | planner hooks、constraints、存储估算器（794 行） |
| [`torcheasyrec/tzrec/utils/plan_util.py:887-916`](../torcheasyrec/tzrec/utils/plan_util.py#L887-L916) | `_emit_dynamicemb_variants` |
| [`torcheasyrec/tzrec/utils/checkpoint_util.py:705-751`](../torcheasyrec/tzrec/utils/checkpoint_util.py#L705-L751) | save/restore DynamicEmb checkpoint |
| [`torcheasyrec/tzrec/features/feature.py:631,657`](../torcheasyrec/tzrec/features/feature.py#L631) | 嵌入配置上的 `use_dynamicemb` 标志 |
| [`torcheasyrec/tzrec/protos/feature.proto`](../torcheasyrec/tzrec/protos/feature.proto) | `DynamicEmbedding` 消息 |
| [`torcheasyrec/tzrec/tools/dynamicemb/zch_to_dynamicemb_convert.py`](../torcheasyrec/tzrec/tools/dynamicemb/zch_to_dynamicemb_convert.py) | ZCH → DynamicEmb checkpoint 迁移 |
| [`torcheasyrec/tzrec/tools/dynamicemb/create_dynamicemb_init_ckpt.py`](../torcheasyrec/tzrec/tools/dynamicemb/create_dynamicemb_init_ckpt.py) | 从稠密 FBGEMM 冷启动 DynamicEmb |
