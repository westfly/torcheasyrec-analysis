---
title: DynamicEmb Integration
nav_order: 11
---

# DynamicEmb Integration

## Overview

TorchEasyRec integrates [DynamicEmb](https://github.com/NVIDIA/recsys-examples/tree/main/corelib/dynamicemb), an NVIDIA open-source GPU hash-table embedding library, as an alternative sparse embedding backend alongside TorchRec's `EmbeddingBagCollection` (EBC) / `EmbeddingCollection` (EC) backed by FBGEMM. The integration is **optional and additive** — features configured with `dynamicemb { ... }` go through the DynamicEmb path; everything else stays on the standard EBC/EC path.

DynamicEmb is a **customized compute kernel** in TorchRec's sharding framework, not a replacement for EBC. It plugs into the existing planner / sharder / checkpoint machinery through monkey-patched hooks and a small integration shim.

This document covers two perspectives:

1. **NVIDIA upstream** — what DynamicEmb provides as a library
2. **TorchEasyRec integration** — what the integration adds on top

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

The upstream code lives in this analysis repo as a submodule:
[`external/recsys-examples/corelib/dynamicemb/`](../external/recsys-examples/corelib/dynamicemb/) (commit `2091502`, Apache-2.0, copyright NVIDIA 2025).

The TorchEasyRec integration lives at:
[`torcheasyrec/tzrec/utils/dynamicemb_util.py`](../torcheasyrec/tzrec/utils/dynamicemb_util.py) (794 lines, 2024 Alibaba, Apache-2.0).

## Why DynamicEmb?

Standard FBGEMM-backed embedding tables are **pre-allocated dense arrays** indexed by integer ID. This works well when the ID space is small and bounded (e.g., ≤ 10M), but breaks down for:

- **Massive cardinality** — long-tail features with billions of unique IDs (cold-start IDs that may never appear in training)
- **Highly skewed distributions** — most rows are sparse / never touched, wasting HBM
- **Dynamic workloads** — production traffic patterns shift over time, pre-sizing is hard

DynamicEmb solves this with a **GPU hash table** that:

- **Grows on demand** — buckets are added as new IDs arrive
- **Evicts cold IDs** — LRU / LFU / epoch-based / customized scoring policies
- **Spills to host memory** — values can live on host (DDR) with an HBM cache, controlled by `cache_load_factor`
- **Uses no FBGEMM rows** — the kernel (`DynamicEmbStorage` / `HybridStorage` / `DynamicEmbCache` in C++/CUDA) is a custom implementation in `dynamicemb_extensions`

The trade-off: DynamicEmb has higher per-lookup latency than dense FBGEMM (hash probe + possible host spill), but eliminates the memory waste of pre-allocation.

## Storage Modes (HBM / HYBRID / CACHING)

The runtime has three storage modes, selected by the planner based on `cache_load_factor` and `caching` flag:

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

The mode is **implicit**, derived from `(cache_load_factor, caching)`:

| `cache_load_factor` | `caching` | Mode | Notes |
|---|---|---|---|
| `1.0` | either | `HBM_ONLY` | Host tier dropped; `dynamicemb` runtime switches to `DynamicEmbStorage` kernel |
| `< 1.0` | `False` | `HYBRID` | Values hash-partitioned across HBM / host |
| `< 1.0` | `True` | `CACHING` | HBM is a hot-row cache over host backing store |

The 8x jump in `_dynamicemb_effective_cache_ratio()` at `cache_load_factor=1.0` ([`dynamicemb_util.py:94-95`](../torcheasyrec/tzrec/utils/dynamicemb_util.py#L94-L95)) reflects this mode switch — the runtime kernel changes, not just the parameters.

## Eviction Strategies

[`dynamicemb/dynamicemb_config.py:104-110`](../external/recsys-examples/corelib/dynamicemb/dynamicemb/dynamicemb_config.py#L104-L110):

```python
class DynamicEmbEvictStrategy(enum.Enum):
    LRU = EvictStrategy.KLru           # standard LRU
    LFU = EvictStrategy.KLfu           # standard LFU
    EPOCH_LRU = EvictStrategy.KEpochLru  # LRU within a training epoch
    EPOCH_LFU = EvictStrategy.KEpochLfu  # LFU within a training epoch
    CUSTOMIZED = EvictStrategy.KCustomized  # user-provided per-row score
```

Pair with `DynamicEmbScoreStrategy` ([`types.py:113`](../external/recsys-examples/corelib/dynamicemb/dynamicemb/types.py#L113)):
- `STEP` — score increases monotonically per step (aging)
- `TIMESTAMP` — score is the step index
- `CUSTOMIZED` — user-supplied scores per row

## Per-Feature Config (protobuf)

A feature opts in to DynamicEmb by setting `dynamicemb { ... }` in its `FeatureConfig` ([`feature.proto`](../torcheasyrec/tzrec/protos/feature.proto)):

```protobuf
message FeatureConfig {
    ...
    oneof embedding {
        RegularEmbedding regular_embedding = 4;
        ZchEmbedding zch_embedding = 5;
        DynamicEmbedding dynamic_embedding = 6;  // <- opt-in to dynamicemb
    }
    ...
}

message DynamicEmbedding {
    int64 max_capacity = 1;                       // per-rank row ceiling
    optional int64 init_capacity_per_rank = 2;    // initial alloc
    optional int64 bucket_capacity = 3;            // bucket size (default 128)
    optional float cache_load_factor = 4;         // HBM share in HYBRID / cache size in CACHING
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

Key fields:
- `max_capacity` — per-rank hard cap on row count. Unlike FBGEMM this is a **storage limit**, not a key range.
- `cache_load_factor` — see storage mode table above.
- `frequency_admission_strategy` — skip inserting a key into the table until it has been seen ≥ `threshold` times (saves memory for one-hit-wonders).

## NVIDIA Upstream: Building Blocks

### Package Layout

```
external/recsys-examples/corelib/dynamicemb/
├── setup.py                                # CUDAExtension build (Bazel/CMake)
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

### Core Types

**`DynamicEmbTableOptions`** ([`dynamicemb_config.py`](../external/recsys-examples/corelib/dynamicemb/dynamicemb/dynamicemb_config.py)) — per-table configuration dataclass containing:

| Field | Purpose |
|---|---|
| `max_capacity` | per-rank row ceiling |
| `init_capacity` | initial allocation (clamped to `max_capacity`) |
| `bucket_capacity` | hash table bucket size (default 128, aligned to 16) |
| `global_hbm_for_values` | HBM budget for values, divided by world_size → `local_hbm_for_values` |
| `local_hbm_for_values` | per-rank HBM budget (filled by planner) |
| `initializer_args` | train-time init (UNIFORM/NORMAL/CONSTANT/...) |
| `eval_initializer_args` | eval-time init (typically CONSTANT=0) |
| `score_strategy` | STEP / TIMESTAMP / CUSTOMIZED |
| `evict_strategy` | LRU / LFU / EPOCH_LRU / EPOCH_LFU / CUSTOMIZED |
| `admit_strategy` | optional `FrequencyAdmissionStrategy` |
| `admission_counter` | optional `KVCounter` for admission |
| `caching` | `False` = HYBRID, `True` = CACHING |
| `dist_type` | `"roundrobin"` (only one supported currently) |
| `index_type` | `torch.int64` (default) |
| `embedding_dtype` | `torch.float32` (default) |
| `training` | set by planner; presence triggers optimizer state |
| `dim` | embedding dim (filled by planner) |
| `check_mode` | `DynamicEmbCheckMode.ERROR / WARNING / IGNORE` for insertion failures |

**`Storage` and `Cache`** ([`types.py:150`](../external/recsys-examples/corelib/dynamicemb/dynamicemb/types.py#L150)) — abstract base classes for the three storage modes:

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

The two concrete backends (`DynamicEmbStorage`, `HybridStorage`, `DynamicEmbCache`) live in the C++/CUDA extension `dynamicemb_extensions`, built via `setup.py`.

### Sharders

[`dynamicemb/shard/embeddingbag.py`](../external/recsys-examples/corelib/dynamicemb/dynamicemb/shard/embeddingbag.py) and `embedding.py` provide `DynamicEmbeddingBagCollectionSharder` / `DynamicEmbeddingCollectionSharder`. These register `DynamicEmbKernel` (the value `"DynamicEmb"`) as a `customized_compute_kernel` for `EmbeddingComputeKernel.CUSTOMIZED_KERNEL`.

### Planner

[`dynamicemb/planner/planner.py:213`](../external/recsys-examples/corelib/dynamicemb/dynamicemb/planner/planner.py#L213) — `DynamicEmbeddingShardingPlanner`:

```python
class DynamicEmbeddingShardingPlanner:
    def __init__(self, eb_configs, topology=None, batch_size=None, ...,
                 constraints=None, debug=True):
        _prepare_dynemb_table_options(constraints, eb_configs)
        # split into dyn_emb + torchrec constraints
        # build self._torchrec_planner = EmbeddingShardingPlanner(...)
        # build self._dyn_emb_plan: name -> DynamicEmbParameterSharding
        #   (each: ROW_WISE sharding, EnumerableShardingSpec across ranks,
        #    compute_kernel=CUSTOMIZED_KERNEL, customized_compute_kernel=DynamicEmb)

    def collective_plan(self, module, sharders, pg=...):
        torchrec_plan = self._torchrec_planner.collective_plan(module, sharders, pg)
        # overlay dyn_emb plans onto matching table names
        return torchrec_plan
```

`DynamicEmbParameterSharding` ([`planner.py:81`](../external/recsys-examples/corelib/dynamicemb/dynamicemb/planner/planner.py#L81)) extends `ParameterSharding` with `compute_kernel=CUSTOMIZED_KERNEL`, `customized_compute_kernel=DynamicEmb`, `dist_type="roundrobin"`, and the `dynamicemb_options` instance.

`pop_additional_fused_params()` ([`planner.py:108`](../external/recsys-examples/corelib/dynamicemb/dynamicemb/planner/planner.py#L108)) strips DynamicEmb-only keys from `fused_params` before they reach `BatchedDynamicEmbeddingTablesV2`.

### Checkpoint Dump/Load

[`dynamicemb/dump_load.py`](../external/recsys-examples/corelib/dynamicemb/dynamicemb/dump_load.py):

- `find_sharded_modules(model)` — DFS for `ShardedEmbeddingCollection` / `ShardedEmbeddingBagCollection`
- `get_dynamic_emb_module(model)` — DFS for `BatchedDynamicEmbeddingTablesV2` (walks through private `_lookups` / `_emb_modules` / `_emb_module` attrs that `nn.Module.children()` won't discover)
- `DynamicEmbDump(path, model, table_names, optim, counter, pg, allow_overwrite)` — per-rank dump to `path/<collection_path>/<table_name>.{keys,values,opt,counter,sizes}` binary files
- `DynamicEmbLoad(path, model, table_names, optim, counter, pg)` — reverse

Currently supports only **row-wise sharding**. TODO comments at `dump_load.py:97-102` and `:205-210` flag this and the all-or-nothing optimizer dump.

## TorchEasyRec Integration

The integration is small — most of the heavy lifting is in upstream `dynamicemb`. TorchEasyRec adds:

1. **Pluggable install** — `has_dynamicemb` flag (dynamicemb_util.py:134-157)
2. **Constraints from protobuf** — `build_dynamicemb_constraints()` (dynamicemb_util.py:216-307)
3. **Planner monkey-patches** — `to_sharding_plan` and `HardwarePerfConfig.get_device_bw` overrides (dynamicemb_util.py:395-531)
4. **Variant emission** — `_emit_dynamicemb_variants()` expanding each option into HYBRID + CACHING × load_factors (plan_util.py:887-916)
5. **Storage estimator** — `dynamicemb_calculate_shard_storages()` (dynamicemb_util.py:637-775)
6. **Checkpoints** — wired into `restore_model()` / `save_model()` (checkpoint_util.py:705-722, 743-751)
7. **Range-check bypass** — `_validate_feature_range_with_dynamicemb` skips TorchRec's key range check (dynamicemb_util.py:777-793)
8. **Tools** — `zch_to_dynamicemb_convert.py` (migrate from ZCH-based tables), `create_dynamicemb_init_ckpt.py` (cold-start from dense init)

### Install Detection

[`dynamicemb_util.py:134-157`](../torcheasyrec/tzrec/utils/dynamicemb_util.py#L134-L157):

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

All subsequent patching is guarded by `if has_dynamicemb:`.

### Per-Feature `use_dynamicemb` Flag

[`features/feature.py:631`](../torcheasyrec/tzrec/features/feature.py#L631) and `:657` — when a feature's `embedding_config` is a `DynamicEmbedding`, the generated `BaseEmbeddingConfig` / `EmbeddingBagConfig` is tagged:

```python
emb_bag_config.use_dynamicemb = hasattr(self.embedding_config, "dynamicemb_options")
# or
emb_config.use_dynamicemb = hasattr(self.embedding_config, "dynamicemb_options")
```

This flag flows through constraints → planner → sharder.

### Building Constraints from Protobuf

[`dynamicemb_util.py:216-307`](../torcheasyrec/tzrec/utils/dynamicemb_util.py#L216-L307) — `build_dynamicemb_constraints()`:

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
        sharding_types=[ShardingType.ROW_WISE.value],   # only row-wise supported
        compute_kernels=[EmbeddingComputeKernel.CUSTOMIZED_KERNEL.value],
        dynamicemb_options=dynamicemb_options,
        **constraints_kwargs,
    )
```

### Planner Monkey-Patches

Four patches make the TorchRec planner DynamicEmb-aware. All are applied once at import time (guarded by `if has_dynamicemb:`).

#### 1. Compute Kernel Enumeration (EBC + EC)

[`dynamicemb_util.py:313-340`](../torcheasyrec/tzrec/utils/dynamicemb_util.py#L313-L340):

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

Without this, the TorchRec enumerator never offers `CUSTOMIZED_KERNEL` as a candidate compute kernel.

#### 2. Customized Sharding Plan Emission

[`dynamicemb_util.py:395-500`](../torcheasyrec/tzrec/utils/dynamicemb_util.py#L395-L500) — overrides `planners.to_sharding_plan`:

```python
def _to_sharding_plan(sharding_options, topology):
    plan = {}
    for sharding_option in sharding_options:
        shards = sharding_option.shards
        # ... build EnumerableShardingSpec from shards
        if sharding_option.use_dynamicemb:
            # compute local_hbm_for_values from the actual shard layout
            dynamicemb_options.local_hbm_for_values = (
                _calculate_dynamicemb_table_storage_specific_size(
                    shards[0].size, tensor.element_size(),
                    optimizer_multipler,
                    sharding_option.cache_load_factor,
                    is_hbm=True, only_values=True,
                    bucket_capacity=dynamicemb_options.bucket_capacity,
                )
            )
            # fill in dim / max_capacity / embedding_dtype / index_type
            # (after NVIDIA recsys-examples PR #343 these are no longer
            # auto-populated by upstream; we set them here)
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
            # default path: standard ParameterSharding
            module_plan[name] = ParameterSharding(...)
    return ShardingPlan(plan)

planners.to_sharding_plan = _to_sharding_plan
```

#### 3. Bandwidth Perf Model Override

[`dynamicemb_util.py:502-531`](../torcheasyrec/tzrec/utils/dynamicemb_util.py#L502-L531):

```python
def _customized_kernel_aware_get_device_bw(
    self, compute_device, compute_kernel,
    hbm_mem_bw, ddr_mem_bw, ssd_mem_bw, hbm_to_ddr_mem_bw,
    caching_ratio=None, prefetch_pipeline=False,
):
    if compute_kernel == EmbeddingComputeKernel.CUSTOMIZED_KERNEL.value:
        cr = caching_ratio if caching_ratio is not None else 0.0
        # (cached portion * hbm + spill portion * hbm->ddr) / 10
        # (the /10 is a calibration factor — see comment in source)
        return (cr * hbm_mem_bw + (1 - cr) * hbm_to_ddr_mem_bw) / 10
    return _orig_hw_perf_config_get_device_bw(...)

HardwarePerfConfig.get_device_bw = _customized_kernel_aware_get_device_bw
```

This makes the TorchRec perf model charge DynamicEmb the right bandwidth — HBM for the cached portion, HBM→DDR for the spill portion.

#### 4. Perf Context Builder

[`dynamicemb_util.py:533-591`](../torcheasyrec/tzrec/utils/dynamicemb_util.py#L533-L591):

```python
def _dynamicemb_aware_build_shard_perf_contexts(cls, config, shard_sizes, sharding_option,
                                                topology, constraints, sharder, *args, **kwargs):
    """Inject the empirical x_eff into the perf estimator for both modes.

    Temporarily replace sharding_option.cache_params with a clone whose
    load_factor is the empirically-fitted x_eff for the (mode, cache_load_factor)
    combination. Restored before returning so the storage estimator still sees
    the un-boosted ratio.
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

### Storage Estimation

`_calculate_dynamicemb_table_storage_specific_size` ([`dynamicemb_util.py:342-393`](../torcheasyrec/tzrec/utils/dynamicemb_util.py#L342-L393)) — per-shard HBM/DDR byte budget:

```
value_bytes_per_row = round_up16(dim * (1 + opt_mult) * element)
total_value_memory  = align(rows) * value_bytes_per_row
num_buckets         = align(rows) / bucket_capacity

HBM budget = cache_ratio * total_value_memory                 # values
           + align(rows) * (key<8B> + score<8B> + digest<1B>) # per-row
           + num_buckets * bucket_header<4B>                  # per-bucket

DDR budget = HYBRID  (caching=False): (1 - cache_ratio) * total_value_memory
             CACHING (caching=True):  total_value_memory      # full backing
```

`dynamicemb_calculate_shard_storages` ([`dynamicemb_util.py:637-775`](../torcheasyrec/tzrec/utils/dynamicemb_util.py#L637-L775)) is the drop-in replacement for the TorchRec `shard_estimators.calculate_shard_storages`. It computes the HBM / DDR breakdown above, then adds the pipeline I/O cost (`shard_estimators.calculate_pipeline_io_cost(...)`).

### Variant Emission

[`plan_util.py:887-916`](../torcheasyrec/tzrec/utils/plan_util.py#L887-L916) — `_emit_dynamicemb_variants`:

```python
def _emit_dynamicemb_variants(base_option: ShardingOption) -> List[ShardingOption]:
    """Expand a dynamicemb ShardingOption into HYBRID + CACHING variants.

    Sweeps both placement modes (caching=False and caching=True) and,
    when base_option.cache_params is unset, ten cache_load_factor values
    (0.1, 0.2, ..., 1.0). The downstream 2D DP proposer picks per table the
    best (mode, ratio) that fits both HBM and host topology budgets.
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

When user specifies `cache_load_factor` explicitly, the sweep collapses to a single point. Otherwise the proposer sees up to 20 variants per dynamicemb table (2 modes × 10 ratios) and picks the cheapest fitting one.

### Checkpoint Integration

[`checkpoint_util.py:705-722, 743-751`](../torcheasyrec/tzrec/utils/checkpoint_util.py#L705-L722):

```python
# restore_model(): after standard load_state_dict
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

# save_model(): after standard state_dict save
if has_dynamicemb:
    from dynamicemb.dump_load import DynamicEmbDump
    DynamicEmbDump(
        os.path.join(checkpoint_dir, "dynamicemb"),
        model,
        optim=optimizer is not None,
        counter=True,
    )
```

DynamicEmb checkpoints are stored separately at `checkpoint_dir/dynamicemb/<collection_path>/<table_name>.{keys,values,opt,counter,sizes}` and reloaded by the per-rank DFS walk in upstream `dump_load.find_sharded_modules()`.

### Range-Check Bypass

[`dynamicemb_util.py:777-793`](../torcheasyrec/tzrec/utils/dynamicemb_util.py#L777-L793):

```python
def _validate_feature_range_with_dynamicemb(kjt, configs):
    """Skip range check for dynamicemb features.

    DynamicEmb uses hash tables that accept arbitrary uint64 keys.
    max_capacity is a storage limit, not a valid key range.
    """
    filtered_configs = [c for c in configs if not getattr(c, "use_dynamicemb", False)]
    if not filtered_configs:
        return True
    return _orig_validate_feature_range(kjt, filtered_configs)

_jtv._validate_feature_range = _validate_feature_range_with_dynamicemb
```

TorchRec normally rejects KJT inputs with key IDs ≥ `num_embeddings` (assumed to be a key range). DynamicEmb has no such constraint — keys are hashed into buckets of unbounded size up to `max_capacity`.

## Migration Tools

### `tools/dynamicemb/zch_to_dynamicemb_convert.py`

Migrates a checkpoint from a ZCH-based (`ManagedCollisionEmbeddingBagCollection`) table to a DynamicEmb table. Reads the ZCH checkpoint, hashes the original IDs to a target `max_capacity` layout, and writes a `dynamicemb/` directory that `DynamicEmbLoad` can ingest.

Why this is needed: ZCH uses a fixed `max_id` collision module; DynamicEmb uses an open hash table. Different ID spaces → different checkpoints.

### `tools/dynamicemb/create_dynamicemb_init_ckpt.py`

Cold-starts a DynamicEmb checkpoint from a dense FBGEMM init (e.g., a pre-trained EBC) by writing the initial keys + values in the on-disk format `DynamicEmbLoad` expects. Useful for warming up a new deployment without retraining from scratch.

## When to Use DynamicEmb (vs FBGEMM)

| Criterion | FBGEMM (EBC/EC) | DynamicEmb |
|---|---|---|
| Cardinality | bounded (≤ 10M usually) | unbounded |
| Memory waste | high if sparse | only pays for what's hot |
| Cold-start cost | none | one-hit wonders admitted at threshold |
| Latency (hot path) | lowest (dense index) | higher (hash probe + possible spill) |
| Exportable to RTP / TensorRT | yes (JIT script) | limited (hash kernel is custom) |
| Checkpoint format | one big `state_dict` | per-rank binary + `state_dict` |
| Operating mode | ROW_WISE / TABLE_WISE / COLUMN_WISE / TP / DP | ROW_WISE only |

Recommended:
- **Frequent, hot IDs in a bounded space** → FBGEMM
- **Long-tail, cold-start, unbounded space** → DynamicEmb
- **Mixed** → per-feature choose, even within the same model

## Known Limitations

- **Row-wise sharding only** — upstream TODO at `dump_load.py:97`, `:205` (and a comment in planner `planner.py:333`: `TODO:0 is we don't have column-wise sharding now`)
- **All-or-nothing optimizer dump** — `DynamicEmbDump(optim=True)` dumps all opt state for the table (no per-param opt control)
- **No FBGEMM-style `bounds_check_mode`** — DynamicEmb uses its own `DynamicEmbCheckMode` enum (ERROR/WARNING/IGNORE) for insertion failures, not TorchRec's
- **No cross-table fusion in planner** — each table is enumerated independently, even if they could share storage (the table_fusion logic in `KeyedJaggedTensor` / `FeatureGroup` is not extended to DynamicEmb)
- **Limited export compatibility** — the `BatchedDynamicEmbeddingTablesV2` CUDA kernel is not directly JIT-scriptable. For RTP/TRT export, features with `dynamicemb { }` typically need to fall back to FBGEMM at export time, or be exported with custom RTP hooks (the latter is work-in-progress and not the default path)

## References

### NVIDIA Upstream (submodule)

| File | Purpose |
|---|---|
| [`external/recsys-examples/corelib/dynamiceb/dynamicemb/__init__.py`](../external/recsys-examples/corelib/dynamicemb/dynamicemb/__init__.py) | Public API |
| [`external/recsys-examples/corelib/dynamicemb/dynamicemb/dynamicemb_config.py`](../external/recsys-examples/corelib/dynamicemb/dynamicemb/dynamicemb_config.py) | `DynamicEmbTableOptions`, enums |
| [`external/recsys-examples/corelib/dynamicemb/dynamicemb/types.py`](../external/recsys-examples/corelib/dynamicemb/dynamicemb/types.py) | `Storage`, `Cache` ABCs |
| [`external/recsys-examples/corelib/dynamicemb/dynamicemb/planner/planner.py`](../external/recsys-examples/corelib/dynamicemb/dynamicemb/planner/planner.py) | `DynamicEmbeddingShardingPlanner` |
| [`external/recsys-examples/corelib/dynamicemb/dynamicemb/shard/embeddingbag.py`](../external/recsys-examples/corelib/dynamicemb/dynamicemb/shard/embeddingbag.py) | `DynamicEmbeddingBagCollectionSharder` |
| [`external/recsys-examples/corelib/dynamicemb/dynamicemb/shard/embedding.py`](../external/recsys-examples/corelib/dynamicemb/dynamicemb/shard/embedding.py) | `DynamicEmbeddingCollectionSharder` |
| [`external/recsys-examples/corelib/dynamicemb/dynamicemb/dump_load.py`](../external/recsys-examples/corelib/dynamicemb/dynamicemb/dump_load.py) | `DynamicEmbDump` / `DynamicEmbLoad` |
| [`external/recsys-examples/corelib/dynamicemb/dynamicemb/embedding_admission.py`](../external/recsys-examples/corelib/dynamicemb/dynamicemb/embedding_admission.py) | `FrequencyAdmissionStrategy`, `KVCounter` |
| [`external/recsys-examples/corelib/dynamicemb/dynamicemb/optimizer.py`](../external/recsys-examples/corelib/dynamicemb/dynamicemb/optimizer.py) | sparse optim state |

### TorchEasyRec Integration

| File | Purpose |
|---|---|
| [`torcheasyrec/tzrec/utils/dynamicemb_util.py`](../torcheasyrec/tzrec/utils/dynamicemb_util.py) | planner hooks, constraints, storage estimator (794 lines) |
| [`torcheasyrec/tzrec/utils/plan_util.py:887-916`](../torcheasyrec/tzrec/utils/plan_util.py#L887-L916) | `_emit_dynamicemb_variants` |
| [`torcheasyrec/tzrec/utils/checkpoint_util.py:705-751`](../torcheasyrec/tzrec/utils/checkpoint_util.py#L705-L751) | save/restore DynamicEmb checkpoints |
| [`torcheasyrec/tzrec/features/feature.py:631,657`](../torcheasyrec/tzrec/features/feature.py#L631) | `use_dynamicemb` flag on embedding configs |
| [`torcheasyrec/tzrec/protos/feature.proto`](../torcheasyrec/tzrec/protos/feature.proto) | `DynamicEmbedding` message |
| [`torcheasyrec/tzrec/tools/dynamicemb/zch_to_dynamicemb_convert.py`](../torcheasyrec/tzrec/tools/dynamicemb/zch_to_dynamicemb_convert.py) | ZCH → DynamicEmb checkpoint migration |
| [`torcheasyrec/tzrec/tools/dynamicemb/create_dynamicemb_init_ckpt.py`](../torcheasyrec/tzrec/tools/dynamicemb/create_dynamicemb_init_ckpt.py) | cold-start DynamicEmb from dense FBGEMM |
