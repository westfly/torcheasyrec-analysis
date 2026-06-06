---
layout: default
title: 09. Export & RTP Pipeline
nav_order: 10
parent: TorchEasyRec Source Walkthrough
---

# Export & RTP Pipeline

This document covers how TorchEasyRec converts a trained model into a deployable
serving graph. It traces the code from the CLI entry point down through the
four supported export backends, and then drills into the special RTP path
used for Alibaba's PAI / EAS serving stack.

The goal is not to recite every line. The goal is to give a reader enough
structural understanding to (1) know which file owns which decision, (2) know
which environment variable flips which switch, and (3) know exactly where the
RTP-specific graph rewrites happen.

---

## 1. Where Export Lives

| Concern | File | Role |
| --- | --- | --- |
| CLI | [`tzrec/export.py`](torcheasyrec/tzrec/export.py) | 1-line shim that calls `tzrec.main.main` with `["export", ...]` |
| Orchestrator | [`tzrec/main.py`](torcheasyrec/tzrec/main.py) `export()` at L881 | Builds features/model, picks checkpoint, dispatches to per-tower export for `MatchModel` / `TDM` |
| Public entry | [`tzrec/utils/export_util.py`](torcheasyrec/tzrec/utils/export_util.py) `export_model()` at L80 | Top-level dispatcher: `use_rtp ? export_rtp_model : export_model_normal` |
| Normal backend | [`tzrec/utils/export_util.py`](torcheasyrec/tzrec/utils/export_util.py) `export_model_normal()` at L148 | Loads checkpoint, quantizes, splits sparse/dense, calls TRT/AOTI backend |
| AOTI legacy | [`tzrec/acc/aot_utils.py`](torcheasyrec/tzrec/acc/aot_utils.py) `export_model_aot()` at L142 | Two-stage export: sparse JIT + dense AOTI |
| AOTI unified | [`tzrec/acc/aot_utils.py`](torcheasyrec/tzrec/acc/aot_utils.py) `export_unified_model_aot()` at L437 | Single `.pt2` containing sparse+dense fused |
| TensorRT | [`tzrec/acc/trt_utils.py`](torcheasyrec/tzrec/acc/trt_utils.py) `export_model_trt()` at L107 | Splits dense graph for `torch_tensorrt.dynamo.compile` |
| Acc flags | [`tzrec/acc/utils.py`](torcheasyrec/tzrec/acc/utils.py) | Pure env-var predicates (`is_trt`, `is_aot`, `is_unified_aot`, `is_quant`, ...) plus `export_acc_config()` which writes `model_acc.json` |
| Env predicates | [`tzrec/utils/env_util.py`](torcheasyrec/tzrec/utils/env_util.py) | `use_rtp()`, `use_hash_node_id()`, `enable_tma()` — read at module level |
| Proto | [`tzrec/protos/export.proto`](torcheasyrec/tzrec/protos/export.proto) | `ExportConfig` with `exporter_type`, `mixed_precision`, TF32 flags |

---

## 2. CLI Surface

```bash
# Standard JIT-only export
python -m tzrec.export --pipeline_config_path=P.config --export_dir=out --checkpoint_path=ckpt

# TensorRT dense + JIT sparse
ENABLE_TRT=1 python -m tzrec.export ...

# AOTI legacy (sparse JIT + dense AOTI)
ENABLE_AOT=1 python -m tzrec.export ...

# AOTI unified (single .pt2)
ENABLE_AOT=2 python -m tzrec.export ...

# RTP export (PAladdin / EAS serving)
USE_RTP=1 USE_FARM_HASH_TO_BUCKETIZE=true python -m tzrec.export ...
```

`tzrec/export.py` is intentionally trivial — it just calls into `main.main`
with the `export` subcommand. All real logic is in `tzrec/main.py` and the
acc/export utility modules.

---

## 3. The Four Backends — Decision Tree

`export_model()` in [`export_util.py:80`](torcheasyrec/tzrec/utils/export_util.py#L80) makes the
first fork: `USE_RTP=1` → `export_rtp_model`, otherwise → `export_model_normal`.

Inside `export_model_normal()` (L148), the second fork is `is_cuda_export()`,
which is `is_trt() or is_aot()` ([`acc/utils.py:152`](torcheasyrec/tzrec/acc/utils.py#L152)):

```
USE_RTP=1                →  export_rtp_model
USE_RTP=0, ENABLE_TRT=1  →  split_model + export_model_trt
USE_RTP=0, ENABLE_AOT=2  →  export_unified_model_aot   (single .pt2)
USE_RTP=0, ENABLE_AOT=1  →  split_model + export_model_aot   (legacy 2-stage)
USE_RTP=0, no acc flags  →  torch.jit.script on full model
```

Mixed precision (`export_config.mixed_precision = "BF16" | "FP16"`) is
applied **only to the dense sub-graph** via `CudaAutocastWrapper` before
`torch.export`. The sparse sub-graph is left untouched because it consists
only of embedding lookups and AMP there adds no speedup but complicates
`torch.jit.script` compilation.

---

## 4. Normal Export Path (`export_model_normal`)

[`export_util.py:148`](torcheasyrec/tzrec/utils/export_util.py#L148) — the workhorse for non-RTP serving.

### 4.1 Pre-export Setup

1. **Rank gating.** Only `RANK=0` actually exports; the rest are silent.
2. **Dataloader in PREDICT mode.** Builds a single batch sized to
   `min(data_config.batch_size, MAX_EXPORT_BATCH_SIZE)` to avoid OOM during
   inductor compile.
3. **Checkpoint restore.** `checkpoint_util.restore_model()` loads weights
   into the model on CPU.
4. **Quantization (optional).** If `QUANT_EMB` or `QUANT_EC_EMB` is set,
   `torchrec.inference.modules.quantize_embeddings` converts EBC/EC tables
   to fbgemm IntNBit representations. This **must happen on CPU** — the CUDA
   kernel uses int32 pointer arithmetic that overflows for large tables.
5. **Move to CUDA.** `is_cuda_export()` is true, so quantized fbgemm modules
   are migrated via `move_to_device_with_cache(load_factor=1.0)` (the
   default 0.0 places weights in UVM which is incompatible with the
   inference forward path), then the rest of the model goes to CUDA.

### 4.2 Backend Dispatch

The `is_trt / is_aot` switch lives at L248–L282:

```python
if acc_utils.is_trt() or acc_utils.is_aot():
    data = OrderedDict(sorted(data.items()))           # canonical key order
    with torch.no_grad(), torch.amp.autocast(...):     # AMP if requested
        result = model(data, "cuda:0")                 # sanity forward
    if acc_utils.is_trt():
        sparse, dense, meta_info = split_model(data, model, save_dir)
        export_model_trt(sparse, dense, data, save_dir, mixed_precision=...)
    elif acc_utils.is_unified_aot():
        export_unified_model_aot(model, data, save_dir, mixed_precision=...)
    else:
        sparse, dense, meta_info = split_model(data, model, save_dir)
        export_model_aot(sparse, dense, data, meta_info, save_dir, mixed_precision=...)
```

`split_model()` ([`export_util.py:1068`](torcheasyrec/tzrec/utils/export_util.py#L1068)) is the helper that
decomposes a single full model FX graph into a sparse sub-graph and a dense
sub-graph by matching the `fx_mark_*` sentinel nodes that the model code
inserts at the sparse→dense boundary (see §6).

### 4.3 JIT-only fallback

If no acc flag is set, the model is symbolic-traced then `torch.jit.script`ed
and saved as `scripted_model.pt` (L283–L293). This is the slowest path but the
simplest artifact.

### 4.4 Artifacts

| File | Owner | When |
| --- | --- | --- |
| `pipeline.config` | `config_util.save_message` | always |
| `fg.json` | `create_fg_json` (in-feature) | always |
| `model_acc.json` | `acc_utils.export_acc_config` | always |
| `scripted_model.pt` | `torch.jit.script` | JIT path |
| `scripted_sparse_model.pt` | sparse side of split | legacy AOTI / TRT |
| `aoti/aoti_model.pt2` | `torch._inductor.aoti_compile_and_package` | AOTI path (legacy dense or unified) |
| `gm*.code` | `GraphModule.code` dump | TRT / AOTI (debug) |
| `graph/gm_*.graph` | `GraphModule.graph` dump | split paths |
| `emb_ckpt_mapping.txt` | `write_mapping_file_for_input_tile` | `INPUT_TILE=3*` |

`model_acc.json` is the **key/values contract for the predict-time loader**.
Its `ENABLE_AOT`, `ENABLE_TRT`, `INPUT_TILE`, `QUANT_EMB`, etc. values are
read by the matching `*_predict()` predicates in [`acc/utils.py`](torcheasyrec/tzrec/acc/utils.py)
(L33, L92, L123, L142) to decide which loader to use at predict time. The
export side writes the same keys; the predict side re-reads them. This is the
only mechanism for cross-process acc configuration — the predict process
starts from a saved dir, not the original env.

---

## 5. TensorRT Backend

[`acc/trt_utils.py:107`](torcheasyrec/tzrec/acc/trt_utils.py#L107) `export_model_trt()`.

1. Run sparse model once to materialize the embedding output dict.
2. `symbolic_trace` + `torch.jit.script` the sparse sub-graph → save
   `scripted_sparse_model.pt` *and* `gm_sparse.code`.
3. For each key in the embedding output, build a dynamic-shape spec:
   - axis 0: `batch` dim up to `MAX_EXPORT_BATCH_SIZE` (default 512)
   - axis 1 (for 3-D outputs): `seq_len` dim up to `TRT_MAX_SEQ_LEN` (default 100)
4. Wrap the dense layer in `CudaAutocastWrapper` if mixed precision is set
   ([`trt_utils.py:167`](torcheasyrec/tzrec/acc/trt_utils.py#L167)).
5. `torch.export.export()` → `trt_convert()` ([`trt_utils.py:35`](torcheasyrec/tzrec/acc/trt_utils.py#L35)) → `torch_tensorrt.dynamo.compile()`
   with `enabled_precisions={float32}`, `disable_tf32=not torch.backends.cuda.matmul.allow_tf32`,
   `workspace_size=2 GiB`, `min_block_size=2`, `hardware_compatible=True`.
6. `torch.jit.trace` then `torch.jit.script` the TRT result.
7. Wrap both in `CombinedModelWrapper(sparse_scripted, dense_trt_scripted)`
   ([`models/model.py:453`](torcheasyrec/tzrec/models/model.py#L453)) and save as `scripted_model.pt`.

`DEBUG_TRT=1` adds full CPU+CUDA profiling traces around the dense and
combined forward paths.

---

## 6. AOTI Backend

There are **two** AOTI export entry points; the choice is the `ENABLE_AOT`
env var. Both go through `_aoti_compile_cfg()` ([`aot_utils.py:32`](torcheasyrec/tzrec/acc/aot_utils.py#L32)):

```python
{
    "scalar_asserts": False,                                  # AssertScalar codegen bug
    "unsafe_ignore_unsupported_triton_autotune_args": True,  # HSTU pre_hook
    "_use_fp64_for_unbacked_floats": False,                  # fp64 sigmoid on Ada/Ampere
}
```

And both call `_backport_pt178147_int_array_dedup()` ([`aot_utils.py:48`](torcheasyrec/tzrec/acc/aot_utils.py#L48)),
a patch for `pytorch/pytorch#178147` (keying `CppWrapperCpu.codegen_int_array_var`
on `id(writeline.__self__)` rather than `id(writeline)` to avoid intermittent
`'int_array_NN' was not declared in this scope` errors).

### 6.1 Legacy two-stage — `ENABLE_AOT=1` (a.k.a. `UNIFIED_AOT=0`)

[`aot_utils.py:142`](torcheasyrec/tzrec/acc/aot_utils.py#L142) `export_model_aot()`.

1. Sparse side: `symbolic_trace` + `torch.jit.script` → `scripted_sparse_model.pt`.
2. Build per-key dynamic shapes using `seq_tensor_names`,
   `jagged_seq_tensor_names`, `seq_share_groups` from `split_model`'s
   `meta_info`. Features in the same `FeatureGroupConfig` or `SeqGroupConfig`
   share a `torch.export.Dim` so they can grow together; non-sequence sparse
   features get their own Dim with `min=0`.
3. Wrap dense in `CudaAutocastWrapper` for AMP, then
   `torch.export.export(dense, (sparse_output,), dynamic_shapes=...)`.
4. `torch._inductor.aoti_compile_and_package()` writes `aoti/aoti_model.pt2`.

Loading at predict time uses `CombinedModelWrapper(sparse_scripted, dense_aoti)`.
The predict path calls [`aot_utils.py:101`](torcheasyrec/tzrec/acc/aot_utils.py#L101) `load_model_aot()`, which
branches on `is_unified_aot_predict()` reading `model_acc.json` and falls
back to a file-presence heuristic (`scripted_sparse_model.pt` exists →
legacy) for unit tests that bypass the full pipeline.

### 6.2 Unified — `ENABLE_AOT=2` (a.k.a. `UNIFIED_AOT=1`)

[`aot_utils.py:437`](torcheasyrec/tzrec/acc/aot_utils.py#L437) `export_unified_model_aot()`.

1. Set `model.set_is_inference(True)` and `model.eval()`.
2. Bind device and autocast into `CudaAutocastWrapper(model, mixed_precision, device="cuda:0")` —
   the wrapped forward takes only the data dict (no device arg).
3. `symbolic_trace` and dump `gm.code`.
4. `_pad_empty_sparse_values()` ([`aot_utils.py:237`](torcheasyrec/tzrec/acc/aot_utils.py#L237)) inflates any 0-element non-sequence sparse
   `.values` tensor to 2 elements (with corresponding length bump) so
   `torch.export` does not specialize on the size-0 sentinel.
5. `_build_dynamic_shapes()` ([`aot_utils.py:287`](torcheasyrec/tzrec/acc/aot_utils.py#L287)) uses **structural knowledge** of
   features and feature groups to assign dims:
   - Sequence features in a `SequenceFeature` config → share one Dim per
     `sequence_name`.
   - Sequence features in a `JAGGED_SEQUENCE` group → share one Dim per
     `group_name` (single-valued only; multi-valued keep their own Dim).
   - `SEQUENCE` group standalone sequence features → independent Dims
     (they don't share lengths).
   - Non-sequence sparse → own Dim, `min=0`.
   - Dense features, scalars, labels, sample weights → batch dim.
6. `torch.export.export(full_gm, args=(data,), dynamic_shapes=(dynamic_shapes,))`
   with `unsafe_ignore_unsupported_triton_autotune_args=True`.
7. AOTI compile and package → `aoti/aoti_model.pt2`.

The result is a single `.pt2` that the predict-time
`UnifiedAOTIModelWrapper` ([`models/model.py:515`](torcheasyrec/tzrec/models/model.py#L515)) loads with
`torch._inductor.aoti_load_package()`.

---

## 7. RTP Backend — The Alibaba Serving Specialization

RTP ("RealTime Prediction") is Alibaba's serving stack for PAI / EAS. It
differs from generic TorchScript / TensorRT serving in two important ways:

1. **Sparse parameters live as `.safetensors` files**, not inside the FX
   graph, and may be sharded across multiple ranks. The serving runtime
   loads them with its own custom kernels.
2. **The dense graph is exported with a proprietary tool** (`torch_fx_tool`)
   that produces an RTP-compatible model directory, not a standard
   `scripted_model.pt`.

This means the export pipeline has to (a) split sparse and dense, (b) save
sparse weights to a separate safetensors file, (c) rewrite the dense FX
graph so it can be re-traced with RTP-friendly ops, and (d) emit
RTP-style `fg.json` with `shared_name` / `gen_key_type` / `gen_val_type` keys
the server understands.

### 7.1 Detection

`use_rtp()` ([`env_util.py:24`](torcheasyrec/tzrec/utils/env_util.py#L24)) is the single source of truth:

```python
def use_rtp() -> bool:
    flag = os.environ.get("USE_RTP", "0") == "1"
    if flag and os.environ.get("USE_FARM_HASH_TO_BUCKETIZE", "false") != "true":
        logger.warning("you should set USE_FARM_HASH_TO_BUCKETIZE=true for "
                       "train/eval/export when use rtp for online inference.")
    return flag
```

`USE_FARM_HASH_TO_BUCKETIZE=true` must also be set during **training and
evaluation** for the bucketization function to match what RTP applies at
serving time. Mismatch leads to silent embedding mis-alignment.

The feature layer also queries `use_rtp()` at construction time
([`features/feature.py:426`](torcheasyrec/tzrec/features/feature.py#L426)) to switch its sequence-name
separator: RTP uses `seq_feat` (single underscore); PyFG / Aliyun FG uses
`seq__feat` (double underscore). This is why `_to_real_input_name` /
`_to_pyfg_input_name` / `_to_pyfg_feat_name` exist in
[`datasets/data_parser.py:954`](torcheasyrec/tzrec/datasets/data_parser.py#L954) — they bridge the two
conventions when `USE_RTP=1` so the in-process PyFG calls return values that
match what RTP will name them on the wire.

### 7.2 Top-level flow

[`export_util.py:697`](torcheasyrec/tzrec/utils/export_util.py#L697) `export_rtp_model()`:

```
USE_RTP=1 + checkpoint_path
  → load checkpoint, build DataLoader (PREDICT, batch_size=MAX_EXPORT_BATCH_SIZE)
  → build planner, sharded model (DistributedModelParallel)
  → trace sharded model with Tracer → full_graph
  → feature_to_embedding_info = _get_rtp_feature_to_embedding_info(model)
       (validates that no two modules share an embedding name; no feature is
        served by two different EBCs — RTP needs a 1:1:1 mapping)
  → split full_graph into sparse graph + dense graph
  → sparse_model = DMP(sparse_gm)
  → restore sparse checkpoint
  → run sparse → sparse_output, sparse_attrs
  → save sparse weights: model-NNNNNN-of-NNNNNN.safetensors + .json meta
  → init_parameters + checkpoint restore for dense
  → ExportTorchFxTool().export_fx_model(gm, sparse_output, mc_config)
       writes save_dir/fx_user_model/ (RTP dense package)
  → save adjusted fg.json
```

The sharded-model step is the part that differs most from `export_model_normal`.
RTP serving runs on potentially many nodes, and each node needs only the
shards it owns. So export is run under `DistributedModelParallel` so the
shard plan is identical between training and serving.

### 7.3 The FX graph rewrite

This is the heart of RTP integration. The model code inserts four
`@torch.fx.wrap`-tagged sentinel calls at the sparse→dense boundary:

| Sentinel | Meaning | Where it's used |
| --- | --- | --- |
| `fx_mark_keyed_tensor(name, KJT)` | A pooled `KeyedTensor` (EBC output) | sparse output dict |
| `fx_mark_tensor(name, t)` | A dense feature | sparse output dict |
| `fx_mark_seq_tensor(seq_name, t, max_seq_len, is_jagged_seq, keys)` | A sequence feature | sparse output dict |
| `fx_mark_seq_len(seq_name, t)` | A sequence-length tensor | sparse output dict |

These come from [`utils/fx_util.py`](torcheasyrec/tzrec/utils/fx_util.py). They are
opaque to PyTorch's tracer (because of `@torch.fx.wrap`) so the tracer
preserves them as `call_function` nodes with the original Python callable
as target — that's how the rewrite code finds them.

#### Sparse side (graph at L805–L869)

For each sentinel:
- `fx_mark_keyed_tensor(name, KJT)` → call `KJT.values()`, `KJT.length_per_key()`,
  `KJT.keys()` and stash each in the output dict under `name`,
  `name + "__length_per_key"`, `name + "__keys"`.
- `fx_mark_tensor(name, t)` → `t` flows through.
- `fx_mark_seq_tensor(seq_name, t, ...)` → if `is_jagged_seq`, prepend
  `torch.unsqueeze(t, 0)`; then call `_rtp_pad_to_max_seq_len` ([L639](torcheasyrec/tzrec/utils/export_util.py#L639))
  to right-pad the sequence to `max_seq_len`. Output key becomes
  `seq_name + "_sequence"`.
- `fx_mark_seq_len(seq_name, t)` → `torch.unsqueeze(t, 1)` (RTP can't accept
  rank-1 tensors). Output key becomes `seq_name + "_sequence_length"`.

The result is wrapped in a new `GraphModule`, dead-code-eliminated, and
pruned of unused params/buffers via `_prune_unused_param_and_buffer`
([L426](torcheasyrec/tzrec/utils/export_util.py#L426)).

#### Dense side (graph at L889–L1006)

For each sentinel, the dense graph replaces the original `call_function` node
with an input-from-dict lookup. Concretely:
- `fx_mark_keyed_tensor(name, KJT)` → `KeyedTensor(keys=sparse_attrs[name+"__keys"],
  length_per_key=sparse_attrs[name+"__length_per_key"],
  values=input_node[name])` — so at serve time the dense graph reads pooled
  embeddings from the input dict and rebuilds a `KeyedTensor` of the
  expected shape.
- `fx_mark_tensor(name, t)` → `input_node[name]`.
- `fx_mark_seq_len(seq_name, t)` → first, replace the `t` placeholder with
  `input_node[seq_name + "_sequence_length"]` squeezed on axis 1 (so
  rank-1 again inside the graph); then **also** add a
  `user:seq_name_sequence_length` raw_feature to the FG config so PyFG
  exposes it as a normal input. The user must remember to register this
  feature in their `qinfo` (the code logs a warning at L925).
- `fx_mark_seq_tensor(seq_name, t, max_seq_len, ...)` → `input_node[seq_name + "_sequence"]`
  followed by `_rtp_slice_with_seq_len` ([L646](torcheasyrec/tzrec/utils/export_util.py#L646)) which slices
  axis 1 to the actual runtime sequence length (capped at `max_seq_len`).
  If `is_jagged_seq`, `torch.squeeze(_, 0)` is applied at the end.
- Any `call_function` whose target is a `torch.ops.fbgemm.*` op is
  replaced with the corresponding `FBGEMM_RTP_TORCH_OP_MAPPING` entry
  ([L689](torcheasyrec/tzrec/utils/export_util.py#L689)), which provides pure-PyTorch
  fallbacks RTP understands. The four supported ops are
  `asynchronous_complete_cumsum`, `jagged_to_padded_dense`, `dense_to_jagged`,
  and `jagged_dense_elementwise_add_jagged_output`. Anything else raises
  `RuntimeError("... is not supported by rtp")`.

Jagged-sequence and fbgemm-op fallbacks **require `MAX_EXPORT_BATCH_SIZE=1`**
(assertions at L837 and L984). The trace-time example batch must look like
a single request so the symbolic shapes match the runtime assumption.

`mc_config` ([L892](torcheasyrec/tzrec/utils/export_util.py#L892)) is the dict mapping each output key to its
underlying feature names. It's passed to `ExportTorchFxTool` so the
generated RTP model knows which MC-managed-collision tables each output
reads from.

### 7.4 Saving sparse parameters

[`export_util.py:463`](torcheasyrec/tzrec/utils/export_util.py#L463) `_get_rtp_embedding_tensor()` walks the sharded
model's `state_dict()` and writes one safetensors file per rank:

```
save_dir/model-RRRRRR-of-WWWWWW.safetensors
save_dir/model-RRRRRR-of-WWWWWW.json      ← per-tensor metadata
```

For each tensor:
- If it's a `ShardedTensor` and this rank owns a shard: save the local shard
  under name `param_name/part_{idx}_{num_shards}`.
- If it's a plain tensor and last dim == embedding dim: save directly.
- DynamicEmb tables (sparse Adam-style) are loaded from
  `checkpoint_path/dynamicemb/*/*_emb_{keys,values}.rank_*.world_size_*` and
  paired up; the JSON entry gets `is_hashmap=True`, `hashmap_key`,
  `hashmap_value`, `hashmap_key_dtype` fields so the server knows to look
  up the values via the keys tensor.

**Constraint:** only `float32` is supported for sparse weights. The function
asserts this at L551.

### 7.5 The FG JSON adjustment

[`export_util.py:617`](torcheasyrec/tzrec/utils/export_util.py#L617) `_adjust_fg_json_for_rtp()` rewrites the
FG JSON in three ways:

1. **Bucket key swap.** For each feature:
   - `boundaries` → `gen_key_type = "boundary"`
   - `hash_bucket_size` → `gen_key_type = "hash"`
   - `num_buckets` → renamed to `hash_bucket_size`, `gen_key_type = "mod"`
   - else → `gen_key_type = "idle"`
   - Vocab-based features (`vocab_dict` / `vocab_list` / `vocab_file`) raise
     `ValueError` because RTP cannot load them.
2. **Embedding lookup hint.** If the feature has a corresponding entry in
   `feature_to_embedding_info`, add `shared_name`, `embedding_dimension`,
   `gen_val_type = "lookup"`. Otherwise `gen_val_type = "idle"`.
3. **Sequence bookkeeping.** Sequence groups get `sequence_table = "item"`
   (user/item split), `value_dim` → `value_dimension`, `need_discrete` →
   `needDiscrete`.

Additional raw features for the synthetic sequence lengths
(`<seq_name>_sequence_length`) are appended inside `export_rtp_model` at
L918 before this adjustment runs.

### 7.6 Why an FX wrap sentinel design?

The `fx_mark_*` nodes are visible to the tracer as ordinary
`call_function` nodes, but invisible to inductor / AOTI (they get
inlined). The `export_rtp_model` rewrite then walks the graph, finds those
nodes, and either (a) re-routes their `args[1]` (the actual tensor) to a
new `getitem(input_node, name)` lookup (dense side) or (b) extracts their
output as a fresh top-level graph output (sparse side). This is why the
boundary between sparse and dense in the trained model is explicit at the
Python level rather than implicit — the rewrites are graph-local, not
heuristic.

The same sentinel design is what `split_model()` in the normal export path
uses (L1068), so the boundary markers are reused across all four backends.

---

## 8. Inputs That Disagree With the Other Backends

RTP places a few non-obvious requirements on the rest of the framework:

1. **No name collisions** in EBCs/ECs ([L386](torcheasyrec/tzrec/utils/export_util.py#L386), [L394](torcheasyrec/tzrec/utils/export_util.py#L394)). Two features
   served by two different embedding modules fail at export time. Workaround
   is to create a second `Feature` with the same `feature_config` and use it
   in the second group.
2. **No vocab-based features.** `RTP_INVALID_BUCKET_KEYS` ([L573](torcheasyrec/tzrec/utils/export_util.py#L573)) lists `vocab_dict`,
   `vocab_list`, `vocab_file`. Use hash-bucketized variants.
3. **`USE_FARM_HASH_TO_BUCKETIZE=true`** must be set in train/eval/export
   for hashing parity. The warning fires at env-var-read time.
4. **`MAX_EXPORT_BATCH_SIZE=1`** if the model has jagged sequences or
   fbgemm ops that need the RTP fallbacks.
5. **Sparse weights are float32 only** ([L551](torcheasyrec/tzrec/utils/export_util.py#L551)).
   Quantized EBC tables must be dequantized before saving for RTP.
6. **Each user must register `<seq_name>_sequence_length` in their `qinfo`**
   because the rewritten dense graph treats it as a user-supplied input.
   The exporter logs each one at L925.

---

## 9. Predict-Time Loading

For all backends, the predict-time loader (`tzrec.main.predict` →
`create_predict_fn`) re-reads `model_acc.json` and dispatches:

- `is_trt_predict(model_path)` ([`acc/utils.py:142`](torcheasyrec/tzrec/acc/utils.py#L142)) → `torch.jit.load(scripted_model.pt)`, call as before.
- `is_unified_aot_predict(model_path)` ([`acc/utils.py:92`](torcheasyrec/tzrec/acc/utils.py#L92)) → `torch._inductor.aoti_load_package(aoti/aoti_model.pt2)` wrapped in `UnifiedAOTIModelWrapper`.
- Otherwise AOTI legacy → `CombinedModelWrapper(sparse_scripted, dense_aoti)`.
- RTP path uses a different runtime (not in this repo) that loads the
  `fx_user_model/` directory plus the `model-*-of-*.safetensors` files.

This is why `model_acc.json` exists: the predict process is a fresh
interpreter that knows nothing about the env vars the export process was
started with. The JSON is the only channel for that information.

---

## 10. End-to-End Diagram

```
                                          pipeline.config
                                                |
                                                v
                                     main.export()   (L881)
                                       |          |
                       MatchModel?     |          |   otherwise
                                       v          v
              per-tower export_model   .   export_model()
                                                 |
                          +----------------------+----------------------+
                          |                                             |
                  USE_RTP=1                                       USE_RTP=0
                          |                                             |
                          v                                             v
                export_rtp_model  (L697)                    export_model_normal  (L148)
                ├ split sparse/dense FX                          ├ quantize EBC/EC (CPU)
                ├ DMP, restore, run sparse                       ├ move to CUDA
                ├ save safetensors per rank                      ├ is_trt() / is_aot() / neither
                ├ rewrite dense graph                            ├─ TRT: split_model + export_model_trt
                │   (sentinel replacement, fbgemm fallbacks)      ├─ AOTI legacy: split_model + export_model_aot
                ├ ExportTorchFxTool → fx_user_model/             └─ AOTI unified: export_unified_model_aot
                └ adjusted fg.json (RTP-style)                   or jit.script → scripted_model.pt

                Output dir:                       Output dir:
                  fg.json (RTP)                     pipeline.config
                  model-*.safetensors               fg.json (PyFG)
                  model-*.json (meta)               model_acc.json
                  fx_user_model/                    scripted_model.pt  (JIT)
                  graph/gm_*.graph                  aoti/aoti_model.pt2  (AOTI)
                                                   scripted_sparse_model.pt  (legacy AOTI/TRT)
                                                   graph/gm_*.graph  (debug)
                                                   gm*.code  (debug)
```

---

## 11. Key Environment Variables (Export Side)

| Var | Read at | Effect |
| --- | --- | --- |
| `USE_RTP` | `env_util.use_rtp()` | Switch backend to `export_rtp_model`. **Also requires** `USE_FARM_HASH_TO_BUCKETIZE=true` (warns otherwise). |
| `USE_FARM_HASH_TO_BUCKETIZE` | `env_util.use_rtp()` | Bucketization parity with RTP. Should be set in train, eval, **and** export. |
| `ENABLE_AOT` | `acc.utils.is_aot()` / `is_unified_aot()` | `1` = legacy two-stage, `2` = unified single `.pt2`. `UNIFIED_AOT=1` is a legacy alias for `2`. |
| `ENABLE_TRT` | `acc.utils.is_trt()` | Compile dense sub-graph with `torch_tensorrt.dynamo.compile`. |
| `QUANT_EMB` | `acc.utils.is_quant()` / `quant_dtype()` | EBC quantization dtype (`FP32`/`FP16`/`INT8`/`INT4`/`INT2`). Default `INT8`; `QUANT_EMB=0` disables. |
| `QUANT_EC_EMB` | `acc.utils.is_ec_quant()` | EC quantization dtype. `0` disables. |
| `INPUT_TILE` | `acc.utils.is_input_tile*()` | `2*` = user/item split, `3*` = user/item + split embedding tables. Writes `emb_ckpt_mapping.txt` for `3*`. |
| `INPUT_TILE_3_ONLINE` | `acc.utils.is_input_tile_3_online()` | If `1`, sequential tensor uses `jt.values()` directly; offline predict unsupported. |
| `MAX_EXPORT_BATCH_SIZE` | `acc.utils.get_max_export_batch_size()` | Caps the trace batch size to avoid OOM during inductor compile. |
| `TRT_MAX_BATCH_SIZE` | same | Legacy alias for `MAX_EXPORT_BATCH_SIZE`. |
| `TRT_MAX_SEQ_LEN` | `trt_utils.get_trt_max_seq_len()` | Dynamic-shape bound for sequence length in TRT. |
| `AOTI_AUTOTUNE_WITH_SAMPLE_INPUTS` | `acc.utils.is_autotune_with_sample_inputs()` | `1` enables inductor's `triton.autotune_with_sample_inputs`. |
| `ENABLE_TMA` | `env_util.enable_tma()` | TMA in Triton kernels (needs SM 9.0+ and Triton ≥ 3.5). |
| `USE_HASH_NODE_ID` | `env_util.use_hash_node_id()` | Hash-based node IDs (distributed setups). |
| `DEBUG_TRT` | `acc.utils.is_debug_trt()` | Enable TRT graph logging + profiler dumps. |
| `FORCE_LOAD_SHARDING_PLAN` | `env_util.force_load_sharding_plan()` | Reuse the sharding plan in the checkpoint. |
| `LOCAL_CACHE_DIR` | `export_model` | Local scratch dir when `save_dir` is a remote URL (fsspec). |

The predict-time loader reads the same variables from `model_acc.json` to
pick the matching deserializer — see `*_predict()` predicates in
[`acc/utils.py`](torcheasyrec/tzrec/acc/utils.py).

---

## 12. `INPUT_TILE` Inference Modes

Beyond the four backends, a second axis controls how the dense graph is
**deployed** (separate from how it's **compiled**): the `INPUT_TILE` env
var. This applies to AOTI / TRT backends, not RTP.

```
┌─────────────────────────────────────────────────────────────────────────┐
│                        INPUT_TILE 推理模式                                │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  ┌─────────────────────────────────────────────────────────────────┐  │
│  │ INPUT_TILE=1 (Default): 无分片                                   │  │
│  │                                                                  │  │
│  │   User Features ──┬──▶ Model ──▶ Output                         │  │
│  │   Item Features ──┘                                              │  │
│  └─────────────────────────────────────────────────────────────────┘  │
│                                                                         │
│  ┌─────────────────────────────────────────────────────────────────┐  │
│  │ INPUT_TILE=2: User/Item 分别推理                                 │  │
│  │                                                                  │  │
│  │   ┌─────────────┐                                                │  │
│  │   │ User Input │                                                │  │
│  │   └──────┬─────┘                                                │  │
│  │          ▼                                                       │  │
│  │   ┌─────────────┐     ┌─────────────┐                           │  │
│  │   │ User Tower  │────▶│ Item Tower  │                           │  │
│  │   │ (稀疏+MLP) │     │ (稀疏+MLP) │                           │  │
│  │   └──────┬─────┘     └──────┬─────┘                           │  │
│  │          │                   │                                  │  │
│  │          ▼                   ▼                                  │  │
│  │   User Embed ──────▶ Similarity Calculation                     │  │
│  │                                                                  │  │
│  │   用于: Match 模型的两阶段推理                                    │  │
│  └─────────────────────────────────────────────────────────────────┘  │
│                                                                         │
│  ┌─────────────────────────────────────────────────────────────────┐  │
│  │ INPUT_TILE=3: User/Item + Embedding 分片                         │  │
│  │                                                                  │  │
│  │   User Features ──▶ User Tower ──▶ User Embed                    │  │
│  │        │                                                          │  │
│  │        │ (embedding 分片到不同 GPU)                               │  │
│  │        ▼                                                          │  │
│  │   Item Features ──▶ Item Tower ──▶ Item Embed                   │  │
│  │                                                                  │  │
│  │   用于: 超大规模 Embedding 的分布式推理                           │  │
│  └─────────────────────────────────────────────────────────────────┘  │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

| `INPUT_TILE` | Use Case | Artifacts |
|---|---|---|
| `1` (default) | Single-call inference; user + item in one batch | standard `scripted_model.pt` |
| `2*` | Two-stage matching (user offline, item online) | separate user/item `.pt` files |
| `3*` | Distributed embedding sharding for huge tables | `emb_ckpt_mapping.txt` for runtime embedding loading |

`INPUT_TILE=3*` writes `emb_ckpt_mapping.txt` (via
`write_mapping_file_for_input_tile` in `export_util.py`) that maps logical
embedding names to physical sharded checkpoint files — the predict loader
uses this to look up weights by their actual shard location.

### AOT/TRT vs RTP — Key Differences

| Feature | AOT/TRT | RTP |
|---|---|---|
| **Distributed support** | Not supported (forces `WORLD_SIZE=1`) | Supported (`WORLD_SIZE=N`) |
| **Embedding sharding** | Not supported | Supported (via `ShardedModule`) |
| **Weight export format** | Inlined in JIT model | Sharded safetensors per rank |
| **Leaf modules** | Flattened | Kept as `ShardedModule` |
| **Sequence handling** | Standard | Replaced with RTP-compatible impls |
| **Graph format** | JIT Script | `ExportTorchFxTool` |
| **Runtime** | Single-GPU CUDA | Multi-GPU / multi-node distributed |

### `split_model` Logic (AOT/TRT, [`export_util.py:1068`](torcheasyrec/tzrec/utils/export_util.py#L1068))

```
┌─────────────────────────────────────────────────────────────────────────┐
│                 split_model() - AOT/TRT 共用                            │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  【第一步】FX Tracing                                                  │
│  ┌────────────────────────────────────────────────────────────────┐  │
│  │  tracer.trace(model) → full_graph                              │  │
│  │  捕获完整计算图，保留所有 fx_mark_* 调用                        │  │
│  └────────────────────────────────────────────────────────────────┘  │
│                              │                                          │
│                              ▼                                          │
│  【第二步】Sparse Part 提取                                            │
│  ┌────────────────────────────────────────────────────────────────┐  │
│  │  for node in graph:                                            │  │
│  │      if fx_mark_keyed_tensor:                                  │  │
│  │          → outputs[name] = node.args[1].values()               │  │
│  │          → output_attrs[name + "__keys"] = keys()               │  │
│  │          → output_attrs[name + "__length_per_key"] = ...        │  │
│  │      elif fx_mark_tensor (query):                              │  │
│  │          → outputs[name] = node.args[1]                        │  │
│  │      elif fx_mark_seq_tensor:                                  │  │
│  │          → outputs[name + "__sequence"] = node.args[1]         │  │
│  │      elif fx_mark_seq_len:                                     │  │
│  │          → outputs[name + "__sequence_length"] = ...            │  │
│  │                                                                 │  │
│  │  sparse_output = {**outputs, **output_attrs}                   │  │
│  └────────────────────────────────────────────────────────────────┘  │
│                              │                                          │
│                              ▼                                          │
│  【第三步】Dense Part 提取                                             │
│  ┌────────────────────────────────────────────────────────────────┐  │
│  │  for node in graph:                                            │  │
│  │      if fx_mark_keyed_tensor:                                  │  │
│  │          # 用真实 keys/length 替换                              │  │
│  │          new_node = KeyedTensor(                                │  │
│  │              keys=sparse_attrs["__keys"],                       │  │
│  │              length_per_key=sparse_attrs["__length"],           │  │
│  │              values=getitem(input, name)                        │  │
│  │          )                                                      │  │
│  │      elif fx_mark_tensor/fx_mark_seq_tensor:                   │  │
│  │          # 直接从 input 获取                                    │  │
│  │          node_t.replace_all_uses_with(                         │  │
│  │              getitem(input_node, name)                          │  │
│  │          )                                                      │  │
│  └────────────────────────────────────────────────────────────────┘  │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

### AOT/TRT vs RTP Data Flow

```
【AOT/TRT】
┌─────────────┐     ┌─────────────┐     ┌─────────────┐
│ Input Data │────▶│ Sparse Part │────▶│ Dense Part  │
│ (原始ID)    │     │ (Embedding) │     │ (MLP+交互)  │
└─────────────┘     └─────────────┘     └─────────────┘
                           │                    │
                           ▼                    │
                    KeyedTensor ─────────────────┤
                    (values + keys + length)    │
                                                  │
                                                  ▼
                                          ┌─────────────┐
                                          │   Output    │
                                          └─────────────┘

【RTP】
┌─────────────────────────────────────────────────────────────────┐
│                    分布式训练/导出                                 │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  ┌──────────────┐                                               │
│  │ Embedding    │──▶ safetensors (按 rank 分片)               │
│  │ (分片)       │                                               │
│  └──────────────┘                                               │
│                                                                  │
│  ┌──────────────────────────────────────────────────────────┐   │
│  │ Dense Graph (FX) │──▶ ExportTorchFxTool ──▶ fx_user_model/ │   │
│  └──────────────────────────────────────────────────────────┘   │
│                                                                  │
│  运行时: 加载 safetensors + 解析 fx_user_model/                 │
└─────────────────────────────────────────────────────────────────┘
```
