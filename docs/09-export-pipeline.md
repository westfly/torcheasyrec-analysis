---
title: 导出与 RTP 管线
nav_order: 9
has_children: true
---

# 导出与 RTP 管线

本文档介绍 TorchEasyRec 如何将训练好的模型转换为可部署的 serving 图。它从 CLI 入口向下追踪代码，覆盖四种支持的导出后端，并深入到用于阿里巴巴 PAI / EAS serving 栈的特殊 RTP 路径。

目标不是逐行复述。目标是给读者足够的结构性理解，使其能够 (1) 知道哪个文件拥有哪个决策，(2) 知道哪个环境变量翻转哪个开关，(3) 准确知道 RTP 特定的图重写发生在哪里。

---

## 1. 导出代码所在

| 关注点 | 文件 | 角色 |
| --- | --- | --- |
| CLI | [`tzrec/export.py`](torcheasyrec/tzrec/export.py) | 一行 shim，调用 `tzrec.main.main` 并传入 `["export", ...]` |
| 编排器 | [`tzrec/main.py`](torcheasyrec/tzrec/main.py) 中 L881 的 `export()` | 构建 features/model，选择 checkpoint，为 `MatchModel` / `TDM` 派发到 per-tower export |
| 公共入口 | [`tzrec/utils/export_util.py`](torcheasyrec/tzrec/utils/export_util.py) 中 L80 的 `export_model()` | 顶层分发器：`use_rtp ? export_rtp_model : export_model_normal` |
| Normal 后端 | [`tzrec/utils/export_util.py`](torcheasyrec/tzrec/utils/export_util.py) 中 L148 的 `export_model_normal()` | 加载 checkpoint、量化、拆分稀疏/稠密、调用 TRT/AOTI 后端 |
| AOTI 旧版 | [`tzrec/acc/aot_utils.py`](torcheasyrec/tzrec/acc/aot_utils.py) 中 L142 的 `export_model_aot()` | 两阶段导出：稀疏 JIT + 稠密 AOTI |
| AOTI 统一 | [`tzrec/acc/aot_utils.py`](torcheasyrec/tzrec/acc/aot_utils.py) 中 L437 的 `export_unified_model_aot()` | 包含稀疏+稠密融合的单一 `.pt2` |
| TensorRT | [`tzrec/acc/trt_utils.py`](torcheasyrec/tzrec/acc/trt_utils.py) 中 L107 的 `export_model_trt()` | 为 `torch_tensorrt.dynamo.compile` 拆分稠密图 |
| Acc 标志 | [`tzrec/acc/utils.py`](torcheasyrec/tzrec/acc/utils.py) | 纯环境变量谓词（`is_trt`、`is_aot`、`is_unified_aot`、`is_quant` 等）以及写入 `model_acc.json` 的 `export_acc_config()` |
| 环境谓词 | [`tzrec/utils/env_util.py`](torcheasyrec/tzrec/utils/env_util.py) | `use_rtp()`、`use_hash_node_id()`、`enable_tma()` — 在模块级读取 |
| Proto | [`tzrec/protos/export.proto`](torcheasyrec/tzrec/protos/export.proto) | 带 `exporter_type`、`mixed_precision`、TF32 标志的 `ExportConfig` |

---

## 2. CLI 表面

```bash
# 标准仅 JIT 导出
python -m tzrec.export --pipeline_config_path=P.config --export_dir=out --checkpoint_path=ckpt

# TensorRT 稠密 + JIT 稀疏
ENABLE_TRT=1 python -m tzrec.export ...

# AOTI 旧版 (sparse JIT + dense AOTI)
ENABLE_AOT=1 python -m tzrec.export ...

# AOTI 统一 (单一 .pt2)
ENABLE_AOT=2 python -m tzrec.export ...

# RTP 导出 (PAladdin / EAS serving)
USE_RTP=1 USE_FARM_HASH_TO_BUCKETIZE=true python -m tzrec.export ...
```

`tzrec/export.py` 故意保持简单——它只是用 `export` 子命令调用 `main.main`。所有真实逻辑在 `tzrec/main.py` 与 acc/export 工具模块中。

---

## 3. 四个后端 — 决策树

[`export_util.py:80`](torcheasyrec/tzrec/utils/export_util.py#L80) 中的 `export_model()` 进行第一层分叉：`USE_RTP=1` → `export_rtp_model`，否则 → `export_model_normal`。

`export_model_normal()`（L148）内部，第二层分叉是 `is_cuda_export()`，它等于 `is_trt() or is_aot()`（[`acc/utils.py:152`](torcheasyrec/tzrec/acc/utils.py#L152)）：

```
USE_RTP=1                →  export_rtp_model
USE_RTP=0, ENABLE_TRT=1  →  split_model + export_model_trt
USE_RTP=0, ENABLE_AOT=2  →  export_unified_model_aot   (单一 .pt2)
USE_RTP=0, ENABLE_AOT=1  →  split_model + export_model_aot   (旧版 2 阶段)
USE_RTP=0, no acc flags  →  torch.jit.script 作用于完整模型
```

混合精度（`export_config.mixed_precision = "BF16" | "FP16"`）**仅作用于稠密子图**，通过 `CudaAutocastWrapper` 在 `torch.export` 之前应用。稀疏子图保持不变，因为它只由嵌入查找组成，AMP 在那里不会带来加速但会使 `torch.jit.script` 编译复杂化。

---

## 4. Normal 导出路径（`export_model_normal`）

[`export_util.py:148`](torcheasyrec/tzrec/utils/export_util.py#L148) — 非 RTP serving 的主力。

### 4.1 导出前准备

1. **Rank 门控。** 仅 `RANK=0` 实际执行导出；其他静默。
2. **PREDICT 模式 Dataloader。** 构建一个大小为 `min(data_config.batch_size, MAX_EXPORT_BATCH_SIZE)` 的单 batch，以避免 inductor compile 期间的 OOM。
3. **Checkpoint 恢复。** `checkpoint_util.restore_model()` 将权重加载到 CPU 上的模型中。
4. **量化（可选）。** 如果设置了 `QUANT_EMB` 或 `QUANT_EC_EMB`，`torchrec.inference.modules.quantize_embeddings` 将 EBC/EC 表转换为 fbgemm IntNBit 表示。**必须在 CPU 上完成** —— CUDA kernel 使用 int32 指针算术，对大表会溢出。
5. **迁移到 CUDA。** `is_cuda_export()` 为真，因此量化后的 fbgemm 模块通过 `move_to_device_with_cache(load_factor=1.0)` 迁移（默认 0.0 会将权重放在 UVM 中，与推理前向路径不兼容），然后模型其余部分进入 CUDA。

### 4.2 后端分派

`is_trt / is_aot` 开关位于 L248–L282：

```python
if acc_utils.is_trt() or acc_utils.is_aot():
    data = OrderedDict(sorted(data.items()))           # 规范化 key 顺序
    with torch.no_grad(), torch.amp.autocast(...):     # 若请求则 AMP
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

`split_model()`（[`export_util.py:1068`](torcheasyrec/tzrec/utils/export_util.py#L1068)）是一个辅助函数，通过匹配模型代码在稀疏→稠密边界插入的 `fx_mark_*` 哨兵节点（见 §6）将单一完整模型 FX 图分解为稀疏子图与稠密子图。

### 4.3 仅 JIT 回退

如果未设置 acc 标志，则模型经过 symbolic-trace 然后 `torch.jit.script` 并保存为 `scripted_model.pt`（L283–L293）。这是最慢的路径但产物最简单。

### 4.4 产物

| 文件 | 所有者 | 时机 |
| --- | --- | --- |
| `pipeline.config` | `config_util.save_message` | 总是 |
| `fg.json` | `create_fg_json` (in-feature) | 总是 |
| `model_acc.json` | `acc_utils.export_acc_config` | 总是 |
| `scripted_model.pt` | `torch.jit.script` | JIT 路径 |
| `scripted_sparse_model.pt` | 拆分的稀疏侧 | 旧版 AOTI / TRT |
| `aoti/aoti_model.pt2` | `torch._inductor.aoti_compile_and_package` | AOTI 路径（旧版稠密或统一） |
| `gm*.code` | `GraphModule.code` dump | TRT / AOTI（debug） |
| `graph/gm_*.graph` | `GraphModule.graph` dump | split 路径 |
| `emb_ckpt_mapping.txt` | `write_mapping_file_for_input_tile` | `INPUT_TILE=3*` |

`model_acc.json` 是**用于 predict-time loader 的 key/values 契约**。它的 `ENABLE_AOT`、`ENABLE_TRT`、`INPUT_TILE`、`QUANT_EMB` 等值由 [`acc/utils.py`](torcheasyrec/tzrec/acc/utils.py) 中对应的 `*_predict()` 谓词（L33、L92、L123、L142）读取，以决定 predict 时使用哪个 loader。导出端写入相同的 key；predict 端重新读取它们。这是跨进程 acc 配置的唯一机制 —— predict 进程从保存目录而非原始环境启动。

---

## 5. TensorRT 后端

[`acc/trt_utils.py:107`](torcheasyrec/tzrec/acc/trt_utils.py#L107) `export_model_trt()`。

1. 跑一次 sparse 模型以物化嵌入输出字典。
2. `symbolic_trace` + `torch.jit.script` 稀疏子图 → 保存 `scripted_sparse_model.pt` **以及** `gm_sparse.code`。
3. 对嵌入输出中的每个 key，构建动态 shape spec：
   - axis 0：`batch` 维，最大 `MAX_EXPORT_BATCH_SIZE`（默认 512）
   - axis 1（对 3-D 输出）：`seq_len` 维，最大 `TRT_MAX_SEQ_LEN`（默认 100）
4. 若设置了混合精度，将稠密层包装在 `CudaAutocastWrapper` 中（[`trt_utils.py:167`](torcheasyrec/tzrec/acc/trt_utils.py#L167)）。
5. `torch.export.export()` → `trt_convert()`（[`trt_utils.py:35`](torcheasyrec/tzrec/acc/trt_utils.py#L35)）→ `torch_tensorrt.dynamo.compile()`，参数为 `enabled_precisions={float32}`、`disable_tf32=not torch.backends.cuda.matmul.allow_tf32`、`workspace_size=2 GiB`、`min_block_size=2`、`hardware_compatible=True`。
6. `torch.jit.trace` 然后 `torch.jit.script` TRT 结果。
7. 将两者包装到 `CombinedModelWrapper(sparse_scripted, dense_trt_scripted)`（[`models/model.py:453`](torcheasyrec/tzrec/models/model.py#L453)）并保存为 `scripted_model.pt`。

`DEBUG_TRT=1` 在稠密和 combined 前向路径周围添加完整的 CPU+CUDA profiling trace。

---

## 6. AOTI 后端

有**两个** AOTI 导出入口；选择由 `ENABLE_AOT` 环境变量决定。两者都经过 `_aoti_compile_cfg()`（[`aot_utils.py:32`](torcheasyrec/tzrec/acc/aot_utils.py#L32)）：

```python
{
    "scalar_asserts": False,                                  # AssertScalar codegen bug
    "unsafe_ignore_unsupported_triton_autotune_args": True,  # HSTU pre_hook
    "_use_fp64_for_unbacked_floats": False,                  # fp64 sigmoid on Ada/Ampere
}
```

两者都调用 `_backport_pt178147_int_array_dedup()`（[`aot_utils.py:48`](torcheasyrec/tzrec/acc/aot_utils.py#L48)），这是 `pytorch/pytorch#178147` 的一个补丁（对 `CppWrapperCpu.codegen_int_array_var` 使用 `id(writeline.__self__)` 而非 `id(writeline)` 进行 key，以避免间歇性的 `'int_array_NN' was not declared in this scope` 错误）。

### 6.1 旧版两阶段 — `ENABLE_AOT=1`（即 `UNIFIED_AOT=0`）

[`aot_utils.py:142`](torcheasyrec/tzrec/acc/aot_utils.py#L142) `export_model_aot()`。

1. 稀疏侧：`symbolic_trace` + `torch.jit.script` → `scripted_sparse_model.pt`。
2. 使用 `split_model` 的 `meta_info` 中的 `seq_tensor_names`、`jagged_seq_tensor_names`、`seq_share_groups` 构建 per-key 动态 shape。同一 `FeatureGroupConfig` 或 `SeqGroupConfig` 中的特征共享一个 `torch.export.Dim`，以便它们可以一起增长；非序列稀疏特征获得自己的 Dim， `min=0`。
3. 用 `CudaAutocastWrapper` 包装稠密以应用 AMP，然后用 `torch.export.export(dense, (sparse_output,), dynamic_shapes=...)`。
4. `torch._inductor.aoti_compile_and_package()` 写入 `aoti/aoti_model.pt2`。

predict 时加载使用 `CombinedModelWrapper(sparse_scripted, dense_aoti)`。predict 路径调用 [`aot_utils.py:101`](torcheasyrec/tzrec/acc/aot_utils.py#L101) `load_model_aot()`，它根据读取 `model_acc.json` 的 `is_unified_aot_predict()` 进行分支，并对绕过完整管线的单元测试回退到文件存在性启发式（`scripted_sparse_model.pt` 存在 → 旧版）。

### 6.2 统一 — `ENABLE_AOT=2`（即 `UNIFIED_AOT=1`）

[`aot_utils.py:437`](torcheasyrec/tzrec/acc/aot_utils.py#L437) `export_unified_model_aot()`。

1. 设置 `model.set_is_inference(True)` 并 `model.eval()`。
2. 将 device 和 autocast 绑定到 `CudaAutocastWrapper(model, mixed_precision, device="cuda:0")`——包装后的前向只接受数据字典（不需要 device 参数）。
3. `symbolic_trace` 并 dump `gm.code`。
4. `_pad_empty_sparse_values()`（[`aot_utils.py:237`](torcheasyrec/tzrec/acc/aot_utils.py#L237)）将任何 0 元素的非序列稀疏 `.values` 张量膨胀为 2 元素（对应的 length 也增加），这样 `torch.export` 不会特化在 size-0 哨兵上。
5. `_build_dynamic_shapes()`（[`aot_utils.py:287`](torcheasyrec/tzrec/acc/aot_utils.py#L287)）使用特征和特征组的**结构知识**分配 dim：
   - `SequenceFeature` 配置中的序列特征 → 每个 `sequence_name` 共享一个 Dim。
   - `JAGGED_SEQUENCE` 组中的序列特征 → 每个 `group_name` 共享一个 Dim（仅单值；多值保留自己的 Dim）。
   - `SEQUENCE` 组中的独立序列特征 → 独立的 Dim（它们不共享长度）。
   - 非序列稀疏 → 自己的 Dim，`min=0`。
   - 稠密特征、标量、labels、sample weights → batch dim。
6. `torch.export.export(full_gm, args=(data,), dynamic_shapes=(dynamic_shapes,))` 带 `unsafe_ignore_unsupported_triton_autotune_args=True`。
7. AOTI 编译并打包 → `aoti/aoti_model.pt2`。

结果是单个 `.pt2`，predict 时 `UnifiedAOTIModelWrapper`（[`models/model.py:515`](torcheasyrec/tzrec/models/model.py#L515)）用 `torch._inductor.aoti_load_package()` 加载。

---

## 7. RTP 后端 — 阿里巴巴 Serving 专用化

RTP（"RealTime Prediction"）是阿里巴巴的 PAI / EAS serving 栈。它在两个重要方面与通用 TorchScript / TensorRT serving 不同：

1. **稀疏参数以 `.safetensors` 文件存在**，不在 FX 图内，可跨多个 rank 分片。Serving 运行时用其自定义 kernel 加载它们。
2. **稠密图通过专有工具**（`torch_fx_tool`）导出，产生 RTP 兼容的模型目录，而非标准的 `scripted_model.pt`。

这意味着导出管线必须 (a) 拆分稀疏与稠密，(b) 将稀疏权重保存到单独的 safetensors 文件，(c) 重写稠密 FX 图使其能用 RTP 友好的 op 重新追踪，(d) 发出 RTP 风格的 `fg.json`，带服务器理解的 `shared_name` / `gen_key_type` / `gen_val_type` key。

### 7.1 检测

`use_rtp()`（[`env_util.py:24`](torcheasyrec/tzrec/utils/env_util.py#L24)）是唯一权威来源：

```python
def use_rtp() -> bool:
    flag = os.environ.get("USE_RTP", "0") == "1"
    if flag and os.environ.get("USE_FARM_HASH_TO_BUCKETIZE", "false") != "true":
        logger.warning("you should set USE_FARM_HASH_TO_BUCKETIZE=true for "
                       "train/eval/export when use rtp for online inference.")
    return flag
```

`USE_FARM_HASH_TO_BUCKETIZE=true` 也必须在**训练与评估**期间设置，以使分桶函数与 RTP 在 serving 时应用的一致。不匹配会导致静默的嵌入错位。

特征层在构建时也会查询 `use_rtp()`（[`features/feature.py:426`](torcheasyrec/tzrec/features/feature.py#L426)）以切换其序列名称分隔符：RTP 使用 `seq_feat`（单下划线）；PyFG / Aliyun FG 使用 `seq__feat`（双下划线）。这就是为什么 `_to_real_input_name` / `_to_pyfg_input_name` / `_to_pyfg_feat_name` 存在于 [`datasets/data_parser.py:954`](torcheasyrec/tzrec/datasets/data_parser.py#L954)——当 `USE_RTP=1` 时它们桥接两种约定，使进程内 PyFG 调用返回的值与 RTP 将在 wire 上命名的值匹配。

### 7.2 顶层流程

[`export_util.py:697`](torcheasyrec/tzrec/utils/export_util.py#L697) `export_rtp_model()`：

```
USE_RTP=1 + checkpoint_path
  → 加载 checkpoint，构建 DataLoader (PREDICT, batch_size=MAX_EXPORT_BATCH_SIZE)
  → 构建 planner，分片模型 (DistributedModelParallel)
  → 用 Tracer 追踪分片模型 → full_graph
  → feature_to_embedding_info = _get_rtp_feature_to_embedding_info(model)
       (验证没有两个模块共享嵌入名；没有特征由两个不同 EBC 服务——RTP 需要 1:1:1 映射)
  → 将 full_graph 拆分为 sparse graph + dense graph
  → sparse_model = DMP(sparse_gm)
  → 恢复 sparse checkpoint
  → 跑 sparse → sparse_output, sparse_attrs
  → 保存 sparse 权重：model-NNNNNN-of-NNNNNN.safetensors + .json meta
  → 为 dense 初始化参数 + 恢复 checkpoint
  → ExportTorchFxTool().export_fx_model(gm, sparse_output, mc_config)
       写入 save_dir/fx_user_model/ (RTP dense package)
  → 保存调整后的 fg.json
```

分片模型步骤是与 `export_model_normal` 差异最大的部分。RTP serving 运行在潜在的多个节点上，每个节点只需要它拥有的分片。因此在 `DistributedModelParallel` 下运行导出，使 shard 计划在训练与 serving 之间完全相同。

### 7.3 FX 图重写

这是 RTP 集成的核心。模型代码在稀疏→稠密边界插入四个 `@torch.fx.wrap` 标记的哨兵调用：

| 哨兵 | 含义 | 使用位置 |
| --- | --- | --- |
| `fx_mark_keyed_tensor(name, KJT)` | 池化的 `KeyedTensor`（EBC 输出） | sparse 输出字典 |
| `fx_mark_tensor(name, t)` | 稠密特征 | sparse 输出字典 |
| `fx_mark_seq_tensor(seq_name, t, max_seq_len, is_jagged_seq, keys)` | 序列特征 | sparse 输出字典 |
| `fx_mark_seq_len(seq_name, t)` | 序列长度张量 | sparse 输出字典 |

它们来自 [`utils/fx_util.py`](torcheasyrec/tzrec/utils/fx_util.py)。对 PyTorch 的 tracer 是不透明的（因为有 `@torch.fx.wrap`），所以 tracer 将它们保留为以原始 Python callable 作为 target 的 `call_function` 节点——这就是重写代码找到它们的方式。

#### 稀疏侧（图位于 L805–L869）

对每个哨兵：
- `fx_mark_keyed_tensor(name, KJT)` → 调用 `KJT.values()`、`KJT.length_per_key()`、`KJT.keys()`，并将每个存储在输出字典中，key 分别为 `name`、`name + "__length_per_key"`、`name + "__keys"`。
- `fx_mark_tensor(name, t)` → `t` 直接流过。
- `fx_mark_seq_tensor(seq_name, t, ...)` → 如果 `is_jagged_seq`，前置 `torch.unsqueeze(t, 0)`；然后调用 `_rtp_pad_to_max_seq_len`（[L639](torcheasyrec/tzrec/utils/export_util.py#L639)）将序列右侧填充到 `max_seq_len`。输出 key 变为 `seq_name + "_sequence"`。
- `fx_mark_seq_len(seq_name, t)` → `torch.unsqueeze(t, 1)`（RTP 不能接受 rank-1 张量）。输出 key 变为 `seq_name + "_sequence_length"`。

结果被包装到新的 `GraphModule` 中，进行 dead-code-elimination，并通过 `_prune_unused_param_and_buffer`（[L426](torcheasyrec/tzrec/utils/export_util.py#L426)）剪除未使用的 params/buffers。

#### 稠密侧（图位于 L889–L1006）

对每个哨兵，稠密图用 input-from-dict 查找替换原始 `call_function` 节点。具体地：
- `fx_mark_keyed_tensor(name, KJT)` → `KeyedTensor(keys=sparse_attrs[name+"__keys"], length_per_key=sparse_attrs[name+"__length_per_key"], values=input_node[name])`——这样在 serve 时稠密图从输入字典读取池化嵌入，并重建一个具有期望 shape 的 `KeyedTensor`。
- `fx_mark_tensor(name, t)` → `input_node[name]`。
- `fx_mark_seq_len(seq_name, t)` → 首先，将 `t` 占位符替换为 `input_node[seq_name + "_sequence_length"]`，在 axis 1 上 squeeze（在图内重新成为 rank-1）；然后**也**向 FG 配置添加一个 `user:seq_name_sequence_length` raw_feature，以便 PyFG 将其作为普通输入暴露。用户必须记得在其 `qinfo` 中注册此特征（代码在 L925 记录警告）。
- `fx_mark_seq_tensor(seq_name, t, max_seq_len, ...)` → `input_node[seq_name + "_sequence"]` 后跟 `_rtp_slice_with_seq_len`（[L646](torcheasyrec/tzrec/utils/export_util.py#L646)），它在 axis 1 上切片到实际的运行时序列长度（上限为 `max_seq_len`）。如果 `is_jagged_seq`，最后应用 `torch.squeeze(_, 0)`。
- 任何 target 是 `torch.ops.fbgemm.*` op 的 `call_function` 都被替换为对应的 `FBGEMM_RTP_TORCH_OP_MAPPING` 条目（[L689](torcheasyrec/tzrec/utils/export_util.py#L689)），它提供 RTP 理解的纯 PyTorch fallback。四个支持的 op 是 `asynchronous_complete_cumsum`、`jagged_to_padded_dense`、`dense_to_jagged` 和 `jagged_dense_elementwise_add_jagged_output`。其他任何 op 都会抛出 `RuntimeError("... is not supported by rtp")`。

Jagged-sequence 和 fbgemm-op fallback **要求 `MAX_EXPORT_BATCH_SIZE=1`**（在 L837 和 L984 处的 assert）。追踪时示例 batch 必须看起来像单个请求，这样符号化 shape 与运行时假设匹配。

`mc_config`（[L892](torcheasyrec/tzrec/utils/export_util.py#L892)）是将每个输出 key 映射到其底层特征名的字典。它被传递给 `ExportTorchFxTool`，以便生成的 RTP 模型知道每个输出读取哪些 MC 管理的 collision 表。

### 7.4 保存稀疏参数

[`export_util.py:463`](torcheasyrec/tzrec/utils/export_util.py#L463) `_get_rtp_embedding_tensor()` 遍历分片模型的 `state_dict()`，并为每个 rank 写入一个 safetensors 文件：

```
save_dir/model-RRRRRR-of-WWWWWW.safetensors
save_dir/model-RRRRRR-of-WWWWWW.json      ← per-tensor metadata
```

对每个张量：
- 如果它是 `ShardedTensor` 且该 rank 拥有一个分片：在名称 `param_name/part_{idx}_{num_shards}` 下保存本地分片。
- 如果它是普通张量且最后一维 == embedding dim：直接保存。
- DynamicEmb 表（sparse Adam 风格）从 `checkpoint_path/dynamicemb/*/*_emb_{keys,values}.rank_*.world_size_*` 加载并配对；JSON 条目获得 `is_hashmap=True`、`hashmap_key`、`hashmap_value`、`hashmap_key_dtype` 字段，这样服务器知道通过 keys 张量查找 values。

**约束：** 稀疏权重仅支持 `float32`。该函数在 L551 处 assert 这一点。

### 7.5 FG JSON 调整

[`export_util.py:617`](torcheasyrec/tzrec/utils/export_util.py#L617) `_adjust_fg_json_for_rtp()` 以三种方式重写 FG JSON：

1. **Bucket key 交换。** 对每个特征：
   - `boundaries` → `gen_key_type = "boundary"`
   - `hash_bucket_size` → `gen_key_type = "hash"`
   - `num_buckets` → 重命名为 `hash_bucket_size`，`gen_key_type = "mod"`
   - 其他 → `gen_key_type = "idle"`
   - 基于 vocab 的特征（`vocab_dict` / `vocab_list` / `vocab_file`）抛出 `ValueError`，因为 RTP 无法加载它们。
2. **嵌入查找提示。** 如果该特征在 `feature_to_embedding_info` 中有对应条目，添加 `shared_name`、`embedding_dimension`、`gen_val_type = "lookup"`。否则 `gen_val_type = "idle"`。
3. **序列簿记。** 序列组获得 `sequence_table = "item"`（user/item 分割），`value_dim` → `value_dimension`，`need_discrete` → `needDiscrete`。

合成序列长度的额外 raw 特征（`<seq_name>_sequence_length`）在 `export_rtp_model` 的 L918 附加，然后此调整运行。

### 7.6 为什么采用 FX wrap 哨兵设计？

`fx_mark_*` 节点对 tracer 可见为普通 `call_function` 节点，但对 inductor / AOTI 不可见（它们会被内联）。`export_rtp_model` 重写然后遍历图，找到这些节点，并 (a) 将其 `args[1]`（实际张量）重新路由到新的 `getitem(input_node, name)` 查找（稠密侧）或 (b) 将其输出提取为新的顶级图输出（稀疏侧）。这就是为什么训练好的模型中稀疏与稠密之间的边界在 Python 级别是显式的而非隐式的——重写是图本地的，不是启发式的。

同一哨兵设计是 normal 导出路径中 `split_model()`（L1068）使用的，因此边界标记在所有四个后端中重用。

---

## 8. 与其他后端不一致的输入

RTP 对框架的其他部分有一些不明显的要求：

1. **EBCs/ECs 中无名称冲突**（[L386](torcheasyrec/tzrec/utils/export_util.py#L386)、[L394](torcheasyrec/tzrec/utils/export_util.py#L394)）。由两个不同嵌入模块服务的两个特征在导出时失败。解决方法是用相同的 `feature_config` 创建第二个 `Feature` 并在第二组中使用。
2. **无 vocab-based 特征。** `RTP_INVALID_BUCKET_KEYS`（[L573](torcheasyrec/tzrec/utils/export_util.py#L573)）列出 `vocab_dict`、`vocab_list`、`vocab_file`。使用哈希分桶变体。
3. **`USE_FARM_HASH_TO_BUCKETIZE=true`** 必须在 train/eval/export 中设置以确保哈希一致性。警告在环境变量读取时触发。
4. **`MAX_EXPORT_BATCH_SIZE=1`** 如果模型具有 jagged 序列或需要 RTP fallback 的 fbgemm op。
5. **稀疏权重仅支持 float32**（[L551](torcheasyrec/tzrec/utils/export_util.py#L551)）。量化的 EBC 表在为 RTP 保存之前必须去量化。
6. **每个用户必须在其 `qinfo` 中注册 `<seq_name>_sequence_length`**，因为重写后的稠密图将其视为用户提供的输入。导出器在 L925 记录每一个。

---

## 9. Predict-Time 加载

对于所有后端，predict-time loader（`tzrec.main.predict` → `create_predict_fn`）重新读取 `model_acc.json` 并分派：

- `is_trt_predict(model_path)`（[`acc/utils.py:142`](torcheasyrec/tzrec/acc/utils.py#L142)）→ `torch.jit.load(scripted_model.pt)`，像以前一样调用。
- `is_unified_aot_predict(model_path)`（[`acc/utils.py:92`](torcheasyrec/tzrec/acc/utils.py#L92)）→ `torch._inductor.aoti_load_package(aoti/aoti_model.pt2)` 包装在 `UnifiedAOTIModelWrapper` 中。
- 否则 AOTI 旧版 → `CombinedModelWrapper(sparse_scripted, dense_aoti)`。
- RTP 路径使用不同的运行时（不在本仓库中），它加载 `fx_user_model/` 目录以及 `model-*-of-*.safetensors` 文件。

这就是为什么存在 `model_acc.json`：predict 进程是一个全新解释器，对启动导出进程时的环境变量一无所知。JSON 是该信息的唯一通道。

---

## 10. 端到端图示

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

## 11. 关键环境变量（导出侧）

| 变量 | 读取位置 | 效果 |
| --- | --- | --- |
| `USE_RTP` | `env_util.use_rtp()` | 切换后端到 `export_rtp_model`。**还需要** `USE_FARM_HASH_TO_BUCKETIZE=true`（否则警告）。 |
| `USE_FARM_HASH_TO_BUCKETIZE` | `env_util.use_rtp()` | 与 RTP 的分桶一致性。应在 train、eval 和 export 中设置。 |
| `ENABLE_AOT` | `acc.utils.is_aot()` / `is_unified_aot()` | `1` = 旧版两阶段，`2` = 统一单一 `.pt2`。`UNIFIED_AOT=1` 是 `2` 的旧版别名。 |
| `ENABLE_TRT` | `acc.utils.is_trt()` | 用 `torch_tensorrt.dynamo.compile` 编译稠密子图。 |
| `QUANT_EMB` | `acc.utils.is_quant()` / `quant_dtype()` | EBC 量化 dtype（`FP32`/`FP16`/`INT8`/`INT4`/`INT2`）。默认 `INT8`；`QUANT_EMB=0` 禁用。 |
| `QUANT_EC_EMB` | `acc.utils.is_ec_quant()` | EC 量化 dtype。`0` 禁用。 |
| `INPUT_TILE` | `acc.utils.is_input_tile*()` | `2*` = user/item 分割，`3*` = user/item + 拆分嵌入表。为 `3*` 写入 `emb_ckpt_mapping.txt`。 |
| `INPUT_TILE_3_ONLINE` | `acc.utils.is_input_tile_3_online()` | 如果为 `1`，序列张量直接使用 `jt.values()`；不支持离线 predict。 |
| `MAX_EXPORT_BATCH_SIZE` | `acc.utils.get_max_export_batch_size()` | 限制追踪 batch 大小以避免 inductor compile 期间的 OOM。 |
| `TRT_MAX_BATCH_SIZE` | 同上 | `MAX_EXPORT_BATCH_SIZE` 的旧版别名。 |
| `TRT_MAX_SEQ_LEN` | `trt_utils.get_trt_max_seq_len()` | TRT 中序列长度的动态 shape 上界。 |
| `AOTI_AUTOTUNE_WITH_SAMPLE_INPUTS` | `acc.utils.is_autotune_with_sample_inputs()` | `1` 启用 inductor 的 `triton.autotune_with_sample_inputs`。 |
| `ENABLE_TMA` | `env_util.enable_tma()` | Triton kernel 中的 TMA（需要 SM 9.0+ 且 Triton ≥ 3.5）。 |
| `USE_HASH_NODE_ID` | `env_util.use_hash_node_id()` | 基于哈希的节点 ID（分布式设置）。 |
| `DEBUG_TRT` | `acc.utils.is_debug_trt()` | 启用 TRT 图日志 + profiler dump。 |
| `FORCE_LOAD_SHARDING_PLAN` | `env_util.force_load_sharding_plan()` | 重用 checkpoint 中的分片计划。 |
| `LOCAL_CACHE_DIR` | `export_model` | 当 `save_dir` 是远程 URL（fsspec）时的本地临时目录。 |

> C++ IO（Scripted Model 加载、safetensors 导出）不可被 fsspec 透传，必须用 `LOCAL_CACHE_DIR` 中转。完整机制见 [USE_FSSPEC 与 fsspec 透传机制](11-use-fsspec) 第三节。

predict-time loader 从 `model_acc.json` 读取相同的变量以选择匹配的反序列化器——参见 [`acc/utils.py`](torcheasyrec/tzrec/acc/utils.py) 中的 `*_predict()` 谓词。

---

## 12. `INPUT_TILE` 推理模式

除四个后端外，第二个轴控制稠密图的**部署**方式（与**编译**方式分开）：`INPUT_TILE` 环境变量。这适用于 AOTI / TRT 后端，不适用于 RTP。

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

| `INPUT_TILE` | 用途 | 产物 |
|---|---|---|
| `1` (默认) | 单调用推理；user + item 在一个 batch 中 | 标准 `scripted_model.pt` |
| `2*` | 两阶段匹配（user 离线，item 在线） | 独立的 user/item `.pt` 文件 |
| `3*` | 巨大表的分布式嵌入分片 | `emb_ckpt_mapping.txt` 用于运行时嵌入加载 |

`INPUT_TILE=3*` 写入 `emb_ckpt_mapping.txt`（通过 `export_util.py` 中的 `write_mapping_file_for_input_tile`），将逻辑嵌入名映射到物理分片 checkpoint 文件——predict loader 使用它来按其实际分片位置查找权重。

### AOT/TRT vs RTP — 关键差异

| 特性 | AOT/TRT | RTP |
|---|---|---|
| **分布式支持** | 不支持（强制 `WORLD_SIZE=1`） | 支持（`WORLD_SIZE=N`） |
| **嵌入分片** | 不支持 | 支持（通过 `ShardedModule`） |
| **权重导出格式** | 内联在 JIT 模型中 | 每个 rank 分片 safetensors |
| **叶子模块** | 已展平 | 保留为 `ShardedModule` |
| **序列处理** | 标准 | 替换为 RTP 兼容的实现 |
| **图格式** | JIT Script | `ExportTorchFxTool` |
| **运行时** | 单 GPU CUDA | 多 GPU / 多节点分布式 |

### `split_model` 逻辑（AOT/TRT，[`export_util.py:1068`](torcheasyrec/tzrec/utils/export_util.py#L1068)）

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

### AOT/TRT vs RTP 数据流

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
