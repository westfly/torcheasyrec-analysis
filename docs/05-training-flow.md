---
title: Training Flow
nav_order: 6
---

# Training Flow

## Overview

The training flow starts in [`_train_and_evaluate()`](../torcheasyrec/tzrec/main.py#L317-L530). Each training step processes:

```
DataLoader → pipeline.progress(iterator) → (losses, predictions, batch)
                                           → optimizer.step()
                                           → update_metric()
                                           → log()
```

## The Training Loop

```
For each epoch:
    pipeline = create_train_pipeline(model, optimizer)
    For each step:
        losses, predictions, batch = pipeline.progress(iterator)
        model.update_train_metric(predictions, batch)
        log metrics
        lr_scheduler.step()
        if save_checkpoints:
            ckpt_manager.save(step, model, optimizer)
            _evaluate(model, eval_dataloader)
```

### `create_train_pipeline()`

[`torcheasyrec/tzrec/utils/dist_util.py`](../torcheasyrec/tzrec/utils/dist_util.py)

Creates a pipeline function that:
1. Fetches next batch from iterator
2. Moves batch to GPU
3. Calls model (TrainWrapper.forward)
4. Calls backward on total_loss
5. Calls optimizer.step()
6. Returns (losses_dict, predictions_dict, batch)

### `TrainWrapper.forward()`

[`torcheasyrec/tzrec/models/model.py`](../torcheasyrec/tzrec/models/model.py#L262-L288)

```python
def forward(self, batch):
    with torch.amp.autocast(device_type=self._device_type, dtype=self._mixed_dtype):
        predictions = self.model.predict(batch)
        losses = self.model.loss(predictions, batch)
        if self.training and self.pareto:
            total_loss = self.pareto(losses, self.model)
        else:
            total_loss = torch.stack(list(losses.values())).sum()
    losses = {k: v.detach() for k, v in losses.items()}
    predictions = {k: v.detach() for k, v in predictions.items()}
    return total_loss, (losses, predictions, batch)
```

Key points:
- **Autocast**: FP16/BF16 mixed precision if configured
- **Pareto MTL**: Optional Pareto-efficient multi-task loss weighting
- **Loss aggregation**: All losses are summed
- **Detach**: Losses and predictions are detached for logging/metrics

## Data Flow Through a Step

### 1. Dataset → Batch

The dataset yields `RecordBatch` objects (pyarrow tables). `DataParser.to_batch()` converts them:

[`torcheasyrec/tzrec/datasets/data_parser.py`](../torcheasyrec/tzrec/datasets/data_parser.py)

```python
class DataParser:
    def to_batch(self, data: Dict[str, pa.Array]) -> Batch:
        # For each feature, parse raw data → SparseData/DenseData
        # Group by data group (BASE_DATA_GROUP, NEG_DATA_GROUP)
        # Build KeyedJaggedTensor (sparse) and KeyedTensor (dense)
        # Return Batch(sparse_features, dense_features, labels, ...)
```

### 2. Batch → Embedding

`EmbeddingGroup.forward(batch)` processes the batch:

[`torcheasyrec/tzrec/modules/embedding.py`](../torcheasyrec/tzrec/modules/embedding.py#L419-L508)

```
For each embedding implementation:
    Extract sparse KJT (KeyedJaggedTensor) from batch.sparse_features
    Extract dense KT (KeyedTensor) from batch.dense_features
    Forward through:
        - EmbeddingBagCollection (sparse categorical features)
        - ManagedCollisionEmbeddingBagCollection (ZCH features)
        - DenseEmbeddingCollection (AutoDis/MLP embedded dense features)
        - SequenceEmbeddingGroupImpl (sequence features)
    Combine outputs into grouped_features dict
For each sequence encoder:
    Apply sequence encoder to grouped_features
Return grouped_features
```

### 3. Embedding → Predictions

Each model implements `predict(batch)`:

[`torcheasyrec/tzrec/models/deepfm.py`](../torcheasyrec/tzrec/models/deepfm.py#L72-L108)

```python
def predict(self, batch):
    grouped_features = self.build_input(batch)
    # Wide
    wide_feat = grouped_features["wide"]
    y_wide = torch.sum(wide_feat, dim=1, keepdim=True)
    # Deep
    deep_feat = grouped_features["deep"]
    y_deep = self.deep_mlp(deep_feat)
    # FM
    fm_feat = grouped_features.get("fm", deep_feat)
    fm_feat = fm_feat.reshape(-1, n_fields, emb_dim)
    y_fm = self.fm(fm_feat)
    # Combine
    y = y_wide + y_fm.sum(dim=1, keepdim=True) + self.output_mlp(y_deep)
    return self._output_to_prediction(y)
```

`_output_to_prediction()` converts logits to predictions dict with `logits`, `probs`, etc., based on loss type.

### 4. Predictions → Loss

`RankModel.loss()` or `MatchModel.loss()` computes losses:

[`torcheasyrec/tzrec/models/rank_model.py`](../torcheasyrec/tzrec/models/rank_model.py#L264-L287)

```python
def loss(self, predictions, batch):
    for loss_cfg in self._base_model_config.losses:
        loss_type = loss_cfg.WhichOneof("loss")
        if loss_type == "binary_cross_entropy":
            losses["binary_cross_entropy"] = BCEWithLogitsLoss(pred, label)
        elif loss_type == "softmax_cross_entropy":
            losses["softmax_cross_entropy"] = CrossEntropyLoss(pred, label)
        ...
    return losses
```

### 5. Backward + Optimizer

```
total_loss.backward()
GradientClippingOptimizer.step()
```

The optimizer is built in [`torcheasyrec/tzrec/optim/optimizer_builder.py`](../torcheasyrec/tzrec/optim/optimizer_builder.py). It uses TorchRec's `CombinedOptimizer` which handles:
- **Sparse parameters**: optimized in-backward (via `apply_optimizer_in_backward`)
- **Dense parameters**: standard optimizers (SGD, Adam, Adagrad)

## Evaluation Flow

[`_evaluate()`](../torcheasyrec/tzrec/main.py#L162-L226)

```
model.eval()
pipeline = create_train_pipeline(model)
with torch.no_grad():
    for step in range(num_steps):
        losses, predictions, batch = pipeline.progress(iterator)
        model.update_metric(predictions, batch, losses)
metric_result = model.compute_metric()
```

Metrics are computed via `torchmetrics` objects stored in `model._metric_modules`.

## Checkpointing

[`torcheasyrec/tzrec/utils/checkpoint_util.py`](../torcheasyrec/tzrec/utils/checkpoint_util.py)

- `CheckpointManager` saves model + optimizer + dataloader state
- Supports `fine_tune_checkpoint` for transfer learning
- `save_checkpoints_steps` / `save_checkpoints_epochs` control frequency
- Also handles model export during checkpointing

## Logging

Train metrics (loss, learning rate, gradient norm) are logged via:

- **ProgressLogger**: text progress bar
- **SummaryWriter**: TensorBoard (loss, learning_rate, gradient, parameter histograms)

Train metrics like `DecayAUC` are collected during training and logged periodically.

## Key Files

| File | Purpose |
|------|---------|
| [`torcheasyrec/tzrec/main.py`](../torcheasyrec/tzrec/main.py#L317-L530) | `_train_and_evaluate()`, `_evaluate()`, `_log_train()` |
| [`torcheasyrec/tzrec/models/model.py`](../torcheasyrec/tzrec/models/model.py#L235-L288) | `TrainWrapper.forward()` |
| [`torcheasyrec/tzrec/utils/dist_util.py`](../torcheasyrec/tzrec/utils/dist_util.py) | `create_train_pipeline()`, `DistributedModelParallel` |
| [`torcheasyrec/tzrec/modules/embedding.py`](../torcheasyrec/tzrec/modules/embedding.py#L419-L508) | `EmbeddingGroup.forward()` |
| [`torcheasyrec/tzrec/optim/optimizer_builder.py`](../torcheasyrec/tzrec/optim/optimizer_builder.py) | Optimizer construction |
| [`torcheasyrec/tzrec/utils/checkpoint_util.py`](../torcheasyrec/tzrec/utils/checkpoint_util.py) | Checkpoint management |

## Two-Phase Initialization: Planner + DMP

TorchEasyRec's distributed training uses a **two-phase initialization** to
wire up TorchRec sharding — a static planning step followed by a dynamic
execution step:

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                      两阶段初始化架构                                         │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  【第一阶段】Planner 规划 (静态)                                             │
│  ┌─────────────────────────────────────────────────────────────────────┐  │
│  │                                                                      │  │
│  │   输入:                                                              │  │
│  │   ├── 模型结构 (EmbeddingBagCollection)                              │  │
│  │   ├── 特征配置 (vocab_size, embedding_dim)                           │  │
│  │   ├── 设备拓扑 (world_size, local_size)                               │  │
│  │   └── 内存约束 (max_shard_size_mb)                                   │  │
│  │                                                                      │  │
│  │   规划器决策:                                                        │  │
│  │   ├── 选择分片策略 (Table-wise / Row-wise / Column-wise)            │  │
│  │   ├── 确定每个 embedding 表的分片数量                                  │  │
│  │   ├── 分配分片到物理设备                                              │  │
│  │   └── 生成 ShardingPlan                                               │  │
│  │                                                                      │  │
│  │   输出: ShardingPlan (描述如何分片)                                    │  │
│  │                                                                      │  │
│  └─────────────────────────────────────────────────────────────────────┘  │
│                                    │                                        │
│                                    ▼                                        │
│  【第二阶段】DMP 执行 (动态)                                                 │
│  ┌─────────────────────────────────────────────────────────────────────┐  │
│  │                                                                      │  │
│  │   DistributedModelParallel(model, plan)                              │  │
│  │                                                                      │  │
│  │   执行内容:                                                          │  │
│  │   ├── 将 embedding 参数按 plan 分片                                   │  │
│  │   ├── 注入通信模块 (AllToAll, ReduceScatter)                         │  │
│  │   ├── 注册 Sharder 用于 checkpoint                                    │  │
│  │   └── 建立分布式执行上下文                                            │  │
│  │                                                                      │  │
│  │   关键机制:                                                          │  │
│  │   ├── Lazy Input Dist: 延迟初始化输入分布                             │  │
│  │   ├── Collective Plan: 所有 rank 协同执行分片                         │  │
│  │   └── ShardedModule: 封装分片后的子模块                               │  │
│  │                                                                      │  │
│  └─────────────────────────────────────────────────────────────────────┘  │
│                                                                             │
│  代码位置: tzrec/utils/dist_util.py:154-183                                  │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

[`torcheasyrec/tzrec/utils/dist_util.py:154-183`](../torcheasyrec/tzrec/utils/dist_util.py#L154-L183):

```python
def DistributedModelParallel(
    module: nn.Module,
    env: Optional[ShardingEnv] = None,
    device: Optional[torch.device] = None,
    plan: Optional[ShardingPlan] = None,
    sharders: Optional[List[ModuleSharder]] = None,
    init_data_parallel: bool = True,
    init_parameters: bool = True,
    data_parallel_wrapper: Optional[DataParallelWrapper] = None,
) -> _DistributedModelParallel:
    """模型并行的入口点."""
    model = _DistributedModelParallel(
        module, env, device, plan, sharders,
        init_data_parallel, init_parameters, data_parallel_wrapper,
    )
    # 保持稀疏模块的 input_dist 延迟初始化
    for _, m in model.named_modules():
        if hasattr(m, "_has_uninitialized_input_dist"):
            m._has_uninitialized_input_dist = True
    return model
```

**Phase 1 (Planner)** — A static pass that inspects the model topology,
device mesh, and memory budget to decide how each embedding table is
sharded. The output is a `ShardingPlan` describing the placement.

**Phase 2 (DMP)** — A dynamic pass that consumes the `ShardingPlan` and
applies it to the live module: it shards the parameter tensors across
ranks, injects AllToAll / ReduceScatter collectives for cross-rank
communication, and wraps each sharded sub-module in a `ShardedModule`
that handles per-rank execution.

The split into two phases is necessary because:

- The plan is deterministic given a model + topology + budget, so it can
  be computed once and reused (e.g., shared across training and export,
  see `FORCE_LOAD_SHARDING_PLAN`).
- The execution step requires live distributed context (process groups,
  NCCL handles) that only exists after `torch.distributed.init_process_group`.

**Lazy Input Dist**: DMP keeps input distribution modules
(`_has_uninitialized_input_dist = True`) uninitialized until the first
forward pass. This avoids materializing communication buffers for sparse
modules that might never be called in a given model variant.

## Mixed Precision Training

The `TrainWrapper` autocast block (see §`TrainWrapper.forward()` above)
runs the dense forward in FP16 or BF16 if configured, but leaves the
sparse embedding lookups in their native FP32 (the lookup itself is a
gather, not a matmul, so AMP adds no speedup but does complicate
`torch.jit.script`).

For FP16 (where the smaller dynamic range risks gradient underflow),
`TZRecOptimizer` wraps the combined optimizer with a `GradScaler` that
multiplies the loss by a `scale_factor` (e.g., 65536) before backward,
checks the resulting gradients for inf/nan, and:

- On inf/nan: shrinks `scale_factor` by `backoff_factor` (default 0.5)
  and skips the step.
- After N consecutive clean steps: grows `scale_factor` by
  `growth_factor` (default 2.0).

This is a standard PyTorch AMP pattern; see
[`torcheasyrec/tzrec/main.py:709-714`](../torcheasyrec/tzrec/main.py#L709-L714) for the
scaler construction.

## Pipeline Parallelism

[`torcheasyrec/tzrec/utils/dist_util.py:304-345`](../torcheasyrec/tzrec/utils/dist_util.py#L304-L345) — `create_train_pipeline()`:

```python
def create_train_pipeline(
    model: nn.Module,
    optimizer: Optional[torch.optim.Optimizer] = None,
    check_all_workers_data_status: bool = False,
) -> TrainPipeline:
    """根据模型类型创建训练流水线."""
    has_sparse_module = False
    q = Queue()
    q.put(model.module)
    while not q.empty():
        m = q.get()
        if isinstance(m, ShardedModule):
            has_sparse_module = True
            break
        else:
            for child in m.children():
                q.put(child)

    if not has_sparse_module:
        return TrainPipelineBase(model, optimizer, model.device)
    else:
        return TrainPipelineSparseDist(
            model, optimizer, model.device,
            execute_all_batches=True,
            check_all_workers_data_status=check_all_workers_data_status,
        )
```

- `TrainPipelineBase` — vanilla pipeline for models with no sharded
  modules. Used when the model has no embedding tables (rare; the
  pipeline still has an optimizer / scaler wrapper).
- `TrainPipelineSparseDist` — TorchRec's pipeline that pipelines
  data-loading on a separate CUDA stream so the next batch's host-side
  preprocessing overlaps the current batch's GPU forward/backward. Only
  valid when the model contains `ShardedModule`s (i.e., embedding
  tables), because the pipelined iteration needs sharded data-parallel
  input distribution.

`execute_all_batches=True` ensures all ranks see all batches (vs. sharding
batches by rank), which is required for some negative samplers and
for `check_all_workers_data_status` debugging.

## Optimizer Construction

[`torcheasyrec/tzrec/main.py:656-741`](../torcheasyrec/tzrec/main.py#L656-L741):

```python
# 稀疏优化器用于 embedding 参数
sparse_optim_cls, sparse_optim_kwargs = optimizer_builder.create_sparse_optimizer(
    train_config.sparse_optimizer
)
trainable_params, frozen_params = model.model.sparse_parameters()
apply_optimizer_in_backward(sparse_optim_cls, trainable_params, sparse_optim_kwargs)
if len(frozen_params) > 0:
    apply_optimizer_in_backward(SGD, frozen_params, {"lr": 0.0})

# 稠密优化器用于模型参数
dense_optim_cls, dense_optim_kwargs = optimizer_builder.create_dense_optimizer(
    train_config.dense_optimizer
)
dense_optimizer = KeyedOptimizerWrapper(
    remaining_params,
    lambda params: dense_optim_cls(params, **dense_optim_kwargs),
)

# 组合优化器 + 梯度裁剪
combined_optimizer = CombinedOptimizer(
    [model.fused_optimizer, dense_optimizer, *part_optimizers]
)
if train_config.HasField("grad_clipping"):
    combined_optimizer = GradientClippingOptimizer(
        optimizer=combined_optimizer,
        clipping=GradientClipping[clipping_type_str],
        max_gradient=gc_config.max_gradient,
        norm_type=gc_config.norm_type,
        enable_global_grad_clip=gc_config.enable_global_grad_clip,
    )

# 最终封装器 + 梯度缩放
optimizer = TZRecOptimizer(
    combined_optimizer,
    grad_scaler=grad_scaler,
    gradient_accumulation_steps=train_config.gradient_accumulation_steps,
)
```

**Optimizer hierarchy:**

```
TZRecOptimizer
├── CombinedOptimizer
│   ├── fused_optimizer (稀疏 embeddings)
│   ├── dense_optimizer (稠密参数)
│   └── part_optimizers (参数分组)
└── GradScaler (混合精度)
```

The three-way split:

- **Sparse optimizers** are applied in-backward via
  `apply_optimizer_in_backward` — this avoids materializing the full
  embedding gradient as a dense tensor (sparse Adam has O(num_lookups)
  state, not O(table_size)).
- **Dense optimizer** is a standard per-parameter PyTorch optimizer.
- **Frozen parameters** (e.g., the SSTable in a `ManagedCollisionModule`)
  are bound to `SGD(lr=0.0)` so the framework still tracks them but they
  don't actually update.

## Distributed Training — Datasets

| Dataset | File | Notes |
|---|---|---|
| `CsvDataset` | `csv_dataset.py` | CSV files with custom delimiter |
| `ParquetDataset` | `parquet_dataset.py` | Apache Parquet columnar |
| `OdpsDataset` | `odps_dataset.py` | Alibaba MaxCompute (ODPS) |
| `OdpsDatasetV1` | `odps_dataset_v1.py` | Legacy ODPS Tunnel API |
| `KafkaDataset` | `kafka_dataset.py` | Kafka streaming |
