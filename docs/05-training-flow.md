---
title: 训练流程
nav_order: 6
---

# 训练流程

## 概览

训练流程从 [`_train_and_evaluate()`](../torcheasyrec/tzrec/main.py#L317-L530) 启动。每一步处理：

```
DataLoader → pipeline.progress(iterator) → (losses, predictions, batch)
                                            → optimizer.step()
                                            → update_metric()
                                            → log()
```

## 训练循环

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

创建一个 pipeline 函数，依次：

1. 从 iterator 拉取下一个 batch
2. 将 batch 移动到 GPU
3. 调用模型（`TrainWrapper.forward`）
4. 在 total_loss 上调用 backward
5. 调用 `optimizer.step()`
6. 返回 `(losses_dict, predictions_dict, batch)`

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

关键点：
- **Autocast**：若已配置则使用 FP16/BF16 混合精度
- **Pareto MTL**：可选的 Pareto-efficient 多任务损失加权
- **损失聚合**：所有损失求和
- **Detach**：losses 与 predictions 被 detach 以用于日志/指标

## 单步数据流

### 1. Dataset → Batch

Dataset 产出 `RecordBatch` 对象（pyarrow 表）。`DataParser.to_batch()` 进行转换：

[`torcheasyrec/tzrec/datasets/data_parser.py`](../torcheasyrec/tzrec/datasets/data_parser.py)

```python
class DataParser:
    def to_batch(self, data: Dict[str, pa.Array]) -> Batch:
        # 对每个 feature，解析原始数据 → SparseData/DenseData
        # 按 data group 分组 (BASE_DATA_GROUP, NEG_DATA_GROUP)
        # 构建 KeyedJaggedTensor (sparse) 与 KeyedTensor (dense)
        # 返回 Batch(sparse_features, dense_features, labels, ...)
```

### 2. Batch → Embedding

`EmbeddingGroup.forward(batch)` 处理 batch：

[`torcheasyrec/tzrec/modules/embedding.py`](../torcheasyrec/tzrec/modules/embedding.py#L419-L508)

```
对每个嵌入实现:
    从 batch.sparse_features 抽取 sparse KJT (KeyedJaggedTensor)
    从 batch.dense_features 抽取 dense KT (KeyedTensor)
    前向传播通过:
        - EmbeddingBagCollection (sparse 类别特征)
        - ManagedCollisionEmbeddingBagCollection (ZCH 特征)
        - DenseEmbeddingCollection (AutoDis/MLP 嵌入的 dense 特征)
        - SequenceEmbeddingGroupImpl (sequence 特征)
    将输出合并为 grouped_features 字典
对每个序列编码器:
    对 grouped_features 应用序列编码器
返回 grouped_features
```

### 3. Embedding → Predictions

每个模型实现 `predict(batch)`：

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
    # 合并
    y = y_wide + y_fm.sum(dim=1, keepdim=True) + self.output_mlp(y_deep)
    return self._output_to_prediction(y)
```

`_output_to_prediction()` 基于 loss 类型将 logits 转换为 predictions 字典，包含 `logits`、`probs` 等。

### 4. Predictions → Loss

`RankModel.loss()` 或 `MatchModel.loss()` 计算损失：

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

optimizer 在 [`torcheasyrec/tzrec/optim/optimizer_builder.py`](../torcheasyrec/tzrec/optim/optimizer_builder.py) 中构建。它使用 TorchRec 的 `CombinedOptimizer`，处理：

- **稀疏参数**：在 backward 中优化（通过 `apply_optimizer_in_backward`）
- **稠密参数**：标准 optimizer（SGD、Adam、Adagrad）

## 评估流程

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

指标通过存储在 `model._metric_modules` 中的 `torchmetrics` 对象计算。

## Checkpoint

[`torcheasyrec/tzrec/utils/checkpoint_util.py`](../torcheasyrec/tzrec/utils/checkpoint_util.py)

- `CheckpointManager` 保存模型 + optimizer + dataloader 状态
- 支持 `fine_tune_checkpoint` 用于迁移学习
- `save_checkpoints_steps` / `save_checkpoints_epochs` 控制频率
- 同时处理 checkpoint 期间的模型导出

## 日志

训练指标（loss、learning rate、gradient norm）通过以下方式记录：

- **ProgressLogger**：文本进度条
- **SummaryWriter**：TensorBoard（loss、learning_rate、gradient、参数直方图）

训练指标（如 `DecayAUC`）在训练期间收集并定期记录。

## 关键文件

| 文件 | 用途 |
|------|------|
| [`torcheasyrec/tzrec/main.py`](../torcheasyrec/tzrec/main.py#L317-L530) | `_train_and_evaluate()`、`_evaluate()`、`_log_train()` |
| [`torcheasyrec/tzrec/models/model.py`](../torcheasyrec/tzrec/models/model.py#L235-L288) | `TrainWrapper.forward()` |
| [`torcheasyrec/tzrec/utils/dist_util.py`](../torcheasyrec/tzrec/utils/dist_util.py) | `create_train_pipeline()`、`DistributedModelParallel` |
| [`torcheasyrec/tzrec/modules/embedding.py`](../torcheasyrec/tzrec/modules/embedding.py#L419-L508) | `EmbeddingGroup.forward()` |
| [`torcheasyrec/tzrec/optim/optimizer_builder.py`](../torcheasyrec/tzrec/optim/optimizer_builder.py) | Optimizer 构建 |
| [`torcheasyrec/tzrec/utils/checkpoint_util.py`](../torcheasyrec/tzrec/utils/checkpoint_util.py) | Checkpoint 管理 |

## 两阶段初始化：Planner + DMP

TorchEasyRec 的分布式训练采用**两阶段初始化**来连接 TorchRec 分片——一个静态规划步骤，紧随其后一个动态执行步骤：

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

[`torcheasyrec/tzrec/utils/dist_util.py:154-183`](../torcheasyrec/tzrec/utils/dist_util.py#L154-L183)：

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

**Phase 1 (Planner)** — 一个静态遍历，检视模型拓扑、设备 mesh 和内存预算，决定每个嵌入表如何分片。输出是描述分片位置的 `ShardingPlan`。

**Phase 2 (DMP)** — 一个动态遍历，消耗 `ShardingPlan` 并将其应用到实时模块：跨 rank 分片参数张量，注入用于跨 rank 通信的 AllToAll / ReduceScatter 集合通信，将每个分片子模块包装到 `ShardedModule` 中处理 per-rank 执行。

将流程拆分为两个阶段是必要的，原因如下：

- 计划在给定模型 + 拓扑 + 预算后是确定性的，因此可以计算一次并复用（例如在训练和导出之间共享，参见 `FORCE_LOAD_SHARDING_PLAN`）。
- 执行步骤需要只有 `torch.distributed.init_process_group` 之后才存在的实时分布式上下文（进程组、NCCL 句柄）。

**Lazy Input Dist**：DMP 保持输入分布模块（`_has_uninitialized_input_dist = True`）未初始化，直到第一次前向传播。这避免了为在给定模型变体中可能永远不会被调用的稀疏模块物化通信缓冲。

## 混合精度训练

`TrainWrapper` autocast 块（见上文 `TrainWrapper.forward()`）在配置时以 FP16 或 BF16 运行稠密前向，但将稀疏嵌入查找保留在其原生 FP32 中（查找本身是 gather 而非 matmul，因此 AMP 不会带来加速，但会使 `torch.jit.script` 编译复杂化）。

对于 FP16（动态范围较小，有梯度下溢风险），`TZRecOptimizer` 用 `GradScaler` 包装组合 optimizer，该 scaler 在反向传播前将 loss 乘以 `scale_factor`（例如 65536），检查结果梯度是否有 inf/nan，并：

- 检测到 inf/nan 时：将 `scale_factor` 按 `backoff_factor`（默认 0.5）缩小并跳过该步骤。
- 连续 N 步无异常后：将 `scale_factor` 按 `growth_factor`（默认 2.0）增大。

这是标准 PyTorch AMP 模式；scaler 构造见 [`torcheasyrec/tzrec/main.py:709-714`](../torcheasyrec/tzrec/main.py#L709-L714)。

## 流水线并行

[`torcheasyrec/tzrec/utils/dist_util.py:304-345`](../torcheasyrec/tzrec/utils/dist_util.py#L304-L345) — `create_train_pipeline()`：

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

- `TrainPipelineBase` — 无分片模块模型的 vanilla pipeline。当模型没有嵌入表（罕见；pipeline 仍具有 optimizer / scaler 包装器）时使用。
- `TrainPipelineSparseDist` — TorchRec 的 pipeline，在独立 CUDA stream 上对数据加载进行流水线化处理，使下一个 batch 的主机端预处理与当前 batch 的 GPU 前向/反向重叠。仅在模型包含 `ShardedModule`（即嵌入表）时有效，因为流水线迭代需要分片数据并行输入分布。

`execute_all_batches=True` 确保所有 rank 看到所有 batch（而不是按 rank 分片 batch），这对于某些负采样器和 `check_all_workers_data_status` 调试是必需的。

## Optimizer 构建

[`torcheasyrec/tzrec/main.py:656-741`](../torcheasyrec/tzrec/main.py#L656-L741)：

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

**Optimizer 层级：**

```
TZRecOptimizer
├── CombinedOptimizer
│   ├── fused_optimizer (稀疏 embeddings)
│   ├── dense_optimizer (稠密参数)
│   └── part_optimizers (参数分组)
└── GradScaler (混合精度)
```

三层分割：

- **稀疏优化器** 通过 `apply_optimizer_in_backward` 在 backward 中应用——避免将完整嵌入梯度物化为稠密张量（稀疏 Adam 的状态为 O(num_lookups) 而非 O(table_size)）。
- **稠密优化器** 是标准的 per-parameter PyTorch 优化器。
- **冻结参数**（例如 `ManagedCollisionModule` 中的 SSTable）被绑定到 `SGD(lr=0.0)`，框架仍跟踪它们但它们实际上不更新。

## 分布式训练 — Datasets

| Dataset | 文件 | 说明 |
|---|---|---|
| `CsvDataset` | `csv_dataset.py` | CSV 文件，自定义分隔符 |
| `ParquetDataset` | `parquet_dataset.py` | Apache Parquet 列式 |
| `OdpsDataset` | `odps_dataset.py` | 阿里云 MaxCompute (ODPS) |
| `OdpsDatasetV1` | `odps_dataset_v1.py` | 旧版 ODPS Tunnel API |
| `KafkaDataset` | `kafka_dataset.py` | Kafka 流式 |
