---
title: 离线预测/评估入口
parent: 推理篇
nav_order: 4
---

# 离线预测/评估入口

## 评估入口：`tzrec.eval`

### CLI 用法

```bash
torchrun --nnodes=1 --nproc-per-node=2 -m tzrec.eval \
  --pipeline_config_path my_model.config \
  --checkpoint_path experiments/my_model/ckpt-1000/ \
  --eval_input_path data/eval/*.parquet
```

### Proto `EvalConfig`

```protobuf
message EvalConfig {
    optional int32 num_steps = 1;
    optional int32 log_step_count_steps = 2;
}
```

- `num_steps`: 限制评估步数（一般不设，表示全量评估）
- `log_step_count_steps`: 日志输出间隔

### 内部流程（`main.py:781-878`）

```
evaluate()
  ├── load_pipeline_config()
  ├── init_process_group()          # 分布式初始化
  ├── create_dataloader(mode=EVAL)  # 构建评估 DataLoader
  ├── _create_model()               # 构建模型
  ├── TrainWrapper                  # 包装训练组件（loss/ metric）
  ├── create_planner()              # 构建 sharding plan
  ├── DistributedModelParallel()    # 分布式包装
  ├── ckpt_manager.restore()        # 加载 checkpoint
  ├── _evaluate()                   # 循环评估
  │     ├── for batch in dataloader
  │     ├──   model(batch) → predictions
  │     ├──   update_metric(predictions)
  │     └── compute_metric() → 写入 eval_result.txt
  └── summary_writer (rank 0 only)  # TensorBoard 日志
```

评估与训练共享同一套数据 pipeline 和模型定义，但**跳过优化器初始化**、**跳过采样器训练态配置**。

### 分布式支持

- 评估使用 `init_process_group()`，支持多 GPU 分布式评估
- 分布式指标计算（`dist.all_reduce`）确保跨 rank 的 AUC 精确汇总

## 预测入口：`tzrec.predict`

### CLI 用法

两种模式：

**模式 1：从导出模型预测（冻结逻辑，无需代码依赖）**

```bash
torchrun --nnodes=1 --nproc-per-node=1 -m tzrec.predict \
  --scripted_model_path experiments/my_model/export/ \
  --predict_input_path data/test/*.parquet \
  --predict_output_path data/predict_result/ \
  --output_columns "probs,probs_neg"
```

**模式 2：从 checkpoint 直接预测**

```bash
torchrun --nnodes=1 --nproc-per-node=1 -m tzrec.predict \
  --pipeline_config_path my_model.config \
  --checkpoint_path experiments/my_model/ckpt-1000/ \
  --predict_input_path data/test/*.parquet \
  --predict_output_path data/predict_result/
```

### CLI 标志

| 标志 | 默认 | 说明 |
|------|------|------|
| `--scripted_model_path` | None | 导出模型路径（模式 1） |
| `--pipeline_config_path` | None | 配置文件（模式 2） |
| `--checkpoint_path` | None | checkpoint 路径（模式 2） |
| `--predict_input_path` | — | 预测数据输入路径 |
| `--predict_output_path` | — | 预测结果输出路径 |
| `--reserved_columns` | None | 输入中需要保留在输出中的列 |
| `--output_columns` | None | 模型输出的指定列 |
| `--batch_size` | 配置值 | 预测 batch size |
| `--predict_threads` | num_workers | 预测线程数 |
| `--dataset_type` | 配置值 | 数据集类型（如 ParquetDataset） |
| `--writer_type` | 同 dataset_type | 输出写入类型 |
| `--predict_steps` | None | 限制预测步数 |
| `--is_profiling` | false | 性能分析 |

### 内部流程

`predict()`（`main.py:1038-?`）：

```
predict()
  ├── init_process_group()
  ├── 加载 scripted model 或 checkpoint
  ├── create_dataloader(mode=PREDICT)
  ├── for batch in dataloader
  │     model(batch) → predictions
  │     _write_predictions(writer, predictions, ...)
  └── writer.close()
```

`predict_checkpoint()`（`main.py:1276-?`）是 predict 的变体，直接从 checkpoint 加载模型，支持导出前的预测验证。

### 预测结果写入

```python
# main.py:1000-1035 — _write_predictions()
def _write_predictions(writer, predictions, reserves, output_cols):
    for c in output_cols:
        v = predictions[c]
        output_dict[c] = pa.array(v)
    reserve_batch_record = reserves.get()
    # 合并保留的输入列
    writer.write(output_dict)
```

通过 `BaseWriter` 子类输出到不同目标：

| writer_type | 输出格式 | 适用 |
|------------|---------|------|
| `ParquetWriter` | Parquet 文件 | 大规模预测|
| `CsvWriter` | CSV 文件 | 调试/小规模 |
| `OdpsWriter` | ODPS 表 | 阿里云环境 |

### 限制

1. **INPUT_TILE 不兼容**：`INPUT_TILE=2/3` 模式下 predict 支持有限
2. **DynamicEmb 不支持**：scripted model 路径不支持 DynamicEmb（需要 RTP 路径）
3. **分布式**：predict 支持分布式但通常建议 `nproc-per-node=1`（预测通常不依赖模型并行）
