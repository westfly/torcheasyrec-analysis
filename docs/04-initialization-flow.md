---
title: 初始化流程
nav_order: 5
---

# 初始化流程

## 入口点

典型入口为 `train_eval.py`：

[`torcheasyrec/tzrec/train_eval.py`](../torcheasyrec/tzrec/train_eval.py#L16-L72)

```python
# CLI 用法:
# python -m tzrec.train_eval --pipeline_config_path=config.yaml --model_dir=experiments/

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--pipeline_config_path", ...)
    parser.add_argument("--model_dir", ...)
    parser.add_argument("--train_input_path", ...)
    parser.add_argument("--eval_input_path", ...)
    parser.add_argument("--continue_train", action="store_true")
    parser.add_argument("--fine_tune_checkpoint", ...)
    parser.add_argument("--edit_config_json", ...)
    parser.add_argument("--ignore_restore_optimizer", ...)
    args = parser.parse_known_args()
    train_and_evaluate(...)
```

## 初始化序列

[`train_and_evaluate()`](../torcheasyrec/tzrec/main.py#L533-L616) 中的完整 init 序列：

```
1. 加载 pipeline 配置
        │
2. 解析 CLI 覆盖项 (train_input_path, model_dir, fine_tune_checkpoint, edit_config_json)
        │
3. init_process_group() → device, backend
   (torch.distributed, NCCL/GLOO)
        │
4. allow_tf32(train_config)
   (在 Ampere GPU 上启用 TF32)
        │
5. _create_features(feature_configs, data_config) → List[BaseFeature]
   (构建特征对象，检测负采样字段，初始化 fg handler)
        │
6. create_dataloader(data_config, features, train_input_path, TRAIN) → DataLoader
   (创建 BaseDataset → 包装为 DataLoader)
        │
7. create_dataloader(data_config, features, eval_input_path, EVAL) → DataLoader
        │
8. CheckpointManager(model_dir, keep_checkpoint_max)
   (初始化 checkpoint 跟踪)
        │
9. 确定 ckpt_path (fine_tune_checkpoint 或现有 model_dir)
        │
10. _create_model(model_config, features, labels, sampler_type) → BaseModel
    (按名称实例化模型类，设置 kernel)
        │
11. 包装为 TrainWrapper → DMP (DistributedModelParallel)
    (autocast, sharding)
        │
12. 构建 optimizer + LR scheduler
    (create_optimizer / TZRecOptimizer)
        │
13. _train_and_evaluate(model, optimizer, dataloaders, ...)
    (主训练循环)
```

## 详细步骤

### Step 3：`init_process_group()`

设置 torch.distributed。关键文件：[`torcheasyrec/tzrec/utils/dist_util.py`](../torcheasyrec/tzrec/utils/dist_util.py)。

- 从环境变量检测 world size（RANK、WORLD_SIZE、LOCAL_RANK、MASTER_ADDR）
- 使用 NCCL（GPU）或 GLOO（CPU）初始化默认进程组
- 返回 device 与 backend 字符串

### Step 5：`_create_features()`

[`torcheasyrec/tzrec/main.py`](../torcheasyrec/tzrec/main.py#L94-L112)

```python
def _create_features(feature_configs, data_config):
    neg_fields = None
    if data_config.HasField("sampler"):
        sampler_type = data_config.WhichOneof("sampler")
        if sampler_type != "tdm_sampler":
            neg_fields = list(getattr(data_config, sampler_type).attr_fields)
    features = create_features(
        feature_configs,
        fg_mode=data_config.fg_mode,
        neg_fields=neg_fields,
        fg_encoded_multival_sep=data_config.fg_encoded_multival_sep,
        force_base_data_group=data_config.force_base_data_group,
    )
    return features
```

`create_features()` 遍历 `FeatureConfig` proto，创建 `BaseFeature` 子类实例，标记负采样特征，并检测 DAG 特征以确定 user/item 侧。

### Step 6：`create_dataloader()`

[`torcheasyrec/tzrec/datasets/dataset.py`](../torcheasyrec/tzrec/datasets/dataset.py)

- 创建相应的 `BaseDataset` 子类（`CSVDataset`、`ParquetDataset`、`ODPSDataset`、`KafkaDataset`）
- 用可配置的 num_workers、batch_size、prefetch 包装为 `DataLoader`
- Dataset 包含一个 `DataParser`，将原始 pyarrow 数据转换为 `Batch` 对象

### Step 10：`_create_model()`

[`torcheasyrec/tzrec/main.py`](../torcheasyrec/tzrec/main.py#L127-L159)

```python
def _create_model(model_config, features, labels, sampler_type):
    model_cls_name = config_util.which_msg(model_config, "model")
    model_cls = BaseModel.create_class(model_cls_name)
    model = model_cls(model_config, features, labels, sampler_type=sampler_type)
    kernel = Kernel[KernelProto.Name(model_config.kernel)]
    model.set_kernel(kernel)
    return model
```

模型类通过 [`model.py`](../torcheasyrec/tzrec/models/model.py#L37-L38) 中的 `get_register_class_meta()` 自动注册。`config_util.which_msg()` 提取 oneof 字段名（如 `"deepfm"`、`"dssm"`），它通过自动注册装饰器映射到类名。

### Step 11：DistributedModelParallel

[`torcheasyrec/tzrec/utils/dist_util.py`](../torcheasyrec/tzrec/utils/dist_util.py)

- 用 TorchRec 的 `DistributedModelParallel` 包装模型
- 基于 `ParameterConstraints` 通过 `create_planner()` 创建分片计划
- 处理嵌入表分片（行分片、列分片、表分片）

## 包初始化（`__init__.py`）

[`torcheasyrec/tzrec/__init__.py`](../torcheasyrec/tzrec/__init__.py)

在任何用户代码运行前，包 init 会：

1. **禁用 ECS metadata**（避免不必要的网络调用）
2. **抑制 fbgemm 警告**（已知的 autograd kernel 缺失）
3. **设置 OMP_NUM_THREADS=1**（防止线程争用）
4. **导入 graphlearn + pyfg**（将 glog 设置到 stderr）
5. **配置 logging**（LOG_LEVEL 环境变量）
6. **设置随机种子**（TORCH_MANUAL_SEED、NUMPY_MANUAL_SEED）
7. **启用确定性算法**（USE_DETERMINISTIC_ALGORITHMS）
8. **自动导入所有类**（load_class.auto_import）
9. **注册外部文件系统**（OSS、ODPS）
10. **应用 RAM 凭证补丁**（针对阿里云）

## 关键文件

| 文件 | 用途 |
|------|------|
| [`torcheasyrec/tzrec/train_eval.py`](../torcheasyrec/tzrec/train_eval.py) | CLI 入口点 |
| [`torcheasyrec/tzrec/main.py`](../torcheasyrec/tzrec/main.py) | `train_and_evaluate()`、init 序列 |
| [`torcheasyrec/tzrec/__init__.py`](../torcheasyrec/tzrec/__init__.py) | 包级初始化 |
| [`torcheasyrec/tzrec/utils/dist_util.py`](../torcheasyrec/tzrec/utils/dist_util.py) | `init_process_group()`、`DistributedModelParallel` |
| [`torcheasyrec/tzrec/utils/config_util.py`](../torcheasyrec/tzrec/utils/config_util.py) | 配置加载、`which_msg()` |
| [`torcheasyrec/tzrec/utils/plan_util.py`](../torcheasyrec/tzrec/utils/plan_util.py) | TorchRec 分片规划器 |
