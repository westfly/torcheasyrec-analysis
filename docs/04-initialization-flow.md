---
title: Initialization Flow
nav_order: 5
---

# Initialization Flow

## Entry Point

The typical entry point is `train_eval.py`:

[`torcheasyrec/tzrec/train_eval.py`](../torcheasyrec/tzrec/train_eval.py#L16-L72)

```python
# CLI usage:
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

## Initialization Sequence

The full init sequence in [`train_and_evaluate()`](../torcheasyrec/tzrec/main.py#L533-L616):

```
1. Load pipeline config
        │
2. Parse CLI overrides (train_input_path, model_dir, fine_tune_checkpoint, edit_config_json)
        │
3. init_process_group() → device, backend
   (torch.distributed, NCCL/GLOO)
        │
4. allow_tf32(train_config)
   (enable TF32 on Ampere GPUs)
        │
5. _create_features(feature_configs, data_config) → List[BaseFeature]
   (build feature objects, detect neg fields, init fg handlers)
        │
6. create_dataloader(data_config, features, train_input_path, TRAIN) → DataLoader
   (create BaseDataset → wrap in DataLoader)
        │
7. create_dataloader(data_config, features, eval_input_path, EVAL) → DataLoader
        │
8. CheckpointManager(model_dir, keep_checkpoint_max)
   (initialize checkpoint tracking)
        │
9. Determine ckpt_path (fine_tune_checkpoint or existing model_dir)
        │
10. _create_model(model_config, features, labels, sampler_type) → BaseModel
    (instantiate model class by name, set kernel)
        │
11. Wrap in TrainWrapper → DMP (DistributedModelParallel)
    (autocast, sharding)
        │
12. Build optimizer + LR scheduler
    (create_optimizer / TZRecOptimizer)
        │
13. _train_and_evaluate(model, optimizer, dataloaders, ...)
    (main training loop)
```

## Step-by-Step Detail

### Step 3: `init_process_group()`

Sets up torch.distributed. Key file: [`torcheasyrec/tzrec/utils/dist_util.py`](../torcheasyrec/tzrec/utils/dist_util.py).

- Detects world size from env vars (RANK, WORLD_SIZE, LOCAL_RANK, MASTER_ADDR)
- Initializes the default process group with NCCL (GPU) or GLOO (CPU)
- Returns device and backend strings

### Step 5: `_create_features()`

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

`create_features()` iterates `FeatureConfig` protos, creates `BaseFeature` subclass instances, marks negative-sampled features, and detects DAG features to determine user/item side.

### Step 6: `create_dataloader()`

[`torcheasyrec/tzrec/datasets/dataset.py`](../torcheasyrec/tzrec/datasets/dataset.py)

- Creates appropriate `BaseDataset` subclass (CSVDataset, ParquetDataset, ODPSDataset, KafkaDataset)
- Wraps with `DataLoader` with configurable num_workers, batch_size, prefetch
- Dataset includes a `DataParser` that converts raw pyarrow data to `Batch` objects

### Step 10: `_create_model()`

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

Model classes are auto-registered via `get_register_class_meta()` in [`model.py`](../torcheasyrec/tzrec/models/model.py#L37-L38). The `config_util.which_msg()` extracts the oneof field name (e.g., "deepfm", "dssm"), which maps to a class name via the auto-registration decorator.

### Step 11: DistributedModelParallel

[`torcheasyrec/tzrec/utils/dist_util.py`](../torcheasyrec/tzrec/utils/dist_util.py)

- Wraps the model with TorchRec's `DistributedModelParallel`
- Creates a sharding plan via `create_planner()` based on `ParameterConstraints`
- Handles embedding table sharding (row-wise, column-wise, table-wise)

## Package Init (`__init__.py`)

[`torcheasyrec/tzrec/__init__.py`](../torcheasyrec/tzrec/__init__.py)

Before any user code runs, the package init:

1. **Disables ECS metadata** (to avoid unnecessary network calls)
2. **Suppresses fbgemm warnings** (known autograd kernel missing)
3. **Sets OMP_NUM_THREADS=1** (prevents thread contention)
4. **Imports graphlearn + pyfg** (sets glog to stderr)
5. **Configures logging** (LOG_LEVEL env var)
6. **Sets random seeds** (TORCH_MANUAL_SEED, NUMPY_MANUAL_SEED)
7. **Enables deterministic algorithms** (USE_DETERMINISTIC_ALGORITHMS)
8. **Auto-imports all classes** (load_class.auto_import)
9. **Registers external filesystem** (OSS, ODPS)
10. **Applies RAM credential patch** (for Alibaba Cloud)

## Key Files

| File | Purpose |
|------|---------|
| [`torcheasyrec/tzrec/train_eval.py`](../torcheasyrec/tzrec/train_eval.py) | CLI entry point |
| [`torcheasyrec/tzrec/main.py`](../torcheasyrec/tzrec/main.py) | `train_and_evaluate()`, init sequence |
| [`torcheasyrec/tzrec/__init__.py`](../torcheasyrec/tzrec/__init__.py) | Package-level initialization |
| [`torcheasyrec/tzrec/utils/dist_util.py`](../torcheasyrec/tzrec/utils/dist_util.py) | `init_process_group()`, `DistributedModelParallel` |
| [`torcheasyrec/tzrec/utils/config_util.py`](../torcheasyrec/tzrec/utils/config_util.py) | Config loading, `which_msg()` |
| [`torcheasyrec/tzrec/utils/plan_util.py`](../torcheasyrec/tzrec/utils/plan_util.py) | TorchRec sharding planner |
