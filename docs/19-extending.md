---
title: 扩展机制与自定义模型开发
parent: 训练篇
nav_order: 11
---

# 扩展机制与自定义模型开发

## 两大扩展机制

TorchEasyRec 有两种不同的扩展机制：

### 1. 元类自动注册（Model / Feature / Dataset / Sampler / LR Scheduler / SeqEncoder）

```python
# tzrec/utils/load_class.py:117-145
_MODEL_CLASS_MAP = {}
_meta_cls = get_register_class_meta(_MODEL_CLASS_MAP)

class BaseModel(BaseModule, metaclass=_meta_cls):
    ...
```

机制原理：

- `get_register_class_meta(class_map)` 创建一个元类 `RegisterABCMeta`
- 任何继承该元类基类的子类，在类定义时自动执行 `register_class(class_map, name, newclass)`（`load_class.py:130`）
- 注册的 key 是 **Python 类名**（字符串），存储在全局 `_CLASS_MAP` 字典中
- 基类自动获得 `create_class(name)` 类方法，用于按名查找（`load_class.py:132-141`）
- `auto_import()`（`load_class.py:53-99`）在包初始化时扫描 `tzrec/models/`、`tzrec/datasets/`、`tzrec/features/` 目录，导入所有非 `_test` 的 Python 文件，触发类定义和注册

```python
# tzrec/__init__.py:74 — 包初始化时自动导入
_load_class.auto_import()
```

支持 `user_path` 参数导入用户自定义代码目录：

```python
# tzrec/__init__.py:76-80
if hasattr(config, 'user_define_path') and config.user_define_path:
    _load_class.auto_import(config.user_define_path)
```

### 2. 硬编码 if-elif 链（Loss / Optimizer）

Loss 和 Optimizer 不使用注册表，而是通过 **proto `oneof` 字段名**在 if-elif 链中分发：

```python
# tzrec/models/rank_model.py:181-211
loss_type = loss_cfg.WhichOneof("loss")
if loss_type == "binary_cross_entropy":
    ...
elif loss_type == "binary_focal_loss":
    ...
```

```python
# tzrec/optim/optimizer_builder.py:30-97
optimizer_type = optimizer_cfg.WhichOneof("optimizer")
if optimizer_type == "sgd_optimizer":
    ...
elif optimizer_type == "adam_optimizer":
    ...
```

原因：Loss 需要同时修改 `_init_loss_impl()` 初始化、`_loss_impl()` 前向计算、`_output_to_prediction_impl()` 输出映射三处逻辑，无法通过纯继承 + 注册完成。

## 完整扩展步骤

### 添加新模型（Model）

| 步骤 | 修改位置 | 说明 |
|------|---------|------|
| 1 | `protos/models/rank_model.proto` 或自有 `.proto` | 添加模型配置消息，如 `message MyModel { ... }` |
| 2 | `protos/model.proto` oneof model | 添加 `MyModel my_model = N;` 字段 |
| 3 | 运行 `scripts/gen_proto.sh` | 重新生成 `model_pb2.py` |
| 4 | `tzrec/models/my_model.py` | 继承 `RankModel`（排序）/ `MatchModel`（召回）/ `MultiTaskRank`（多任务） |
| 5 | 实现 `__init__()` | 解析 `model_config`，构建 feature groups、MLP towers |
| 6 | 实现 `predict()` | 定义前向计算逻辑 |

```python
# tzrec/models/my_model.py
from tzrec.models.rank_model import RankModel

class MyModel(RankModel):
    def __init__(self, model_config, features, labels, ...):
        super().__init__(model_config, features, labels, ...)
        # 构建网络层
        self.my_mlp = MLP(...)

    def predict(self, batch):
        # 前向计算
        sparse, dense = self._parse_features(batch)
        output = self.my_mlp(dense)
        return {"probs": output}
```

**无需手动注册** — 文件放在 `tzrec/models/` 目录下，`auto_import()` 自动发现，元类自动注册 `MyModel` 到 `_MODEL_CLASS_MAP`。

### 添加新 Loss 函数

| 步骤 | 修改位置 | 说明 |
|------|---------|------|
| 1 | `protos/loss.proto` | 添加 loss 配置消息 + `oneof loss` 新字段 |
| 2 | 运行 `scripts/gen_proto.sh` | |
| 3 | `tzrec/loss/my_loss.py` | 实现 `nn.Module` 子类 |
| 4 | `tzrec/models/rank_model.py` | `_init_loss_impl()` 添加 elif |
| 5 | `tzrec/models/rank_model.py` | `_loss_impl()` 添加前向分支 |
| 6 | `tzrec/models/rank_model.py` | `_output_to_prediction_impl()` 添加输出映射 |

### 添加新 Optimizer

| 步骤 | 修改位置 | 说明 |
|------|---------|------|
| 1 | `protos/optimizer.proto` | 添加 optimizer 配置消息 + `oneof optimizer` 新字段 |
| 2 | 运行 `scripts/gen_proto.sh` | |
| 3 | `tzrec/optim/optimizer_builder.py` | `create_sparse_optimizer()` 或 `create_dense_optimizer()` 添加 elif |

### 添加新 Feature

同模型，继承 `BaseFeature` 自动注册。文件放在 `tzrec/features/` 下。

### 添加新 Dataset / Reader / Writer

同模型，继承 `BaseDataset` / `BaseReader` / `BaseWriter` 自动注册（`dataset.py:46-51`）。

## 注册的组件一览

| 组件 | 基类 | 注册文件 | 注册位置 | 发现方式 |
|------|------|---------|---------|---------|
| Model | `BaseModel` | `models/model.py:37-41` | `model.py` | `auto_import()` 扫描 `models/` |
| Feature | `BaseFeature` | `features/feature.py:67-68` | `feature.py` | `auto_import()` 扫描 `features/` |
| Dataset | `BaseDataset` | `datasets/dataset.py:46-51` | `dataset.py` | `auto_import()` 扫描 `datasets/` |
| Reader | `BaseReader` | `datasets/dataset.py:47` | `dataset.py` | `auto_import()` |
| Writer | `BaseWriter` | `datasets/dataset.py:48` | `dataset.py` | `auto_import()` |
| Sampler | `BaseSampler` | `datasets/sampler.py:129-130` | `sampler.py` | 手动 import（不通过 auto_import） |
| LR Scheduler | `BaseLR` | `optim/lr_scheduler.py:22-23` | `lr_scheduler.py` | 手动 import |
| SeqEncoder | `SequenceEncoder` | `modules/sequence.py:29-30` | `sequence.py` | 手动 import |
| Loss | — | `models/rank_model.py:181-211` | 硬编码 if-elif | proto `WhichOneof()` |
| Optimizer | — | `optim/optimizer_builder.py:30-136` | 硬编码 if-elif | proto `WhichOneof()` |

## Config 到 Python 类的映射流程

```python
# tzrec/main.py:145-148 — 模型创建
model_cls_name = config_util.which_msg(model_config, "model")
model_cls = BaseModel.create_class(model_cls_name)
```

`config_util.which_msg()`（`config_util.py:73-75`）提取 proto `oneof` 中选定消息的**类名**：

```python
def which_msg(config, oneof_name):
    return getattr(config, config.WhichOneof(oneof_name)).__class__.__name__
```

例如，如果配置设定 `model { multi_tower { ... } }`，则返回 `"MultiTower"`，然后 `BaseModel.create_class("MultiTower")` 从 `_MODEL_CLASS_MAP` 中查到 `MultiTower` 类。

## Proto 生成流程

框架依赖 `protoc` 将 `.proto` → `_pb2.py`。所有 proto 文件在 `tzrec/protos/` 目录下，由 `scripts/gen_proto.sh` 批量编译：

```bash
python -m grpc_tools.protoc \
  -I tzrec/protos \
  --python_out=tzrec/protos \
  tzrec/protos/*.proto
```

修改任何 `.proto` 文件后都需要重新运行此脚本。
