---
title: 架构总览
nav_order: 2
---

# 架构总览

## 分层

TorchEasyRec 的架构组织为六层，每层职责清晰：

```
 Pipeline Config (protobuf)
        │
        ▼
 ┌─────────────┐    ┌──────────────────┐
 │  Data Layer  │───▶│  Feature Layer   │
 │  (Dataset)   │    │  (Parse + FG)    │
 └─────────────┘    └────────┬─────────┘
                             │
                             ▼
 ┌─────────────────────────────────────┐
 │         Embedding Layer             │
 │  (EmbeddingGroup → EBC/EC + ZCH)    │
 └──────────────┬──────────────────────┘
                │
                ▼
 ┌─────────────────────────────────────┐
 │         Model Layer                 │
 │  BaseModel → RankModel/MatchModel   │
 │  → DeepFM, DSSM, MMoE, HSTU...     │
 └──────────────┬──────────────────────┘
                │
                ▼
 ┌─────────────────────────────────────┐
 │    Loss / Metric / Export Layer     │
 └─────────────────────────────────────┘
```

## 整体系统视图

```
┌────────────────────────────────────────────────────────────────────────────┐
│                              TorchEasyRec                                    │
├────────────────────────────────────────────────────────────────────────────┤
│                                                                            │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────┐                   │
│  │  Pipeline    │───▶│   Feature    │───▶│    Model     │                   │
│  │   Config     │    │   System     │    │   System     │                   │
│  │  (Protobuf)  │    │              │    │              │                   │
│  └──────────────┘    └──────────────┘    └──────────────┘                   │
│         │                   │                   │                            │
│         ▼                   ▼                   ▼                            │
│  ┌─────────────────────────────────────────────────────────────┐           │
│  │                     Data Pipeline                             │           │
│  │  CSV │ Parquet │ ODPS │ Kafka  ──▶  PyArrow  ──▶  Batch     │           │
│  └─────────────────────────────────────────────────────────────┘           │
│                              │                                              │
│                              ▼                                              │
│  ┌─────────────────────────────────────────────────────────────┐           │
│  │              Training / Evaluation / Export / Predict          │           │
│  └─────────────────────────────────────────────────────────────┘           │
│                                                                            │
└────────────────────────────────────────────────────────────────────────────┘
```

三个顶层子系统（Pipeline Config、Feature System、Model System）都是 protobuf 驱动的独立模块，共享通用的 `Batch` 数据结构。Data Pipeline 在运行时将三者连接，Training/Evaluation/Export/Predict 层驱动模型完成整个生命周期。

## 数据流（训练）

单个训练步骤遵循以下管线：

```
1. Dataset.__iter__() → 原始 RecordBatch (pyarrow)
        │
        ▼
2. DataParser.to_batch() → Batch(sparse_features=KJT,
        │                     dense_features=KT,
        │                     labels=dict, sample_weights=dict)
        │
        ▼
3. TrainWrapper.forward(batch)
        │
        ├─▶ model.predict(batch)
        │       │
        │       ├─▶ EmbeddingGroup(batch) → grouped_features dict
        │       │       │
        │       │       ├─ EmbeddingBagCollection (sparse)
        │       │       ├─ ManagedCollisionEmbeddingBagCollection (ZCH)
        │       │       ├─ DenseEmbeddingCollection (dense)
        │       │       └─ SequenceEmbeddingGroupImpl (sequence)
        │       │
        │       └─▶ 模型专属 forward (DeepFM、DSSM 等) → predictions
        │
        ├─▶ model.loss(predictions, batch) → 损失字典
        │
        └─▶ total_loss = sum(losses) → backward
```

## Pipeline Config（`EasyRecConfig`）

整个管线通过一个 protobuf 配置：

[`torcheasyrec/tzrec/protos/pipeline.proto`](../torcheasyrec/tzrec/protos/pipeline.proto#L11-L29)：

```protobuf
message EasyRecConfig {
    required string train_input_path = 1;
    required string eval_input_path = 2;
    required string model_dir = 3;
    optional TrainConfig train_config = 4;
    optional EvalConfig eval_config = 5;
    optional ExportConfig export_config = 6;
    optional DataConfig data_config = 7;
    repeated FeatureConfig feature_configs = 8;
    optional ModelConfig model_config = 9;
}
```

## Model Config（`ModelConfig`）

[`torcheasyrec/tzrec/protos/model.proto`](../torcheasyrec/tzrec/protos/model.proto#L45-L96) 将模型定义为 oneof：

```protobuf
message ModelConfig {
    repeated FeatureGroupConfig feature_groups = 1;
    oneof model {
        DLRM dlrm = 100;
        DeepFM deepfm = 101;
        MultiTower multi_tower = 102;
        ...
        DSSM dssm = 301;
        TDM tdm = 400;
    }
    optional uint32 num_class = 2;
    repeated LossConfig losses = 3;
    repeated MetricConfig metrics = 4;
}
```

## 关键架构决策

### 1. 配置驱动

模型、特征、数据源、训练、导出都通过 protobuf 配置定义。这带来：
- 零代码模型实验
- 便捷的参数扫描
- 训练与推理的配置一致性

### 2. 特征生成（FG）封装

每个 `BaseFeature` 子类自带 FG 逻辑（`_fg_json()`）。特征可处于：
- **FG_NONE**：预编码（ID 已经哈希、值已经归一化）
- **FG_NORMAL**：特征配置由 pyfg 解析
- **FG_DAG**：基于 DAG 的 FG，可引用中间特征

### 3. 嵌入分离

稀疏特征（类别型）使用 TorchRec 的 `EmbeddingBagCollection`。需要自定义嵌入的稠密特征（AutoDis、MLP）使用 `DenseEmbeddingCollection`。`EmbeddingGroup` 类协调两者，并按 data group（base、neg、user）组织特征。

### 4. 与模型无关的训练循环

[`main.py`](../torcheasyrec/tzrec/main.py#L317-L530) 中的训练循环（`_train_and_evaluate()`）与模型无关。所有模型特定逻辑集中在：
- `model.predict(batch)` → predictions 字典
- `model.loss(predictions, batch)` → losses 字典
- `model.update_metric(predictions, batch, losses)` → metric 状态

### 5. Wrapper 模式

三个 wrapper 将模型与管线关注点解耦：
- [`TrainWrapper`](../torcheasyrec/tzrec/models/model.py#L235-L288)：添加 autocast、损失聚合、Pareto MTL
- [`PredictWrapper`](../torcheasyrec/tzrec/models/model.py#L291-L340)：添加输出列过滤、CPU offload
- [`ScriptWrapper`](../torcheasyrec/tzrec/models/model.py#L343-L393)：为 JIT/torch.export 添加数据解析

## 推理架构

对于生产服务，TorchEasyRec 支持多条导出路径：

```
Training Checkpoint
        │
        ├─▶ JIT Script (TorchScript) → TensorRT
        ├─▶ torch.export → AOTInductor
        └─▶ 组合：JIT 稀疏 + AOTI 稠密
```

[`CombinedModelWrapper`](../torcheasyrec/tzrec/models/model.py#L453-L512) 将模型拆分为稀疏（嵌入，已 script）和稠密（MLP，AOTInductor）两部分以优化推理。

## 参考资料

| 文件 | 用途 |
|------|------|
| [`torcheasyrec/tzrec/main.py`](../torcheasyrec/tzrec/main.py) | `train_and_evaluate()`、`_train_and_evaluate()`、`_evaluate()` |
| [`torcheasyrec/tzrec/models/model.py`](../torcheasyrec/tzrec/models/model.py) | `BaseModel`、`TrainWrapper`、`PredictWrapper`、`ScriptWrapper` |
| [`torcheasyrec/tzrec/protos/pipeline.proto`](../torcheasyrec/tzrec/protos/pipeline.proto) | `EasyRecConfig` |
| [`torcheasyrec/tzrec/protos/model.proto`](../torcheasyrec/tzrec/protos/model.proto) | `ModelConfig`、`FeatureGroupConfig` |
| [`torcheasyrec/tzrec/utils/config_util.py`](../torcheasyrec/tzrec/utils/config_util.py) | 配置加载与编辑工具 |
