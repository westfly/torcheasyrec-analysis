---
title: Checkpoint 与 Export 产物结构
parent: 导出与 RTP 管线
nav_order: 2
---

# Checkpoint 与 Export 产物结构

## Checkpoint 结构

训练过程中保存的检查点：

```
model_dir/
├── model.ckpt-0/
│   ├── metadata.json
│   ├── model-0-of-4.pt          # 模型参数分片（按 rank）
│   ├── model-1-of-4.pt
│   ├── model-2-of-4.pt
│   ├── model-3-of-4.pt
│   ├── optimizer-0-of-4.pt      # 优化器状态分片（按 rank）
│   ├── optimizer-1-of-4.pt
│   ├── optimizer-2-of-4.pt
│   └── optimizer-3-of-4.pt
├── model.ckpt-1000/
└── model.ckpt-2000/
```

### metadata.json 示例

```json
{
    "model": ["model.0", "model.1", "model.2", "model.3"],
    "optimizer": ["optimizer.0", "optimizer.1", "optimizer.2", "optimizer.3"],
    "step": 1000,
    "feature_configs_hash": "abc123"
}
```

## Default 导出

```
export_default/
├── fg.json                      # 特征配置（FG JSON）
├── gm.code                      # GraphModule Python 代码
├── scripted_model.pt           # TorchScript 推理模型
├── pipeline.config             # 训练配置快照
└── model_acc.json              # ACC 配置
```

## RTP 导出

### 顶层结构

```
export_rtp/
├── fg.json                              # RTP 版特征配置
├── model-000000-of-000001.safetensors  # 稀疏权重 + dynamicemb 合并容器
├── model-000000-of-000001.json          # 元信息 + hashmap 键值映射
├── graph/
│   ├── gm_full.graph                   # 全图
│   ├── gm_sparse.graph                 # 稀疏子图
│   └── gm_dense.graph                  # 稠密子图
└── fx_user_model/                       # RTP 稠密图导出包
    ├── exported_model.pt               # ExportedProgram
    ├── graph.txt                       # FX 图文本
    ├── input_columns.json
    ├── inputs_dict.json / .pkl
    ├── inputs_dict_lite.json
    ├── output_info.json
    ├── output_base.pkl
    ├── params_order.json
    └── user_model/
        ├── graph_model.pt
        ├── state_dict.pt
        ├── module.py
        └── __init__.py
```

### fx_user_model/exported_model.pt 结构

`exported_model.pt` 是 `torch.export` 的 ExportedProgram 产物（zip 包）：

```
exported_model.pt/
├── models/
│   └── model.json
├── data/
│   ├── weights/
│   │   ├── model_weights_config.json
│   │   └── weight_0 ~ weight_n
│   ├── constants/
│   │   └── model_constants_config.json
│   └── sample_inputs/
│       └── model.pt
```

## safetensors 格式

RTP 导出下的 `model-*.safetensors` 与同名 `model-*.json`。

### 写入入口

- 写入函数：`safetensors.torch.save_file(...)`（[`export_util.py:779`](../torcheasyrec/tzrec/utils/export_util.py#L779)）
- 数据构造：`_get_rtp_embedding_tensor(...)`（[`export_util.py:363`](../torcheasyrec/tzrec/utils/export_util.py#L363)）

两类 tensor 合并写入同一文件：

1. 静态 embedding（来自 `model.state_dict()`）
2. dynamicemb keys/values（来自 `checkpoint_path/dynamicemb`）

### 命名与分片

- `model-{rank:06d}-of-{world_size:06d}.safetensors`
- `model-{rank:06d}-of-{world_size:06d}.json`

tensor 命名：

- `ShardedTensor` 追加 `part_{idx}_{num_shards}`
- dynamicemb 采用 `<table>.<emb>.keys/part_{idx}_{num_shards}`、`<table>.<emb>.values/part_{idx}_{num_shards}`

### meta.json 字段

```json
{
    "name": "user_id_emb",
    "dense": false,
    "dimension": 16,
    "dtype": "float32",
    "memory": 18267680,
    "shape": [1141730, 16],
    "is_hashmap": false
}
```

DynamicEmb 额外字段：`hashmap_key`、`hashmap_value`、`hashmap_key_dtype`（固定 `int64`）。

## Default vs RTP 对比

| 特性 | Default | RTP |
|------|---------|-----|
| **模型载体** | `scripted_model.pt` | `fx_user_model/` + `safetensors` |
| **稀疏权重** | 不单独导出 | 独立导出 `model-*.safetensors` |
| **配置快照** | `pipeline.config` + `model_acc.json` | 仅 `fg.json` |
| **图文件** | `gm.code` | `gm_full/gm_sparse/gm_dense` |
| **适用场景** | 通用推理 | 在线实时推理 |


