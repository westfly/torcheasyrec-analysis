---
title: 数据管线：Dataset 体系
parent: 训练篇
nav_order: 12
---

# 数据管线：Dataset 体系

## 架构总览

数据管线的核心是 `BaseDataset`（`datasets/dataset.py:87`） — 一个 `IterableDataset` 子类，通过工厂模式按配置创建具体 Dataset 实例。

```
DataLoader
  └── BaseDataset (IterableDataset)
        ├── CsvDataset         — CSV 文件读取
        ├── ParquetDataset     — Parquet 文件读取
        ├── OdpsDataset        — ODPS/MaxCompute 表（Storage API v1）
        ├── OdpsDatasetV1      — ODPS 表（传统 tunnel）
        └── KafkaDataset       — Kafka/DataHub 流式数据
              └── Reader (BaseReader)
                    ├── CsvReader
                    ├── ParquetReader
                    ├── OdpsReader
                    └── KafkaReader
```

## BaseDataset 抽象层

`BaseDataset.__init__()`（`dataset.py:100-198`）负责：

- **DataParser 构建**: 每个 Dataset 封装一个 `DataParser` 实例，将 RecordBatch 中的原始数据解析为特征张量
- **输入字段选择**: 根据 `selected_input_names` 过滤 reader schema 中的列，支持 `ALL_COLUMNS` 保留所有列
- **Sampler 集成**: 如果配置了负采样，初始化 `BaseSampler` 实例
- **Batch size 管理**: 训练用 `data_config.batch_size`，评估用 `data_config.eval_batch_size`
- **FG 模式**: 存储 `fg_mode` 和 `fg_encoded_multival_sep` 配置

### 核心方法

| 方法 | 行号 | 功能 |
|------|------|------|
| `launch_sampler_cluster()` | 200-246 | 启动 graphlearn 采样集群（仅负采样模式下） |
| `get_worker_info()` | 273-290 | 返回 (worker_id, num_workers)，考虑分布式多进程 |
| `load_state_dict()` | 292-299 | 从 checkpoint 恢复 reader 的消费位置 |
| `_init_input_fields()` | 253-264 | 从 reader schema 推断输入字段类型 |

### 数据迭代

每个 Dataset 子类通过 `BaseReader` 读取原始数据，yield `pa.RecordBatch`。`__iter__()` 方法由子类实现，基本模式：

1. Reader 从数据源读取 RecordBatch
2. DataParser 将 RecordBatch 解析为 `Batch`（特征张量 + 标签）
3. 如果配置了负采样，Sampler 在解析后的 batch 上执行负采样
4. 产出的 `Batch` 进入训练/评估 pipeline

## DatasetScanner（`dataset.py:327-？`）

用于自动发现数据目录中的文件，支持以下模式：

- 通配符路径 (e.g. `/data/train/*.parquet`)
- 目录扫描
- OSS 等远程存储的 glob 语义

## 各 Dataset 详解

### CsvDataset（~200 行）

```python
# tzrec/datasets/csv_dataset.py
class CsvDataset(BaseDataset):
    reader_cls = CsvReader
```

- **输入格式**: 文本 CSV，支持自定义分隔符、表头
- **配置字段**: `delimiter`、`with_header`、`input_fields`
- **适用**: 小规模测试数据、开发调试

### ParquetDataset（~300 行）

```python
# tzrec/datasets/parquet_dataset.py
class ParquetDataset(BaseDataset):
    reader_cls = ParquetReader
```

- **输入格式**: Apache Parquet 列存文件
- **分片**: 支持 `input_path` 通配符自动分片。每个 worker 负责一个文件分片，避免重复读取
- **字段推断**: 自动读取 Parquet schema 推断字段类型
- **适用**: 推荐系统标准训练/评估数据

### OdpsDataset（~400 行）

```python
# tzrec/datasets/odps_dataset.py
class OdpsDataset(BaseDataset):
    reader_cls = OdpsReader
```

- **输入格式**: 阿里云 MaxCompute（原 ODPS）表
- **认证**: 通过 `ODPS_ENDPOINT`、`ODPS_CONFIG_FILE_PATH` 环境变量配置
- **读取方式**: 使用 ODPS Storage API（SplitReader），支持大规模表的分片读取
- **适用**: 阿里云内部生产环境

### OdpsDatasetV1（~200 行）

```python
# tzrec/datasets/odps_dataset_v1.py
class OdpsDatasetV1(BaseDataset):
    reader_cls = OdpsReader
```

- 与 OdpsDataset 基本相同，但使用传统 ODPS Tunnel SDK

### KafkaDataset（~200 行）

```python
# tzrec/datasets/kafka_dataset.py
class KafkaDataset(BaseDataset):
    reader_cls = KafkaReader
```

- **输入格式**: Kafka / DataHub 流式数据
- **序列化**: 支持 Arrow IPC 和 schema-less 两种格式
- **Flink 集成**: 通过 Flink ArrowBatchUDTF 写入
- **适用**: 在线学习、实时特征更新

## Reader 与 Writer 基类

```python
# tzrec/datasets/dataset.py:46-48
_DATASET_CLASS_MAP = {}
_READER_CLASS_MAP = {}
_WRITER_CLASS_MAP = {}
```

Reader 和 Writer 同样使用元类注册模式：

- **Reader**: 实现特定数据源的读取逻辑，返回 `pa.RecordBatch`
- **Writer**: 实现预测结果输出，支持 CSV / Parquet / ODPS 等目标

Reader 关键能力：

| 能力 | 方法 | 说明 |
|------|------|------|
| Schema 推断 | `read_schema()` | 从数据源自动推断列类型 |
| 分片读取 | `_iter()` | 支持多 worker 并行读取 |
| Checkpoint | `load_state_dict()` | 保存/恢复读取位置，支持断点续训 |
| 数据过滤 | `_filter_input_fields()` | 只读取需要的列 |

## Batch 数据结构（`datasets/utils.py:923`）

```python
@dataclass
class Batch:
    # 基础数据组
    dense: Dict[str, torch.Tensor]     # 稠密特征（DenseData）
    sparse: KeyedJaggedTensor          # 稀疏特征（SparseData）
    sequence_dense: KeyedTensor        # 稠密序列特征
    sequence_sparse: KeyedJaggedTensor # 稀疏序列特征（SequenceSparseData）

    # 负采样数据组
    neg_dense: Optional[Dict[str, torch.Tensor]] = None
    neg_sparse: Optional[KeyedJaggedTensor] = None
    neg_sequence_sparse: Optional[KeyedJaggedTensor] = None

    # 交叉负采样数据组
    cross_neg_dense: Optional[Dict[str, torch.Tensor]] = None
    cross_neg_sparse: Optional[KeyedJaggedTensor] = None
    cross_neg_sequence_sparse: Optional[KeyedJaggedTensor] = None

    # 标签和样本权重
    labels: Dict[str, torch.Tensor] = field(default_factory=dict)
    sample_weights: Dict[str, torch.Tensor] = field(default_factory=dict)

    # 元数据（用于断点续训）
    checkpoint_meta: Optional[Dict[str, torch.Tensor]] = None
```

Batch 实现了 `Pipelineable` 接口，可在 `TrainPipeline` 的 GPU pipeline prefetch 中使用。

### 数据组常量

| 常量 | 值 | 用途 |
|------|-----|------|
| `BASE_DATA_GROUP` | `"__BASE__"` | 正样本/基础数据 |
| `NEG_DATA_GROUP` | `"__NEG__"` | 负采样数据 |
| `CROSS_NEG_DATA_GROUP` | `"__CNEG__"` | 交叉负采样数据 |

## 配置示例

```protobuf
data_config {
    input_fields: [
        {field_name: "user_id", field_type: STRING},
        {field_name: "item_id", field_type: STRING},
        {field_name: "price", field_type: FLOAT},
    ]
    label_fields: ["click"]
    batch_size: 4096
    num_workers: 4
    dataset_type: "ParquetDataset"
    fg_mode: FG_NONE
}
```

- `dataset_type` 指定使用哪种 Dataset 类（字符串 → `BaseDataset.create_class()`）
- `input_fields` 定义输入 schema（在 `FG_NONE` 模式下必须显式配置）
- `fg_mode` 控制特征生成模式：`FG_DAG`（生产）/ `FG_NORMAL` / `FG_NONE`（测试）
- `num_workers` 控制 DataLoader 的多进程 worker 数量
