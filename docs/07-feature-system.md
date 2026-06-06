---
title: 特征系统
nav_order: 7
---

# 特征系统

## 概览

TorchEasyRec 的特征系统处理从原始输入数据到解析张量再到嵌入查找的完整管线。它通过通用的 `BaseFeature` 接口支持 10+ 特征类型。

```
Raw Input (pyarrow table)
        │
        ▼
   Feature.parse()  ─── FG_NONE: 直接解析
        │              ─── FG_NORMAL: pyfg handler
        │              ─── FG_DAG: 基于 DAG 的 FG
        ▼
   ParsedData (SparseData / DenseData / SequenceSparseData / SequenceDenseData)
        │
        ▼
   DataParser.to_batch() → Batch(sparse_features=KJT, dense_features=KT)
        │
        ▼
   EmbeddingGroup → EmbeddingBagCollection / DenseEmbeddingCollection
        │
        ▼
   Grouped feature tensors (per feature group)
```

## BaseFeature

[`torcheasyrec/tzrec/features/feature.py`](../torcheasyrec/tzrec/features/feature.py#L380-L1149)

`BaseFeature` 是所有特征类型的抽象基类：

```python
class BaseFeature(object, metaclass=_meta_cls):
    def __init__(self, feature_config, fg_mode, fg_encoded_multival_sep,
                 is_sequence, sequence_name, ...):
        self.fg_mode = fg_mode      # FG_NONE, FG_NORMAL, FG_DAG, FG_BUCKETIZE
        self._is_sparse = None      # True = 类别型, False = 数值型
        self._is_sequence = False   # True = 序列特征
        self._is_weighted = False   # True = 加权 ID 特征
        self._data_group = BASE_DATA_GROUP  # 或 NEG_DATA_GROUP

    def parse(self, input_data, is_training=False) -> ParsedData:
        # FG 特定解析逻辑
    def fg_json(self) -> List[Dict[str, Any]]:
        # 特征生成 JSON 配置
    def _fg_json(self) -> List[Dict[str, Any]]:
        # 子类特定的 FG 配置
```

## 特征类型

[`torcheasyrec/tzrec/protos/feature.proto`](../torcheasyrec/tzrec/protos/feature.proto)

每种特征类型都有自己的 proto 配置和 Python 类：

| 特征类型 | Proto 配置 | Python 类 | 说明 |
|----------|-----------|-----------|------|
| IdFeature | `IdFeatureConfig` | `IdFeature` | 类别型（稀疏）特征，带嵌入 |
| RawFeature | `RawFeatureConfig` | `RawFeature` | 数值型（稠密）特征 |
| ComboFeature | `ComboFeatureConfig` | `ComboFeature` | 多个类别特征的交叉 |
| SequenceFeature | `SequenceFeature` | (容器) | 将特征分组为序列 |
| LookupFeature | `LookupFeatureConfig` | `LookupFeature` | 键值查找特征 |
| ExprFeature | `ExprFeatureConfig` | `ExprFeature` | 基于表达式的特征 |
| TokenizeFeature | `TokenizeFeatureConfig` | `TokenizeFeature` | 文本分词 |
| MatchFeature | `MatchFeatureConfig` | `MatchFeature` | 检索匹配特征 |
| CombineFeature | `CombineFeatureConfig` | `CombineFeature` | 特征组合 |
| BoolMaskFeature | `BoolMaskFeatureConfig` | `BoolMaskFeature` | 布尔遮罩 |
| OverlapFeature | `OverlapFeatureConfig` | `OverlapFeature` | 集合重叠特征 |
| KV Dot Product | (在 config 中) | `KVDotProduct` | 键值点积 |

### IdFeature（类别型）

最常见的特征类型。每个唯一 ID 映射到一个嵌入向量。

```python
# 配置:
feature_config {
    id_feature {
        feature_name: "user_id"
        embedding_dim: 16
        hash_bucket_size: 100000
    }
}
```

关键属性：
- `num_embeddings`：来自 `hash_bucket_size`、`vocab_list` 或 `vocab_dict`
- `embedding_dim`：输出嵌入维度
- `pooling`：SUM 或 MEAN（针对多值特征）
- 可选 ZCH（Zero-Collision Hash）用于动态嵌入表

### RawFeature（数值型）

用于稠密数值：

```python
# 配置:
feature_config {
    raw_feature {
        feature_name: "price"
        value_dim: 1      # 标量，或 >1 表示向量
        boundaries: [0, 10, 100, 1000]  # 可选分桶
    }
}
```

RawFeature 可被：
- 直接用作稠密输入
- 通过 `boundaries` 分桶为类别 ID
- 通过 `AutoDisEmbedding`（自动离散化）或 `MLPEmbedding` 嵌入

### SequenceFeature

将特征分组为序列：

```python
# 配置:
feature_config {
    sequence_feature {
        sequence_name: "click_history"
        sequence_delim: ";"
        sequence_length: 50
        features: [
            { id_feature { feature_name: "item_id" ... } },
            { id_feature { feature_name: "category_id" ... } }
        ]
    }
}
```

序列特征由 `SequenceEmbeddingGroupImpl` 自动分割为 query（标量）和 sequence（列表）部分。

## 特征生成（FG）模式

TorchEasyRec 支持四种 FG 模式：

### FG_NONE

输入数据**已预编码**。特征直接从 pyarrow 数组解析：
- Sparse：`"1,2,3"` → `SparseData(values=[1,2,3], lengths=[3])`
- Dense：`"0.5,1.2"` → `DenseData(values=[[0.5, 1.2]])`

### FG_NORMAL

使用 `pyfg`（阿里巴巴的特征生成库）处理特征。每个特征的 `_fg_json()` 返回其 FG 配置，`pyfg.FgArrowHandler` 处理原始数据。

### FG_DAG

基于 DAG 的特征生成。特征可通过 `"feature:..."` 侧输入引用中间特征，形成计算 DAG。DAG 在 `create_features()` 中检测，并为所有 DAG 特征创建单个 `FgArrowHandler`。

### FG_BUCKETIZE

用于 raw 特征分桶的特殊模式。

## 特征分组

特征在模型配置中组织为**特征组**：

```protobuf
feature_groups {
    group_name: "wide"
    feature_names: ["user_id", "item_id"]
    group_type: WIDE
}
feature_groups {
    group_name: "deep"
    feature_names: ["price", "category"]
    group_type: DEEP
}
```

`EmbeddingGroup` 为每个组创建独立的嵌入查找。组可以是：
- **DEEP**：标准嵌入 + MLP
- **WIDE**：wide（线性）模型
- **SEQUENCE**：带 query/sequence 分割的序列特征
- **JAGGED_SEQUENCE**：可变长序列

## 数据类型

[`torcheasyrec/tzrec/datasets/utils.py`](../torcheasyrec/tzrec/datasets/utils.py)

| Python 类 | 说明 | 用途 |
|----------|------|------|
| `SparseData` | 多值类别型 | IdFeature |
| `DenseData` | 定维数值 | RawFeature |
| `SequenceSparseData` | 多值类别型序列 | SequenceFeature（sparse） |
| `SequenceDenseData` | 定维数值序列 | SequenceFeature（dense） |
| `Batch` | 完整批容器 | 所有模型输入 |
| `KeyedJaggedTensor` | TorchRec sparse 格式 | 嵌入查找 |
| `KeyedTensor` | TorchRec dense 格式 | 稠密特征传递 |

## Dense Embedding 类型

对于非稀疏特征，TorchEasyRec 提供专门的稠密嵌入：

[`torcheasyrec/tzrec/modules/dense_embedding_collection.py`](../torcheasyrec/tzrec/modules/dense_embedding_collection.py)

- **AutoDisEmbedding**：自动离散化 + 软嵌入
- **MLPDenseEmbedding**：基于 MLP 的稠密特征嵌入

## 特征创建管线

[`torcheasyrec/tzrec/features/feature.py`](../torcheasyrec/tzrec/features/feature.py#L1151-L1230)

```python
def create_features(feature_configs, fg_mode, neg_fields, ...):
    features = []
    for feat_config in feature_configs:
        feat_type = feat_config.WhichOneof("feature")
        if feat_type == "sequence_feature":
            # 将 sequence feature 展开为子特征
            for sub_feature in sequence_feature.features:
                features.append(BaseFeature.create_sub_feature(...))
        else:
            features.append(BaseFeature.create(feat_type, feat_config, ...))

    # 检测 DAG 特征并确定 user/item 侧
    if has_dag:
        fg_handler = pyfg.FgArrowHandler(fg_json, 1)
        user_feats = fg_handler.user_features()
        for feature in features:
            feature.is_user_feat = feature.name in user_feats

    return features
```

## 关键文件

| 文件 | 用途 |
|------|------|
| [`torcheasyrec/tzrec/features/feature.py`](../torcheasyrec/tzrec/features/feature.py) | `BaseFeature`、`create_features()`、解析函数 |
| [`torcheasyrec/tzrec/features/id_feature.py`](../torcheasyrec/tzrec/features/id_feature.py) | IdFeature 实现 |
| [`torcheasyrec/tzrec/features/raw_feature.py`](../torcheasyrec/tzrec/features/raw_feature.py) | RawFeature 实现 |
| [`torcheasyrec/tzrec/protos/feature.proto`](../torcheasyrec/tzrec/protos/feature.proto) | 所有特征 proto 定义 |
| [`torcheasyrec/tzrec/datasets/utils.py`](../torcheasyrec/tzrec/datasets/utils.py) | 数据类型（SparseData、Batch 等） |
| [`torcheasyrec/tzrec/datasets/data_parser.py`](../torcheasyrec/tzrec/datasets/data_parser.py) | DataParser — 原始 → Batch |
| [`torcheasyrec/tzrec/modules/dense_embedding_collection.py`](../torcheasyrec/tzrec/modules/dense_embedding_collection.py) | AutoDis、MLP 稠密嵌入 |

## 附录：纯 Python FG 计划（从 pyfg 迁移）

**背景：** 标准 `FG_NORMAL` / `FG_DAG` / `FG_BUCKETIZE` 模式依赖于 **pyfg**——阿里巴巴闭源的 `libfg.so`（833 MB 二进制）。这种依赖在以下场景中存在问题：

- 在线服务（无法在所有地方分发 `libfg.so`）
- 开源部署（需要宽松许可的 FG）
- pyfg 不支持的一些高级特性

计划的解决方案是一个**纯 Python FG** 实现，在无二进制依赖的前提下覆盖 ~80% 的用例。

### 覆盖计划

| 层级 | 特征类型 | 核心逻辑 | 优先级 |
|------|---------|---------|--------|
| 核心 (P0) | `id_feature` | 哈希分桶 / vocab 查找 | P0 |
| 核心 (P0) | `raw_feature` | 归一化（log10 / zscore / minmax） | P0 |
| 核心 (P0) | `combo_feature` | 笛卡尔积 | P0 |
| 核心 (P1) | `lookup_feature` | KV map 查找 | P1 |
| 核心 (P1) | `expr_feature` | 表达式解析 (`eval`) | P1 |
| 高级 (P2) | `match_feature` | 嵌套 dict 匹配 | P2 |
| 高级 (P2) | `overlap_feature` | 字符串相似度 | P2 |
| 高级 (P2) | `sequence_feature` | 序列解析 + 组合 | P2 |
| 高级 (P2) | `custom_feature` | 用户自定义 op | P2 |

### 架构

```
+-------------------------------------------------------------+
|              PurePythonFG (纯 Python FG)                     |
+-------------------------------------------------------------+
|                                                              |
|  +---------------------+    +---------------------+            |
|  |  FG Config          -->|  PyArrowHandler     |            |
|  |  (JSON/Proto)       |    |                    |            |
|  +---------------------+    +---------+-----------+            |
|                                      |                    |
|  +-----------------------------------+--------------------+  |
|  |              Feature Processors       |                    |
|  +-----------------------------------+--------------------+  |
|  | IdFeatureProcessor  | RawFeatureProcessor       |  |
|  | ComboFeatureProc |  LookupFeatureProcessor   |  |
|  | ExprFeatureProc  |  MatchFeatureProcessor   |  |
|  | ...                                            |  |
|  +-----------------------------------+----------------+  |
|                                      |                    |
|                                      v                    |
|  +-------------------------------------------------+        |
|  |         ParsedData (输出)                            |        |
|  |  SparseData / DenseData / SequenceSparseData    |        |
|  +-------------------------------------------------+        |
+-------------------------------------------------------------+
```

### 新 FG 模式

```protobuf
// tzrec/protos/data.proto
enum FgMode {
    FG_NONE = 0;
    FG_NORMAL = 1;
    FG_DAG = 2;
    FG_BUCKETIZE = 3;
    FG_PURE_PYTHON = 4;  // 新增
}
```

### DataParser 集成

```python
# tzrec/datasets/data_parser.py
if self._fg_mode == FgMode.FG_PURE_PYTHON:
    from tzrec.features.pure_python_fg import PurePythonFGHandler
    self._fg_handler = PurePythonFGHandler(fg_json)
```

### 风险与缓解

| 风险 | 影响 | 缓解 |
|------|------|------|
| 表达式安全性 | `expr_feature` 可能执行任意代码 | 沙箱化 `eval` |
| 性能 | Python 可能比 C++ 慢 | PyArrow 向量化 + 多线程 |
| 兼容性 | 可能与 pyfg 行为不同 | 针对 pyfg 的回归测试 |

### 成功标准

- 覆盖 80% 真实场景
- 与 `FG_NONE`（不应用 FG）行为 100% 一致
- 性能回退 < 20%
- 零 pyfg 依赖
