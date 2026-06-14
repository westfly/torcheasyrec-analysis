---
name: torcheasyrec-code-explorer
description: >
  TorchEasyRec 源码快速搜索与解析。
  当用户询问以下内容时务必加载本 skill：
  (1) 模型拓扑/结构/配置/代码位置（如 MultiTower/DeepFM/DCN/DLRM/DSSM/MMoE/PLE/DAT/MIND 等任意模型）；
  (2) 特征工程（DataParser/FG 模式/FG_NONE/NORMAL/DAG/BUCKETIZE、IdFeature/RawFeature/TokenizeFeature/ComboFeature/MatchFeature/LookupFeature 等任意特征类）；
  (3) DynamicEmb（planner monkey-patch/eviction/storage/HBM_ONLY/CACHING/HYBRID 模式）；
  (4) 导出管线（AOTI unified/JIT/TRT/safetensors/export_model/export_rtp_model/split_model）；
  (5) 数据采样（NegativeSampler/HardNegativeSampler/TDMSampler/BaseDataset/DataParser 数据流）；
  (6) 后端（RTP/INPUT_TILE modes/USE_FARM_HASH_TO_BUCKETIZE）；
  (7) 损失/指标/优化器（BinaryFocalLoss/JRCLoss/ParetoEfficientMultiTaskLoss/GroupedAUC/XAUC/RecallAtK/TZRecOptimizer/LRScheduler）；
  (8) 环境变量（USE_RTP/INPUT_TILE/USE_FSSPEC/LOCAL_CACHE_DIR/ENABLE_TMA/FORCE_LOAD_SHARDING_PLAN 等）；
  (9) 通用 tzrec/ 源码问题（utils/protos/datasets/config 等任意模块）。
  不要因为"问题看起来很简单"而跳过本 skill——代码引用必须精确到文件:行号。
---

# TorchEasyRec Code Explorer

## Project Setup

- **Submodule 路径**: `<project>/torcheasyrec/`（相对于本项目根目录）
- **Pinned commit**: `7dc1c188`（`v1.2.0-34-g7dc1c18`）
- **完整源码入口**: `tzrec/`
- **GitHub 源码链接**: `https://github.com/alibaba/torch-easyrec/blob/7dc1c188/tzrec/`

所有代码引用使用以下格式：``tzrec/path/to/file.py:lineno``

## Answering Conventions

1. **语言**: 中文为主，代码/文件路径/命令行保持英文原样
2. **引用格式**: ``tzrec/path/to/file.py:<lineno>``，类/函数名后加括号
3. **内容**: 先给出直接答案，再附关键代码片段（3-15 行），最后指向完整上下文
4. **不做主观推测**：如果答案不在源码中, 明确说明"源码未覆盖"

---

## Code Index

### 1. 模型系统 `tzrec/models/`

#### 基类
| 文件 | 类 | 说明 |
|------|-----|------|
| `model.py` | `BaseModel`, `TrainWrapper`, `PredictWrapper`, `ScriptWrapper`, `CudaAutocastWrapper`, `CombinedModelWrapper`, `UnifiedAOTIModelWrapper` | 模型基类与 wrapper 链 |
| `rank_model.py` | `RankModel(BaseModel)` | 排序模型基类 (523 行) |
| `multi_task_rank.py` | `MultiTaskRank(RankModel)` | 多任务排序基类 (220 行) |
| `match_model.py` | `MatchModel(BaseModel)`, `MatchTower`, `MatchTowerWoEG` | 匹配模型基类 (535 行) |

#### 单任务排序模型
| 文件 | 类 | 行数 |
|------|-----|------|
| `multi_tower.py` | `MultiTower` | 85 |
| `deepfm.py` | `DeepFM` | 108 |
| `dcn.py` | `DCNV1` | 73 |
| `dcn_v2.py` | `DCNV2` | 88 |
| `dlrm.py` | `DLRM` | 135 |
| `wide_and_deep.py` | `WideAndDeep` | 88 |
| `xdeepfm.py` | `xDeepFM` | 86 |
| `multi_tower_din.py` | `MultiTowerDIN` | 104 |
| `wukong.py` | `WuKong` | 130 |
| `masknet.py` | `MaskNet` | 65 |
| `tdm.py` | `TDM`, `TDMEmbedding` | 156 |
| `dlrm_hstu.py` | `DlrmHSTU` | 375 |
| `rocket_launching.py` | `RocketLaunching` | 323 |

#### 多任务排序模型
| 文件 | 类 | 行数 |
|------|-----|------|
| `mmoe.py` | `MMoE` | 86 |
| `ple.py` | `PLE` | 109 |
| `dbmtl.py` | `DBMTL` | 175 |
| `dc2vr.py` | `DC2VR` | 165 |
| `pepnet.py` | `PEPNet` | 244 |

#### 召回/匹配模型
| 文件 | 类 | 行数 |
|------|-----|------|
| `dssm.py` | `DSSMTower`, `DSSM` | 155 |
| `dssm_v2.py` | `DSSMTower(MatchTowerWoEG)`, `DSSMV2` | 202 |
| `dat.py` | `DATTower`, `DAT` | 259 |
| `mind.py` | `MINDUserTower`, `MINDItemTower`, `MIND` | 365 |
| `hstu.py` | `HSTUUserTower`, `HSTUMatchItemTower`, `HSTUMatch` | 429 |

#### 其他
| 文件 | 类 | 行数 |
|------|-----|------|
| `sid_model.py` | `BaseSidModel` | 120 |
| `ultra_hstu.py` | `_HSTUTransducerStack`, `UltraHSTU` | 118 |

#### 特定模型配置 proto
| 文件 | 关键 message | 行数 |
|------|-------------|------|
| `protos/models/rank_model.proto` | Rank 模型专属配置 | 79 |
| `protos/models/match_model.proto` | Match 模型专属配置 | 83 |
| `protos/models/multi_task_rank.proto` | MMoE/PLE/DBMTL/DC2VR/PEPNet/UltraHSTU 配置 | 119 |
| `protos/models/general_rank_model.proto` | 通用 Rank 配置 | 15 |

---

### 2. 特征系统 `tzrec/features/`

> **术语辨析**: 
> - "FG 模式/FG 流水线" → 特征生成（Feature Generation）方式，定义在 `data_parser.py` 中的 `FgMode`（FG_NONE/NORMAL/DAG/BUCKETIZE）
> - "FeatureGroup/特征组" → 模型配置中的 `FeatureGroupType`（DEEP/WIDE/SEQUENCE/JAGGED_SEQUENCE），定义在 `protos/model.proto`
> - 两者是完全不同的概念。用户说"FG 模式"通常指前者。

#### 基类
| 文件 | 关键内容 | 行数 |
|------|---------|------|
| `feature.py` | `BaseFeature`（元类注册）、`create_features()`、`create_fg_json()`、四种 `_parse_fg_encoded_*` 解析函数 | 1367 |

#### 特征实现
| 文件 | 类 | 基类 | 行数 |
|------|-----|------|------|
| `id_feature.py` | `IdFeature` | `BaseFeature` | 140 |
| `raw_feature.py` | `RawFeature` | `BaseFeature` | 99 |
| `tokenize_feature.py` | `TokenizeFeature` | `IdFeature` | 224 |
| `combo_feature.py` | `ComboFeature` | `IdFeature` | 106 |
| `bool_mask_feature.py` | `BoolMaskFeature` | `IdFeature` | 94 |
| `combine_feature.py` | `CombineFeature` | `BaseFeature` | 111 |
| `match_feature.py` | `MatchFeature` | `BaseFeature` | 189 |
| `lookup_feature.py` | `LookupFeature` | `BaseFeature` | 204 |
| `overlap_feature.py` | `OverlapFeature` | `RawFeature` | 82 |
| `expr_feature.py` | `ExprFeature` | `RawFeature` | 72 |
| `kv_dot_product.py` | `KvDotProduct` | `RawFeature` | 84 |
| `custom_feature.py` | `CustomFeature` | `BaseFeature` | 194 |

#### 特征配置 proto
| 文件 | 行数 | 关键 message |
|------|------|-------------|
| `protos/feature.proto` | 1036 | `FeatureConfig`, `IdFeature`, `RawFeature`, `TokenizeFeature`, `ComboFeature`, `MatchFeature`, `LookupFeature`, `OverlapFeature`, `ExprFeature`, `BoolMaskFeature`, `CustomFeature`, `CombineFeature`, `KvDotProduct`, `SequenceFeature`, `DynamicEmbedding`, `ZchConfig`, `ParameterConstraints` |

---

### 3. DynamicEmb `tzrec/utils/dynamicemb_util.py` (794 行)

| 符号 | 说明 |
|------|------|
| `_to_sharding_plan()` | **Planner 覆写** — 将 `ShardingOption` 转为 `ShardingPlan`，注入 `DynamicEmbParameterSharding` |
| `_customized_kernel_aware_get_device_bw()` | 带宽估计 — monkey-patch `HardwarePerfConfig.get_device_bw` |
| `_dynamicemb_aware_build_shard_perf_contexts()` | 性能估计 — 向 `cache_params` 注入经验 `x_eff` |
| `_calculate_dynamicemb_storage_specific_sizes()` | 存储估计 — 计算 HBM + DDR 字节数 |
| `dynamicemb_calculate_shard_storages()` | 存储入口 — 对每个 shard 计算 `Storage(hbm, ddr)` |
| `_calculate_dynamicemb_table_storage_specific_size()` | 单 shard 存储计算 |
| `build_dynamicemb_constraints()` | 从 proto `DynamicEmbedding` 中构建 `DynamicEmbParameterConstraints` |
| `_build_dynamicemb_initializer()` | 构建 `DynamicEmbInitializerArgs` |
| `_log_dynamicemb_table_plan()` | 日志打印每个表的模式（HBM_ONLY/CACHING/HYBRID） |
| `_dynamicemb_effective_cache_ratio()` | 经验 HBM 命中率 |
| `_validate_feature_range_with_dynamicemb()` | 跳过 dynamicemb 特征的 range 校验 |

依赖外部包: `dynamicemb`（NVIDIA 开源 GPU hash-table embedding）

---

### 4. 导出管线 `tzrec/utils/export_util.py`、`tzrec/acc/aot_utils.py`、`tzrec/acc/trt_utils.py`

#### `export_util.py` (1206 行)
| 函数 | 说明 |
|------|------|
| `export_model()` | 顶层入口 — 根据 `USE_RTP` 分派 RTP / 正常导出 |
| `export_model_normal()` | 标准导出: quantize → JIT script → AOT/TRT → 保存 pipeline.config + fg.json |
| `export_rtp_model()` | RTP 在线导出: sparse/dense 拆分 → FX tracing → safetensors |
| `split_model()` | 将 EasyRec 模型拆分为 sparse + dense 两部分 |

#### `aot_utils.py` (506 行)
| 函数 | 说明 |
|------|------|
| `export_model_aot()` | 2 阶段 AOT: sparse + dense 分别编译 |
| `export_unified_model_aot()` | Unified AOT: 融合模型单 `.pt2` |
| `_aoti_compile_cfg()` | AOTI inductor 配置覆写 |
| `_backport_pt178147_int_array_dedup()` | Torch bug 兼容 |

#### `trt_utils.py` (230 行)
| 函数 | 说明 |
|------|------|
| `export_model_trt()` | TensorRT 导出 |

#### `fx_util.py` (126 行)
| 函数 | 说明 |
|------|------|
| `symbolic_trace()` | 封装 `torchrec.fx.symbolic_trace` |
| `fx_mark_keyed_tensor()` / `fx_mark_tensor()` / `fx_mark_seq_tensor()` / `fx_mark_seq_len()` | FX trace marker |

---

### 5. 损失/指标/优化器

#### 损失 `tzrec/loss/`
| 文件 | 类 | 行数 |
|------|-----|------|
| `focal_loss.py` | `BinaryFocalLoss` | 72 |
| `jrc_loss.py` | `JRCLoss` | 117 |
| `pe_mtl_loss.py` | `ParetoEfficientMultiTaskLoss` | 110 |

#### 指标 `tzrec/metrics/`
| 文件 | 类 | 行数 |
|------|-----|------|
| `grouped_auc.py` | `GroupedAUC` | 125 |
| `xauc.py` | `XAUC`, `sampling_xauc()` | 173 |
| `grouped_xauc.py` | `GroupedXAUC` | 168 |
| `decay_auc.py` | `DecayAUC` | 60 |
| `normalized_entropy.py` | `NormalizedEntropy` | 72 |
| `recall_at_k.py` | `RecallAtK` | 54 |
| `unique_ratio.py` | `UniqueRatio` | 50 |
| `train_metric_wrapper.py` | `TrainMetricWrapper` | 62 |

#### 优化器 `tzrec/optim/`
| 文件 | 关键内容 | 行数 |
|------|---------|------|
| `optimizer.py` | `TZRecOptimizer(OptimizerWrapper)` | 237 |
| `optimizer_builder.py` | `create_sparse_optimizer()`, `create_dense_optimizer()`, `create_optimizer()` | 260 |
| `lr_scheduler.py` | `BaseLR(LRScheduler)` + 子类（ExponentialLR/StepLR/CosineAnnealingLR/CyclicLR/PolynomialLR 等） | 274 |

---

### 6. 数据集 `tzrec/datasets/`

| 文件 | 类/函数 | 行数 |
|------|---------|------|
| `dataset.py` | `BaseDataset(IterableDataset)`, `BaseReader`, `BaseWriter`, `create_reader()`, `create_writer()`, `create_dataloader()` | 827 |
| `data_parser.py` | `DataParser` — 4 种 FG 模式（FG_NONE/FG_NORMAL/FG_DAG/FG_BUCKETIZE）、`parse→to_batch` 流程、tile/input_tile 逻辑 | 977 |
| `sampler.py` | `NegativeSampler`, `NegativeSamplerV2`, `HardNegativeSampler`, `HardNegativeSamplerV2`, `TDMSampler`, `BaseSampler` | 1108 |
| `utils.py` | `Batch`, `SparseData`, `DenseData`, `SequenceSparseData`, `SequenceDenseData`, `ParsedData`, `RecordBatchTensor` | 923 |
| `odps_dataset.py` | `OdpsDataset`, `OdpsReader`, `OdpsWriter` | 812 |
| `parquet_dataset.py` | `ParquetDataset`, `ParquetReader` | 335 |
| `csv_dataset.py` | `CsvDataset`, `CsvReader` | 209 |
| `kafka_dataset.py` | `KafkaDataset`, `KafkaReader` | 473 |

---

### 7. 关键 Proto 配置 `tzrec/protos/`

| 文件 | 行数 | 主要 message |
|------|------|-------------|
| `pipeline.proto` | 29 | `EasyRecConfig` — 顶层管线配置 |
| `model.proto` | 96 | `ModelConfig`, `FeatureGroupConfig` |
| `data.proto` | 164 | `DataConfig`, `FgMode`, `DatasetType`, `FieldType` |
| `feature.proto` | 1036 | 全特征类型配置、DynamicEmbedding、ZchConfig |
| `module.proto` | 319 | MLP/FM/CIN/Cross/attention 等模块配置 |
| `tower.proto` | 192 | `TaskTower`, `BayesTaskTower`, `InterventionTaskTower` |
| `optimizer.proto` | 262 | `Optimizer`, `SparseOptimizer`, `DenseOptimizer` |
| `train.proto` | 76 | `TrainConfig` |
| `loss.proto` | 33 | `LossConfig` |
| `metric.proto` | 82 | `MetricConfig`, `TrainMetricConfig` |
| `sampler.proto` | 141 | `Sampler` 系列配置 |
| `export.proto` | 24 | `ExportConfig` |
| `eval.proto` | 9 | `EvalConfig` |
| `seq_encoder.proto` | 71 | `SeqEncoderConfig` (DIN/DIEN/SIM/GRU/Pooling) |
| `simi.proto` | 8 | `Similarity` (COSINE/INNER_PRODUCT) |

---

### 8. 关键工具模块 `tzrec/utils/`

| 文件 | 关键内容 | 行数 |
|------|---------|------|
| `plan_util.py` | `create_planner()`, `get_default_sharders()`, `EmbeddingEnumerator` | 1227 |
| `checkpoint_util.py` | `save_model()`, `restore_model()`, 分布式 checkpoint | 879 |
| `config_util.py` | `load_pipeline_config()`, `config_to_kwargs()`, `edit_config()` | 298 |
| `env_util.py` | `use_rtp()`, `use_hash_node_id()`, `enable_tma()`, `force_load_sharding_plan()` | 57 |
| `filesystem_util.py` | `url_to_fs()` — fsspec 集成 | 291 |
| `dist_util.py` | `DistributedModelParallel`, `init_process_group()` | 376 |
| `state_dict_util.py` | `fix_mch_state()`, `init_parameters()` | 59 |
| `load_class.py` | `get_register_class_meta()`, `import_pkg()` — 插件注册机制 | 171 |
| `logging_util.py` | Logger 配置 | 45 |
| `misc_util.py` | `get_free_port()` | 98 |
| `faiss_util.py` | FAISS 索引工具 | 168 |
| `test_util.py` | 测试工具 | 290 |

---

### 9. 环境变量

| 变量 | 类型 | 默认值 | 位置 | 说明 |
|------|------|--------|------|------|
| `USE_RTP` | bool | `0` | `env_util.py:24` | 启用 RTP（阿里在线服务）模式 |
| `USE_FARM_HASH_TO_BUCKETIZE` | bool | `false` | `env_util.py:27` | RTP 模式下推荐设为 true |
| `INPUT_TILE` | int | — | `export_util.py` | 1=不拆分, 2=user/item, 3=+embedding sharding |
| `INPUT_TILE_3_ONLINE` | bool | — | — | INPUT_TILE=3 时 seq tensor 用 jt.values() |
| `MAX_EXPORT_BATCH_SIZE` | int | `512` | `export_util.py:362` | 导出最大 batch size |
| `USE_FSSPEC` | bool | `0` | `filesystem_util.py:279` | 启用 fsspec 文件系统集成 |
| `LOCAL_CACHE_DIR` | str | — | `filesystem_util.py` | 远程文件的本地缓存目录 |
| `ENABLE_TMA` | bool | `0` | `env_util.py:35` | 启用 TMA（需要 CUDA arch>=9.0 + triton>=3.5） |
| `FORCE_LOAD_SHARDING_PLAN` | bool | `0` | `env_util.py:55` | 强制从 checkpoint 加载 sharding plan |
| `USE_HASH_NODE_ID` | bool | `0` | `env_util.py:19` | 使用 hash node id |
| `OMP_NUM_THREADS` | int | — | `export_util.py` | OpenMP 线程数（推荐 16） |
| `RANK` / `WORLD_SIZE` / `MASTER_ADDR` / `MASTER_PORT` | — | — | `dist_util.py` | 分布式训练标准 env |
| `LOCAL_RANK` / `LOCAL_WORLD_SIZE` / `GROUP_RANK` | — | — | `dist_util.py` | 本地/组 rank |
| `PREDICT_QUEUE_TIMEOUT` | int | `600` | `constant.py` | predict 队列超时（秒） |
| `NPROC_PER_NODE` | int | — | `export_util.py` | 每节点进程数 |
| `EVALUATE_MODELS` | — | — | — | eval 模式模型列表 |
| `TRAIN_MODELS` | — | — | — | train 模式模型列表 |

---

### 10. 常量 `tzrec/constant.py` (40 行)

| 符号 | 值 |
|------|-----|
| `Mode` | `TRAIN=1`, `EVAL=2`, `PREDICT=3` |
| `EASYREC_VERSION` | `"0.7.5"` |
| `TENSORBOARD_SUMMARIES` | `["loss", "learning_rate", "parameter", "global_gradient_norm", "gradient_norm", "gradient"]` |

---

## Ambiguity Resolution (必读)

以下术语在 TorchEasyRec 中有多个含义，引用时务必区分：

| 歧义词 | 含义 A | 含义 B |
|--------|--------|--------|
| **FeatureGroup 模式** | `protos/model.proto` 中 `FeatureGroupType` (DEEP/WIDE/SEQUENCE/JAGGED_SEQUENCE)，定义特征组如何分组/embedding | `datasets/data_parser.py` 中 `FgMode` (FG_NONE/FG_NORMAL/FG_DAG/FG_BUCKETIZE)，定义特征生成(FG)流水线模式 |
| **FG** | 缩写: Feature Generation（特征生成），对应 data_parser 中 FgMode | 读音相近: FeatureGroup，对应 model.proto |
| **AOT** | `ENABLE_AOT=1` 2 阶段导出 (sparse+dense 分开) | `ENABLE_AOT=2` Unified 导出 (单 .pt2) |
| **RTP 模式** | 阿里在线服务，USE_RTP=1 | 普通易用模式 |

**规则**: 当用户说"FeatureGroup 模式"且上下文不明确时，优先解释 FgMode（数据流水线），
同时补充说明 FeatureGroupType（模型配置），让用户确认。

## Common Lookup Patterns

当用户问以下问题时，按对应模式搜索：

1. **"xxx 模型在哪/拓扑是什么"** → `tzrec/models/<model_name>.py` 找类定义 + protos/models/ 下找配置
2. **"特征怎么配/xxx feature 是什么"** → `tzrec/features/<feature_name>.py` + `protos/feature.proto`
3. **"DynamicEmb 的 planner/storage/eviction"** → `tzrec/utils/dynamicemb_util.py` 四个 monkey-patch 函数
4. **"导出流程/AOT/TRT/RTP"** → `tzrec/utils/export_util.py` + `tzrec/acc/`
5. **"FG 模式/特征生成模式/DataParser 解析"** → `tzrec/datasets/data_parser.py`（四模式: FG_NONE/NORMAL/DAG/BUCKETIZE + parse→to_batch）
6. **"负采样怎么工作"** → `tzrec/datasets/sampler.py` + `dataset.py:_apply_negative_sampler()`
7. **"xxx loss/metric/optimizer"** → `tzrec/loss/` / `tzrec/metrics/` / `tzrec/optim/`
8. **"环境变量 xxx 什么意思"** → `tzrec/utils/env_util.py` + `tzrec/constant.py`
9. **"RTP 和普通模式有什么区别"** → `env_util.py:use_rtp()`（需同时设置 `USE_FARM_HASH_TO_BUCKETIZE=true`）→ `export_util.py:export_rtp_model()`
