---
title: Code Structure
nav_order: 4
---

# Code Structure

## Top-Level Layout

```
torcheasyrec/                  # Git submodule (pinned commit)
├── .claude/                   # Claude AI assistant config
├── .github/workflows/         # CI: nightly unit tests
├── data/                      # Sample data
├── docker/                    # Docker images
├── docs/                      # ReadTheDocs documentation
├── examples/                  # Example configs and tutorials
├── requirements/              # Dependency groups
├── scripts/                   # Utility scripts
├── setup.py                   # Package build
├── tzrec/                     # *** MAIN SOURCE ***
└── README.md
```

## Source Package (`tzrec/`)

```
tzrec/
├── __init__.py                 # Env setup, auto-import, determinism
├── version.py                  # __version__ = "1.2.0"
├── constant.py                 # Enums (Mode), constants
├── main.py                     # Training/eval/export entry point (1478 lines)
├── train_eval.py               # CLI entry point: train_and_evaluate()
├── eval.py                     # Evaluation-only entry
├── predict.py                  # Prediction entry
├── export.py                   # Model export entry
│
├── protos/                     # *** PROTOBUF CONFIG DEFINITIONS ***
│   ├── pipeline.proto          # EasyRecConfig (top-level)
│   ├── data.proto              # DataConfig, DatasetType
│   ├── feature.proto           # FeatureConfig, all feature types
│   ├── model.proto             # ModelConfig, FeatureGroupConfig
│   ├── train.proto             # TrainConfig
│   ├── eval.proto              # EvalConfig
│   ├── export.proto            # ExportConfig
│   ├── loss.proto              # LossConfig
│   ├── metric.proto            # MetricConfig
│   ├── optimizer.proto         # Optimizer config
│   ├── module.proto            # MLP, Cross, FM, etc. module configs
│   ├── seq_encoder.proto       # Sequence encoder configs
│   ├── simi.proto              # Similarity config
│   ├── tower.proto             # Tower configs
│   ├── sampler.proto           # Sampler configs
│   └── models/                 # Model-specific protos
│       ├── rank_model.proto    # DeepFM, DCN, etc.
│       ├── match_model.proto   # DSSM, DAT, MIND, etc.
│       ├── multi_task_rank.proto # MMoE, PLE, PEPNet
│       └── general_rank_model.proto # Custom models
│
├── features/                   # *** FEATURE IMPLEMENTATIONS ***
│   ├── __init__.py
│   ├── feature.py              # BaseFeature, create_features(), FG parsing
│   ├── id_feature.py           # IdFeature (categorical)
│   ├── raw_feature.py          # RawFeature (numerical)
│   ├── combo_feature.py        # ComboFeature (cross)
│   ├── lookup_feature.py       # LookupFeature
│   ├── sequence_feature.py     # SequenceFeature (grouped)
│   ├── expr_feature.py         # ExprFeature (expression-based)
│   ├── tokenize_feature.py     # TokenizeFeature (text)
│   ├── match_feature.py        # MatchFeature (retrieval)
│   ├── combine_feature.py      # CombineFeature
│   ├── bool_mask_feature.py    # BoolMaskFeature
│   ├── overlap_feature.py      # OverlapFeature
│   ├── kv_dot_product.py       # KV Dot Product
│   └── custom_feature.py       # CustomFeature (user-defined)
│
├── models/                     # *** MODEL IMPLEMENTATIONS ***
│   ├── __init__.py
│   ├── model.py                # BaseModel + Wrappers
│   ├── rank_model.py           # RankModel base
│   ├── match_model.py          # MatchModel base + tower classes
│   ├── sid_model.py            # Sparse ID model base
│   ├── multi_task_rank.py      # MultiTaskRank base
│   │
│   │   # Ranking Models
│   ├── deepfm.py               # DeepFM
│   ├── multi_tower.py          # MultiTower
│   ├── multi_tower_din.py      # MultiTowerDIN
│   ├── wide_and_deep.py        # Wide & Deep
│   ├── dcn.py                  # DCN v1
│   ├── dcn_v2.py               # DCN v2
│   ├── dlrm.py                 # DLRM
│   ├── masknet.py              # MaskNet
│   ├── xdeepfm.py              # xDeepFM
│   ├── wukong.py               # WuKong
│   ├── rocket_launching.py     # RocketLaunching
│   │
│   │   # Multi-Task Models
│   ├── mmoe.py                 # MMoE
│   ├── ple.py                  # PLE
│   ├── dbmtl.py                # DBMTL
│   ├── pepnet.py               # PEPNet
│   ├── dc2vr.py                # DC2VR
│   │
│   │   # Generative Rec Models
│   ├── dlrm_hstu.py            # DLRM-HSTU
│   ├── ultra_hstu.py           # ULTRA-HSTU
│   ├── hstu.py                 # HSTU base
│   │
│   │   # Matching Models
│   ├── dssm.py                 # DSSM
│   ├── dssm_v2.py              # DSSM v2
│   ├── dat.py                  # DAT
│   ├── mind.py                 # MIND
│   │
│   │   # Tree-based Models
│   ├── tdm.py                  # TDM
│   └── match_model_test.py
│
├── modules/                    # *** REUSABLE MODULES ***
│   ├── __init__.py
│   ├── mlp.py                  # MLP, FinalMLP
│   ├── fm.py                   # Factorization Machine
│   ├── interaction.py          # Feature interaction (CIN, Cross)
│   ├── masknet.py              # MaskNet block
│   ├── mmoe.py                 # MMoE gate
│   ├── extraction_net.py       # Extraction networks
│   ├── personalized_net.py     # Personalized network (PEPNet)
│   ├── sequence.py             # Sequence encoders (LSTM, Pooling, etc.)
│   ├── hstu.py                 # HSTU transducer
│   ├── capsule.py              # Capsule network (MIND)
│   ├── attention.py            # Attention mechanisms (DIN)
│   ├── embedding.py            # EmbeddingGroup, EmbeddingGroupImpl
│   ├── dense_embedding_collection.py # AutoDis, MLP dense embedding
│   ├── norm.py                 # Normalization layers
│   ├── activation.py           # Activation functions
│   ├── intervention.py         # Intervention modules
│   ├── variational_dropout.py  # Variational dropout
│   ├── task_tower.py           # Task-specific towers
│   ├── utils.py                # BaseModule, div_no_nan
│   │
│   ├── gr/                     # Generative Recommendation
│   │   ├── hstu_transducer.py  # HSTU transducer core
│   │   ├── action_encoder.py   # Action encoder
│   │   ├── content_encoder.py  # Content encoder
│   │   ├── preprocessors.py    # Input preprocessors
│   │   ├── postprocessors.py   # Output postprocessors
│   │   ├── stu.py              # STU module
│   │   └── contextualize_mlps.py # Contextual MLPs
│   │
│   └── sid/                    # Sparse ID
│       └── residual_quantizer.py # Residual quantizer
│
├── datasets/                   # *** DATA LOADING ***
│   ├── __init__.py
│   ├── dataset.py              # BaseDataset, create_dataloader()
│   ├── data_parser.py          # DataParser (raw → Batch)
│   ├── utils.py                # Batch, ParsedData, SparseData, etc.
│   ├── sampler.py              # Negative samplers
│   ├── csv_dataset.py          # CSV reader
│   ├── parquet_dataset.py      # Parquet reader
│   ├── odps_dataset.py         # MaxCompute/ODPS reader
│   ├── kafka_dataset.py        # Kafka streaming reader
│   └── *_test.py               # Tests
│
├── loss/                       # *** LOSS FUNCTIONS ***
│   ├── __init__.py
│   ├── focal_loss.py           # BinaryFocalLoss
│   ├── jrc_loss.py             # JRCLoss (session-based)
│   └── pe_mtl_loss.py          # Pareto-efficient MTL loss
│
├── metrics/                    # *** METRICS ***
│   ├── __init__.py
│   ├── decay_auc.py            # DecayAUC (training)
│   ├── grouped_auc.py          # GroupedAUC
│   ├── xauc.py                 # XAUC
│   ├── grouped_xauc.py         # GroupedXAUC
│   ├── normalized_entropy.py   # NormalizedEntropy
│   ├── recall_at_k.py          # Recall@K
│   ├── unique_ratio.py         # Unique ratio
│   └── train_metric_wrapper.py # TrainMetricWrapper (decay)
│
├── ops/                        # *** KERNEL OPERATIONS ***
│   ├── __init__.py
│   ├── hstu_attention.py       # HSTU attention dispatch
│   ├── hstu_attention_utils.py # HSTU attention utilities
│   ├── hstu_compute.py         # HSTU compute
│   ├── jagged_tensors.py       # Jagged tensor ops
│   ├── layer_norm.py           # LayerNorm ops
│   ├── mm.py                   # Matrix multiply ops
│   ├── position.py             # Position encoding
│   ├── utils.py                # Kernel utilities
│   │
│   ├── _cuda/                  # CUDA (CUTLASS) implementations
│   │   └── cutlass_hstu_attention.py
│   │
│   ├── _pytorch/               # PyTorch native implementations
│   │   ├── pt_hstu_attention.py
│   │   ├── pt_hstu_linear.py
│   │   ├── pt_jagged_tensors.py
│   │   ├── pt_layer_norm.py
│   │   └── pt_position.py
│   │
│   ├── _triton/                # Triton implementations
│   │   ├── triton_addmm.py
│   │   ├── triton_hstu_attention.py
│   │   ├── triton_hstu_linear.py
│   │   ├── triton_hstu_preprocess_and_attention.py
│   │   ├── triton_jagged_tensors.py
│   │   ├── triton_layer_norm.py
│   │   └── triton_position.py
│   │
│   └── benchmarks/             # Kernel benchmarks
│       └── hstu_attention_bench.py
│
├── optim/                      # *** OPTIMIZERS ***
│   ├── __init__.py
│   ├── optimizer.py            # TZRecOptimizer
│   ├── optimizer_builder.py    # Build optimizer from config
│   └── lr_scheduler.py         # LR schedulers
│
├── acc/                        # *** ACCELERATION ***
│   ├── __init__.py
│   ├── utils.py                # allow_tf32, mixed_precision_to_dtype
│   ├── trt_utils.py            # TensorRT utilities
│   └── aot_utils.py            # AOTInductor utilities
│
├── utils/                      # *** UTILITIES ***
│   ├── __init__.py
│   ├── config_util.py          # Config loading, editing, parsing
│   ├── checkpoint_util.py      # Checkpoint save/restore
│   ├── dist_util.py            # Distributed training utilities
│   ├── export_util.py          # Model export
│   ├── filesystem_util.py      # File system abstraction (local/OSS/ODPS)
│   ├── load_class.py           # Auto-registration + class loading
│   ├── plan_util.py            # TorchRec sharding planner
│   ├── misc_util.py            # RAM credential patch
│   ├── env_util.py             # Environment detection
│   ├── logging_util.py         # Logging + progress
│   ├── state_dict_util.py      # State dict utilities
│   ├── fx_util.py              # torch.fx utilities
│   ├── dynamicemb_util.py      # Dynamic embedding utilities
│   └── faiss_util.py           # FAISS index utilities
│
├── tools/                      # *** TOOLS ***
│   ├── create_faiss_index.py   # FAISS index builder
│   ├── create_fg_json.py       # FG json config generator
│   ├── create_online_infer_data.py # Online inference data
│   ├── feature_selection.py    # Feature importance
│   ├── hitrate.py              # Hit rate computation
│   ├── add_feature_info_to_config.py # Config enrichment
│   ├── convert_easyrec_config_to_tzrec_config.py # Migration tool
│   └── tdm/                    # TDM utilities
│       ├── cluster_tree.py
│       ├── init_tree.py
│       ├── retrieval.py
│       └── gen_tree/           # Tree generation
│           ├── tree_builder.py
│           ├── tree_cluster.py
│           ├── tree_generator.py
│           └── tree_search_util.py
│
├── benchmark/                  # Benchmarks
│   └── benchmark.py
│
└── tests/                      # Integration tests
    ├── __init__.py
    ├── run.py                  # Test runner
    ├── utils.py                # Test utilities
    ├── match_integration_test.py
    └── rank_integration_test.py
```

## Proto Config System

The protobuf definitions form a hierarchy:

```
pipeline.proto: EasyRecConfig
    ├── train.proto: TrainConfig
    ├── eval.proto: EvalConfig
    ├── export.proto: ExportConfig
    ├── data.proto: DataConfig
    ├── feature.proto: FeatureConfig (IdFeatureConfig, RawFeatureConfig, ...)
    ├── model.proto: ModelConfig
    │   ├── models/rank_model.proto: DeepFM, DCN, DLRM, ...
    │   ├── models/match_model.proto: DSSM, DAT, MIND, ...
    │   └── models/multi_task_rank.proto: MMoE, PLE, PEPNet
    ├── loss.proto: LossConfig
    ├── metric.proto: MetricConfig
    ├── optimizer.proto: Optimizer config
    └── sampler.proto: Sampler config
```

Each proto is compiled to Python by `protoc` and imported via `tzrec.protos.*_pb2`.

## Key Stats

| Metric | Count |
|--------|-------|
| Python files | 295 |
| Proto files | 20 |
| Model implementations | 22+ |
| Feature types | 12 |
| Loss functions | 5 |
| Metric types | 8 |
| Custom ops | 15+ (Triton, CUDA, PyTorch) |
