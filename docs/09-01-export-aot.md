---
title: AOT 编译
parent: 导出与 RTP 管线
nav_order: 1
---

# AOT 编译

## 输入与目标

- 输入：`fx_user_model/exported_model.pt`（`torch.export` 产物）
- 目标：`.so`（AOTInductor 共享库）和 / 或 `.pt2`（AOT Package，供 C++/Python 加载）

## 工具链与默认值

入口脚本 `tools/export_aot.py`：

| 参数 | 默认值 |
|------|--------|
| `--artifacts` | `so`（`export_aot.py:35-39`） |
| `--device` | `cuda` 若可用，否则 `cpu`（`export_aot.py:51-54`） |
| `--cxx` | `g++-13`（`export_aot.py:61-64`） |

包装脚本 `run_export_aot.sh` 默认 `ARTIFACTS=both`（同时产 `.so` + `.pt2`）。

## 编译主流程

### 输入解析

- 优先 `--exported-model`，否则按 `--export-dir/fx_user_model/exported_model.pt` 寻址
- 输入样例优先 `--input-pt`，否则从 zip 内 `exported_model/data/sample_inputs/model.pt` 读取

（`export_aot.py:68-104`、`270-276`）

### 设备迁移与编译

```
args/kwargs → 递归迁移到目标设备
.so   → torch._inductor.aot_compile(...)
.pt2  → torch._inductor.aoti_compile_and_package(...)
```

若遇到 `Device mismatch` 或 `found two different devices`，自动在目标设备重新 export 再编译（`export_aot.py:227-235`、`304-320`）。

## AOT 产物

```
fx_user_model/
├── exported_model.pt               # 原始导出模型
├── exported_model_aot.so           # AOTInductor 共享库
├── exported_model_aot.pt2          # AOT Package
├── exported_model_aot.so.meta.json # Sidecar 元信息
└── exported_model_aot.pt2.meta.json
```

### .pt2 归档内结构

```
exported_model.pt/
├── models/
│   └── model.json                  # 图结构 + 输入/输出描述
├── data/
│   ├── weights/
│   │   ├── model_weights_config.json
│   │   └── weight_0 ~ weight_n
│   ├── constants/
│   │   └── model_constants_config.json
│   └── sample_inputs/
│       └── model.pt               # 导出时输入样例
```

### meta.json 字段

```json
{
    "schema_version": "1.0",
    "input_tensor_name": ["user__item__ebc"],
    "input_shape": [[128, 256]],
    "input_dtype": ["float32"],
    "batch_size": 128,
    "feature_dim": 256,
    "device": "cuda",
    "artifacts": ["so", "pt2"]
}
```

## 常用命令

```bash
# 导出 so + pt2（run_export_aot.sh 默认 artifacts=both）
bash tools/run_export_aot.sh

# 仅导出 pt2（CPU）
bash tools/run_export_aot.sh --artifacts pt2 --device cpu

# 查看 pt2 结构
bash tools/run_inspect_pt2_archive.sh

# C++ AOTI 推理验证
bash tools/run_fireworks_test_aot_infer.sh
```

## 检查与调试工具

| 脚本 | 功能 |
|------|------|
| `tools/inspect_pt2_archive.py` | 解析 .pt2 归档结构 |
| `tools/run_exported_forward.py` | 用 exported_model.pt 做前向推理 |
| `tests/fireworks/test_aot_infer.cpp` | C++ 推理验证 |

## 与 TorchEasyRec AOT 后端的关联

TorchEasyRec 在导出管线中自动调用 AOTI 编译的流程见 [09-导出与 RTP 管线](09-export-pipeline) §6。AOTI 编译用 `torch._inductor.aoti_compile_and_package()` 生成单个 `.pt2`，predict 时 `UnifiedAOTIModelWrapper` 用 `torch._inductor.aoti_load_package()` 加载。
