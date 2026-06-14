---
title: AOT 编译
parent: 导出与 RTP 管线
nav_order: 1
---

# AOT 编译

## 概述

AOT（Ahead-of-Time）编译将导出的模型进一步优化为可部署产物，由 TorchEasyRec 导出管线自动触发（`ENABLE_AOT=2` → 统一 AOTI 路径）。

## 编译流程

```
训练模型 → torch.export.export() → ExportedProgram
                                  → torch._inductor.aoti_compile_and_package()
                                  → aoti/aoti_model.pt2
```

核心调用在 [`aot_utils.py:437`](../torcheasyrec/tzrec/acc/aot_utils.py#L437) `export_unified_model_aot()`。

### 编译前准备

1. `_pad_empty_sparse_values()` — 将 0 元素非序列稀疏 `.values` 张量膨胀为 2 元素，防止 `torch.export` 特化在 size-0 哨兵上
2. `_build_dynamic_shapes()` — 利用特征和特征组的结构知识分配动态 shape：序列特征共享 Dim、非序列稀疏独立 Dim、稠密特征 batch dim
3. `CudaAutocastWrapper` — 绑定 device 和混合精度

### 编译配置

```python
{
    "scalar_asserts": False,
    "unsafe_ignore_unsupported_triton_autotune_args": True,
    "_use_fp64_for_unbacked_floats": False,
}
```

## 产物

```
export_dir/
├── model_acc.json                  # 记录 ENABLE_AOT=2 供 predict 端读取
└── aoti/
    └── aoti_model.pt2             # AOTI Package（含编译后 .so 和元信息）
```

predict 时 `UnifiedAOTIModelWrapper` 用 `torch._inductor.aoti_load_package()` 加载。

## 与 TorchEasyRec 导出管线

AOTI 编译的完整触发链路（决策树、split_model、旧版两阶段 vs 统一路径）见 [09-导出与 RTP 管线 ⇒ AOTI 后端](14-export-pipeline#6-aoti-后端)。
