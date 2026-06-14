---
title: 故障排查
parent: 参考
nav_order: 1
---

# 故障排查

## 导出问题

### ZCH 导出断裂

ZCH（Zero-Collision Hash）目前无法通过 `export_unified_model_aot()` 导出。问题位于 `export_util.py:544` 的 TODO `mczch`。

**症状**: 导出时报错 `NotImplementedError: MCZCH export not supported`

**临时方案**:
- 使用 `zch_to_dynamicemb_convert.py` 将 ZCH checkpoint 迁移为 DynamicEmb 格式再导出
- 或等待上游支持

详见 [ZCH 零碰撞哈希 → 导出限制](09-01-zch#33-导出断裂)。

### AOTI 编译失败

AOTI 统一路径需要 torch 版本 ≥ 2.4.0。使用旧版 torch 时 fallback 到两阶段路径。

**检查**:
```
# 确认 torch 版本
python -c "import torch; print(torch.__version__)"
# 检查 meta.json 中 backend 字段
cat export/fx_user_model/meta.json | python -m json.tool
```

### DynamicEmb 加载失败

```python
# 确认 CUDA 可用
import torch
assert torch.cuda.is_available()
# 确认 CUB 和 CUTLASS 头文件路径
echo $CUB_HOME
echo $CUTLASS_HOME
```

## FSSPEC 问题

### C++ IO 不被 fsspec 透传

Scripted Model 加载、safetensors 导出使用 C++ 文件 API，不会被 `fsspec` 拦截。

**方案**: 设置 `LOCAL_CACHE_DIR` 环境变量，让 Python 侧负责下载到本地缓存，C++ 侧读取本地文件。

## INPUT_TILE 问题

| 模式 | 适用场景 | 限制 |
|------|---------|------|
| `INPUT_TILE=1` | 默认，单 batch | 大 batch 时显存压力大 |
| `INPUT_TILE=2` | user/item FG tile | item 特征不可包含 user embedding |
| `INPUT_TILE=3` | + embedding tile | 仅支持特定模型结构 |

## 分布式训练

### NCCL 超时

```python
# 增加超时时间
os.environ["NCCL_TIMEOUT"] = "600"
os.environ["TORCH_NCCL_BLOCKING_WAIT"] = "1"
```

## 参考源码

| 文件 | 关键内容 |
|------|---------|
| `tzrec/utils/export_util.py:544` | MCZCH TODO |
| `tzrec/tools/dynamicemb/zch_to_dynamicemb_convert.py` | ZCH → DynamicEmb 迁移工具 |
| `tzrec/utils/export_util.py:1037-1065` | INPUT_TILE 兼容性检查 |
