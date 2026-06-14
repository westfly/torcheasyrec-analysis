# 测试 3: AOT Unified 入口

**问题**: AOT 导出的 Unified 流程入口在哪？和 2 阶段 AOT 有什么区别？

**验证点**:
- 入口: `tzrec/acc/aot_utils.py:export_unified_model_aot()`
- 2 阶段: `export_model_aot()` (sparse + dense 分开)
- Unified: 融合模型单 `.pt2` 文件
- 引用 `export_util.py` 中的分派逻辑

**期望回答质量**: 对比两种方式，附函数签名和调用链路
