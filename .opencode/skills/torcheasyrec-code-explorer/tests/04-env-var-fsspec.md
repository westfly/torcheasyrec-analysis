# 测试 4: 环境变量 USE_FSSPEC

**问题**: 环境变量 USE_FSSPEC 的作用是什么？它的实现机制是怎样的？

**验证点**:
- 引用 `tzrec/utils/filesystem_util.py` 中的实现
- 说明 register_external_filesystem 自动调用机制
- 列出被 monkeypatch 的 10 个 Python IO 函数
- 提到 LOCAL_CACHE_DIR 的作用
- 说明 3 个 C++ IO  pitfalls 的绕过方式

**期望回答质量**: 完整描述 fsspec 集成机制，包括注册时机、patch 范围、缓存策略
