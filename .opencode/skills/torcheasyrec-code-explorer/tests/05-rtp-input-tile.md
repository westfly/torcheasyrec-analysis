# 测试 5: RTP + INPUT_TILE

**问题**: RTP 模式下 INPUT_TILE = 2 是什么意思？和 INPUT_TILE = 3 有什么区别？

**验证点**:
- INPUT_TILE=2: user/item 分流
- INPUT_TILE=3: + embedding sharding
- INPUT_TILE=1: 不拆分
- 引用 `export_util.py` 中的实现
- 关联 USE_RTP 和 USE_FARM_HASH_TO_BUCKETIZE

**期望回答质量**: 解释每个模式的用途和导出产物的区别
