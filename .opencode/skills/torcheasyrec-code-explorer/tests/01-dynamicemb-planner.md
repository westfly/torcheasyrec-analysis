# 测试 1: DynamicEmb Planner

**问题**: DynamicEmb 的四个 planner monkey-patch 在哪？

**验证点**:
- 定位到 `tzrec/utils/dynamicemb_util.py`
- 指出 4 个函数: `_to_sharding_plan`、`_customized_kernel_aware_get_device_bw`、`_dynamicemb_aware_build_shard_perf_contexts`、行为 `_calculate_dynamicemb_storage_specific_sizes`/`dynamicemb_calculate_shard_storages`
- 每个函数说明用途

**期望回答质量**: 精确到文件名和行号，说明每个函数在 planner 中的作用
