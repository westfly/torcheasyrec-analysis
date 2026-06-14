---
title: 案例分析
nav_order: 6
has_children: true
---

# 案例分析

基于 `multi_tower_din_taobao.config` 的端到端流水线分析，覆盖训练、导出、推理全链路。包含当前 `num_buckets` 方案的瓶颈分析和集成 DynamicEmb 的迁移方案。

| 章节 | 标题 | 简介 |
|------|------|------|
| [17](17-multi-tower-din-current) | MultiTowerDIN 当前现状 | 配置概览、数据流水线、Embedding 瓶颈、DIN 注意力性能、分布式通信、导出路径 |
| [18](18-multi-tower-din-dynamicemb) | MultiTowerDIN 集成 DynamicEmb | 迁移策略、Config 示例、训练/导出/推理变化、分布式通信分析 |
