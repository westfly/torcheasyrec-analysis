---
title: 推理篇
nav_order: 5
has_children: true
---

# 推理篇

模型导出与在线推理全链路，包括导出管线、RTP 稀疏模型重建、DynamicEmb GPU 服务端嵌入、FSSPEC 透传和环境变量。

| 章节 | 标题 | 简介 |
|------|------|------|
| [13](13-export-pipeline) | 导出管线 | JIT / TRT / AOTI / RTP 四后端对比、FX 切图、INPUT_TILE |
| [14](14-fsspec) | USE_FSSPEC 透传 | 外部文件系统抽象、`fsspec` 协议解析、10 个 IO 函数 monkeypatch、C++ IO 绕行 |
| [15](15-env) | 环境变量 | 分布式/推理/特征/导出/日志/数据源/测试全表 |
