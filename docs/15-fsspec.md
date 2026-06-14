---
title: USE_FSSPEC 与 fsspec 透传机制
nav_order: 2
parent: 推理篇
---

# USE_FSSPEC 与 fsspec 透传机制

TorchEasyRec 的训练/评估/导出链路会触碰多种外部存储（数据文件、checkpoint、导出图）。本地 POSIX IO 无法直接处理 `s3://`、`hdfs://`、`oss://` 等 URI 协议——必须借助统一抽象层。本文档剖析 `tzrec` 框架在原生 Python IO 与外部存储之间的透传机制：`USE_FSSPEC` 环境变量、`fsspec` 协议解析、以及为了让业务代码"无感"而对 10 个内置 IO 函数做的 monkeypatch。

目标读者：需要在非阿里云环境（OSS / S3 / GCS / Azure / HDFS）跑通训练或导出的工程师；或想理解 monkeypatch 影响面的维护者。

---

## 1. 背景与定位

### 1.1 问题

- 训练数据可能落在 `oss://bucket/data/`、`s3://bucket/train/`、`hdfs:///path/` 等位置
- 模型输出 `model_dir`、`export_dir` 也可能要求写到远程
- checkpoint save/load 同样要支持远程协议
- 业务代码（数据加载、TFRecord 解析、FG 特征生成、safetensors dump）遍布 `open()` / `os.path.exists()` / `shutil.rmtree()` 等原生调用——**逐处改写协议分支不现实**

### 1.2 解法

通过 [fsspec](https://filesystem-spec.readthedocs.io/) 提供统一文件系统抽象，再用 **monkeypatch** 把 10 个 Python 内置 IO 函数**在进程级**换皮。这样：

- 业务代码无需修改，继续写 `open(path)` / `os.path.exists(path)`
- patch 自动按 `path` 的协议（`s3://` / `hdfs://` / `oss://` / ...）分发到 fsspec 对应实现
- 对**第三方库**也透明（TorchRec、TensorBoard、PyTorch Lightning 内部 IO 同样被劫持）

---

## 2. 入口：自动注册机制

[`torcheasyrec/tzrec/utils/filesystem_util.py:277-291`](../torcheasyrec/tzrec/utils/filesystem_util.py#L277-L291) 的 `register_external_filesystem()` 在 [`tzrec/__init__.py:80`](../torcheasyrec/tzrec/__init__.py#L80) 被**自动调用**——任何 `import tzrec` 都会触发：

```python
# tzrec/__init__.py:80
_register_external_filesystem()
```

```python
# filesystem_util.py:277-291
def register_external_filesystem():
    use_fsspec = int(os.environ.get("USE_FSSPEC", "0")) == 1
    try:
        from pangudfs_client.high_level_client.extern.fsspec import PanguDfs
        from tensorboard.compat.tensorflow_stub.io import gfile

        gfile.register_filesystem("dfs", PanguGFile())   # tensorboard 兼容
        fsspec.register_implementation("dfs", PanguDfs)  # dfs:// 协议支持
        use_fsspec = True                                 # 自动开启
    except ImportError:
        pass                                              # 阿里云外部署时静默跳过

    if use_fsspec:
        apply_monkeypatch()                               # 10 个内置函数换皮
```

### 2.1 阿里元 vs 非阿里元

| 维度 | 阿里元（有 `pangudfs_client`） | 非阿里元（本次询问） |
|------|------------------------------|---------------------|
| `pangudfs_client` import | ✅ 成功 | ❌ ImportError（被 try 静默吞） |
| `dfs://` 协议 | ✅（PanguDfs + PanguGFile 双注册） | ❌（fsspec 不认识 `dfs://`） |
| 是否需设 `USE_FSSPEC` | 不需要（自动开启） | **必须**显式 `export USE_FSSPEC=1` |
| 支持的协议 | `dfs://` + fsspec 原生协议 | 仅有 fsspec 原生协议（见 §4） |
| tensorboard gfile | ✅ PanguGFile 注入 | 默认 tf 实现 |

> **核心陷阱**：非阿里元**必须**显式设 `USE_FSSPEC=1`，否则 fsspec 仅被 import、**完全没起作用**——所有 IO 仍走本地 POSIX，远程路径会直接报 `FileNotFoundError`。

---

## 3. 协议支持与安装

### 3.1 `url_to_fs()` 协议解析

[`filesystem_util.py:37-50`](../torcheasyrec/tzrec/utils/filesystem_util.py#L37-L50)：

```python
def url_to_fs(path):
    protocol = None
    rpath = path
    if isinstance(path, str):
        protocol, rpath = fsspec.core.split_protocol(path)   # 拆出 (protocol, rpath)
    if protocol is None:
        return None, rpath                                  # 没协议 → 走原 Python IO
    elif protocol in _CACHED_FSSPEC_FILESYSTEMS:
        return _CACHED_FSSPEC_FILESYSTEMS[protocol], rpath  # 命中缓存
    else:
        fs, _ = fsspec.core.url_to_fs(path)                 # 查 fsspec 注册表
        _CACHED_FSSPEC_FILESYSTEMS[protocol] = fs           # 缓存实例
        return fs, rpath
```

- `split_protocol("s3://bucket/x")` → `("s3", "bucket/x")`
- `split_protocol("/local/path")` → `(None, "/local/path")` —— **纯本地路径透传**
- 缓存避免反复建 S3/GCS client 浪费 TLS 握手；分布式多 rank 各建各的（fsspec 标准行为）

### 3.2 协议与依赖

TorchEasyRec **不参与注册自定义协议**——所有 fsspec 原生支持均可直接用，按需装对应 extras：

| 协议 | 用途 | 装包命令 |
|------|------|----------|
| `file://` | 本地（带 URI 前缀，与裸路径等价） | 标配 |
| `s3://` | AWS S3 / 兼容协议（**OSS 走 S3 兼容端点**） | `pip install fsspec[s3]` |
| `gs://` | Google Cloud Storage | `pip install fsspec[gcs]` |
| `abfs://` / `az://` | Azure Data Lake / Blob | `pip install fsspec[abfs]` |
| `hdfs://` | HDFS（经 PyArrow 内置） | `pip install fsspec[hdfs]` |
| `http(s)://` | HTTP(S) 只读 | `pip install fsspec[http]` |
| `sftp://` | SFTP | `pip install fsspec[sftp]` |
| `webhdfs://` | WebHDFS REST | `pip install requests` |

**OSS 特别说明**：阿里云 OSS 原生协议 `oss://` 不在 fsspec 默认表里。生产做法是用 **S3 兼容端点**：

```bash
export AWS_ENDPOINT_URL=https://oss-cn-hangzhou.aliyuncs.com
export AWS_ACCESS_KEY_ID=<your-ak>
export AWS_SECRET_ACCESS_KEY=<your-sk>
# 然后用 s3://<bucket>/<key> 即可
```

---

## 4. 透明劫持：10 个被 monkeypatch 的 IO 函数

[`filesystem_util.py:141-152`](../torcheasyrec/tzrec/utils/filesystem_util.py#L141-L152)：

```python
def apply_monkeypatch():
    builtins.open        = _patched_open
    os.makedirs          = _patched_makedirs
    os.path.isdir        = _patched_isdir
    os.listdir           = _patched_listdir
    os.remove            = _patched_remove
    os.path.exists       = _patched_exists
    shutil.copy          = _patched_copy
    shutil.rmtree        = _patched_rmtree
    glob_module.glob     = _patched_glob
    os.path.getsize      = _patch_get_size
```

每个 patch 模式统一：

```python
def _patched_xxx(path, *args, **kwargs):
    fs, _ = url_to_fs(path)
    if fs is not None:                  # 有 protocol → 走 fsspec
        return fs.xxx(path, *args, **kwargs)
    else:                                # 无 protocol → 退回原 Python IO
        return _original_xxx(path, *args, **kwargs)
```

### 4.1 影响面

| 函数 | 原生语义 | fsspec 等价 | 常见调用方 |
|------|----------|-------------|------------|
| `builtins.open` | 本地文件 IO | `fs.open(path, mode)` | 数据加载、TFRecord 读、yaml/json 配置 |
| `os.makedirs` | 创建目录 | `fs.makedirs(path, exist_ok)` | 导出目录创建、checkpoint 目录 |
| `os.path.isdir` | 判断目录 | `fs.isdir(path)` | 训练初始化（判断 checkpoint 是否存在） |
| `os.listdir` | 列目录 | `fs.ls(path, detail=False)` | checkpoint 恢复时扫 shard 文件 |
| `os.remove` | 删除单文件 | `fs.rm(path)` | 导出清理 |
| `os.path.exists` | 判断存在 | `fs.exists(path, check_dir=True)` | **高频** — 几乎所有路径前都调 |
| `shutil.copy` | 复制 | `fs.open(src) + fs.open(dst)` 双向流 | 配置拷贝、模型导出 |
| `shutil.rmtree` | 递归删 | `fs.rm(path, recursive=True)` + `ignore_errors` 抑制 | 导出前清空目录 |
| `glob.glob` | 通配 | `fs.glob(pattern)` | 数据文件扫描 |
| `os.path.getsize` | 文件大小 | `fs.info(path)["size"]` | checkpoint 元信息收集 |

### 4.2 第三方库透明

> monkeypatch 是**进程级**的，影响整个 Python 运行时。TorchRec 的 `DistributedModelParallel` 内部用 `os.path.exists` 判断 checkpoint dir、TensorBoard 用 `gfile` 写 summary、PyTorch Lightning 用 `shutil.copy` 拷贝权重——**全部被劫持**，无需框架主动适配。

### 4.3 反向卸载

`remove_monkeypatch()`（[L155-166](../torcheasyrec/tzrec/utils/filesystem_util.py#L155-L166)）恢复全部 10 个原函数。用于测试场景隔离。

---

## 5. 三个 C++ IO 坑（必须用 `LOCAL_CACHE_DIR`）

**monkeypatch 只能劫持 Python 层 IO**。以下三处走 **C++ 层 IO**（PyTorch C++ binding），fsspec 透明机制**管不到**：

### 5.1 Scripted Model 加载

[`main.py:1081-1088`](../torcheasyrec/tzrec/main.py#L1081-L1088)：

```python
fs, local_path = url_to_fs(scripted_model_path)
if fs is not None:
    # scripted model use io in cpp, so that we can not path to fsspec
    local_path = os.environ.get("LOCAL_CACHE_DIR", local_path)
    if int(os.environ.get("LOCAL_RANK", 0)) == 0:
        logger.info(f"downloading {scripted_model_path} to {local_path}.")
        fs.download(scripted_model_path, local_path, recursive=True)
    dist.barrier()
    scripted_model_path = local_path
```

- `torch.jit.load()` 走 C++ IO，**Python 层 `open()` patch 失效**
- 解决：rank-0 先 `fs.download(recursive=True)` 到 `LOCAL_CACHE_DIR`，所有 rank `dist.barrier()` 后从本地加载
- 仅 `LOCAL_RANK=0` 做下载，省流量

### 5.2 导出 safetensors / TorchScript 图

[`export_util.py:97-103`](../torcheasyrec/tzrec/utils/export_util.py#L97-L103)：

```python
fs, local_path = url_to_fs(save_dir)
if fs is not None:
    # scripted model and safetensors use io in cpp,
    # so that we can not use fsspec to patch cpp io operations.
    local_path = os.environ.get("LOCAL_CACHE_DIR", local_path)
    use_local_cache_dir = True
```

- safetensors C++ 写盘、`torch.save` C++ 序列化**绕过** Python 层
- 解决：`save_dir` 落到 `LOCAL_CACHE_DIR` 的**本地路径**，导出完成后再 `fs.put()` 上传
- 上传逻辑不在此函数内，由调用方（CLI / pipeline）按需处理

### 5.3 fsspec glob 语义差异

[`checkpoint_util.py:182`](../torcheasyrec/tzrec/utils/checkpoint_util.py#L182) 注释：

```python
# fsspec glob need endswith os.path.sep
```

- `fs.glob(pattern)` 与 `glob.glob(pattern)` 在 `*` 递归匹配上语义不同
- 路径末尾是否带 `os.path.sep` 影响结果
- 调用方需手动补 `/`，或用 `fs.glob(path + "/**")` 显式递归

### 5.4 三坑对比

| 坑 | 文件 | 触发条件 | 解决 |
|----|------|----------|------|
| Scripted Model 加载 | `main.py:1081` | `scripted_model_path` 是远端 URL | `LOCAL_CACHE_DIR` + `fs.download()` |
| Safetensors 导出 | `export_util.py:97` | `save_dir` 是远端 URL | `LOCAL_CACHE_DIR` + 导出后上传 |
| Glob 语义 | `checkpoint_util.py:182` | `fs.glob()` 通配目录 | 路径末尾补 `os.path.sep` |

---

## 6. 端到端实操示例（OSS）

```bash
# 1. 装好对应协议包
pip install fsspec[s3,http]

# 2. 开启 fsspec 透传（非阿里元必须显式开）
export USE_FSSPEC=1

# 3. 配置 OSS S3 兼容端点
export AWS_ENDPOINT_URL=https://oss-cn-hangzhou.aliyuncs.com
export AWS_ACCESS_KEY_ID=<your-ak>
export AWS_SECRET_ACCESS_KEY=<your-sk>
export AWS_REGION=cn-hangzhou

# 4. 配置 C++ IO 绕行（用于脚本化模型加载/导出）
export LOCAL_CACHE_DIR=/tmp/te_cache
mkdir -p $LOCAL_CACHE_DIR

# 5. 直接用 s3:// 路径跑训练
python -m tzrec.main \
  --pipeline_config_path s3://my-bucket/config/dssm.config \
  --train_input_path s3://my-bucket/data/train/ \
  --model_dir s3://my-bucket/model/ \
  ...

# 6. 同样用 s3:// 路径导出
python -m tzrec.export \
  --pipeline_config_path s3://my-bucket/config/dssm.config \
  --export_dir s3://my-bucket/export/dssm/ \
  --checkpoint_path s3://my-bucket/model/ckpt \
  ...
```

`dssm.config` 配置文件中所有的 `feature_input_path`、`model_dir`、`output_dir` 字段同样被 monkeypatch 接管，可直接用 s3:// 路径。

---

## 7. 与阿里元环境的差异

| 维度 | 阿里元 | 非阿里元 |
|------|--------|----------|
| `pangudfs_client` | 可 import | ImportError 吞掉 |
| 默认开启 | ✅（无需 env） | ❌（必须 `USE_FSSPEC=1`） |
| `dfs://` 协议 | ✅（PanguDfs 注册） | ❌（fsspec 不认识） |
| tensorboard gfile | ✅（PanguGFile 注入） | ❌（用 tf 默认） |
| `s3://` 等标准协议 | 同样支持（s3fs 已装） | 同样支持（装 extras 即可） |
| `LOCAL_CACHE_DIR` 逻辑 | 一样 | 一样 |
| 训练/导出主流程 | 一样 | 一样 |

**实际差异仅 3 点**：① `dfs://` 用不了；② `USE_FSSPEC` 需显式开；③ `PanguGFile` 类的 tensorboard 兼容层不存在（一般不影响）。

---

## 8. 设计含义

1. **零侵入**：业务代码无需写 "if use_fsspec: ..." 分支；用普通 `open()` / `os.path.exists()` 即可支持远端协议
2. **第三方库透明**：进程级 monkeypatch 自动覆盖 TorchRec / TensorBoard / PyTorch Lightning 等所有依赖
3. **C++ IO 不可劫持**：Scripted Model、safetensors 导出必须用 `LOCAL_CACHE_DIR` 中转
4. **测试隔离**：单测可用 `remove_monkeypatch()` 还原 IO，确保 fsspec 路径不被污染
5. **协议缓存**：`_CACHED_FSSPEC_FILESYSTEMS` 减少 S3/GCS client 重建开销，但跨进程不共享

---

## 9. 关键文件

| 文件 | 用途 |
|------|------|
| [`torcheasyrec/tzrec/utils/filesystem_util.py`](../torcheasyrec/tzrec/utils/filesystem_util.py) | 核心：`register_external_filesystem()`、`url_to_fs()`、10 个 patch、协议缓存 |
| [`torcheasyrec/tzrec/utils/filesystem_util.py:169-274`](../torcheasyrec/tzrec/utils/filesystem_util.py#L169-L274) | `PanguGFile`：tensorboard 兼容层（仅阿里元使用） |
| [`torcheasyrec/tzrec/__init__.py:76-80`](../torcheasyrec/tzrec/__init__.py#L76-L80) | 自动调用入口（任何 `import tzrec` 都会触发） |
| [`torcheasyrec/tzrec/main.py:1081-1088`](../torcheasyrec/tzrec/main.py#L1081-L1088) | Scripted Model 加载：C++ IO 绕行（`LOCAL_CACHE_DIR`） |
| [`torcheasyrec/tzrec/utils/export_util.py:97-103`](../torcheasyrec/tzrec/utils/export_util.py#L97-L103) | 导出 safetensors：C++ IO 绕行（`LOCAL_CACHE_DIR`） |
| [`torcheasyrec/tzrec/utils/checkpoint_util.py:182`](../torcheasyrec/tzrec/utils/checkpoint_util.py#L182) | fsspec glob 语义差异注释 |
| [`torcheasyrec/tzrec/utils/filesystem_util_test.py`](../torcheasyrec/tzrec/utils/filesystem_util_test.py) | `USE_FSSPEC=1` 本地 file:// 测试样本 |
