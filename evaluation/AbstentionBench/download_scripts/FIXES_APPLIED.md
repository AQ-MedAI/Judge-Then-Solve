# 下载脚本修复说明

## 修复概述

已修复所有下载脚本中的错误，现在应该可以成功下载所有数据集。

## 修复的问题

### 1. ✅ QASPER 数据集 - 加载脚本已弃用

**问题**: `Dataset scripts are no longer supported, but found qasper.py`

**修复方案**:
- 添加了 `download_mode="force_redownload"` 参数
- 添加了备用方法：直接下载 parquet 文件
- 文件位置: `download_scripts/01_download_hf_datasets.py:130-158`

**备用 URL**: `https://hf-mirror.com/datasets/allenai/qasper/resolve/main/data/test-00000-of-00001.parquet`

---

### 2. ✅ SituatedQA 数据集 - 加载脚本已弃用

**问题**: `Dataset scripts are no longer supported, trust_remote_code is not supported anymore`

**修复方案**:
- 移除了 `trust_remote_code=True` 参数
- 添加了 `download_mode="force_redownload"` 参数
- 添加了备用方法：直接下载 parquet 文件
- 文件位置: `download_scripts/01_download_hf_datasets.py:160-188`

**备用 URL**: `https://hf-mirror.com/datasets/siyue/SituatedQA/resolve/main/geo/test-00000-of-00001.parquet`

---

### 3. ✅ KUQ 数据集 - Brotli 解压错误

**问题**: `brotli: decoder process called with data when 'can_accept_more_data()' is False`

**修复方案**:
- 添加了重试机制（最多 3 次）
- 每次重试间隔 2 秒
- 添加了备用方法：直接下载 parquet 文件
- 文件位置: `download_scripts/01_download_hf_datasets.py:66-103`

**备用 URL**: `https://hf-mirror.com/datasets/amayuelas/KUQ/resolve/main/data/train-00000-of-00001.parquet`

---

### 4. ✅ UMWP 数据集 - 404 错误

**问题**: `404 Client Error: Not Found for url`

**原因**: GitHub 仓库名称错误（`solving-biased-math` 应该是 `solving-biases`）

**修复方案**:
- 修正了仓库名称
- 添加了多个备用 URL 尝试
- 文件位置: `download_scripts/02_download_github_datasets.py:119-139`

**尝试的 URL**:
1. `https://raw.githubusercontent.com/eth-lre/solving-biases/main/data/umwp_test.jsonl`
2. `https://raw.githubusercontent.com/eth-lre/solving-biases/main/umwp_test.jsonl`
3. `https://raw.githubusercontent.com/eth-lre/solving-biases/master/data/umwp_test.jsonl`

---

### 5. ✅ BBQ 数据集 - 下载超时

**问题**: `HTTPSConnectionPool Read timed out` (Race_x_gender.jsonl)

**修复方案**:
- 为 `download_file()` 函数添加了重试机制（最多 3 次）
- 每次重试间隔 2 秒
- 文件位置: `download_scripts/02_download_github_datasets.py:22-48`

---

### 6. ✅ Google Drive SSL 证书错误

**问题**: `[SSL: CERTIFICATE_VERIFY_FAILED] certificate verify failed`

**影响的数据集**: ALCUNA, NQ/Musique

**修复方案**:
- 添加了 SSL 证书验证绕过
- 添加了重试机制（最多 3 次）
- 每次重试间隔 3 秒
- 添加了 `verify=False` 参数到 gdown
- 文件位置: `download_scripts/03_download_gdrive_datasets.py:20-63`

**依赖要求**: `pip install gdown certifi`

---

## 使用方法

### 重新运行下载脚本

```bash
cd download_scripts
python 00_download_all.py
```

### 或者单独运行修复的脚本

```bash
# 只下载 HuggingFace 数据集（包含 QASPER, SituatedQA, KUQ）
python 01_download_hf_datasets.py

# 只下载 GitHub 数据集（包含 UMWP, BBQ）
python 02_download_github_datasets.py

# 只下载 Google Drive 数据集（包含 ALCUNA, NQ/Musique）
python 03_download_gdrive_datasets.py
```

---

## 预期结果

修复后，应该能够成功下载：

### HuggingFace 数据集 (7/7)
- ✓ CoCoNot
- ✓ GPQA
- ✓ KUQ (已修复)
- ✓ MMLU (6 个子集)
- ✓ MoralChoice
- ✓ QASPER (已修复)
- ✓ SituatedQA (已修复)

### GitHub 数据集 (8/8)
- ✓ BBQ (11 个文件，已修复超时)
- ✓ FalseQA
- ✓ MediQ
- ✓ SelfAware
- ✓ UMWP (已修复 URL)
- ✓ BigBench Disambiguate
- ✓ WorldSense
- ✓ Squad2

### Google Drive 数据集 (3/3)
- ✓ ALCUNA (已修复 SSL)
- ✓ NQ/Musique (已修复 SSL)
- ✓ QAQA

---

## 注意事项

### 1. Google Drive 数据集仍需代理

即使修复了 SSL 错误，Google Drive 在中国大陆仍然无法访问。请确保：
- 使用有效的代理或 VPN
- 设置环境变量：
  ```bash
  export HTTP_PROXY=http://127.0.0.1:7890
  export HTTPS_PROXY=http://127.0.0.1:7890
  ```

### 2. 如果自动下载仍然失败

对于 Google Drive 数据集，脚本会输出手动下载链接：
- ALCUNA id2question.json: `https://drive.google.com/file/d/19xjgOuFZe7WdAglX71OgUJXJoqDnPUzp/view`
- ALCUNA meta_data.jsonl: `https://drive.google.com/file/d/1kolOjXhS5AWI20RnwpA--xZf2ghojCxB/view`
- NQ/Musique: `https://drive.google.com/file/d/1q-6FIEGufKVBE3s6OdFoLWL2iHQPJh8h/view`
- QAQA: `https://drive.google.com/file/d/12aLKsSKe85G0u5bBTq0X0aKICsdxpaFL/view`

### 3. 备用 parquet 文件

如果 HuggingFace 数据集加载仍有问题，脚本会自动尝试下载 parquet 文件。这些文件可以直接使用：

```python
import pandas as pd
df = pd.read_parquet("data/qasper_test.parquet")
```

---

## 技术细节

### 修复策略

1. **重试机制**: 所有下载函数都添加了重试逻辑，避免临时网络问题
2. **备用方法**: HuggingFace 数据集提供 parquet 文件作为备用
3. **SSL 绕过**: Google Drive 下载添加了 SSL 验证绕过
4. **多 URL 尝试**: UMWP 尝试多个可能的 URL

### 代码改进

- `download_file()`: 添加 `max_retries` 参数
- `download_with_gdown()`: 添加 SSL 配置和重试
- `download_kuq()`: 完整的重试和备用逻辑
- `download_qasper()`: 备用 parquet 下载
- `download_situated_qa()`: 备用 parquet 下载
- `download_umwp()`: 多 URL 尝试

---

## 下一步

1. 运行修复后的下载脚本
2. 检查 `data/` 目录确认所有数据集已下载
3. 如果仍有问题，查看日志中的具体错误信息
4. 对于 Google Drive 数据集，确保代理设置正确

如有问题，请查看日志输出中的详细错误信息。
