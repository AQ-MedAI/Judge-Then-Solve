# AbstentionBench 数据集下载脚本

## 📦 脚本说明

本目录包含用于下载 AbstentionBench 所有数据集的脚本。

### 脚本列表

1. **00_download_all.py** - 主下载脚本（推荐使用）
2. **01_download_hf_datasets.py** - HuggingFace 数据集（7个）
3. **02_download_github_datasets.py** - GitHub 数据集（8个）
4. **03_download_gdrive_datasets.py** - Google Drive 数据集（3个）

## 🚀 快速开始

### 前置要求

```bash
# 安装依赖
pip install datasets requests gdown
```

### 使用主脚本下载所有数据集

```bash
cd download_scripts
python 00_download_all.py
```

这个脚本会：
1. 自动下载 HuggingFace 数据集（使用镜像站）
2. 自动下载 GitHub 数据集
3. 询问是否下载 Google Drive 数据集（需要代理）

## 📋 数据集分类

### HuggingFace 数据集（7个）

使用 `https://hf-mirror.com` 镜像站，在中国大陆可以访问：

- CoCoNot
- GPQA
- KUQ
- MMLU (Math + History)
- MoralChoice
- QASPER
- SituatedQA

### GitHub 数据集（8个）

从 GitHub raw content 下载，可能需要代理：

- BBQ (11个文件)
- FalseQA
- MediQ
- SelfAware
- UMWP
- BigBench Disambiguate
- WorldSense
- Squad2

### Google Drive 数据集（3个）

**需要代理或 VPN**：

- ALCUNA
- NQ/Musique
- QAQA

## 🔧 单独运行脚本

如果只需要下载特定类型的数据集：

```bash
# 只下载 HuggingFace 数据集
python 01_download_hf_datasets.py

# 只下载 GitHub 数据集
python 02_download_github_datasets.py

# 只下载 Google Drive 数据集（需要代理）
python 03_download_gdrive_datasets.py
```

## 📁 数据保存位置

所有数据集将保存到：
```
AbstentionBench/
└── data/
    ├── coconot_original/
    ├── coconot_contrast/
    ├── gpqa_diamond/
    ├── kuq_train/
    ├── mmlu_*/
    ├── moralchoice_*/
    ├── qasper_test/
    ├── situated_qa_geo/
    ├── bbq_raw/
    ├── falseqa_test.csv
    ├── mediq_raw/
    ├── selfaware_raw.json
    ├── umwp_test.jsonl
    ├── big_bench_disambiguate.json
    ├── world_sense_raw.jsonl
    ├── squad2_validation.parquet
    ├── alcuna/
    ├── nq_musique/
    └── qaqa/
```

## ⚠️ 注意事项

### 在中国大陆使用

1. **HuggingFace 数据集**：脚本已配置使用镜像站 `https://hf-mirror.com`，可以直接下载
2. **GitHub 数据集**：可能需要代理，但通常可以访问
3. **Google Drive 数据集**：必须使用代理或 VPN

### 设置代理

如果需要使用代理：

```bash
# 设置环境变量
export HTTP_PROXY=http://127.0.0.1:7890
export HTTPS_PROXY=http://127.0.0.1:7890

# 然后运行下载脚本
python 00_download_all.py
```

## 📤 推送到服务器

下载完成后，使用 git 或 zeta 推送到服务器：

```bash
# 方法 1: 使用 git
git add data/
git commit -m "Add downloaded datasets"
git push

# 方法 2: 使用 zeta（如果配置了）
zeta push
```

## 🐛 故障排除

### 问题 1: HuggingFace 连接失败

**解决方案**：确认镜像站设置
```python
import os
print(os.environ.get('HF_ENDPOINT'))  # 应该输出: https://hf-mirror.com
```

### 问题 2: Google Drive 下载失败

**解决方案**：
1. 确认已安装 gdown: `pip install gdown`
2. 确认代理设置正确
3. 尝试手动下载文件

### 问题 3: 磁盘空间不足

**解决方案**：
- 所有数据集大约需要 **10-20 GB** 空间
- 可以选择只下载部分数据集

## 📊 下载进度

脚本会显示详细的下载进度和统计信息：

```
✓ CoCoNot
✓ GPQA
✓ KUQ
...
成功: 15/18
```

## 💡 提示

1. 首次下载可能需要较长时间（取决于网络速度）
2. 如果下载中断，可以重新运行脚本（已下载的数据会被跳过）
3. 建议在网络稳定时下载
