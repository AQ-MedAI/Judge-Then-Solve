# 数据集评估失败原因分析

## 🔍 核心问题

你的服务器**无法连接 HuggingFace**，但是代码在运行时**仍然尝试从 HuggingFace 在线加载数据集**！

## 📊 问题详情

### 1. CoCoNot 数据集失败

**代码位置**: `recipe/abstention_datasets/coconot.py:85-86`

```python
def __init__(self, max_num_samples=None):
    super().__init__()

    # ❌ 这里会尝试从 HuggingFace 在线加载！
    coconot_original = datasets.load_dataset("allenai/coconot", "original", split="test")
    coconot_contrast = datasets.load_dataset("allenai/coconot", "contrast", split="test")
```

**问题**:
- 代码使用 `datasets.load_dataset()` 从 HuggingFace 在线加载
- 即使你已经下载了数据到 `data/coconot/` 目录
- 代码**完全忽略了本地数据**，仍然尝试在线加载
- 服务器无法连接 HuggingFace，所以加载失败

**你下载的数据**:
```
data/coconot/original_dataset/
├── data-00000-of-00001.arrow
├── dataset_info.json
└── state.json

data/coconot_original/  # 你的下载脚本保存的位置
data/coconot_contrast/  # 你的下载脚本保存的位置
```

**为什么失败**: 代码期望的路径和你下载的路径不匹配！

---

### 2. KUQ 数据集失败

**代码位置**: `recipe/abstention_datasets/kuq.py:40-42`

```python
def __init__(self, categories: List[str] = None, max_num_samples=None, category_map_path: Optional[str] = None):
    super().__init__()

    # ❌ 这里会尝试从 HuggingFace 在线加载！
    self.dataset = datasets.load_dataset(
        "amayuelas/KUQ", data_files="knowns_unknowns.jsonl"
    )["train"]
```

**问题**:
- 同样使用 `datasets.load_dataset()` 在线加载
- 服务器无法连接 HuggingFace
- 即使你有 `data/kuq/new-category-mapping.csv`，但缺少实际的数据文件

**你下载的数据**:
```
data/kuq/
└── new-category-mapping.csv  # ✅ 有这个文件
    # ❌ 但缺少 knowns_unknowns.jsonl 数据文件！
```

**为什么失败**:
1. 代码尝试在线加载，但服务器无法连接 HuggingFace
2. 你的下载脚本遇到了 Brotli 解压错误，没有成功下载数据文件

---

### 3. SituatedQA 数据集失败

**代码位置**: `recipe/abstention_datasets/situated_qa.py:18-20`

```python
def __init__(self, max_num_samples=None):
    super().__init__()

    # ❌ 这里会尝试从 HuggingFace 在线加载！
    self.dataset = datasets.load_dataset(
        "siyue/SituatedQA", "geo", trust_remote_code=True
    )["test"]
```

**问题**:
- 同样使用 `datasets.load_dataset()` 在线加载
- 服务器无法连接 HuggingFace
- 你的下载脚本遇到了"数据集加载脚本已弃用"的错误

**你下载的数据**:
```
# ❌ 没有找到 SituatedQA 相关的数据文件
```

**为什么失败**: 下载脚本失败，没有成功下载数据

---

### 4. QASPER 数据集失败

**代码位置**: `recipe/abstention_datasets/qasper.py:39`

```python
try:
    # Load the formatted dataset from disk
    self.dataset = datasets.Dataset.load_from_disk(data_dir)
except:
    logger.info("Fetching and processing allenai/qasper")
    # ❌ 这里会尝试从 HuggingFace 在线加载！
    dataset = datasets.load_dataset("allenai/qasper")["test"]
```

**问题**:
- 代码首先尝试从本地 `data/qasper` 加载
- 如果失败，会尝试从 HuggingFace 在线加载
- 你的下载脚本遇到了"数据集加载脚本已弃用"的错误

**你下载的数据**:
```
# ❌ 没有找到 QASPER 相关的数据文件
```

**为什么失败**: 下载脚本失败，没有成功下载数据

---

## 🎯 根本原因总结

### 问题1: 代码设计问题
**所有 HuggingFace 数据集的加载代码都直接调用 `datasets.load_dataset()`**，这会：
1. 首先尝试从 HuggingFace Hub 在线下载
2. 如果本地有缓存才使用缓存
3. **不支持**从你预先下载的 `data/` 目录加载

### 问题2: 下载脚本问题
你的下载脚本遇到了多个问题：
- **KUQ**: Brotli 解压错误
- **QASPER**: 数据集加载脚本已弃用
- **SituatedQA**: 数据集加载脚本已弃用
- **CoCoNot**: 下载成功，但保存路径与代码期望不匹配

### 问题3: HuggingFace 缓存机制
`datasets.load_dataset()` 使用的缓存路径通常是：
- Linux: `~/.cache/huggingface/datasets/`
- 不是你的 `data/` 目录！

即使你下载了数据到 `data/`，代码也不会使用它们。

---

## 💡 解决方案

### 方案1: 修改数据集加载代码（推荐）

修改这些数据集的 Python 文件，让它们从本地 `data/` 目录加载，而不是从 HuggingFace 在线加载。

#### 需要修改的文件：
1. `recipe/abstention_datasets/coconot.py`
2. `recipe/abstention_datasets/kuq.py`
3. `recipe/abstention_datasets/situated_qa.py`
4. `recipe/abstention_datasets/qasper.py`

我可以为你创建修复脚本。

---

### 方案2: 将数据放到 HuggingFace 缓存目录

将你下载的数据复制到 HuggingFace 的缓存目录：
```bash
# 在服务器上执行
mkdir -p ~/.cache/huggingface/datasets/
# 然后将下载的数据复制到这个目录
```

但这个方案**不推荐**，因为：
- 需要了解 HuggingFace 的缓存结构
- 路径和格式必须完全匹配
- 很容易出错

---

### 方案3: 在本地下载完整数据，然后上传到服务器缓存目录

1. 在本地（能访问 HuggingFace 的机器）运行一次评估
2. 这会自动下载数据到 `~/.cache/huggingface/datasets/`
3. 将整个缓存目录打包上传到服务器
4. 在服务器上解压到相同位置

但这个方案也**不推荐**，因为：
- 缓存目录可能很大
- 需要完整运行一次才能获取所有数据

---

## 🚀 推荐的解决方案

**方案1是最佳选择**：修改数据集加载代码，让它们从本地 `data/` 目录加载。

### 具体步骤：

1. **首先完成数据下载**
   - 修复 KUQ、QASPER、SituatedQA 的下载问题
   - 确保所有数据都正确保存到 `data/` 目录

2. **修改数据集加载代码**
   - 修改 4 个数据集的 Python 文件
   - 改为从本地 `data/` 目录加载
   - 我会为你创建修复脚本

3. **重新运行评估**
   - 使用修改后的代码运行评估
   - 这次应该能成功加载本地数据

