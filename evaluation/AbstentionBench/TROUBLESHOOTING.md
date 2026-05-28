# 故障排除指南

## 常见问题

### 1. `trust_remote_code` 参数错误

**错误信息：**
```
`trust_remote_code` is not supported anymore.
```

**原因：**
新版本的 `datasets` 库（>=3.0）已经移除了 `trust_remote_code` 参数。

**解决方案：**
脚本已经修复，会自动处理这个问题。如果仍然遇到错误，可以：

1. **降级 datasets 库**（不推荐）：
   ```bash
   pip install datasets==2.14.0
   ```

2. **使用修复后的脚本**（推荐）：
   脚本已经自动检测并处理此问题，无需手动操作。

### 2. 网络超时问题

**错误信息：**
```
'timed out' thrown while requesting HEAD https://huggingface.co/...
```

**原因：**
- 网络连接不稳定
- 访问 HuggingFace 速度慢
- 防火墙限制

**解决方案：**

#### 方案 1：使用 HuggingFace 镜像（推荐）

```bash
# 设置镜像环境变量
export HF_ENDPOINT=https://hf-mirror.com

# 然后运行脚本
python download_all_datasets.py
```

#### 方案 2：使用 VPN 或代理

如果在中国大陆，建议使用 VPN 或代理来访问 HuggingFace。

#### 方案 3：增加超时时间

脚本已经设置了合理的超时时间（5分钟），如果仍然超时，可以：

1. 检查网络连接
2. 使用更稳定的网络环境
3. 分批下载数据集（修改脚本，注释掉部分数据集）

#### 方案 4：手动下载

对于特别大的数据集，可以手动从 HuggingFace 下载：

```python
import datasets

# 手动下载单个数据集
dataset = datasets.load_dataset("openai/gsm8k", "main", split="test")
```

### 3. Google Drive 下载失败

**错误信息：**
```
Failed to download from Google Drive
```

**解决方案：**

1. **检查网络**：确保可以访问 Google Drive
2. **使用代理**：如果在中国大陆，需要使用 VPN
3. **手动下载**：从脚本中获取 Google Drive 文件 ID，手动下载

### 4. GitHub 下载失败

**错误信息：**
```
Failed to download from GitHub
```

**解决方案：**

1. **检查网络**：确保可以访问 GitHub
2. **使用代理**：如果在中国大陆，可能需要使用代理
3. **手动下载**：从脚本中获取 URL，使用浏览器或 wget 手动下载

### 5. 依赖库缺失

**错误信息：**
```
ModuleNotFoundError: No module named 'xxx'
```

**解决方案：**

安装所有必需的依赖：

```bash
pip install datasets gdown requests wget pandas jsonlines
```

或者使用项目的 requirements.txt：

```bash
pip install -r requirements.txt
```

## 脚本改进

最新版本的脚本已经包含以下改进：

1. ✅ **自动处理 `trust_remote_code` 问题**：检测并自动移除不支持的参数
2. ✅ **重试机制**：网络失败时自动重试（最多 2 次）
3. ✅ **超时设置**：HTTP 请求设置合理的超时时间
4. ✅ **更好的错误提示**：提供具体的解决方案建议
5. ✅ **指数退避**：重试时使用指数退避策略

## 推荐的下载策略

### 策略 1：使用镜像（最快）

```bash
# 设置镜像
export HF_ENDPOINT=https://hf-mirror.com

# 运行脚本
python download_all_datasets.py
```

### 策略 2：分批下载

如果网络不稳定，可以修改脚本，分批下载：

1. 先下载 HuggingFace 数据集
2. 再下载 GitHub 数据集
3. 最后下载 Google Drive 数据集

### 策略 3：手动下载特定数据集

如果某个数据集一直失败，可以：

1. 查看脚本中该数据集的下载函数
2. 手动运行该函数
3. 或者使用其他工具（如浏览器）手动下载

## 验证下载

下载完成后，检查 `data/` 目录：

```bash
ls -lh data/
```

每个数据集应该有对应的目录或文件。

## 获取帮助

如果问题仍然存在：

1. 查看脚本输出的详细错误信息
2. 检查网络连接
3. 查看 [DOWNLOAD_DATASETS.md](DOWNLOAD_DATASETS.md) 获取更多信息
4. 查看项目的 GitHub Issues

## 环境变量参考

有用的环境变量：

```bash
# HuggingFace 镜像
export HF_ENDPOINT=https://hf-mirror.com

# HuggingFace 缓存目录
export HF_HOME=/path/to/cache

# 代理设置（如果需要）
export HTTP_PROXY=http://proxy.example.com:8080
export HTTPS_PROXY=http://proxy.example.com:8080
```

