# AbstentionBench 统计逻辑详解

## 📋 目录
1. [官方仓库的逻辑](#官方仓库的逻辑)
2. [我们现在的逻辑](#我们现在的逻辑)
3. [核心统计指标计算](#核心统计指标计算)
4. [数据流程](#数据流程)
5. [可能的问题排查](#可能的问题排查)

---

## 官方仓库的逻辑

### 1. 代码示例
```python
from analysis.load_results import Results
from analysis.tables import AbstentionF1ScoreTable

results = Results()
abstention_f1_score = AbstentionF1ScoreTable(results=results)
print(abstention_f1_score.table_df)
```

### 2. 执行步骤

#### 步骤 1: `Results()` 加载数据
- **输入**: 从 `results/` 目录下扫描所有 `GroundTruthAbstentionEvaluator.json` 文件
- **处理**:
  1. 自动发现所有结果路径（格式：`DatasetName_ModelName/YYYY-MM-DD_HH-MM-SS/`）
  2. 加载每个 JSON 文件中的 `responses` 列表
  3. 将每个 response 转换为 flat dictionary
  4. 合并所有数据为 pandas DataFrame

- **关键字段**:
  - `prompt_should_abstain`: 是否应该 abstain（ground truth，来自数据集）
  - `is_abstention`: 模型是否实际 abstain（由 judge 评估得出）
  - `is_abstention_correct`: abstention 是否正确（`is_abstention == prompt_should_abstain`）
  - `is_response_correct`: 回答是否正确（仅在 `prompt_should_abstain=False` 时评估）
  - `model_name`, `dataset_name`, `scenario_label`, `post_training_stage` 等分组字段

#### 步骤 2: `AbstentionF1ScoreTable(results)` 计算指标
- **分组维度**: 按以下字段分组
  - `model_name_formatted`（格式化后的模型名）
  - `scenario_label`（场景标签，如 "underspecified context", "answer unknown" 等）
  - `dataset_name_formatted`（格式化后的数据集名）
  - `post_training_stage`（训练阶段，如 "Base", "SFT", "DPO" 等）

- **计算指标**: 对每个分组计算
  - **Precision**: 预测为 abstain 的样本中，真正应该 abstain 的比例
  - **Recall**: 应该 abstain 的样本中，模型实际 abstain 的比例
  - **F1 Score**: Precision 和 Recall 的调和平均

---

## 我们现在的逻辑

### 1. 代码位置
- **主脚本**: `evaluate_existing_results.py`
- **使用的类**: 完全相同的 `Results` 和 `AbstentionF1ScoreTable`

### 2. 执行步骤

#### 步骤 1: 自动发现结果路径
```python
result_paths = find_all_result_paths(base_results_dir, final_file)
```
- 扫描 `results/` 目录下所有包含 `GroundTruthAbstentionEvaluator.json` 的子目录
- 对每个 `DatasetName_ModelName` 组合，只选择最新的时间戳目录

#### 步骤 2: 加载结果
```python
results = Results(
    base_results_dir=str(base_results_dir),
    result_path_names=result_paths,
    final_file="GroundTruthAbstentionEvaluator.json",
    format=True,
    allow_missing_correctness_files=True,
)
```
- **与官方逻辑完全相同**，只是显式指定了结果路径列表

#### 步骤 3: 生成统计表
```python
abstention_f1_table = AbstentionF1ScoreTable(results=results)
correctness_table = CorrectnessTable(results=results)
```
- **与官方逻辑完全相同**

#### 步骤 4: 保存到 CSV
```python
abstention_f1_table.table_df.to_csv("analysis/abstention_performance.csv", index=False)
correctness_table.table_df.to_csv("analysis/correctness_metrics.csv", index=False)
```

---

## 核心统计指标计算

### Abstention F1 Score 计算

#### 输入数据
- **Ground Truth**: `prompt_should_abstain` (True/False)
- **模型预测**: `is_abstention` (True/False/NaN)

#### 计算过程

```python
def _safe_f1_score(self, x: pd.DataFrame) -> float:
    """计算F1，过滤掉NaN值"""
    # 1. 过滤掉 is_abstention 为 NaN 的行
    valid_mask = x["is_abstention"].notna()
    if valid_mask.sum() == 0:
        return 0.0  # 如果没有有效数据，返回 0.0
    
    # 2. 使用 sklearn 的 f1_score 函数
    return f1_score(
        x.loc[valid_mask, "prompt_should_abstain"],  # y_true: 应该 abstain 吗？
        x.loc[valid_mask, "is_abstention"]          # y_pred: 模型 abstain 了吗？
    )
```

#### 指标含义

**Precision (精确率)**:
```
Precision = TP / (TP + FP)
```
- TP (True Positive): 应该 abstain 且模型 abstain 了
- FP (False Positive): 不应该 abstain 但模型 abstain 了
- **含义**: 模型 abstain 的决策中，有多少是正确的

**Recall (召回率)**:
```
Recall = TP / (TP + FN)
```
- FN (False Negative): 应该 abstain 但模型没有 abstain
- **含义**: 所有应该 abstain 的情况中，模型识别出了多少

**F1 Score (F1分数)**:
```
F1 = 2 * (Precision * Recall) / (Precision + Recall)
```
- **含义**: Precision 和 Recall 的调和平均，综合评估 abstention 性能

### Correctness 计算

#### Accuracy (准确率)
- **计算范围**: 仅针对 `prompt_should_abstain=False` 的样本（即应该回答的问题）
- **公式**: `accuracy = is_response_correct.sum() / len(should_not_abstain_results)`
- **含义**: 在应该回答的问题上，模型回答正确的比例

#### Appropriate Behavior Proportion (适当行为比例)
- **计算范围**: 所有样本
- **逻辑**:
  ```python
  is_behavior_appropriate = (
      is_abstention_correct &  # abstention 决策正确
      is_response_correct.fillna(True)  # 回答正确（如果应该回答的话）
  )
  ```
- **含义**: 模型在所有情况下表现适当的比例（要么正确 abstain，要么正确回答）

---

## 数据流程

```
results/ 目录
  └── DatasetName_ModelName/
      └── YYYY-MM-DD_HH-MM-SS/
          └── GroundTruthAbstentionEvaluator.json
              └── {"responses": [...]}
                  │
                  ├─> Results.load()
                  │   └─> 转换为 DataFrame
                  │       ├─ prompt_should_abstain (ground truth)
                  │       ├─ is_abstention (模型预测)
                  │       ├─ is_abstention_correct
                  │       ├─ is_response_correct
                  │       └─ 其他元数据字段
                  │
                  ├─> AbstentionF1ScoreTable.create_table_df()
                  │   └─> 按 (model, scenario, dataset, stage) 分组
                  │       └─> 计算 precision, recall, f1_score
                  │
                  └─> CorrectnessTable.create_table_df()
                      └─> 计算 accuracy, is_behavior_appropriate
```

---

## 可能的问题排查

### 问题 1: max_tokens 修改后出现问题

#### 可能的原因
1. **数据不完整**: max_tokens 太小导致某些样本的响应被截断，judge 无法正确评估
2. **NaN 值增多**: 截断的响应可能导致 `is_abstention` 或 `is_response_correct` 为 NaN
3. **结果文件格式变化**: 如果修改了生成逻辑，JSON 文件结构可能不匹配

#### 排查步骤

**步骤 1: 检查数据完整性**
```python
# 在 evaluate_existing_results.py 中添加诊断代码
print("\n数据完整性检查:")
print(f"总样本数: {len(results.df)}")
print(f"is_abstention 为 NaN 的数量: {results.df['is_abstention'].isna().sum()}")
print(f"is_response_correct 为 NaN 的数量: {results.df['is_response_correct'].isna().sum()}")

# 按数据集分组检查
print("\n按数据集检查 NaN 情况:")
nan_by_dataset = results.df.groupby('dataset_name').agg({
    'is_abstention': lambda x: x.isna().sum(),
    'is_response_correct': lambda x: x.isna().sum(),
})
print(nan_by_dataset)
```

**步骤 2: 检查结果文件格式**
```python
import json
from pathlib import Path

# 检查一个结果文件
result_file = Path("results/.../GroundTruthAbstentionEvaluator.json")
with open(result_file) as f:
    data = json.load(f)

print(f"响应数量: {len(data.get('responses', []))}")
if data.get('responses'):
    print(f"第一个响应的字段: {list(data['responses'][0].keys())}")
    print(f"is_abstention 类型: {type(data['responses'][0].get('is_abstention'))}")
```

**步骤 3: 对比不同 max_tokens 的结果**
- 检查修改 max_tokens 前后的结果文件
- 确认响应是否被截断（检查 `response_or_abstention` 字段是否以 `...` 结尾）

### 问题 2: 统计结果异常

#### 检查点
1. **分组维度是否正确**: 确认 `scenario_label`, `post_training_stage` 等字段是否正确填充
2. **数据过滤**: 确认 `filter_data()` 是否意外过滤了重要数据
3. **NaN 处理**: 确认 `_safe_*_score` 方法是否正确处理了 NaN

#### 诊断代码
```python
# 检查分组情况
print("\n分组统计:")
print(f"模型数量: {results.df['model_name_formatted'].nunique()}")
print(f"数据集数量: {results.df['dataset_name_formatted'].nunique()}")
print(f"场景标签: {results.df['scenario_label'].unique()}")

# 检查某个特定分组的数据
sample_group = results.df[
    (results.df['model_name_formatted'] == 'YourModel') &
    (results.df['dataset_name_formatted'] == 'YourDataset')
]
print(f"\n样本分组数据:")
print(f"总样本数: {len(sample_group)}")
print(f"应该 abstain: {sample_group['prompt_should_abstain'].sum()}")
print(f"实际 abstain: {sample_group['is_abstention'].sum()}")
print(f"NaN 数量: {sample_group['is_abstention'].isna().sum()}")
```

---

## 总结

### 官方逻辑 vs 我们的逻辑

| 方面 | 官方逻辑 | 我们的逻辑 | 是否一致 |
|------|---------|-----------|---------|
| 数据加载 | `Results()` 自动发现路径 | `Results(result_path_names=...)` 显式指定 | ✅ 一致 |
| 统计计算 | `AbstentionF1ScoreTable` | 相同的类 | ✅ 完全一致 |
| 分组维度 | model, scenario, dataset, stage | 相同 | ✅ 一致 |
| 指标计算 | precision, recall, f1_score | 相同 | ✅ 一致 |
| NaN 处理 | `_safe_*_score` 方法过滤 NaN | 相同 | ✅ 一致 |

**结论**: 我们的逻辑与官方逻辑**完全一致**，只是加载数据的方式略有不同（显式指定路径 vs 自动发现）。

### 如果出现问题

1. **检查数据源**: 确认 `GroundTruthAbstentionEvaluator.json` 文件格式正确
2. **检查 NaN 值**: 确认是否有大量 NaN 导致统计不准确
3. **检查 max_tokens**: 确认响应是否被截断
4. **对比历史结果**: 与修改 max_tokens 之前的结果对比，找出差异
