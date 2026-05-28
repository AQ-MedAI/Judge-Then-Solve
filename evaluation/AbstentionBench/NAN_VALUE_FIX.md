# NaN 值处理修复

## 🔴 问题

运行 `evaluate_existing_results.py` 时出现错误：

```
ValueError: Input y_pred contains NaN.
```

## 🔍 原因

某些数据集的评估结果中，`is_abstention` 列包含 NaN 值。这通常发生在：
1. 评估任务尚未完成
2. 评估过程中出现错误
3. 某些样本的 judge 评估失败

当 sklearn 的 `precision_score`、`recall_score`、`f1_score` 函数遇到 NaN 值时会报错。

## ✅ 修复方案

修改了 `analysis/tables.py` 中的 `AbstentionF1ScoreTable` 类：

### 修改内容

添加了三个辅助方法来安全地计算指标：

1. `_safe_precision_score()` - 过滤 NaN 后计算 precision
2. `_safe_recall_score()` - 过滤 NaN 后计算 recall
3. `_safe_f1_score()` - 过滤 NaN 后计算 F1

### 修复逻辑

```python
def _safe_precision_score(self, x: pd.DataFrame) -> float:
    """计算precision，过滤掉NaN值"""
    # 过滤掉is_abstention为NaN的行
    valid_mask = x["is_abstention"].notna()
    if valid_mask.sum() == 0:
        return 0.0
    return precision_score(
        x.loc[valid_mask, "prompt_should_abstain"],
        x.loc[valid_mask, "is_abstention"]
    )
```

## 🚀 现在可以运行了

```bash
python evaluate_existing_results.py
```

这个命令现在应该可以成功运行，即使某些样本的评估结果缺失。

## 📊 注意事项

- 如果某个数据集的所有样本都缺失评估结果，该数据集的指标会显示为 0.0
- 建议检查哪些数据集有 NaN 值，确认评估任务是否完成
