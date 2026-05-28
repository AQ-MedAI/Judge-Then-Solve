# 推理模型评测说明

## 推理模型的评测方式

对于推理模型（如 DeepSeek R1、S1、ChatGPT o1），AbstentionBench 支持两种评测方式：

### 1. 仅评测最终答案（默认方式）

**默认情况下**，评测器只使用模型的**最终答案**（`response`）来判断是否 abstention，而不包含推理过程。

- 使用 `LLMJudgeAbstentionDetector` 类
- 评测的是 `response_or_abstention` 字段（即最终答案）
- 推理链（`reasoning`）会被保存，但不用于 abstention 判断

### 2. 评测推理过程 + 最终答案（可选）

**可选地**，可以配置评测器同时考虑**推理过程**和最终答案来判断 abstention。

- 使用 `LLMJudgeAbstentionDetectorWithReasoning` 类
- 评测的是 `response_with_reasoning` 字段（包含推理链和最终答案）
- 这样可以检测模型在推理过程中是否表达了不确定性

### 如何启用推理过程评测

在配置文件中设置 `abstention_detector_with_reasoning`：

```yaml
# configs/default_pipeline.yaml
defaults:
  - abstention_detector_with_reasoning: llm_judge_llama_3_1_8B_instruct_with_reasoning
```

**注意**：推理过程评测只适用于推理模型（DeepSeek R1、S1），对于普通模型会报错。

### 数据存储结构

推理模型生成的数据包含三个字段：

1. **`response`**: 最终答案（不包含推理过程）
2. **`reasoning`**: 推理链（仅推理过程）
3. **`response_with_reasoning`**: 推理链 + 最终答案的组合

示例：
```
Reasoning Chain:
<推理过程内容>

Final Answer:
最终答案内容
```

### 评测流程

1. **推理阶段**：模型生成推理链和最终答案
2. **Abstention 检测阶段**：
   - 默认：使用 `response`（最终答案）判断
   - 可选：使用 `response_with_reasoning`（推理+答案）判断
3. **评估阶段**：比较检测结果与 ground truth

### 代码位置

- 推理模型实现：`recipe/models.py` 中的 `VLLMReasoningChatModelBase`
- 默认检测器：`recipe/evaluation.py` 中的 `LLMJudgeAbstentionDetector`
- 推理检测器：`recipe/evaluation.py` 中的 `LLMJudgeAbstentionDetectorWithReasoning`

