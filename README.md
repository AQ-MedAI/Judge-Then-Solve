# Judge-Then-Solve

This repository contains the training and evaluation code prepared for release.

The evaluation release includes three benchmarks:

- `evaluation/Omni-MATH`
- `evaluation/AbstentionBench`
- `evaluation/MiP-Overthinking`

AIME is intentionally not included.

The training release is in:

- `training/sft_swift`
- `training/train_honest_slime`

Training and evaluation data used by the release scripts are in:

- `data/new_honest_training`

## Setup

Create separate environments if desired, or install the benchmark requirements in your preferred environment:

```bash
pip install -r evaluation/MiP-Overthinking/requirements.txt
pip install -r evaluation/AbstentionBench/requirements.txt
pip install -r training/train_honest_slime/requirements.txt
```

Install `ms-swift` separately for the SFT stage.

Omni-MATH uses standard Python packages plus `openai`, `tqdm`, and data/plotting dependencies used by its scripts.

Configure the model server and judge API:

```bash
export SGLANG_BASE_URL="http://127.0.0.1:40000/v1"
export SGLANG_API_KEY="EMPTY"
export OPENAI_API_KEY="your-judge-api-key"
export OPENAI_BASE_URL="https://api.openai.com/v1"
```

If your judge endpoint uses different variables, set `GPT4O_API_KEY`/`GPT4O_BASE_URL` or `JUDGE_API_KEY`/`JUDGE_BASE_URL`.

## Training

The full training pipeline is:

```text
HuggingFace checkpoint -> Swift SFT -> Megatron conversion -> RL training -> HuggingFace conversion -> evaluation
```

Run SFT first with Swift:

```bash
cd training/sft_swift
MODEL_PATH=/path/to/base-hf-checkpoint \
OUTPUT_DIR=/path/to/output-hf-sft-checkpoint \
bash scripts/run_sft.sh
```

The SFT script uses `data/sft/sft_data_final.jsonl` by default.

The RL training code expects model weights in Megatron torch distributed format. Convert each HuggingFace checkpoint before launching RL training.

Set common paths:

```bash
export MODEL_ROOT=/path/to/models
export MEGATRON_LM_PATH=/root/Megatron-LM
cd training/train_honest_slime
```

Convert a HuggingFace checkpoint to Megatron format:

```bash
HF_CKPT=$MODEL_ROOT/DeepSeek-R1-Distill-Qwen-14B \
OUTPUT_MEGATRON_CKPT=$MODEL_ROOT/train_torch_list/DeepSeek-R1-Distill-Qwen-14B \
bash scripts/convert_hf_to_megatron.sh
```

Repeat conversion for the four model variants used in the experiments:

```text
DeepSeek-R1-Distill-Qwen-14B
DeepSeek-R1-Distill-Qwen-14B-sft
DeepSeek-R1-Distill-Qwen-14B-chat-template
DeepSeek-R1-Distill-Qwen-14B-chat-template-sft
```

For the `*-sft` variants, use the checkpoint produced by the Swift SFT step as the HF input.

Run training:

```bash
bash scripts/run_plain_rl_no_length.sh
bash scripts/run_plain_sft_length.sh
bash scripts/run_jts_rl_no_length.sh
bash scripts/run_jts_sft_length.sh
```

The wrappers call `scripts/run_honesty_training.sh`. You can override any path explicitly:

```bash
HF_CKPT=/path/to/hf \
MEGATRON_CKPT=/path/to/train_torch_list/model \
OUTPUT_MEGATRON_CKPT=/path/to/after_train_torch_list/model \
LENGTH_PENALTY=1 \
bash scripts/run_honesty_training.sh
```

Reward function mapping:

```text
plain, no length penalty: slime.rollout.rm_hub.llm_judge_honesty_2_10.compute_llm_judge_reward
plain, length penalty:    slime.rollout.rm_hub.llm_judge_honesty_2_19.compute_llm_judge_reward
chat template, no length: slime.rollout.rm_hub.llm_judge_honesty_3_2.compute_llm_judge_reward
chat template, length:    slime.rollout.rm_hub.llm_judge_honesty_2_28.compute_llm_judge_reward
```

For chat-template runs, the training script keeps:

```text
--group-rm
--custom-rollout-log-function-path slime.rollout.rm_hub.honesty_logging_2_25.log_honesty_metrics_2_25
```

Convert a trained Megatron checkpoint back to HuggingFace format:

```bash
INPUT_MEGATRON_CKPT=$MODEL_ROOT/after_train_torch_list/my-run/iter_0000299 \
OUTPUT_HF_CKPT=$MODEL_ROOT/my-run-hf \
ORIGIN_HF_CKPT=$MODEL_ROOT/DeepSeek-R1-Distill-Qwen-14B-chat-template-sft \
bash scripts/convert_megatron_to_hf.sh
```

For reproducing the Judge-then-Solve setting, replace the relevant HuggingFace tokenizer/config chat template before HF-to-Megatron conversion. The exact replacement template should be inserted here once finalized.

## Run All Evaluations

Start an OpenAI-compatible inference server first, then run:

```bash
bash run_all_evaluation.sh \
  --model_name Qwen3-30B-A3B-Thinking-2507 \
  --method_name my-model \
  --model_path /path/to/local/model \
  --port 40000
```

`--model_path` is optional. When omitted, the scripts use `--model_name` as the model identifier sent to the API server.

## Benchmark Notes

Omni-MATH runs `inference_sglang.py`, then `run_evaluation.sh`, then `generate_takeaway_csv.py`.

AbstentionBench runs the datasets used in the paper evaluation command:

```text
mmlu_math,mmlu_history,sub_mediq,gpqa
```

MiP-Overthinking defaults to `DATA_DIR=data/test` in the unified script for a lightweight release run. Set `DATA_DIR=data` when running the full local data.

No API keys are stored in this release directory. Generated outputs are ignored by `.gitignore`.
