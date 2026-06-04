# Swift SFT

This directory contains the supervised fine-tuning step that should be run before RL training.

Install the Swift framework in the training environment, then run:

```bash
cd training/sft_swift
MODEL_PATH=/path/to/base-hf-checkpoint \
OUTPUT_DIR=/path/to/output-hf-sft-checkpoint \
bash scripts/run_sft.sh
```

By default the script uses:

```text
data/sft/sft_data_final.jsonl
```

## Judge-then-Solve Chat Template

For Qwen3 Judge-then-Solve runs, apply the provided chat template to the HuggingFace checkpoint before SFT or before converting the HF checkpoint to Megatron format:

```bash
python scripts/apply_jts_chat_template.py \
  --checkpoint-dir /path/to/qwen3-hf-checkpoint
```

This writes `tokenizer_config.json` using `qwen3_jts_tokenizer_config.json` and backs up the previous config as `tokenizer_config.json.bak`.

The original training command used full-parameter SFT with 8 GPUs, `qwen3_moe_thinking`, `bfloat16`, 3 epochs, max length 4096, batch size 1 per GPU, gradient accumulation 8, learning rate `1e-5`, and DeepSpeed ZeRO-3.

After SFT finishes, convert the resulting HF checkpoint to Megatron torch distributed format before RL:

```bash
cd ../train_honest_slime
HF_CKPT=/path/to/output-hf-sft-checkpoint \
OUTPUT_MEGATRON_CKPT=/path/to/models/train_torch_list/DeepSeek-R1-Distill-Qwen-14B-sft \
bash scripts/convert_hf_to_megatron.sh
```

For the chat-template variants, use the checkpoint after applying this template as the source checkpoint for SFT and/or HF-to-Megatron conversion.
