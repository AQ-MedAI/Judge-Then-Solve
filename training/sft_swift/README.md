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

The original training command used full-parameter SFT with 8 GPUs, `qwen3_moe_thinking`, `bfloat16`, 3 epochs, max length 4096, batch size 1 per GPU, gradient accumulation 8, learning rate `1e-5`, and DeepSpeed ZeRO-3.

After SFT finishes, convert the resulting HF checkpoint to Megatron torch distributed format before RL:

```bash
cd ../train_honest_slime
HF_CKPT=/path/to/output-hf-sft-checkpoint \
OUTPUT_MEGATRON_CKPT=/path/to/models/train_torch_list/DeepSeek-R1-Distill-Qwen-14B-sft \
bash scripts/convert_hf_to_megatron.sh
```

For Judge-then-Solve, replace the tokenizer/config chat template before running SFT or before HF-to-Megatron conversion, depending on which checkpoint variant you are preparing. The finalized chat template should be inserted into the tokenizer config once available.

