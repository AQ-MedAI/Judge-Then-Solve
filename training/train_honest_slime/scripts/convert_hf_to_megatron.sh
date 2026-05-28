#!/bin/bash

set -e

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
REPO_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"

MODEL_CONFIG="${MODEL_CONFIG:-${SCRIPT_DIR}/models/DeepSeek-R1-Distill-Qwen-14B.sh}"
HF_CKPT="${HF_CKPT:?Set HF_CKPT to the HuggingFace checkpoint directory}"
OUTPUT_MEGATRON_CKPT="${OUTPUT_MEGATRON_CKPT:?Set OUTPUT_MEGATRON_CKPT to the Megatron output directory}"
MEGATRON_LM_PATH="${MEGATRON_LM_PATH:-/root/Megatron-LM}"
NPROC_PER_NODE="${NPROC_PER_NODE:-8}"

source "$MODEL_CONFIG"

cd "$REPO_DIR"
PYTHONPATH="$MEGATRON_LM_PATH" torchrun --nproc-per-node "$NPROC_PER_NODE" \
  tools/convert_hf_to_torch_dist.py \
  ${MODEL_ARGS[@]} \
  --hf-checkpoint "$HF_CKPT" \
  --save "$OUTPUT_MEGATRON_CKPT"

