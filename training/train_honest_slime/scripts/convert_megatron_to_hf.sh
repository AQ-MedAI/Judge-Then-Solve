#!/bin/bash

set -e

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
REPO_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"

INPUT_MEGATRON_CKPT="${INPUT_MEGATRON_CKPT:?Set INPUT_MEGATRON_CKPT to a Megatron iter_* checkpoint directory}"
OUTPUT_HF_CKPT="${OUTPUT_HF_CKPT:?Set OUTPUT_HF_CKPT to the output HuggingFace checkpoint directory}"
ORIGIN_HF_CKPT="${ORIGIN_HF_CKPT:?Set ORIGIN_HF_CKPT to the original HuggingFace checkpoint directory}"
MEGATRON_LM_PATH="${MEGATRON_LM_PATH:-/root/Megatron-LM}"

cd "$REPO_DIR"
PYTHONPATH="$MEGATRON_LM_PATH" python tools/convert_torch_dist_to_hf.py \
  --input-dir "$INPUT_MEGATRON_CKPT" \
  --output-dir "$OUTPUT_HF_CKPT" \
  --origin-hf-dir "$ORIGIN_HF_CKPT"

