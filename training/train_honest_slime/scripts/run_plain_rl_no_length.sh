#!/bin/bash
set -e

export RUN_NAME="${RUN_NAME:-plain-rl-no-length}"
export CHAT_TEMPLATE=0
export LENGTH_PENALTY=0
export HF_CKPT="${HF_CKPT:-${MODEL_ROOT:?Set MODEL_ROOT or HF_CKPT}/DeepSeek-R1-Distill-Qwen-14B}"
export MEGATRON_CKPT="${MEGATRON_CKPT:-${MODEL_ROOT:?Set MODEL_ROOT or MEGATRON_CKPT}/train_torch_list/DeepSeek-R1-Distill-Qwen-14B}"
export OUTPUT_MEGATRON_CKPT="${OUTPUT_MEGATRON_CKPT:-${MODEL_ROOT:?Set MODEL_ROOT or OUTPUT_MEGATRON_CKPT}/after_train_torch_list/DeepSeek-R1-Distill-Qwen-14B-plain-rl-no-length}"

bash "$(dirname "$0")/run_honesty_training.sh"

