#!/bin/bash
set -e

export RUN_NAME="${RUN_NAME:-jts-sft-length}"
export CHAT_TEMPLATE=1
export LENGTH_PENALTY=1
export HF_CKPT="${HF_CKPT:-${MODEL_ROOT:?Set MODEL_ROOT or HF_CKPT}/DeepSeek-R1-Distill-Qwen-14B-chat-template-sft}"
export MEGATRON_CKPT="${MEGATRON_CKPT:-${MODEL_ROOT:?Set MODEL_ROOT or MEGATRON_CKPT}/train_torch_list/DeepSeek-R1-Distill-Qwen-14B-chat-template-sft}"
export OUTPUT_MEGATRON_CKPT="${OUTPUT_MEGATRON_CKPT:-${MODEL_ROOT:?Set MODEL_ROOT or OUTPUT_MEGATRON_CKPT}/after_train_torch_list/DeepSeek-R1-Distill-Qwen-14B-jts-sft-length}"

bash "$(dirname "$0")/run_honesty_training.sh"

