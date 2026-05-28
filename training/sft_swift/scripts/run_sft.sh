#!/bin/bash

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
OFFICIAL_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"

CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"
export CUDA_VISIBLE_DEVICES
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"

NPROC_PER_NODE="${NPROC_PER_NODE:-8}"
MODEL_PATH="${MODEL_PATH:?Set MODEL_PATH to the base HuggingFace checkpoint directory}"
DATASET_PATH="${DATASET_PATH:-${OFFICIAL_ROOT}/data/sft/sft_data_final.jsonl}"
OUTPUT_DIR="${OUTPUT_DIR:-${OFFICIAL_ROOT}/outputs/sft/qwen3_sft}"
SWIFT_SFT_ENTRY="${SWIFT_SFT_ENTRY:-$(python - <<'PY'
import importlib.util
spec = importlib.util.find_spec("swift")
if spec is None or spec.origin is None:
    raise SystemExit("swift is not installed; install ms-swift first")
print(spec.origin.rsplit("/", 2)[0] + "/cli/sft.py")
PY
)}"

python -m torch.distributed.run --nproc_per_node "$NPROC_PER_NODE" "$SWIFT_SFT_ENTRY" \
  --model "$MODEL_PATH" \
  --dataset "$DATASET_PATH" \
  --train_type full \
  --output_dir "$OUTPUT_DIR" \
  --model_type "${MODEL_TYPE:-qwen3_moe_thinking}" \
  --torch_dtype "${TORCH_DTYPE:-bfloat16}" \
  --num_train_epochs "${NUM_TRAIN_EPOCHS:-3}" \
  --max_length "${MAX_LENGTH:-4096}" \
  --per_device_train_batch_size "${PER_DEVICE_TRAIN_BATCH_SIZE:-1}" \
  --gradient_accumulation_steps "${GRADIENT_ACCUMULATION_STEPS:-8}" \
  --learning_rate "${LEARNING_RATE:-1e-5}" \
  --lora_rank "${LORA_RANK:-16}" \
  --lora_alpha "${LORA_ALPHA:-32}" \
  --target_modules "${TARGET_MODULES:-all-linear}" \
  --deepspeed "${DEEPSPEED:-zero3}" \
  --save_steps "${SAVE_STEPS:-100}" \
  --save_total_limit "${SAVE_TOTAL_LIMIT:-5}" \
  --logging_steps "${LOGGING_STEPS:-5}"

