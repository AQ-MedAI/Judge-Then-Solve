#!/bin/bash

set -e

MODEL_NAME="Qwen3-30B-A3B-Thinking-2507"
METHOD_NAME=""
MODEL_PATH=""
PORT="40000"
NUM_TRIALS="8"

print_usage() {
    echo "Usage:"
    echo "  $0 --method_name <result-name> [--model_name <server-model-name>] [--model_path <path>] [--port <port>] [--num_trials <n>]"
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --model_name)
            MODEL_NAME="$2"
            shift 2
            ;;
        --method_name)
            METHOD_NAME="$2"
            shift 2
            ;;
        --model_path)
            MODEL_PATH="$2"
            shift 2
            ;;
        --port)
            PORT="$2"
            shift 2
            ;;
        --num_trials)
            NUM_TRIALS="$2"
            shift 2
            ;;
        -h|--help)
            print_usage
            exit 0
            ;;
        *)
            echo "Unknown argument: $1"
            print_usage
            exit 1
            ;;
    esac
done

if [ -z "$METHOD_NAME" ]; then
    echo "Missing required argument: --method_name"
    print_usage
    exit 1
fi

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SGLANG_BASE_URL="${SGLANG_BASE_URL:-http://127.0.0.1:${PORT}/v1}"
export SGLANG_BASE_URL
export SGLANG_API_KEY="${SGLANG_API_KEY:-EMPTY}"
export ABSTENTIONBENCH_ROOT="${ABSTENTIONBENCH_ROOT:-${ROOT_DIR}/evaluation/AbstentionBench}"

INFER_MODEL_NAME="${MODEL_PATH:-$MODEL_NAME}"

echo "======================================"
echo "Starting Evaluation Pipeline"
echo "======================================"
echo "Model name : $MODEL_NAME"
echo "Method name: $METHOD_NAME"
echo "Infer model: $INFER_MODEL_NAME"
echo "SGLang URL : $SGLANG_BASE_URL"
echo

echo "========== [1/3] Omni-MATH =========="
cd "$ROOT_DIR/evaluation/Omni-MATH"
python3 inference_sglang.py \
  --input Omni-Math-sampled.jsonl \
  --method_name "$METHOD_NAME" \
  --port "$PORT" \
  --model_name "$INFER_MODEL_NAME" \
  --num-trials "$NUM_TRIALS"
bash run_evaluation.sh
python3 generate_takeaway_csv.py
echo "Omni-MATH done"
echo

echo "========== [2/3] AbstentionBench =========="
cd "$ROOT_DIR/evaluation/AbstentionBench"
python3 main.py \
  -m mode=local model=custom_api_env \
  module.model_name="$MODEL_NAME" \
  dataset="mmlu_math,mmlu_history,sub_mediq,gpqa"
bash run_statistics.sh
echo "AbstentionBench done"
echo

echo "========== [3/3] MiP-Overthinking =========="
cd "$ROOT_DIR/evaluation/MiP-Overthinking"
MIP_ARGS=(
  --short_name "$METHOD_NAME"
  --full_name "$MODEL_NAME"
)
if [ -n "$MODEL_PATH" ]; then
  MIP_ARGS+=(--local_path "$MODEL_PATH")
fi
DATA_DIR="${DATA_DIR:-data/test}" bash run_model.sh "${MIP_ARGS[@]}"
python generate_summary_table.py
echo "MiP-Overthinking done"
echo

echo "======================================"
echo "All evaluations finished"
echo "======================================"

