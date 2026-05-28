#!/bin/bash

set -e

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
REPO_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
OFFICIAL_ROOT="$(cd "${REPO_DIR}/../.." && pwd)"

RUN_NAME="${RUN_NAME:-honesty-training}"
LOG_DIR="${LOG_DIR:-${REPO_DIR}/my_logs}"
mkdir -p "$LOG_DIR"
exec > "${LOG_DIR}/${RUN_NAME}.log" 2>&1

set -x

MODEL_CONFIG="${MODEL_CONFIG:-${SCRIPT_DIR}/models/DeepSeek-R1-Distill-Qwen-14B.sh}"
HF_CKPT="${HF_CKPT:?Set HF_CKPT to the HuggingFace checkpoint directory}"
MEGATRON_CKPT="${MEGATRON_CKPT:?Set MEGATRON_CKPT to the converted Megatron checkpoint directory}"
LOAD_MEGATRON_CKPT="${LOAD_MEGATRON_CKPT:-$MEGATRON_CKPT}"
OUTPUT_MEGATRON_CKPT="${OUTPUT_MEGATRON_CKPT:?Set OUTPUT_MEGATRON_CKPT to the training output directory}"

DATA_DIR="${DATA_DIR:-${OFFICIAL_ROOT}/data/new_honest_training}"
TRAIN_DATA="${TRAIN_DATA:-${DATA_DIR}/train_all.jsonl}"
EVAL_DATA="${EVAL_DATA:-${DATA_DIR}/eval_all.jsonl}"

CHAT_TEMPLATE="${CHAT_TEMPLATE:-0}"
LENGTH_PENALTY="${LENGTH_PENALTY:-1}"
MEGATRON_LM_PATH="${MEGATRON_LM_PATH:-/root/Megatron-LM}"
NUM_GPUS="${NUM_GPUS:-8}"
MASTER_ADDR="${MASTER_ADDR:-127.0.0.1}"
RAY_DASHBOARD_PORT="${RAY_DASHBOARD_PORT:-8265}"

source "$MODEL_CONFIG"

if [ "$CHAT_TEMPLATE" = "1" ]; then
    if [ "$LENGTH_PENALTY" = "1" ]; then
        REWARD_FN="slime.rollout.rm_hub.llm_judge_honesty_2_28.compute_llm_judge_reward"
    else
        REWARD_FN="slime.rollout.rm_hub.llm_judge_honesty_3_2.compute_llm_judge_reward"
    fi
else
    if [ "$LENGTH_PENALTY" = "1" ]; then
        REWARD_FN="slime.rollout.rm_hub.llm_judge_honesty_2_19.compute_llm_judge_reward"
    else
        REWARD_FN="slime.rollout.rm_hub.llm_judge_honesty_2_10.compute_llm_judge_reward"
    fi
fi

if [ "${CLEANUP_BEFORE_RUN:-1}" = "1" ]; then
    pkill -9 sglang || true
    ray stop --force || true
    pkill -9 ray || true
    pkill -9 python || true
    pkill -9 redis || true
    sleep 3
fi

export PYTHONBUFFERED=16
export PYTHONLOGLEVEL="${PYTHONLOGLEVEL:-DEBUG}"

NVLINK_COUNT=$(nvidia-smi topo -m 2>/dev/null | grep -o 'NV[0-9][0-9]*' | wc -l)
if [ "$NVLINK_COUNT" -gt 0 ]; then
    HAS_NVLINK=1
else
    HAS_NVLINK=0
fi
echo "HAS_NVLINK: $HAS_NVLINK (detected $NVLINK_COUNT NVLink references)"

CKPT_ARGS=(
   --hf-checkpoint "$HF_CKPT"
   --ref-load "$MEGATRON_CKPT"
   --load "$LOAD_MEGATRON_CKPT"
   --save "$OUTPUT_MEGATRON_CKPT"
   --save-interval "${SAVE_INTERVAL:-100}"
)

ROLLOUT_ARGS=(
   --prompt-data "$TRAIN_DATA"
   --input-key question
   --label-key answer
   --apply-chat-template
   --rollout-shuffle
   --custom-rm-path "$REWARD_FN"
)

if [ "$CHAT_TEMPLATE" = "1" ]; then
   ROLLOUT_ARGS+=(
      --group-rm
      --custom-rollout-log-function-path slime.rollout.rm_hub.honesty_logging_2_25.log_honesty_metrics_2_25
   )
fi

ROLLOUT_ARGS+=(
   --num-rollout "${NUM_ROLLOUT:-400}"
   --rollout-batch-size "${ROLLOUT_BATCH_SIZE:-8}"
   --n-samples-per-prompt "${N_SAMPLES_PER_PROMPT:-8}"
   --rollout-max-response-len "${ROLLOUT_MAX_RESPONSE_LEN:-8192}"
   --rollout-temperature "${ROLLOUT_TEMPERATURE:-0.7}"
   --global-batch-size "${GLOBAL_BATCH_SIZE:-64}"
   --balance-data
)

EVAL_ARGS=(
   --eval-interval "${EVAL_INTERVAL:-25}"
   --eval-prompt-data "$EVAL_DATA"
   --n-samples-per-eval-prompt "${N_SAMPLES_PER_EVAL_PROMPT:-4}"
   --eval-max-response-len "${EVAL_MAX_RESPONSE_LEN:-8192}"
   --eval-top-p 1
)

PERF_ARGS=(
   --tensor-model-parallel-size "${TP_SIZE:-4}"
   --sequence-parallel
   --pipeline-model-parallel-size "${PP_SIZE:-1}"
   --context-parallel-size "${CP_SIZE:-1}"
   --recompute-granularity full
   --recompute-method uniform
   --recompute-num-layers 1
   --use-dynamic-batch-size
   --max-tokens-per-gpu "${MAX_TOKENS_PER_GPU:-20480}"
)

GRPO_ARGS=(
   --advantage-estimator grpo
   --use-kl-loss
   --kl-loss-coef "${KL_LOSS_COEF:-0.00}"
   --kl-loss-type low_var_kl
   --entropy-coef "${ENTROPY_COEF:-0.00}"
   --eps-clip 0.2
   --eps-clip-high 0.28
   --use-tis
   --calculate-per-token-loss
)

OPTIMIZER_ARGS=(
   --optimizer adam
   --lr "${LR:-1e-6}"
   --lr-decay-style constant
   --weight-decay 0.1
   --adam-beta1 0.9
   --adam-beta2 0.98
   --optimizer-cpu-offload
   --overlap-cpu-optimizer-d2h-h2d
   --use-precision-aware-optimizer
)

WANDB_ARGS=()
if [ "${USE_WANDB:-0}" = "1" ]; then
   WANDB_ARGS=(
      --use-wandb
      --wandb-project "${WANDB_PROJECT:-slime-dev}"
      --wandb-group "${WANDB_GROUP:-honesty-training}"
      --wandb-key "${WANDB_KEY:-}"
   )
fi

SGLANG_ARGS=(
   --rollout-num-gpus-per-engine "${ROLLOUT_NUM_GPUS_PER_ENGINE:-8}"
   --sglang-mem-fraction-static "${SGLANG_MEM_FRACTION_STATIC:-0.7}"
   --sglang-cuda-graph-bs 1 2 4 8 $(seq 16 8 256)
)

MISC_ARGS=(
   --attention-dropout 0.0
   --hidden-dropout 0.0
   --accumulate-allreduce-grads-in-fp32
   --attention-softmax-in-fp32
   --attention-backend flash
   --untie-embeddings-and-output-weights
)

ray start --head --node-ip-address "$MASTER_ADDR" --num-gpus "$NUM_GPUS" --disable-usage-stats --dashboard-host=0.0.0.0 --dashboard-port="$RAY_DASHBOARD_PORT"

RUNTIME_ENV_JSON="{
  \"env_vars\": {
    \"PYTHONPATH\": \"${MEGATRON_LM_PATH}\",
    \"CUDA_DEVICE_MAX_CONNECTIONS\": \"1\",
    \"NCCL_NVLS_ENABLE\": \"${HAS_NVLINK}\"
  }
}"

ray job submit --address="http://127.0.0.1:${RAY_DASHBOARD_PORT}" \
   --runtime-env-json="${RUNTIME_ENV_JSON}" \
   --working-dir "$REPO_DIR" \
   -- python3 train.py \
   --actor-num-nodes "${ACTOR_NUM_NODES:-1}" \
   --actor-num-gpus-per-node "$NUM_GPUS" \
   --colocate \
   ${MODEL_ARGS[@]} \
   ${CKPT_ARGS[@]} \
   ${ROLLOUT_ARGS[@]} \
   ${OPTIMIZER_ARGS[@]} \
   ${GRPO_ARGS[@]} \
   ${WANDB_ARGS[@]} \
   ${PERF_ARGS[@]} \
   ${EVAL_ARGS[@]} \
   ${SGLANG_ARGS[@]} \
   ${MISC_ARGS[@]}

