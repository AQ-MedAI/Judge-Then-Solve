#!/bin/bash

# 通用的 MiP-Overthinking 评测脚本
# 支持通过参数指定模型短名 / 全名 / 本地路径，一键跑完整 pipeline：
# 1. SGLang 生成回答
# 2. count.py 统计 token & 关键词
# 3. eval.py 用 GPT-4o 打分
#
# 用法示例：
#   测试集：
#     ./run_model.sh --short_name qwen3-feb10-600_ptl \
#                    --full_name Qwen3-Feb10-600_ptl \
#                    --local_path /amed/.../Qwen3-Feb10-600_ptl
#   全量集：
#     DATA_DIR=data ./run_model.sh --short_name qwen3-feb10-600_ptl --full_name Qwen3-Feb10-600_ptl
#
# 其他可选参数：
#   --use_prompt_method   使用 Ours 提示词方法（会在 short_name 后加后缀 _ours_prompt）

set -e  # Exit on error

# ============================================================================
# 解析命令行参数
# ============================================================================

MODEL_NAME=""       # 结果目录里用的短名（如 qwen3-feb10-600_ptl）
MODEL_FULL_NAME=""  # SGLang / HF 的完整模型名（如 Qwen3-Feb10-600_ptl 或 Qwen/QwQ-32B）
LOCAL_MODEL_PATH="" # 本地模型路径，可选；为空则尝试用 HF 名称加载
USE_PROMPT_METHOD=false

print_usage() {
    echo "用法："
    echo "  $0 --short_name <短名> --full_name <完整模型名> [--local_path <本地路径>] [--use_prompt_method]"
    echo ""
    echo "示例："
    echo "  DATA_DIR=data/test \\"
    echo "    $0 --short_name qwen3-feb10-600_ptl \\"
    echo "       --full_name Qwen3-Feb10-600_ptl \\"
    echo "       --local_path /path/to/Qwen3-Feb10-600_ptl"
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --short_name)
            MODEL_NAME="$2"
            shift 2
            ;;
        --full_name)
            MODEL_FULL_NAME="$2"
            shift 2
            ;;
        --local_path)
            LOCAL_MODEL_PATH="$2"
            shift 2
            ;;
        --use_prompt_method)
            USE_PROMPT_METHOD=true
            shift
            ;;
        -h|--help)
            print_usage
            exit 0
            ;;
        *)
            echo "未知参数: $1"
            print_usage
            exit 1
            ;;
    esac
done

if [ -z "$MODEL_NAME" ] || [ -z "$MODEL_FULL_NAME" ]; then
    echo "✗ 必须同时提供 --short_name 和 --full_name"
    print_usage
    exit 1
fi

# 当使用 Ours 提示词方法时，在 model_name 上加后缀，便于区分结果目录
if [ "$USE_PROMPT_METHOD" = true ]; then
    MODEL_NAME="${MODEL_NAME}_ours_prompt"
fi

# 如果环境变量 MODEL_PATH 显式设置，则优先使用
if [ -n "${MODEL_PATH:-}" ]; then
    LOCAL_MODEL_PATH="$MODEL_PATH"
fi

# ============================================================================
# 通用配置（从原有 run_evaluation.sh 抽出来）
# ============================================================================

# SGLang configuration
SGLANG_BASE_URL="${SGLANG_BASE_URL:-http://127.0.0.1:40000/v1}"
SGLANG_API_KEY="${SGLANG_API_KEY:-EMPTY}"

# GPT-4o configuration for evaluation
GPT4O_API_KEY="${GPT4O_API_KEY:-${OPENAI_API_KEY:-}}"
GPT4O_BASE_URL="${GPT4O_BASE_URL:-${OPENAI_BASE_URL:-https://api.openai.com/v1}}"

# Dataset configuration
DATASETS=("gsm8k" "math" "svamp" "formula")
# Some datasets only have MiP version (no normal version)
DATASETS_WITH_NORMAL=("gsm8k" "math")
DATASETS_MIP_ONLY=("svamp" "formula")

# Data directory configuration
# 设置 DATA_DIR=data/test 使用测试集
# 设置 DATA_DIR=data 使用全量数据（默认）
DATA_DIR="${DATA_DIR:-data}"

# Concurrency for API calls
CONCURRENCY=16

# ============================================================================
# 帮助函数
# ============================================================================

print_header() {
    echo ""
    echo "========================================================================"
    echo "$1"
    echo "========================================================================"
    echo ""
}

print_step() {
    echo ""
    echo ">>> $1"
    echo ""
}

# ============================================================================
# 主流程
# ============================================================================

print_header "MiP-Overthinking Evaluation Pipeline"
echo "Model full name      : $MODEL_FULL_NAME"
echo "Model short name(dir): $MODEL_NAME"
echo "Use prompt method    : $USE_PROMPT_METHOD"
echo "Local model path     : ${LOCAL_MODEL_PATH:-<none, use HF/cache>}"
echo "Data Directory       : $DATA_DIR"
echo "SGLang URL           : $SGLANG_BASE_URL"
echo "Datasets             : ${DATASETS[@]}"
echo "Datasets with normal : ${DATASETS_WITH_NORMAL[@]}"
echo "Datasets MiP only    : ${DATASETS_MIP_ONLY[@]}"
echo "Concurrency          : $CONCURRENCY"
echo ""

if [ -z "$GPT4O_API_KEY" ]; then
    echo "✗ Error: GPT4O_API_KEY or OPENAI_API_KEY must be set for judge evaluation"
    exit 1
fi

# 检查 SGLang 是否已经启动
print_step "Checking SGLang server..."
if curl -s "$SGLANG_BASE_URL/v1/models" > /dev/null 2>&1; then
    echo "✓ SGLang server is running"
else
    echo "✗ Error: SGLang server is not accessible at $SGLANG_BASE_URL"
    echo "请先在另一块 GPU 上启动 SGLang server（参考已有脚本注释）"
    exit 1
fi

# 逐数据集、逐版本跑评测
for dataset in "${DATASETS[@]}"; do
    # 决定当前数据集要跑哪些版本
    if [[ " ${DATASETS_MIP_ONLY[@]} " =~ " ${dataset} " ]]; then
        VERSIONS_TO_PROCESS=("MiP")
    else
        VERSIONS_TO_PROCESS=("normal" "MiP")
    fi

    for version in "${VERSIONS_TO_PROCESS[@]}"; do
        print_header "Processing: $dataset - $version"

        INPUT_FILE="${DATA_DIR}/${dataset}.json"
        OUTPUT_DIR="results/${dataset}/${version}/${MODEL_NAME}"
        OUTPUT_FILE="${OUTPUT_DIR}/response.jsonl"

        if [ ! -f "$INPUT_FILE" ]; then
            echo "⚠ Warning: Input file $INPUT_FILE not found, skipping..."
            continue
        fi

        mkdir -p "$OUTPUT_DIR"

        # --------------------------------------------------------------------
        # Step 1: SGLang 生成回答
        # --------------------------------------------------------------------
        print_step "Step 1: Generating model responses..."

        echo "[DEBUG] Calling sglang_infer.py with concurrency=$CONCURRENCY"
        # 如果提供了本地路径，优先使用本地路径作为模型名称（SGLang 服务器返回的模型 ID 通常是完整路径）
        MODEL_NAME_FOR_INFER="${LOCAL_MODEL_PATH:-$MODEL_FULL_NAME}"
        INFER_ARGS=(
            --input "$INPUT_FILE"
            --output "$OUTPUT_FILE"
            --model "$MODEL_NAME_FOR_INFER"
            --base-url "$SGLANG_BASE_URL"
            --api-key "$SGLANG_API_KEY"
            --concurrency "$CONCURRENCY"
        )
        # MiP 版本使用 insufficient_question 字段
        if [ "$version" == "MiP" ]; then
            INFER_ARGS+=(--MiP)
        fi
        # 使用 Ours 提示词方法时，增加开关
        if [ "$USE_PROMPT_METHOD" = true ]; then
            INFER_ARGS+=(--use_prompt_method)
        fi

        python api_inference/sglang_infer.py "${INFER_ARGS[@]}"
        echo "✓ Responses saved to $OUTPUT_FILE"

        # --------------------------------------------------------------------
        # Step 2: 统计 token & 关键词
        # --------------------------------------------------------------------
        print_step "Step 2: Counting tokens and analyzing response length..."

        COUNT_ARGS=(
            --model_name "$MODEL_NAME"
            --data_root "$dataset"
            --version "$version"
        )
        # 如果有本地路径，就传给 count.py，它会优先从本地加载 tokenizer
        if [ -n "$LOCAL_MODEL_PATH" ]; then
            COUNT_ARGS+=(--model_path "$LOCAL_MODEL_PATH")
            echo "Using local model path in count.py: $LOCAL_MODEL_PATH"
        fi

        python count.py "${COUNT_ARGS[@]}"
        echo "✓ Token count analysis completed"

        # --------------------------------------------------------------------
        # Step 3: GPT-4o 打分
        # --------------------------------------------------------------------
        print_step "Step 3: Evaluating with GPT-4o judge..."

        ANALYSIS_INFO_FILE="${OUTPUT_DIR}/analysis_info.json"
        
        EVAL_SKIPPED=false
        if [ -f "$ANALYSIS_INFO_FILE" ]; then
            # 如果已经存在 evaluation_results，就跳过打分
            if python3 -c "import json, sys; data = json.load(open('$ANALYSIS_INFO_FILE')); sys.exit(0 if 'evaluation_results' in data else 1)" 2>/dev/null; then
                echo "✓ Evaluation already completed. Skipping..."
                echo "  Results exist in: $ANALYSIS_INFO_FILE"
                EVAL_SKIPPED=true
            else
                echo "⚠ Analysis file exists but missing evaluation_results. Re-running evaluation..."
            fi
        fi

        if [ "$EVAL_SKIPPED" = false ]; then
            python eval.py \
                --model_name "$MODEL_NAME" \
                --data_root "$dataset" \
                --version "$version" \
                --gpt4o_api_key "$GPT4O_API_KEY" \
                --gpt4o_base_url "$GPT4O_BASE_URL"

            echo "✓ Evaluation completed"
        fi

        echo ""
        echo "Results saved to: ${ANALYSIS_INFO_FILE}"

    done
done

# ============================================================================
# 汇总结果
# ============================================================================

print_header "Evaluation Pipeline Completed!"
echo "All per-dataset results have been saved to their respective directories."
echo ""

print_step "Running summary over all datasets..."
python summarize_results.py --model_name "$MODEL_NAME"
echo ""
