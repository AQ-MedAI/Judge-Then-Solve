#!/bin/bash
# Master evaluation script for Omni-MATH benchmark
# Automatically evaluates all methods and generates summary CSV

set -e

# Default parameters
RESULTS_DIR="results"
INPUT_FILE="Omni-Math.jsonl"
API_KEY="${GPT4O_API_KEY:-${OPENAI_API_KEY:-}}"
BASE_URL="${GPT4O_BASE_URL:-${OPENAI_BASE_URL:-https://api.openai.com/v1}}"
MODEL="gpt-4o-2024-11-20"
CONCURRENCY=8  # Number of concurrent API calls
MAX_RETRIES=3  # Maximum number of retries for failed API calls

echo "=========================================="
echo "Omni-MATH Evaluation Pipeline"
echo "=========================================="
echo ""
echo "Concurrency: $CONCURRENCY"
echo "Max retries: $MAX_RETRIES"
echo ""

if [ -z "$API_KEY" ]; then
    echo "API key is not set. Please export GPT4O_API_KEY or OPENAI_API_KEY."
    exit 1
fi

# Check if results directory exists
if [ ! -d "$RESULTS_DIR" ]; then
    echo "Results directory not found: $RESULTS_DIR"
    echo "Please run inference first using inference_sglang.py"
    exit 1
fi

# Find all method directories and check completion status
echo "Scanning for methods in $RESULTS_DIR..."
method_count=0
completed_count=0
pending_methods=()

# First pass: check which methods need evaluation
for method_dir in "$RESULTS_DIR"/*/ ; do
    if [ -d "$method_dir" ]; then
        method_name=$(basename "$method_dir")

        # Find latest timestamp directory
        latest_dir=$(ls -t "$method_dir" | head -n 1)

        if [ -z "$latest_dir" ]; then
            continue
        fi

        result_path="$method_dir$latest_dir"
        gen_file="$result_path/model_generations.jsonl"
        eval_file="$result_path/gpt4o_evaluations.jsonl"

        # Check if generation file exists
        if [ ! -f "$gen_file" ]; then
            continue
        fi

        method_count=$((method_count + 1))

        # Check if evaluation already exists
        if [ -f "$eval_file" ]; then
            completed_count=$((completed_count + 1))
        else
            pending_methods+=("$method_name")
        fi
    fi
done

echo "Found $method_count methods: $completed_count completed, $((method_count - completed_count)) pending"
echo ""

# Check if all evaluations are complete
if [ $completed_count -eq $method_count ] && [ $method_count -gt 0 ]; then
    echo "=========================================="
    echo "All evaluations already completed!"
    echo "Skipping evaluation phase..."
    echo "=========================================="
    echo ""
else
    # Second pass: run evaluations for pending methods
    echo "Running evaluations for pending methods..."
    echo ""

    for method_dir in "$RESULTS_DIR"/*/ ; do
        if [ -d "$method_dir" ]; then
            method_name=$(basename "$method_dir")
            echo ""
            echo "----------------------------------------"
            echo "Processing method: $method_name"
            echo "----------------------------------------"

            # Find latest timestamp directory
            latest_dir=$(ls -t "$method_dir" | head -n 1)

            if [ -z "$latest_dir" ]; then
                echo "No timestamp directories found for $method_name"
                continue
            fi

            result_path="$method_dir$latest_dir"
            gen_file="$result_path/model_generations.jsonl"
            eval_file="$result_path/gpt4o_evaluations.jsonl"

            echo "Latest result directory: $latest_dir"

            # Check if generation file exists
            if [ ! -f "$gen_file" ]; then
                echo "Generation file not found: $gen_file"
                continue
            fi

            # Check if evaluation already exists
            if [ -f "$eval_file" ]; then
                echo "Evaluation already exists: $eval_file"
                echo "Skipping evaluation for this method"
            else
                echo "Running GPT-4o evaluation (concurrency=$CONCURRENCY)..."
                python3 evaluate_gpt4o.py \
                    --input "$gen_file" \
                    --output "$eval_file" \
                    --api_key "$API_KEY" \
                    --base_url "$BASE_URL" \
                    --model "$MODEL" \
                    --concurrency "$CONCURRENCY" \
                    --max_retries "$MAX_RETRIES"

                echo "Evaluation completed!"
            fi
        fi
    done
fi

echo ""
echo "=========================================="
echo "Processed $method_count methods"
echo "=========================================="
echo ""

# Generate pass@k summary CSV and plots
echo "Generating pass@k summary CSV and plots..."
python3 aggregate_results_pass_at_k.py \
    --results_dir "$RESULTS_DIR" \
    --output_csv "evaluation_pass_at_k.csv" \
    --output_plot_dir "plots"

echo ""
echo "=========================================="
echo "Evaluation pipeline completed!"
echo "Pass@k summary saved to: evaluation_pass_at_k.csv"
echo "Plots saved to: plots/"
echo "=========================================="
