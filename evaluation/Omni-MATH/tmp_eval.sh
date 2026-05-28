python3 inference_sglang.py \
--input Omni-Math-sampled.jsonl \
--method_name "Qwen3-30B-baseline" \
--port 40000 \
--model_name "Qwen3-30B-A3B-Thinking-2507" \
--num-trials 8


python3 inference_sglang.py \
  --input Omni-Math-sampled.jsonl \
  --method_name "Qwen3-30B-with-prompt" \
  --port 40000 \
  --model_name "Qwen3-30B-A3B-Thinking-2507" \
  --use_prompt_method \
  --num-trials 8

bash run_evaluation.sh