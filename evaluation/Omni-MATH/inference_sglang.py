#!/usr/bin/env python3
"""
Model inference script using sglang deployed models
Supports optional prompt method and timestamp-based result organization
"""
import json
import argparse
import os
from datetime import datetime
from openai import OpenAI
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading


def load_dataset(input_file):
    """Load the Omni-MATH dataset from JSONL file"""
    data = []
    with open(input_file, 'r', encoding='utf-8') as f:
        for line in f:
            data.append(json.loads(line.strip()))
    return data


def create_prompt(problem, use_prompt_method=False):
    """Create the prompt for the model"""
    base_prompt = problem

    if use_prompt_method:
        additional_prompt = (
            "\n\nBefore reasoning deeply, check whether all necessary "
            "information is available. If any key data is missing or "
            "ambiguous, explicitly state that first; otherwise, answer it "
            "with the minimum number of tokens required."
        )
        return base_prompt + additional_prompt

    return base_prompt


def process_single_item(item, client, model_name, use_prompt_method, max_tokens, max_retries=3):
    """Process a single item with the model with retry logic"""
    problem = item['problem']
    prompt = create_prompt(problem, use_prompt_method)

    for attempt in range(max_retries):
        try:
            response = client.chat.completions.create(
                model=model_name,
                messages=[
                    {"role": "user", "content": prompt}
                ],
                temperature=0.7,
                max_tokens=max_tokens
            )

            model_generation = response.choices[0].message.content

            result = {
                "problem": item['problem'],
                "answer": item['answer'],
                "model_generation": model_generation,
                "domain": item.get('domain', ''),
                "difficulty": item.get('difficulty', ''),
                "source": item.get('source', '')
            }
            return result

        except Exception as e:
            if attempt < max_retries - 1:
                print(f"\nRetry {attempt + 1}/{max_retries - 1} after error: {e}")
            else:
                print(f"\nFailed after {max_retries} attempts: {e}")
                result = {
                    "problem": item['problem'],
                    "answer": item['answer'],
                    "model_generation": "",
                    "domain": item.get('domain', ''),
                    "difficulty": item.get('difficulty', ''),
                    "source": item.get('source', '')
                }
                return result


def inference_with_sglang(data, model_name, port, use_prompt_method=False, max_tokens=2048, num_workers=32, num_trials=1):
    """Run inference using sglang deployed model via OpenAI API with concurrent requests"""
    client = OpenAI(
        api_key="EMPTY",
        base_url=f"http://localhost:{port}/v1",
        timeout=300.0
    )

    results = [None] * len(data)
    lock = threading.Lock()

    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        # Submit all tasks (problem_idx, trial_idx)
        future_to_info = {}
        for idx, item in enumerate(data):
            for trial in range(num_trials):
                future = executor.submit(process_single_item, item, client, model_name, use_prompt_method, max_tokens)
                future_to_info[future] = (idx, trial)

        # Process completed tasks
        with tqdm(total=len(data) * num_trials, desc="Running inference") as pbar:
            for future in as_completed(future_to_info):
                idx, trial = future_to_info[future]
                try:
                    result = future.result()
                    with lock:
                        if results[idx] is None:
                            results[idx] = {
                                "problem": result['problem'],
                                "answer": result['answer'],
                                "domain": result['domain'],
                                "difficulty": result['difficulty'],
                                "source": result['source'],
                                "trials": []
                            }
                        results[idx]['trials'].append({
                            "model_generation": result['model_generation']
                        })
                except Exception as e:
                    print(f"\nError for item {idx} trial {trial}: {e}")
                finally:
                    pbar.update(1)

    return results


def save_results(results, output_dir):
    """Save results to JSONL file"""
    os.makedirs(output_dir, exist_ok=True)
    output_file = os.path.join(output_dir, "model_generations.jsonl")

    with open(output_file, 'w', encoding='utf-8') as f:
        for result in results:
            f.write(json.dumps(result, ensure_ascii=False) + '\n')

    print(f"Results saved to: {output_file}")
    return output_file


def main():
    parser = argparse.ArgumentParser(description="Run inference with sglang deployed model")
    parser.add_argument("--input", type=str, default="Omni-Math.jsonl",
                        help="Input JSONL file with problems")
    parser.add_argument("--method_name", type=str, required=True,
                        help="Method name for result folder naming")
    parser.add_argument("--port", type=int, default=40000,
                        help="Port where sglang server is running")
    parser.add_argument("--model_name", type=str, default="default",
                        help="Model name for API calls")
    parser.add_argument("--use_prompt_method", action="store_true",
                        help="Add additional prompt after question")
    parser.add_argument("--output_base", type=str, default="results",
                        help="Base directory for results")
    parser.add_argument("--max_tokens", type=int, default=260000,
                        help="Maximum tokens for model generation")
    parser.add_argument("--num_workers", type=int, default=16,
                        help="Number of concurrent workers for inference")
    parser.add_argument("--num-trials", type=int, default=1,
                        help="Number of trials per problem")

    args = parser.parse_args()

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = os.path.join(args.output_base, args.method_name, timestamp)

    print(f"Method: {args.method_name}")
    print(f"Use prompt method: {args.use_prompt_method}")
    print(f"Max tokens: {args.max_tokens}")
    print(f"Concurrent workers: {args.num_workers}")
    print(f"Trials per problem: {args.num_trials}")
    print(f"Output directory: {output_dir}")

    print(f"Loading dataset from {args.input}...")
    data = load_dataset(args.input)
    print(f"Loaded {len(data)} problems")

    print("Running inference with concurrent requests...")
    results = inference_with_sglang(
        data,
        args.model_name,
        args.port,
        args.use_prompt_method,
        args.max_tokens,
        args.num_workers,
        args.num_trials
    )

    output_file = save_results(results, output_dir)
    print(f"Inference complete! Results saved to {output_file}")


if __name__ == "__main__":
    main()
