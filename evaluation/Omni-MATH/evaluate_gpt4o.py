#!/usr/bin/env python3
"""
GPT-4o evaluation script for Omni-MATH benchmark
Evaluates model generations using GPT-4o as judge
"""
import json
import argparse
import os
import asyncio
import random
from openai import AsyncOpenAI
from tqdm.asyncio import tqdm as async_tqdm


def load_generations(input_file):
    """Load model generations from JSONL file"""
    data = []
    with open(input_file, 'r', encoding='utf-8') as f:
        for line in f:
            data.append(json.loads(line.strip()))
    return data


def extract_final_answer(model_generation):
    """Extract final answer after </think> tag"""
    if "</think>" in model_generation:
        # Extract everything after the last </think> tag
        final_answer = model_generation.split("</think>")[-1].strip()
        return final_answer
    else:
        # If no </think> tag, return the whole response
        return model_generation.strip()


def create_evaluation_prompt(problem, ground_truth, final_answer):
    """Create the evaluation prompt for GPT-4o"""
    prompt = f"""You are an expert mathematics evaluator. Your task is to determine if a student's answer is equivalent to the ground truth answer.

**Problem:**
{problem}

**Ground Truth Answer:**
{ground_truth}

**Student's Final Answer:**
{final_answer}

Please evaluate whether the student's answer is mathematically equivalent to the ground truth answer. Provide your response in the following format:

## Equivalence Judgement
[Write only "TRUE" if the answers are equivalent, or "FALSE" if they are not]

## Justification
[Provide a brief explanation of your judgement]"""

    return prompt


async def evaluate_single_generation_async(client, problem, ground_truth, model_generation, model_name, semaphore, max_retries=3):
    """Evaluate a single generation asynchronously with retry logic and concurrency control"""
    if not model_generation or model_generation.strip() == '':
        return "## Student Final Answer\n\n## Equivalence Judgement\nFALSE\n## Justification\n超出上下文窗口"

    final_answer = extract_final_answer(model_generation)
    prompt = create_evaluation_prompt(problem, ground_truth, final_answer)

    retry_count = 0
    retry_delay = 1

    while retry_count < max_retries:
        try:
            async with semaphore:
                response = await client.chat.completions.create(
                    model=model_name,
                    messages=[
                        {"role": "system", "content": "You are an expert mathematics evaluator."},
                        {"role": "user", "content": prompt}
                    ],
                    temperature=0.0,
                    max_tokens=300
                )
                return response.choices[0].message.content
        except Exception as e:
            retry_count += 1
            if retry_count < max_retries:
                # Add jitter to avoid thundering herd
                jitter = retry_delay * 0.1 * (2 * random.random() - 1)
                await asyncio.sleep(retry_delay + jitter)
                retry_delay *= 2
            else:
                print(f"Warning: Evaluation failed after {max_retries} retries: {e}")
                return "## Student Final Answer\n\n## Equivalence Judgement\nERROR\n## Justification\nEvaluation failed"

async def process_item_async(item, client, model_name, semaphore, max_retries=3):
    """Process a single item asynchronously"""
    problem = item['problem']
    ground_truth = item['answer']

    if 'trials' in item:
        # Multi-trial format - process all trials concurrently
        trial_tasks = []
        for trial in item['trials']:
            task = evaluate_single_generation_async(
                client, problem, ground_truth, trial['model_generation'], model_name, semaphore, max_retries
            )
            trial_tasks.append(task)
        
        # Wait for all trials to complete
        evaluations = await asyncio.gather(*trial_tasks)
        
        trials_with_eval = []
        for trial, evaluation in zip(item['trials'], evaluations):
            trials_with_eval.append({
                'model_generation': trial['model_generation'],
                'omni_judge': evaluation
            })

        item_copy = item.copy()
        item_copy['trials'] = trials_with_eval
        return {
            "original_json": json.dumps(item_copy),
            "gen": evaluations[-1] if evaluations else ""  # Last trial's evaluation for compatibility
        }
    else:
        # Single trial format
        evaluation = await evaluate_single_generation_async(
            client, problem, ground_truth, item.get('model_generation', ''), model_name, semaphore, max_retries
        )
        return {
            "original_json": json.dumps(item),
            "gen": evaluation
        }


async def evaluate_with_gpt4o_async(data, api_key, base_url, model_name="gpt-4o-2024-11-20", concurrency=8, max_retries=3):
    """Evaluate generations using GPT-4o with concurrent API calls"""
    client = AsyncOpenAI(api_key=api_key, base_url=base_url)
    semaphore = asyncio.Semaphore(concurrency)
    
    print(f"Processing {len(data)} items with concurrency={concurrency}, max_retries={max_retries}")

    # Create tasks for all items
    tasks = [process_item_async(item, client, model_name, semaphore, max_retries) for item in data]

    # Process all items concurrently with progress bar
    results = []
    with async_tqdm(total=len(tasks), desc="Evaluating with GPT-4o") as pbar:
        # Process in batches to show progress
        batch_size = 10
        for i in range(0, len(tasks), batch_size):
            batch = tasks[i:i+batch_size]
            try:
                batch_results = await asyncio.gather(*batch, return_exceptions=True)
                for result in batch_results:
                    if isinstance(result, Exception):
                        print(f"\nError processing item: {result}")
                        results.append({
                            "original_json": "{}",
                            "gen": "## Student Final Answer\n\n## Equivalence Judgement\nERROR\n## Justification\nProcessing failed"
                        })
                    else:
                        results.append(result)
                    pbar.update(1)
            except Exception as e:
                print(f"\nError processing batch: {e}")
                # Add error results for the entire batch
                for _ in batch:
                    results.append({
                        "original_json": "{}",
                        "gen": "## Student Final Answer\n\n## Equivalence Judgement\nERROR\n## Justification\nBatch processing failed"
                    })
                    pbar.update(1)

    return results


def save_evaluations(results, output_file):
    """Save evaluation results to JSONL file"""
    os.makedirs(os.path.dirname(output_file), exist_ok=True)

    with open(output_file, 'w', encoding='utf-8') as f:
        for result in results:
            f.write(json.dumps(result, ensure_ascii=False) + '\n')

    print(f"Evaluations saved to: {output_file}")


def main():
    parser = argparse.ArgumentParser(description="Evaluate model generations with GPT-4o")
    parser.add_argument("--input", type=str, required=True,
                        help="Input JSONL file with model generations")
    parser.add_argument("--output", type=str, default=None,
                        help="Output JSONL file for evaluations (default: input_dir/gpt4o_evaluations.jsonl)")
    parser.add_argument("--api_key", type=str, default=None,
                        help="API key for GPT-4o. Defaults to GPT4O_API_KEY or OPENAI_API_KEY.")
    parser.add_argument("--base_url", type=str, default=None,
                        help="Base URL for API")
    parser.add_argument("--model", type=str, default="gpt-4o-2024-11-20",
                        help="Model name to use")
    parser.add_argument("--concurrency", type=int, default=8,
                        help="Number of concurrent API calls (default: 8)")
    parser.add_argument("--max_retries", type=int, default=3,
                        help="Maximum number of retries for failed API calls (default: 3)")

    args = parser.parse_args()

    args.api_key = args.api_key or os.getenv("GPT4O_API_KEY") or os.getenv("OPENAI_API_KEY")
    args.base_url = args.base_url or os.getenv("GPT4O_BASE_URL") or os.getenv("OPENAI_BASE_URL") or "https://api.openai.com/v1"
    if not args.api_key:
        raise ValueError("API key is required. Set GPT4O_API_KEY/OPENAI_API_KEY or pass --api_key.")

    # Set default output path if not provided
    if args.output is None:
        input_dir = os.path.dirname(args.input)
        args.output = os.path.join(input_dir, "gpt4o_evaluations.jsonl")

    print(f"Input file: {args.input}")
    print(f"Output file: {args.output}")
    print(f"Model: {args.model}")
    print(f"Concurrency: {args.concurrency}")
    print(f"Max retries: {args.max_retries}")

    # Load generations
    print("Loading model generations...")
    data = load_generations(args.input)
    print(f"Loaded {len(data)} generations")

    # Evaluate with GPT-4o (async)
    print("Starting evaluation with GPT-4o...")
    results = asyncio.run(evaluate_with_gpt4o_async(
        data, args.api_key, args.base_url, args.model, args.concurrency, args.max_retries
    ))

    # Save results
    save_evaluations(results, args.output)
    print("Evaluation complete!")


if __name__ == "__main__":
    main()
