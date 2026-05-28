import os
import json
import time
import pathlib
import argparse
import asyncio
from tqdm.asyncio import tqdm as async_tqdm
from openai import AsyncOpenAI

PROMPT_METHOD_TEXT = (
    "[Question] Before reasoning deeply,\n"
    "check whether all necessary\n"
    "information is available. If\n"
    "any key data is missing or\n"
    "ambiguous, explicitly state that\n"
    "first; otherwise, answer it with the\n"
    "minimum number of tokens required.\n\n"
)


async def process_with_sglang_async(args, input_file, output_file, base_url, api_key, prompt_key='insufficient_question', model_name="qwen3-30b-a3b-thinking-2507", concurrency=5):
    """
    Reads data from a JSON file, performs inference using SGLang API with concurrency,
    and saves the results to a JSONL file.

    Args:
        args: Command line arguments
        input_file (str): Path to the input JSON file
        output_file (str): Path where the output JSONL file will be saved
        base_url (str): SGLang base URL
        api_key (str): API key (can be "EMPTY" for local SGLang)
        prompt_key (str): Key in the JSON to use as the prompt
        model_name (str): Model name to use for inference
        concurrency (int): Number of concurrent requests
    """
    # 需要使用 max_completion_tokens 的模型列表
    MODELS_REQUIRING_MAX_COMPLETION_TOKENS = [
        "gpt-5", "o1", "o3", "o4",  # OpenAI 新模型
        "claude-opus-4-6", "claude-sonnet-4-6",  # Claude 新模型
    ]

    # 只允许 temperature=1 的模型列表
    MODELS_REQUIRING_TEMP_1 = [
        "kimi-k2", "kimi-latest",  # Kimi 模型
        "o1", "o3", "o4",  # OpenAI 推理模型
    ]

    # 检查是否需要使用 max_completion_tokens
    use_max_completion_tokens = any(
        model_name.lower().startswith(m.lower()) or m.lower() in model_name.lower()
        for m in MODELS_REQUIRING_MAX_COMPLETION_TOKENS
    )
    if use_max_completion_tokens:
        print(f"Using max_completion_tokens for model: {model_name}")

    # 检查是否需要使用 temperature=1
    use_temp_1 = any(
        model_name.lower().startswith(m.lower()) or m.lower() in model_name.lower()
        for m in MODELS_REQUIRING_TEMP_1
    )
    if use_temp_1:
        print(f"Using temperature=1 for model: {model_name}")
    # Initialize the AsyncOpenAI client with SGLang endpoint
    client = AsyncOpenAI(api_key=api_key, base_url=base_url)

    # Resolve model ID from SGLang server
    print(f"Resolving model ID for: {model_name}")
    model_id = await resolve_model_id_async(client, model_name)
    print(f"Using model: {model_id}")

    # Read the input JSON file
    with open(input_file, 'r') as f:
        data = json.load(f)

    # Define the output key
    output_key = 'model_answer'

    # Load existing processed items if output file exists
    processed_questions = set()
    if os.path.exists(output_file):
        with open(output_file, 'r') as f:
            for line in f:
                if line.strip():
                    item = json.loads(line)
                    if "question" in item:
                        processed_questions.add(item["question"])
        print(f"Loaded {len(processed_questions)} previously processed questions")

    # Filter out already processed items
    items_to_process = []
    for item in data:
        if not isinstance(item, dict) or prompt_key not in item:
            continue
        if "question" in item and item["question"] in processed_questions:
            continue
        items_to_process.append(item)

    print(f"Total items to process: {len(items_to_process)}")

    if len(items_to_process) == 0:
        print("No items to process. Exiting.")
        return

    # Create semaphore for concurrency control
    semaphore = asyncio.Semaphore(concurrency)
    print(f"Using concurrency: {concurrency}")

    # Process items concurrently
    # Create tasks with item references for proper result mapping
    async def process_item_with_ref(item):
        """Wrapper to process item and return both item and response"""
        prompt = item[prompt_key]
        # 当开启 use_prompt_method 时，给问题前面加上 Ours 的提示词
        if getattr(args, "use_prompt_method", False):
            prompt = PROMPT_METHOD_TEXT + prompt
        # 兼容原来的 premise_prompt 开关（可与 use_prompt_method 同时使用）
        if getattr(args, "premise_prompt", False):
            prompt += ' You can tell me if some premise is missing.'
        response = await process_item_async(
            client, model_id, prompt, item, semaphore,
            use_max_completion_tokens=use_max_completion_tokens,
            use_temp_1=use_temp_1
        )
        return item, response

    tasks = [process_item_with_ref(item) for item in items_to_process]

    # Collect results with progress bar using as_completed for true concurrency
    print("Processing items...")
    results = []
    with open(output_file, 'a') as outfile:
        # Use asyncio.as_completed to process results as they complete (true concurrency)
        # This allows tasks to run in parallel and process results as they finish
        completed_tasks = asyncio.as_completed(tasks)
        for coro in async_tqdm(completed_tasks, total=len(tasks), desc="Generating responses"):
            item, response = await coro
            item[output_key] = response
            results.append(item)
            # Write immediately to file
            outfile.write(json.dumps(item, ensure_ascii=False) + '\n')
            outfile.flush()

    print(f"Processing complete. Results saved to {output_file}")
    return results


async def resolve_model_id_async(client, user_model):
    """Resolve model ID from SGLang server."""
    models = await client.models.list()
    ids = [m.id for m in models.data]

    # 1. Exact match
    if user_model in ids:
        return user_model

    # 2. Suffix match (recommended)
    suffix_matches = [mid for mid in ids if mid.endswith(user_model)]
    if len(suffix_matches) == 1:
        return suffix_matches[0]

    # 3. Path-based matching: if user_model is a path, try matching against path components
    # Extract last component(s) from user_model if it looks like a path
    if '/' in user_model:
        # If user_model is a path, try matching the last component or last few components
        path_parts = user_model.rstrip('/').split('/')
        # Try matching with last component
        last_component = path_parts[-1]
        if last_component:
            path_suffix_matches = [mid for mid in ids if mid.endswith(last_component)]
            if len(path_suffix_matches) == 1:
                return path_suffix_matches[0]
        # Try matching with last two components
        if len(path_parts) >= 2:
            last_two = '/'.join(path_parts[-2:])
            path_suffix_matches = [mid for mid in ids if mid.endswith(last_two)]
            if len(path_suffix_matches) == 1:
                return path_suffix_matches[0]

    # 4. Reverse path matching: if server returns paths but user_model is a name,
    # try matching the name against path components
    for mid in ids:
        if '/' in mid:
            # Extract last component from server's model ID
            server_path_parts = mid.rstrip('/').split('/')
            if user_model in server_path_parts:
                return mid

    # 5. Substring match
    contains_matches = [mid for mid in ids if user_model in mid]
    if len(contains_matches) == 1:
        return contains_matches[0]

    # If no match or multiple matches, raise error
    if not suffix_matches and not contains_matches:
        raise ValueError(f"Model '{user_model}' not found. Available models: {ids}")

    raise ValueError(f"Multiple models match '{user_model}': {suffix_matches or contains_matches}")


async def process_item_async(client, model, prompt, original_item, semaphore, max_retries=6, use_max_completion_tokens=False, use_temp_1=False):
    """
    Process a single item with SGLang API asynchronously, with retry logic and concurrency control.

    Args:
        use_max_completion_tokens: If True, use max_completion_tokens instead of max_tokens
                                   (required for some models like gpt-5.2, o1, etc.)
        use_temp_1: If True, use temperature=1 (required for some models like kimi-k2.5, o1, etc.)
    """
    retry_count = 0
    retry_delay = 3

    while retry_count <= max_retries:
        try:
            async with semaphore:
                # 构建请求参数
                temperature = 1 if use_temp_1 else 0.7
                request_params = {
                    "model": model,
                    "messages": [{"role": "user", "content": prompt}],
                    "temperature": temperature,
                }

                # 某些模型（如 gpt-5.2, o1 系列）需要使用 max_completion_tokens 而不是 max_tokens
                if use_max_completion_tokens:
                    request_params["max_completion_tokens"] = 32768
                else:
                    request_params["max_tokens"] = 32768

                completion = await client.chat.completions.create(**request_params)
                response = completion.choices[0].message.content
                return response

        except Exception as e:
            error_str = str(e)
            # 如果错误是关于 max_tokens 不支持，尝试切换到 max_completion_tokens
            if "max_tokens" in error_str and "max_completion_tokens" in error_str and not use_max_completion_tokens:
                print(f"Switching to max_completion_tokens for model {model}")
                use_max_completion_tokens = True
                retry_delay = 1
                continue

            # 如果错误是关于 temperature 不支持，尝试切换到 temperature=1
            if "temperature" in error_str and "only 1" in error_str and not use_temp_1:
                print(f"Switching to temperature=1 for model {model}")
                use_temp_1 = True
                retry_delay = 1
                continue

            retry_count += 1
            if retry_count <= max_retries:
                await asyncio.sleep(retry_delay)
                retry_delay *= 2
            else:
                print(f"Error after {max_retries} attempts: {str(e)}")
                return f"Error after {max_retries} attempts: {str(e)}"


# Main entry point
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Process JSON data with SGLang model (with concurrency)')
    parser.add_argument('--input', type=str, required=True,
                        help='Path to the input JSON file')
    parser.add_argument('--output', type=str, required=True,
                        help='Path where the output JSONL file will be saved')
    parser.add_argument('--MiP', action='store_true', default=False,
                        help='Use "insufficient_question" field instead of "question"')
    parser.add_argument('--model', type=str, default="Qwen3-30B-A3B-Thinking-2507",
                        help='Model name to use (default: Qwen3-30B-A3B-Thinking-2507)')
    parser.add_argument('--base-url', type=str, default=None,
                        help='SGLang base URL (default: from env or http://127.0.0.1:40000/v1)')
    parser.add_argument('--api-key', type=str, default=None,
                        help='API key (default: from env or EMPTY)')
    parser.add_argument('--concurrency', type=int, default=5,
                        help='Number of concurrent requests (default: 5)')
    parser.add_argument("--premise_prompt", action='store_true',
                        help='Add premise prompt to the question')
    parser.add_argument("--use_prompt_method", action='store_true',
                        help='Use Ours prompt method before the question')

    args = parser.parse_args()

    # Get base URL and API key
    base_url = args.base_url or os.getenv("SGLANG_BASE_URL", "http://127.0.0.1:40000/v1")
    api_key = args.api_key or os.getenv("SGLANG_API_KEY", "EMPTY")

    # Ensure output file has the correct extension
    if not args.output.endswith('.jsonl'):
        args.output = args.output + '.jsonl'

    # Create the output directory if it doesn't exist
    output_dir = os.path.dirname(args.output)
    if output_dir:
        pathlib.Path(output_dir).mkdir(parents=True, exist_ok=True)

    # Determine which field to use as prompt
    prompt_key = 'insufficient_question' if args.MiP else 'question'

    # Run the async processing
    asyncio.run(process_with_sglang_async(
        args, args.input, args.output, base_url, api_key,
        prompt_key, args.model, args.concurrency
    ))
