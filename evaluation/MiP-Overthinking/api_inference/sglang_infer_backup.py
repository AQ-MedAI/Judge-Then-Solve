import os
import json
import time
import pathlib
import argparse
from tqdm import tqdm
from openai import OpenAI

def process_with_sglang(args, input_file, output_file, base_url, api_key, prompt_key='insufficient_question', model_name="qwen3-30b-a3b-thinking-2507"):
    """
    Reads data from a JSON file, performs inference using SGLang API
    with the specified field, and saves the results to a JSONL file.
    Each item is saved immediately after processing and only unprocessed items are processed.

    Args:
        args: Command line arguments
        input_file (str): Path to the input JSON file
        output_file (str): Path where the output JSONL file will be saved
        base_url (str): SGLang base URL
        api_key (str): API key (can be "EMPTY" for local SGLang)
        prompt_key (str): Key in the JSON to use as the prompt
        model_name (str): Model name to use for inference
    """
    # Initialize the OpenAI client with SGLang endpoint
    client = OpenAI(api_key=api_key, base_url=base_url)

    # Resolve model ID from SGLang server
    print(f"Resolving model ID for: {model_name}")
    model_id = resolve_model_id(client, model_name)
    print(f"Using model: {model_id}")

    # Read the input JSON file
    with open(input_file, 'r') as f:
        data = json.load(f)

    # Define the output key that will be used to identify processed items
    output_key = 'model_answer'

    # Load existing processed items if output file exists
    processed_questions = set()
    if os.path.exists(output_file):
        with open(output_file, 'r') as f:
            for line in f:
                if line.strip():  # Skip empty lines
                    item = json.loads(line)
                    # Use the "question" field as the identifier
                    if "question" in item:
                        processed_questions.add(item["question"])
        print(f"Loaded {len(processed_questions)} previously processed questions from {output_file}")

    # Process each item in the JSON
    with open(output_file, 'a') as outfile:
        # Process each item in the list if data is a list
        if isinstance(data, list):
            for item in tqdm(data):
                # Skip if the item doesn't have the expected structure
                if not isinstance(item, dict) or prompt_key not in item:
                    print(f"Warning: Item missing '{prompt_key}' field: {item}")
                    continue

                # Skip if the question has already been processed
                if "question" in item and item["question"] in processed_questions:
                    print(f"Skipping already processed question: {item['question'][:50]}...")
                    continue

                # Use the specified key as the prompt
                prompt = item[prompt_key]
                if args.premise_prompt:
                    prompt += ' You can tell me if some premise is missing.'

                response = process_item(client, model_id, prompt, item)

                print(f"Processed item with prompt:\n{prompt[:100]}...")
                print(f"Response length: {len(response)} characters")

                # Add the response to the item
                item[output_key] = response

                # Write the item to the JSONL file
                outfile.write(json.dumps(item, ensure_ascii=False) + '\n')
                outfile.flush()  # Ensure the data is written immediately

                # Update the processed questions set
                if "question" in item:
                    processed_questions.add(item["question"])

    print(f"Processing complete. Results saved to {output_file}")

    # Read all items from the JSONL file to return
    all_items = []
    with open(output_file, 'r') as f:
        for line in f:
            if line.strip():
                all_items.append(json.loads(line))

    return all_items


def resolve_model_id(client, user_model):
    """Resolve model ID from SGLang server."""
    models = client.models.list()
    ids = [m.id for m in models.data]

    # 1. Exact match
    if user_model in ids:
        return user_model

    # 2. Suffix match (recommended)
    suffix_matches = [mid for mid in ids if mid.endswith(user_model)]
    if len(suffix_matches) == 1:
        return suffix_matches[0]

    # 3. Substring match
    contains_matches = [mid for mid in ids if user_model in mid]
    if len(contains_matches) == 1:
        return contains_matches[0]

    # If no match or multiple matches, raise error
    if not suffix_matches and not contains_matches:
        raise ValueError(f"Model '{user_model}' not found. Available models: {ids}")

    raise ValueError(f"Multiple models match '{user_model}': {suffix_matches or contains_matches}")


def strip_thinking_chain(text):
    """Strip <think>...</think> tags from thinking models."""
    if text is None or not isinstance(text, str):
        return text

    marker = "</think>"
    if marker in text:
        # Keep the full response including thinking chain
        # The thinking chain will be stripped during evaluation
        return text
    return text.strip()


def process_item(client, model, prompt, original_item, max_retries=6):
    """
    Process a single item with SGLang API, with retry logic

    Args:
        client: The OpenAI client
        model: The model name
        prompt: The prompt to send
        original_item: The original item data
        max_retries: Maximum number of retry attempts

    Returns:
        The model response text
    """
    retry_count = 0

    while retry_count <= max_retries:
        try:
            # Make the API call
            completion = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "user", "content": prompt},
                ],
                max_tokens=32768,
                temperature=0.7,
            )
            response = completion.choices[0].message.content

            # Keep the full response (including thinking chain if present)
            return response

        except Exception as e:
            retry_count += 1
            if retry_count <= max_retries:
                error_msg = f"Error (attempt {retry_count}/{max_retries}): {str(e)}. Retrying in 3 seconds..."
                print(error_msg)
                time.sleep(3)  # Wait for 3 seconds before retrying
            else:
                # All retries exhausted
                print(f"Error after {max_retries} attempts: {str(e)}. Giving up.")
                return f"Error after {max_retries} attempts: {str(e)}"


# Example usage
if __name__ == "__main__":
    # Setup argument parser
    parser = argparse.ArgumentParser(description='Process JSON data with SGLang model')
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
    parser.add_argument("--premise_prompt", action='store_true',
                        help='Add premise prompt to the question')

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
    if args.MiP:
        key = 'insufficient_question'
    else:
        key = 'question'

    # Process the JSON file
    process_with_sglang(args, args.input, args.output, base_url, api_key, key, args.model)
