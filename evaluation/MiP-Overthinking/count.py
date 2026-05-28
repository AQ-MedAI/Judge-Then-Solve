import os
import json
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--model_name", type=str)
parser.add_argument("--data_root", type=str, default="gsm8k")
parser.add_argument("--version", type=str)
parser.add_argument("--google_api_key", type=str, default=None)
parser.add_argument("--model_path", type=str, default=None,
                    help="Local path to the model (for loading tokenizer). If not specified, will try to use HuggingFace cache or download.")
parser.add_argument("--cache_dir", type=str, default=None,
                    help="Directory to cache the tokenizer (default: use HuggingFace default cache)")
args = parser.parse_args()

word_list = ['alternatively', 'wait', 'check', 'but', 'maybe', 'might', 'perhaps', '\n\n']
model_name = args.model_name
data_root = args.data_root
version = args.version
print(f"Analyzing {model_name} on {data_root}, {version}")

short_model_mapping = {
    "QwQ": "Qwen/QwQ-32B",
    "s1.1": "simplescaling/s1.1-32B",
    "DSQ": "deepseek-ai/DeepSeek-R1-Distill-Qwen-32B",
    "DeepSeek-R1-Distill-Qwen-14B": "deepseek-ai/DeepSeek-R1-Distill-Qwen-14B",
    "Gemma227": "google/gemma-2-27b-it",
    "Limo": "GAIR/LIMO",
    "gpt3_5": "gpt3_5",
    "gpt4o": "gpt4o",
    "o1mini_medium": "o1mini_medium",
    "o3mini_medium": "o3mini_medium",
    "o1": "o1",
    "gemini1_5": "gemini1_5",
    "gemini2": "gemini2",
    "deepseekR1": "deepseek-ai/DeepSeek-R1",
    "sft-qwen-1": "Qwen/Qwen2.5-7B-Instruct",
    "sft-qwen-3": "Qwen/Qwen2.5-7B-Instruct",
        "phi3": "microsoft/Phi-3-medium-128k-instruct",
        "qwen3-30b-thinking": "Qwen/Qwen3-30B-A3B-Thinking-2507",
        "qwen3-30b-thinking_ours_prompt": "Qwen/Qwen3-30B-A3B-Thinking-2507",
        "qwen3-feb3-400": "Qwen/Qwen3-Feb3-400",
        "qwen3-feb9-400": "Qwen/Qwen3-Feb9-400",
        # 注意：训练脚本里传入的是小写的 `qwen3-feb10-600_ptl`
        # 这里为它单独加一条映射到对应的 HF 模型名
        "qwen3-feb10-600_ptl": "Qwen/Qwen3-Feb10-600_ptl",
        # 同时保留原来大写形式，防止其它脚本直接用全名调用
        "Qwen3-Feb10-600_ptl": "Qwen/Qwen3-Feb10-600_ptl",
        # ============ 新增：闭源大模型 ============
        # GPT 系列 - 使用 tiktoken cl100k_base
        "gpt-4o": "gpt-4o",
        "gpt-5.2": "gpt-5.2",
        # Claude 系列 - 使用 tiktoken cl100k_base 近似
        "claude-opus-4-6": "claude-opus-4-6",
        "claude-sonnet-4-6": "claude-sonnet-4-6",
        # Gemini 系列 - 使用 Gemini tokenizer
        "gemini-3.1-pro-preview": "gemini-3.1-pro-preview",
        # Kimi 系列 - 使用 tiktoken cl100k_base 近似
        "kimi-k2.5": "kimi-k2.5",
        "kimi-k2-0905-preview": "kimi-k2-0905-preview",
}

model_type = {
    "Qwen/QwQ-32B": "non-api",
    "simplescaling/s1.1-32B": "non-api",
    "Qwen/Qwen2.5-32B": "non-api",
    "deepseek-ai/DeepSeek-R1-Distill-Qwen-32B": "non-api",
    "deepseek-ai/DeepSeek-R1-Distill-Qwen-14B": "non-api",
    "google/gemma-2-27b-it": "non-api",
    "GAIR/LIMO": "non-api",
    "gpt3_5": "openai",
    "gpt4o": "openai",
    "o1mini_medium": "openai",
    "o3mini_medium": "openai",
    "o1": "openai",
    "gemini1_5": "gemini",
    "gemini2": "gemini",
    "deepseek-ai/DeepSeek-R1": "non-api",
    "Qwen/Qwen2.5-7B-Instruct": "non-api",
    "meta-llama/Llama-3.1-8B-Instruct": "non-api",
    "Qwen/Qwen2.5-32B-Instruct": "non-api",
    "microsoft/Phi-3-medium-128k-instruct": "non-api",
    "Qwen/Qwen2.5-7B-Instruct": "non-api",
    "Qwen/Qwen3-30B-A3B-Thinking-2507": "non-api",
    "Qwen/Qwen3-Feb3-400": "non-api",
    "Qwen/Qwen3-Feb9-400": "non-api",
    "Qwen/Qwen3-Feb10-600_ptl": "non-api",
    # ============ 新增：闭源大模型 ============
    # GPT 系列
    "gpt-4o": "openai",
    "gpt-5.2": "openai",
    # Claude 系列 - 使用 tiktoken 近似
    "claude-opus-4-6": "openai",
    "claude-sonnet-4-6": "openai",
    # Gemini 系列 - 使用 tiktoken 近似（避免依赖 google.generativeai）
    "gemini-3.1-pro-preview": "openai",
    # Kimi 系列 - 使用 tiktoken 近似
    "kimi-k2.5": "openai",
    "kimi-k2-0905-preview": "openai",
}

# 安全获取 long_model_name，给出更友好的报错信息
# 现在支持两种用法：
# 1）传入短名，在 short_model_mapping 里查到对应 HF 模型名（兼容原有脚本）
# 2）直接传入完整 HF 模型名，此时不再强制要求在 short_model_mapping 里登记
if model_name in short_model_mapping:
    long_model_name = short_model_mapping[model_name]
else:
    # 如果没有在映射表中找到，就假定用户已经传入的是完整 HF 模型名
    long_model_name = model_name
    print(
        f"[count.py] 未在 short_model_mapping 中找到 '{model_name}'，"
        f"将其直接视为完整模型名：'{long_model_name}'"
    )
    # 如果 model_type 中也没有登记该模型，默认视为本地非 API 模型
    if long_model_name not in model_type:
        print(
            f"[count.py] 未在 model_type 中找到 '{long_model_name}'，"
            f"默认设置为 'non-api'（本地模型 / HF 模型）。"
        )
        model_type[long_model_name] = "non-api"

# Select the tokenizer or token counter based on the model type.
if model_type[long_model_name] == "non-api":
    from transformers import AutoTokenizer
    # Use local model path if provided, otherwise use HuggingFace model name
    if args.model_path:
        tokenizer_path = args.model_path
        print(f"Loading tokenizer from local path: {tokenizer_path}")
        try:
            tokenizer = AutoTokenizer.from_pretrained(
                tokenizer_path,
                cache_dir=args.cache_dir,
                local_files_only=True  # Only use local files for local paths
            )
        except Exception as e:
            print(f"Warning: Failed to load tokenizer from local path: {e}")
            print("Trying HuggingFace model name instead...")
            tokenizer = AutoTokenizer.from_pretrained(
                long_model_name,
                cache_dir=args.cache_dir,
                local_files_only=True  # Try to use cache only
            )
    else:
        tokenizer_path = long_model_name
        print(f"Loading tokenizer from HuggingFace: {tokenizer_path}")
        try:
            tokenizer = AutoTokenizer.from_pretrained(
                tokenizer_path,
                cache_dir=args.cache_dir,
                local_files_only=False  # Try to download if not in cache
            )
        except Exception as e:
            print(f"Warning: Failed to download tokenizer (network may be unavailable): {e}")
            print("Trying to use cached tokenizer only...")
            tokenizer = AutoTokenizer.from_pretrained(
                tokenizer_path,
                cache_dir=args.cache_dir,
                local_files_only=True  # Fallback to cache only
            )
elif model_type[long_model_name] == "openai":
    import tiktoken
    tokenizer = tiktoken.get_encoding("cl100k_base")
elif model_type[long_model_name] == "gemini":
    import google.generativeai as genai
    genai.configure(api_key=args.google_api_key)
    # Choose the appropriate Gemini endpoint based on the model name.
    if model_name == "gemini1_5":
        gemini_model = genai.GenerativeModel("models/gemini-1.5-flash")
    elif model_name == "gemini2":
        gemini_model = genai.GenerativeModel("models/gemini-2.0-flash")
    elif model_name == "gemini-3.1-pro-preview":
        gemini_model = genai.GenerativeModel("models/gemini-3.1-pro-preview")
    else:
        # 默认使用 gemini-2.0-flash
        gemini_model = genai.GenerativeModel("models/gemini-2.0-flash")

    # Create a simple tokenizer wrapper for Gemini models.
    class GeminiTokenizer:
        def __init__(self, model):
            self.model = model
        def encode(self, text, add_special_tokens=False):
            # Use the Gemini model's count_tokens method.
            token_count = self.model.count_tokens(text).total_tokens
            # Return a dummy list with length equal to the token count.
            return [None] * token_count

    tokenizer = GeminiTokenizer(gemini_model)

result_dir = f"results/{data_root}/{version}/{model_name}"

data = []
for filename in os.listdir(result_dir):
    # Support both .json and .jsonl files, but skip derived/summary files
    # - analysis_info.json: aggregated stats
    # - per_sample_*.jsonl: per-sample evaluation outputs (no model_answer)
    if (
        (filename.endswith('.json') or filename.endswith('.jsonl'))
        and 'analysis_info' not in filename
        and not filename.startswith('per_sample_')
    ):
        file_path = os.path.join(result_dir, filename)
        with open(file_path, 'r', encoding='utf-8') as f:
            if filename.endswith('.jsonl'):
                # Handle JSONL format (line-by-line JSON)
                for line in f:
                    if line.strip():  # Skip empty lines
                        try:
                            data.append(json.loads(line))
                        except json.JSONDecodeError as e:
                            print(f"Warning: Failed to parse line in {filename}: {e}")
                            continue
            else:
                # Handle regular JSON format
                try:
                    file_data = json.load(f)
                    if isinstance(file_data, list):
                        data.extend(file_data)
                    else:
                        data.append(file_data)
                except json.JSONDecodeError:
                    # Fallback: try line-by-line JSON format
                    f.seek(0)  # Reset file pointer to beginning
                    for line in f:
                        if line.strip():  # Skip empty lines
                            try:
                                data.append(json.loads(line))
                            except json.JSONDecodeError as e:
                                print(f"Warning: Failed to parse line in {filename}: {e}")
                                continue

# Check if data is empty
if len(data) == 0:
    print(f"Error: No data found in {result_dir}")
    print(f"Looking for files ending with .json or .jsonl (excluding analysis_info.json and per_sample_*.jsonl)")
    print(f"Found files: {[f for f in os.listdir(result_dir) if (f.endswith('.json') or f.endswith('.jsonl')) and 'analysis_info' not in f and not f.startswith('per_sample_')]}")
    exit(1)

print(f"Loaded {len(data)} items from {result_dir}")

ans_key_tag = 'insufficient_question' if version == 'result' or version == 'prompt_re' else 'question'
model_answer_key = 'model_answer'


def get_answer_length(data, tokenizer, is_openai=False, model_name='', ans_key_tag=''):
    answer_lengths = []
    for item in data:
        if 'o1mini_' in  model_name or 'o3mini_' in  model_name or model_name == 'o1':
            if 'o1mini_' in  model_name:
                item_len = item['o1-mini ({}) all_token_count'.format(ans_key_tag)]
            if 'o3mini_' in  model_name:
                item_len = item['o3-mini ({}) all_token_count'.format(ans_key_tag)]
            if model_name == 'o1':
                item_len = item['o1 ({}) all_token_count'.format(ans_key_tag)]
            answer_lengths.append(item_len)
            pass
        else:
            answer = item[model_answer_key]
            if is_openai:
                # For OpenAI responses, use the tokenizer.encode method without extra arguments
                answer_tokens = tokenizer.encode(answer)
                answer_tokens_len = len(answer_tokens)
            else:
                answer_tokens = tokenizer.encode(answer, add_special_tokens=False)
                answer_tokens_len = len(answer_tokens)
            answer_lengths.append(answer_tokens_len)
    return answer_lengths

def get_word_count(data, word_list, sum_token_length):
    word_counts = {}
    word_freq = {}
    for word in word_list:
        count = 0
        for item in data:
            count += item[model_answer_key].lower().count(word.lower())
        word_counts[word] = count
        word_freq[word] = count / sum_token_length
    return word_counts, word_freq

info = {}
is_openai = model_type[long_model_name] == "openai"
answer_lengths = get_answer_length(data, tokenizer, is_openai=is_openai, model_name=model_name, ans_key_tag=ans_key_tag)

# Check if answer_lengths is empty
if len(answer_lengths) == 0:
    print(f"Error: No answer lengths calculated. Data has {len(data)} items.")
    print(f"Checking if 'model_answer' key exists in data...")
    if data:
        print(f"Sample item keys: {list(data[0].keys())}")
        items_with_model_answer = [item for item in data if 'model_answer' in item]
        print(f"Items with 'model_answer' key: {len(items_with_model_answer)}")
    exit(1)

info['mean_answer_length'] = sum(answer_lengths) / len(answer_lengths)

sum_token_length = sum(answer_lengths)
word_counts, word_freq = get_word_count(data, word_list, sum_token_length)
info['word_counts'] = word_counts
info['word_freq'] = word_freq

# Save the analysis information to a JSON file
output_file = f"{result_dir}/analysis_info.json"
if os.path.exists(output_file):
    with open(output_file, 'r', encoding='utf-8') as f:
        existing_data = json.load(f)
        # Update the existing data with new information
        existing_data.update(info)
        info = existing_data

with open(output_file, 'w', encoding='utf-8') as f:
    json.dump(info, f, indent=4)
