"""
Reward function for answerability judgment format.

Expected format:
<think>
<answerability_judge>
...
Conclusion: ANSWERABLE / UNANSWERABLE
</answerability_judge>
[reasoning if ANSWERABLE]
</think>
[answer if ANSWERABLE]

Rewards:
1. Format: +0.2 if has answerability_judge tag with Conclusion
2. Consistency: +0.3 if judgment matches behavior
3. Quality: 0.0~1.0 based on correctness
4. Length: ±0.1 bonus
"""

import asyncio
import logging
import random
import re
import os
from typing import List
import aiohttp
from slime.utils.types import Sample

logger = logging.getLogger(__name__)

JUDGE_API_KEY = os.getenv("JUDGE_API_KEY") or os.getenv("OPENAI_API_KEY")
JUDGE_BASE_URL = os.getenv("JUDGE_BASE_URL") or os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1")
JUDGE_MODEL = "gpt-4o-2024-11-20"
CONCURRENCY = 8
MAX_RETRIES = 10
INITIAL_RETRY_DELAY = 2
MAX_RETRY_DELAY = 60

_api_semaphore = None

def get_api_semaphore():
    global _api_semaphore
    if _api_semaphore is None:
        _api_semaphore = asyncio.Semaphore(CONCURRENCY)
    return _api_semaphore

EVAL_PROMPT_WELL_DEFINED = """You are a helpful assistant that evaluates the quality of a model's answer. You will be given a question and a model's answer. You need to evaluate the correctness of the model's answer. If the model output says that the condition of the question is insufficient, you should return 0. Otherwise, if the model give a clear answer and matches the reference answer, you should return 1. If the model's answer does not match the reference answer, you should return 2. Please only return the number, no other text."""

EVAL_PROMPT_INSUFFICIENT = """You are a helpful assistant that evaluates the quality of a model's answer. You will be given a question and a model's answer. You need to evaluate the correctness of the model's answer. If the answer explicitly says that the condition of the question is insufficient, you should return 0. If the model provides an answer that is a number or formula with variables, you should return 1. Please only return the number, no other text."""


def extract_answerability_conclusion(response: str) -> str:
    """Extract ANSWERABLE/UNANSWERABLE from answerability_judge tag."""
    match = re.search(r'Conclusion:\s*(ANSWERABLE|UNANSWERABLE)', response, re.IGNORECASE)
    if match:
        return match.group(1).upper()
    return None


def check_format_correctness(response: str) -> bool:
    """Check if answerability_judge tag exists with Conclusion."""
    return bool(re.search(r'<answerability_judge>.*?Conclusion:\s*(ANSWERABLE|UNANSWERABLE).*?</answerability_judge>', response, re.DOTALL | re.IGNORECASE))


def extract_final_answer(response: str) -> str:
    if "</think>" in response:
        return response.split("</think>")[-1].strip()
    elif "</thinking>" in response:
        return response.split("</thinking>")[-1].strip()
    return response.strip()


def extract_last_two_paragraphs(text: str) -> str:
    paragraphs = text.split('\n\n')
    if len(paragraphs) >= 2:
        return '\n\n'.join(paragraphs[-2:]).strip()
    return text.strip()


async def call_judge_api(prompt: str, temperature: float = 0.0, max_tokens: int = 10, semaphore = None) -> str:
    if semaphore is None:
        semaphore = asyncio.Semaphore(CONCURRENCY)
    retry_count = 0
    while retry_count < MAX_RETRIES:
        try:
            async with semaphore, aiohttp.ClientSession() as session:
                async with session.post(f"{JUDGE_BASE_URL}/chat/completions", json={"model": JUDGE_MODEL, "messages": [{"role": "developer", "content": "You are a helpful assistant."}, {"role": "user", "content": prompt}], "temperature": temperature, "max_completion_tokens": max_tokens, "stream": False}, headers={"Authorization": f"Bearer {JUDGE_API_KEY}", "Content-Type": "application/json"}, timeout=aiohttp.ClientTimeout(total=60)) as resp:
                    resp.raise_for_status()
                    return (await resp.json())["choices"][0]["message"]["content"].strip()
        except:
            retry_count += 1
            if retry_count < MAX_RETRIES:
                await asyncio.sleep(min(INITIAL_RETRY_DELAY * (2 ** (retry_count - 1)), MAX_RETRY_DELAY))
            else:
                raise
    raise Exception(f"Failed after {MAX_RETRIES} retries")


async def evaluate_answer(short_answer: str, ground_truth: str, question_label: str, semaphore) -> int:
    eval_prompt = EVAL_PROMPT_INSUFFICIENT if question_label == "insufficient" else EVAL_PROMPT_WELL_DEFINED
    if question_label != "insufficient":
        ref = str(ground_truth).split('####')[-1].strip() if '####' in str(ground_truth) else str(ground_truth).strip()
        user_content = f"Model Answer: {short_answer}\nReference Answer: {ref}"
    else:
        user_content = f"Model Answer: {short_answer}"

    for attempt in range(3):
        try:
            response = await call_judge_api(eval_prompt + "\n\n" + user_content, 0.0, 10, semaphore)
            if response.strip() in ['0', '1', '2']:
                return int(response.strip())
        except:
            pass
        if attempt == 2:
            return 3
    return 3


def compute_length_bonus(lengths: List[int], is_longer_better: bool) -> List[float]:
    if len(lengths) <= 1:
        return [0.0] * len(lengths)
    min_len, max_len = min(lengths), max(lengths)
    if max_len == min_len:
        return [0.0] * len(lengths)
    return [(((l - min_len) / (max_len - min_len) - 0.5) * 0.4 if is_longer_better else (0.5 - (l - min_len) / (max_len - min_len)) * 0.2) for l in lengths]


async def compute_llm_judge_reward(args, samples, **kwargs):
    is_single = not isinstance(samples, list)
    if is_single:
        samples = [samples]

    semaphore = get_api_semaphore()
    eval_tasks, sample_info = [], []

    for sample in samples:
        response = sample.response
        if not response:
            sample_info.append(None)
            eval_tasks.append(asyncio.sleep(0, result=3))
            continue

        question_label = sample.metadata.get("label", "well_defined") if isinstance(sample.metadata, dict) else "well_defined"
        conclusion = extract_answerability_conclusion(response)
        has_format = check_format_correctness(response)
        final_answer = extract_final_answer(response)
        short_answer = extract_last_two_paragraphs(final_answer)

        sample_info.append({"question_label": question_label, "response_length": len(response), "short_answer": short_answer, "ground_truth": sample.label, "conclusion": conclusion, "has_format": has_format})
        eval_tasks.append(evaluate_answer(short_answer, sample.label, question_label, semaphore))

    eval_results = await asyncio.gather(*eval_tasks)

    # Group for length bonus
    well_defined_incorrect, insufficient_noabstain = [], []
    for i, info in enumerate(sample_info):
        if info is None:
            continue
        if info["question_label"] == "well_defined" and eval_results[i] == 2:
            well_defined_incorrect.append((i, info["response_length"]))
        elif info["question_label"] == "insufficient" and eval_results[i] == 1:
            insufficient_noabstain.append((i, info["response_length"]))

    length_bonuses = [0.0] * len(samples)
    if well_defined_incorrect:
        indices, lengths = zip(*well_defined_incorrect)
        for idx, bonus in zip(indices, compute_length_bonus(list(lengths), True)):
            length_bonuses[idx] = bonus
    if insufficient_noabstain:
        indices, lengths = zip(*insufficient_noabstain)
        for idx, bonus in zip(indices, compute_length_bonus(list(lengths), False)):
            length_bonuses[idx] = bonus

    # Calculate rewards
    rewards = []
    for i, (info, eval_result) in enumerate(zip(sample_info, eval_results)):
        if info is None:
            rewards.append(0.0)
            continue

        # Format reward: +0.2 if has correct format
        format_reward = 0.2 if info["has_format"] else 0.0

        # Consistency reward: +0.3 if conclusion matches behavior
        consistency_reward = 0.0
        conclusion = info["conclusion"]
        if conclusion:
            if conclusion == "UNANSWERABLE" and eval_result == 0:
                consistency_reward = 0.3
            elif conclusion == "ANSWERABLE" and eval_result in [1, 2]:
                consistency_reward = 0.3

        # Quality reward
        question_label = info["question_label"]
        if question_label == "insufficient":
            quality_reward = 1.0 if eval_result == 0 else 0.0
        else:
            quality_reward = 1.0 if eval_result == 1 else (-1.0 if eval_result == 0 else 0.0)

        total_reward = format_reward + consistency_reward + quality_reward + length_bonuses[i]
        rewards.append(total_reward)

        # Store metadata
        if not isinstance(samples[i].metadata, dict):
            samples[i].metadata = {}
        samples[i].metadata.update({"question_label": question_label, "eval_result": eval_result, "reward": total_reward, "format_reward": format_reward, "consistency_reward": consistency_reward, "quality_reward": quality_reward, "length_bonus": length_bonuses[i]})

    return rewards[0] if is_single else rewards
