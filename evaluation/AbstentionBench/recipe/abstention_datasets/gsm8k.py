"""
Copyright (c) Meta Platforms, Inc. and affiliates.
All rights reserved.
This source code is licensed under the license found in the
LICENSE file in the root directory of this source tree.
"""
import re
import os
from pathlib import Path

from datasets import Dataset, concatenate_datasets, load_dataset

from recipe.abstention_datasets.abstract_abstention_dataset import (
    AbstentionDataset,
    Prompt,
)


class GSM8K(AbstentionDataset):
    def __init__(self, split="test", max_num_samples=None, data_dir="data/gsm8k"):
        self.max_num_samples = max_num_samples
        self.data_dir = data_dir
        self.gsm8k_generator = GSM8KGenerator(split=split, data_dir=data_dir)
        self.dataset = self.create_dataset()

    def create_dataset(self) -> Dataset:
        dataset_with_context = self.gsm8k_generator.dataset_with_context
        dataset_without_context = self.gsm8k_generator.dataset_without_context
        return concatenate_datasets([dataset_with_context, dataset_without_context])

    def __len__(self):
        if self.max_num_samples is not None:
            return min(len(self.dataset), self.max_num_samples)
        return len(self.dataset)

    def _parse_final_answer(self, answer: str) -> str:
            return answer.split("### ", 1)[1]

    def __getitem__(self, idx) -> Prompt:
        if idx > len(self.dataset):
            raise IndexError(f"Index {idx=}out of range")
        sample = self.dataset[idx]
        question = sample["question"]
        final_answer = self._parse_final_answer(sample["answer"])
        prompt = Prompt(
            question=question,
            reference_answers=[final_answer],
            should_abstain=sample["should_abstain"],
            metadata={"answer_with_explanation": sample["answer"]},
        )
        return prompt


class GSM8KGenerator:
    """
    Filters GSM8K questions that contain
    [context]. [question] ?

    via regex

    then offers two versions of each question
    with and without context

    This is not a multiple choice dataset.
    Answers are numeric
    """

    def __init__(
        self,
        split="test",
        data_dir="data/gsm8k",
    ):
        self.split = split
        # 确保使用绝对路径（Hydra 会改变工作目录）
        self.data_dir = os.path.abspath(data_dir) if not os.path.isabs(data_dir) else data_dir
        os.makedirs(self.data_dir, exist_ok=True)
        
        # 尝试从本地 data 目录加载
        local_cache_path = Path(self.data_dir) / "original_dataset"
        
        # 添加日志输出，方便调试
        import logging
        logger = logging.getLogger(__name__)
        logger.info(f"GSM8K 数据集路径: {local_cache_path}")
        logger.info(f"路径是否存在: {local_cache_path.exists()}")
        
        if local_cache_path.exists():
            # 从本地加载
            logger.info(f"从本地加载数据集: {local_cache_path}")
            from datasets import load_from_disk
            self.original_dataset = load_from_disk(str(local_cache_path))
            logger.info(f"✓ 成功从本地加载数据集，样本数: {len(self.original_dataset)}")
        else:
            # 本地不存在，尝试从 HuggingFace 下载
            logger.warning(f"本地数据集不存在: {local_cache_path}")
            logger.info("尝试从 HuggingFace 下载...")
            try:
                self.original_dataset = load_dataset("openai/gsm8k", "main", split=split)
                # 保存到本地 data 目录
                logger.info(f"保存数据集到本地: {local_cache_path}")
                self.original_dataset.save_to_disk(str(local_cache_path))
                logger.info(f"✓ 下载并保存完成，样本数: {len(self.original_dataset)}")
            except Exception as download_error:
                raise RuntimeError(
                    f"无法加载 GSM8K 数据集。\n"
                    f"本地路径: {local_cache_path}\n"
                    f"路径是否存在: {local_cache_path.exists()}\n"
                    f"请确保：\n"
                    f"1. 已预先下载数据集到 {local_cache_path}，或\n"
                    f"2. 网络连接正常（可以访问 HuggingFace）\n"
                    f"原始错误: {download_error}"
                )
        
        # regex identifies sentences that precede the question
        # [context. ][question?]
        self.context_regex_pattern = r"(?<=\. )[^\.\?\!]*\?$"

        self.dataset_with_context = self.create_dataset()
        self.dataset_without_context = self.create_dataset_without_context()

    def create_dataset(self):
        dataset = []
        for q in self.original_dataset:
            if re.search(self.context_regex_pattern, q["question"]):
                q["should_abstain"] = False
                dataset.append(q)
        return Dataset.from_list(dataset)

    def create_dataset_without_context(self):
        dataset = []
        for q in self.dataset_with_context:
            question_without_context = self.remove_context(q["question"])
            q["should_abstain"] = True
            q["question"] = question_without_context
            dataset.append(q)
        return Dataset.from_list(dataset)

    def remove_context(self, question: str) -> str:
        question_without_context = (
            re.search(self.context_regex_pattern, question).group().strip()
        )
        return question_without_context


if __name__ == "__main__":
    gsm8k_generator = GSM8KGenerator()
    print(len(gsm8k_generator.dataset_with_context))

    for i in range(3):
        print(gsm8k_generator.dataset_with_context[i])
        print("without context")
        print(gsm8k_generator.dataset_without_context[i])
        print()

    gsm8k = GSM8K()
    print(len(gsm8k))
    for i in range(3):
        print(gsm8k[i])
