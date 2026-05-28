"""
Copyright (c) Meta Platforms, Inc. and affiliates.
All rights reserved.
This source code is licensed under the license found in the
LICENSE file in the root directory of this source tree.
"""
import re

from datasets import Dataset, concatenate_datasets, load_dataset

from recipe.abstention_datasets.abstract_abstention_dataset import (
    AbstentionDataset,
    Prompt,
)


class MMLUMath(AbstentionDataset):
    def __init__(self, split="test", max_num_samples=None, data_dir="data/mmlu"):
        self.max_num_samples = max_num_samples
        self.mmlu_generator = MMLUMathGenerator(split=split, data_dir=data_dir)
        self.dataset = self.create_dataset()

    def create_dataset(self) -> Dataset:
        return concatenate_datasets(
            [
                self.mmlu_generator.dataset_with_context,
                self.mmlu_generator.dataset_without_context,
            ]
        )

    def __len__(self):
        if self.max_num_samples is not None:
            return min(len(self.dataset), self.max_num_samples)
        return len(self.dataset)

    def _format_question(self, sample: dict):
        question = sample["question"]

        # chr(65) is 'A'
        choices_text = "\n".join(
            [f"{chr(65+i)}. {choice}" for i, choice in enumerate(sample["choices"])]
        )
        return question + "\n" + choices_text

    def __getitem__(self, idx) -> Prompt:
        if idx > len(self.dataset):
            raise IndexError(f"Index {idx=}out of range")
        sample = self.dataset[idx]
        question = self._format_question(sample)
        answer = [f"{chr(65+i)}" for i in range(len(sample["choices"]))][
            sample["answer"]
        ]
        prompt = Prompt(
            question=question,
            reference_answers=[answer],
            should_abstain=sample["should_abstain"],
            metadata={"subject": sample["subject"]},
        )
        return prompt


class MMLUHistory(MMLUMath):
    def __init__(self, split="test", max_num_samples=None, data_dir="data/mmlu"):
        self.max_num_samples = max_num_samples
        self.mmlu_generator = MMLUHistoryGenerator(split=split, data_dir=data_dir)
        self.dataset = self.create_dataset()


class MMLUMathGenerator:
    SUBSETS = ["college_mathematics", "abstract_algebra", "high_school_mathematics"]

    def __init__(
        self,
        split="test",
        data_dir="data/mmlu",
    ):
        self.subsets = self.SUBSETS
        self.split = split
        self.data_dir = data_dir
        self.original_dataset = self.load_datasets()
        # regex identifies sentences that precede the question
        # [context. ][question?]
        self.context_regex_pattern = r"(?<=\. )[^\.\?\!]*\?$"
        self.dataset_with_context = self.create_dataset()
        self.dataset_without_context = self.create_dataset_without_context()

    def load_datasets(self):
        # 尝试从合并后的数据集加载
        merged_path = f"{self.data_dir}/original_dataset"
        import os
        if os.path.exists(merged_path):
            # 从合并后的数据集加载
            from datasets import Dataset
            return Dataset.load_from_disk(merged_path)
        else:
            # 从按 subset 分别保存的数据集加载
            all_datasets = []
            for subset in self.subsets:
                from datasets import Dataset
                dataset = Dataset.load_from_disk(f"{self.data_dir}_{subset}")
                all_datasets.append(dataset)
            return concatenate_datasets(all_datasets)

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


class MMLUHistoryGenerator(MMLUMathGenerator):
    SUBSETS = ["global_facts", "high_school_world_history", "prehistory"]


if __name__ == "__main__":
    mmlu_history_generator = MMLUHistoryGenerator()
    print(len(mmlu_history_generator.dataset_with_context))

    for i in range(3):
        print(mmlu_history_generator.dataset_with_context[i])
        print("without context")
        print(mmlu_history_generator.dataset_without_context[i])
        print()

    mmlu_history = MMLUHistory()
    print(len(mmlu_history))

    for i in range(3):
        print(mmlu_history[i])
        print()
