"""
Copyright (c) Meta Platforms, Inc. and affiliates.
All rights reserved.
This source code is licensed under the license found in the
LICENSE file in the root directory of this source tree.
"""
import requests
import datasets

from pathlib import Path
import jsonlines
from loguru import logger

from recipe.abstention_datasets.abstract_abstention_dataset import AbstentionDataset
from recipe.abstention_datasets.abstract_abstention_dataset import Prompt


class _MediQSubDataset(
    AbstentionDataset,
):
    """Private class for constructing the MediQ sub-benchmarks, iMedQA and iCRAFT-MD. For evaluation, you probably want `MediQDataset` instead."""

    def __init__(
        self,
        data_dir="data/mediq/icraftmd",
        data_url="https://raw.githubusercontent.com/stellalisy/mediQ/refs/heads/main/data/all_craft_md.jsonl",
        exclude_sample_ids=None,
    ):
        super().__init__()

        # Ensure data_dir is a string (Hydra may pass PosixPath)
        data_dir = str(data_dir)

        try:
            self.dataset = datasets.Dataset.load_from_disk(data_dir)
        except:
            # If we haven't already saved the dataset, fetch and parse the dataset files one-by-one

            logger.info(f"Downloading {data_url}")

            response = requests.get(data_url)
            response.raise_for_status()

            # Response is a jsonlines file, rather than a json file, so parse it appropriately
            lines = response.text.split("\n")
            lines = [line for line in lines if line]  # Filter out any blank lines
            data = jsonlines.Reader(lines)

            self.dataset = datasets.Dataset.from_list(list(data))

            self.dataset.save_to_disk(data_dir)

        if exclude_sample_ids is not None:
            self.dataset = self.dataset.filter(
                lambda x: x["id"] not in exclude_sample_ids
            )

    def __len__(self):
        # We have two samples (one with context, one without) for every entry in the original MediQ
        return len(self.dataset) * 2

    def __getitem__(self, idx) -> Prompt:
        if idx >= len(self):
            raise IndexError

        # Second half of the dataset are abstention samples
        should_abstain = idx >= len(self.dataset)

        if should_abstain:
            idx -= len(self.dataset)

        item = self.dataset[idx]

        # Add a '.' to the end of each context sentence if needed
        context = [(c + "." if not c.endswith(".") else c) for c in item["context"]]

        question = item["question"]

        choices = "\n".join(item["options"].values())

        if should_abstain:
            # Just include the first line of the context, the bare minimum patient information, with the question
            context = context[0]
            reference_answers = None
        else:
            # Include the full patient background with the question
            context = " ".join(context)
            reference_answers = [item["answer"]]

        full_question = (
            f"Context: {context}\nQuestion: {question}\nChoices:\n{choices}\nAnswer: "
        )

        metadata = {
            f"id": item["id"],
        }

        return Prompt(
            question=full_question,
            reference_answers=reference_answers,
            should_abstain=should_abstain,
            metadata=metadata,
        )


class MediQDataset(AbstentionDataset):
    """The MediQ datset, comprising the iCRAFT-MD and iMED-QA sub-benchmarks of multiple-choice medical question answering."""

    def __init__(
        self,
        data_dir="data/mediq",
        icraftmd_url="https://raw.githubusercontent.com/stellalisy/mediQ/refs/heads/main/data/all_craft_md.jsonl",
        imedqa_url="https://raw.githubusercontent.com/stellalisy/mediQ/refs/heads/main/data/all_dev_good.jsonl",
        max_num_samples=None,
    ):
        super().__init__()

        self.max_num_samples = max_num_samples

        # Ensure data_dir is a string (Hydra may pass PosixPath)
        data_dir = str(data_dir)

        self.icraftmd = _MediQSubDataset(Path(data_dir) / "icrafmd", icraftmd_url)

        # Exclude 3 iMedQA samples which don't have a context
        self.imedqa = _MediQSubDataset(
            Path(data_dir) / "imedqa", imedqa_url, exclude_sample_ids={224, 298, 779}
        )

    def __len__(self):
        return self.max_num_samples or (len(self.icraftmd) + len(self.imedqa))

    def __getitem__(self, idx) -> Prompt:
        if idx >= len(self):
            raise IndexError

        if idx < len(self.icraftmd):
            source = "iCRAFT-MD"
            prompt = self.icraftmd[idx]
        else:
            source = "iMEDQA"
            offset_idx = idx - len(self.icraftmd)
            prompt = self.imedqa[offset_idx]

        updated_metadata = {
            "MediQ_source": source,
            f"MediQ_{source}_id": prompt.metadata["id"],
        }

        updated_prompt = Prompt(
            question=prompt.question,
            should_abstain=prompt.should_abstain,
            reference_answers=prompt.reference_answers,
            metadata=updated_metadata,
        )

        return updated_prompt


class SubMediQDataset(AbstentionDataset):
    """MediQ 的随机子集版本（用于快速评测）。

    - 从完整 `MediQDataset` 中按固定随机种子随机采样 `subsample_size` 条样本
    - 采样的是“最终样本索引”（即包含 should_abstain=True/False 的混合样本）
    - 通过 seed 固定，可复现
    """

    def __init__(
        self,
        data_dir="data/mediq",
        icraftmd_url="https://raw.githubusercontent.com/stellalisy/mediQ/refs/heads/main/data/all_craft_md.jsonl",
        imedqa_url="https://raw.githubusercontent.com/stellalisy/mediQ/refs/heads/main/data/all_dev_good.jsonl",
        per_class_samples: int = 150,
    ):
        super().__init__()

        self.base_dataset = MediQDataset(
            data_dir=data_dir,
            icraftmd_url=icraftmd_url,
            imedqa_url=imedqa_url,
            max_num_samples=None,
        )

        # 严格保证 should_abstain=True / False 各 per_class_samples 条
        true_indices = []
        false_indices = []
        for i in range(len(self.base_dataset)):
            prompt = self.base_dataset[i]
            if prompt.should_abstain:
                if len(true_indices) < per_class_samples:
                    true_indices.append(i)
            else:
                if len(false_indices) < per_class_samples:
                    false_indices.append(i)

            if len(true_indices) >= per_class_samples and len(false_indices) >= per_class_samples:
                break

        self.subsample_indices = true_indices + false_indices

    def __len__(self):
        return len(self.subsample_indices)

    def __getitem__(self, idx) -> Prompt:
        if idx >= len(self):
            raise IndexError
        base_idx = self.subsample_indices[idx]
        return self.base_dataset[base_idx]
