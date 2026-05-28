"""
Copyright (c) Meta Platforms, Inc. and affiliates.
All rights reserved.
This source code is licensed under the license found in the
LICENSE file in the root directory of this source tree.
"""
from pathlib import Path
import requests
import datasets
from recipe.abstention_datasets.abstract_abstention_dataset import AbstentionDataset
from recipe.abstention_datasets.abstract_abstention_dataset import Prompt


class SituatedQAGeoDataset(AbstentionDataset):

    def __init__(self, max_num_samples=None, data_dir="data/situated_qa_geo"):
        super().__init__()

        # Check if local data exists
        local_path = Path(data_dir)
        if local_path.exists() and (local_path / "dataset_dict.json").exists():
            # Load from local disk without triggering HuggingFace checks
            import os
            os.environ["HF_DATASETS_OFFLINE"] = "1"
            dataset_dict = datasets.load_from_disk(str(local_path))
            self.dataset = dataset_dict["test"]
        else:
            # Fall back to loading from HuggingFace
            self.dataset = datasets.load_dataset(
                "siyue/SituatedQA", "geo", trust_remote_code=True
            )["test"]

        # Construct the underspecified dataset (which needs to be deduplicated)
        visited_questions = set()
        deduplicated_rows = []
        for row in self.dataset:
            if row["question"] not in visited_questions:
                deduplicated_rows.append(row)
                visited_questions.add(row["question"])

        self.deduplicated_dataset = datasets.Dataset.from_list(list(deduplicated_rows))

        self.max_num_samples = max_num_samples

    def __len__(self):
        return self.max_num_samples or (
            len(self.dataset) + len(self.deduplicated_dataset)
        )

    def __getitem__(self, idx):
        if idx >= len(self):
            raise IndexError

        # Concatenate the deduplicated dataset (for underspecified questions) with
        # the original dataset (for fully specified questions)
        if idx < len(self.dataset):
            item = self.dataset[idx]
            question = item["edited_question"] + "?"
            reference_answers = item["any_answer"]
            should_abstain = False
        else:
            offset_idx = idx - len(self.dataset)
            item = self.deduplicated_dataset[offset_idx]
            question = item["question"] + "?"
            reference_answers = None
            should_abstain = True

        metadata = {
            "SituatedQA_id": item["id"],
            "SituatedQA_location": item["location"],
        }

        return Prompt(
            question=question,
            reference_answers=reference_answers,
            should_abstain=should_abstain,
            metadata=metadata,
        )
