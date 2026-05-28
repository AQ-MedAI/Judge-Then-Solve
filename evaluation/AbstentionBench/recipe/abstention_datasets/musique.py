"""
Copyright (c) Meta Platforms, Inc. and affiliates.
All rights reserved.
This source code is licensed under the license found in the
LICENSE file in the root directory of this source tree.
"""
import json
from pathlib import Path
from typing import List

from recipe.abstention_datasets.nq_dataset import NQDataset


class MusiqueDataset(NQDataset):
    """Implements the Musique dataset from https://aclanthology.org/2023.emnlp-main.220/
    multi-hop dataset with answerable and unanswerable questions.
    Contains paragraphs and corresponding questions that require referencing them

    Inherits from NQDataset since formatting is the same.
    """

    def __init__(
        self,
        data_dir='data/musique',
        file_name="raw_data/musique/test.json",
        max_num_samples=None,
    ):
        # Don't call super().__init__() to avoid the download logic
        self.data_dir = data_dir
        self.file_name = file_name
        self.max_num_samples = max_num_samples
        self.dataset = self.load_dataset()

    def load_dataset(self) -> List[dict]:
        """Load dataset from local file without attempting download"""
        # Construct the full path including subdirectories
        test_file_path = Path(self.data_dir) / self.file_name

        if not test_file_path.exists():
            raise FileNotFoundError(
                f"Musique dataset not found at {test_file_path}. "
                f"Please ensure the data is downloaded to {self.data_dir}"
            )

        with open(test_file_path, mode="r") as f:
            nq_data = json.load(f)

        samples = []
        for raw_sample in nq_data:
            question = self._TEMPLATE.format(
                preprompt=self._PREPROMPT,
                context=raw_sample["context"],
                question=raw_sample["question"],
            )
            sample = {
                "question": question,
                "answer": raw_sample["answer"],
                "should_abstain": True if raw_sample["answerable"] == "no" else False,
                "metadata": json.loads(raw_sample["additional_data"]),
            }
            samples.append(sample)

        return samples