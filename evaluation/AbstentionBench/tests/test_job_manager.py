"""
Copyright (c) Meta Platforms, Inc. and affiliates.
All rights reserved.
This source code is licensed under the license found in the
LICENSE file in the root directory of this source tree.
"""
from recipe.job_manager import JobManager


class TestJobManager:
    def test_dataset_to_dataset_config(self):
        job_manager = JobManager()
        assert len(job_manager.dataset_to_dataset_class) > 10
        assert job_manager.dataset_to_dataset_class["gpqa"] == "GPQA"

    def test_model_to_model_config(self):
        job_manager = JobManager()
        assert len(job_manager.model_to_model_class) > 10
        assert job_manager.model_to_model_class["s1_1_32B"] == "S1_1_32B"

    def test_get_missing_jobs(self):
        job_manager = JobManager(sweep_dir="banana_non_existent")
        missing_jobs = job_manager.show_missing()
        assert len(missing_jobs) == len(job_manager.models)
        assert "s1_1_32B" in missing_jobs
        assert "gpqa" in missing_jobs["s1_1_32B"]

    def test_show_relaunch_commands(self):
        job_manager = JobManager(sweep_dir="banana_non_existent")
        commands = job_manager.show_relaunch_commands()
        assert len(commands) == len(job_manager.models)
