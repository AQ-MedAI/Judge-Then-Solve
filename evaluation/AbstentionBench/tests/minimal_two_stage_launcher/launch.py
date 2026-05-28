"""
Copyright (c) Meta Platforms, Inc. and affiliates.
All rights reserved.
This source code is licensed under the license found in the
LICENSE file in the root directory of this source tree.
"""
from loguru import logger
import os
from pathlib import Path
import hydra
import time
import submitit


@hydra.main(version_base=None, config_path=".", config_name="two_stage_config.yaml")
def launch_evaluation_in_separate_job(config):
    """Reruns the job with same config with flags to load inference
    and run evaluations
    """
    print("Running")

    if config.stage == 1:
        logger.info("Running stage 1")
        time.sleep(3)
        logger.info("Stage 1 complete")

        config.stage = 2
        executor = submitit.AutoExecutor(folder=config.logs_dir)
        executor.update_parameters(
            timeout_min=10,
            slurm_partition="devlab",
            gpus_per_node=1,
            tasks_per_node=1,
            cpus_per_task=8,
            mem_gb=480,
            nodes=1,
            slurm_constraint="volta32gb",
        )

        job = executor.submit(launch_evaluation_in_separate_job, config)

    if config.stage == 2:
        logger.info("Running stage 2")
        time.sleep(3)
        logger.info("Stage 2 complete")


if __name__ == "__main__":
    launch_evaluation_in_separate_job()
