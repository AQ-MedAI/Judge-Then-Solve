"""
Custom logging function for honesty training to track R1 and R2 rewards in WandB.
"""

import logging
import numpy as np

logger = logging.getLogger(__name__)


def log_honesty_metrics(rollout_id, args, samples, rollout_extra_metrics, rollout_time):
    """
    Custom logging function to extract and log R1/R2 metrics from sample metadata.

    This function extracts r1_answer_consistency and r2_thinking_consistency from
    sample metadata and logs them to WandB alongside standard metrics.

    Returns:
        False to allow default logging to continue
    """

    # Extract R1 and R2 values from sample metadata
    r1_values = []
    r2_values = []
    change_counts = []

    for sample in samples:
        if isinstance(sample.metadata, dict):
            if "r1_answer_consistency" in sample.metadata:
                r1_values.append(sample.metadata["r1_answer_consistency"])
            if "r2_thinking_consistency" in sample.metadata:
                r2_values.append(sample.metadata["r2_thinking_consistency"])
            if "thinking_changes_count" in sample.metadata:
                change_counts.append(sample.metadata["thinking_changes_count"])

    # Compute statistics if we have values
    if r1_values:
        r1_mean = np.mean(r1_values).item()
        r1_std = np.std(r1_values).item()
        r1_min = np.min(r1_values).item()
        r1_max = np.max(r1_values).item()

        logger.info(
            f"R1 (Answer Consistency) - "
            f"mean: {r1_mean:.3f}, std: {r1_std:.3f}, "
            f"min: {r1_min:.3f}, max: {r1_max:.3f}"
        )

    if r2_values:
        r2_mean = np.mean(r2_values).item()
        r2_std = np.std(r2_values).item()
        r2_min = np.min(r2_values).item()
        r2_max = np.max(r2_values).item()

        logger.info(
            f"R2 (Thinking Consistency) - "
            f"mean: {r2_mean:.3f}, std: {r2_std:.3f}, "
            f"min: {r2_min:.3f}, max: {r2_max:.3f}"
        )

    if change_counts:
        changes_mean = np.mean(change_counts).item()
        changes_std = np.std(change_counts).item()

        logger.info(
            f"Thinking Changes - "
            f"mean: {changes_mean:.2f}, std: {changes_std:.2f}"
        )

    # Return False to allow default logging to continue
    # The default logging will pick up these metrics from metadata
    return False
