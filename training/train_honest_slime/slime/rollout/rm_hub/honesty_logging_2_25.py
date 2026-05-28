"""
Custom logging function for honesty training with format, consistency, and task rewards.

Logs the following rollout metrics:
- format_success_rate: % samples with correct format
- consistency_success_rate: % samples with consistent conclusion
- insufficient_abstention_rate: for insufficient questions, % that got r_task > 0 (gated)
- well_defined_correct_rate: for well-defined questions, % that got r_task > 0 (gated)
- insufficient_abstention_rate_raw: for insufficient questions, % where eval_result==0 (no gating)
- well_defined_correct_rate_raw: for well-defined questions, % where eval_result==1 (no gating)
- total_reward: average final reward (R_format + R_consistency + R_task)
"""

import logging

logger = logging.getLogger(__name__)


def log_honesty_metrics_2_25(rollout_id, args, samples, rollout_extra_metrics, rollout_time):
    """
    Returns False to allow default logging to continue.
    """
    if not samples:
        return False

    format_valid_count = 0
    consistency_valid_count = 0

    # Gated task performance: based on r_task from metadata (0 if format/consistency failed)
    insufficient_task_correct = 0
    insufficient_total = 0
    well_defined_task_correct = 0
    well_defined_total = 0

    # Raw task performance: purely based on eval_result, ignoring format/consistency
    insufficient_correct_raw = 0
    insufficient_total_raw = 0
    well_defined_correct_raw = 0
    well_defined_total_raw = 0

    total_reward_sum = 0.0
    total_samples = len(samples)

    for sample in samples:
        if not isinstance(sample.metadata, dict):
            continue

        if sample.metadata.get("format_valid", False):
            format_valid_count += 1

        if sample.metadata.get("consistent", False):
            consistency_valid_count += 1

        question_label = sample.metadata.get("question_label", "well_defined")
        eval_result = sample.metadata.get("eval_result", 3)
        r_task = sample.metadata.get("r_task", 0.0)

        if question_label == "insufficient":
            insufficient_total += 1
            insufficient_total_raw += 1
            # Gated: r_task > 0 means format+consistency passed AND correctly abstained
            if r_task > 0:
                insufficient_task_correct += 1
            # Raw: purely eval_result == 0 (abstained), regardless of format/consistency
            if eval_result == 0:
                insufficient_correct_raw += 1
        else:  # well_defined
            well_defined_total += 1
            well_defined_total_raw += 1
            # Gated: r_task > 0 means format+consistency passed AND correctly answered
            if r_task > 0:
                well_defined_task_correct += 1
            # Raw: purely eval_result == 1 (correct answer), regardless of format/consistency
            if eval_result == 1:
                well_defined_correct_raw += 1

        if "reward" in sample.metadata:
            total_reward_sum += sample.metadata["reward"]

    format_success_rate = format_valid_count / total_samples if total_samples > 0 else 0.0
    consistency_success_rate = consistency_valid_count / total_samples if total_samples > 0 else 0.0
    insufficient_abstention_rate = insufficient_task_correct / insufficient_total if insufficient_total > 0 else 0.0
    well_defined_correct_rate = well_defined_task_correct / well_defined_total if well_defined_total > 0 else 0.0
    insufficient_abstention_rate_raw = insufficient_correct_raw / insufficient_total_raw if insufficient_total_raw > 0 else 0.0
    well_defined_correct_rate_raw = well_defined_correct_raw / well_defined_total_raw if well_defined_total_raw > 0 else 0.0
    total_reward_avg = total_reward_sum / total_samples if total_samples > 0 else 0.0

    logger.info(f"=== Honesty Metrics (Rollout {rollout_id}) ===")
    logger.info(f"Format Success Rate:               {format_success_rate:.3f} ({format_valid_count}/{total_samples})")
    logger.info(f"Consistency Success Rate:           {consistency_success_rate:.3f} ({consistency_valid_count}/{total_samples})")
    logger.info(f"Insufficient Abstention (gated):   {insufficient_abstention_rate:.3f} ({insufficient_task_correct}/{insufficient_total})")
    logger.info(f"Insufficient Abstention (raw):     {insufficient_abstention_rate_raw:.3f} ({insufficient_correct_raw}/{insufficient_total_raw})")
    logger.info(f"Well-defined Correct (gated):      {well_defined_correct_rate:.3f} ({well_defined_task_correct}/{well_defined_total})")
    logger.info(f"Well-defined Correct (raw):        {well_defined_correct_rate_raw:.3f} ({well_defined_correct_raw}/{well_defined_total_raw})")
    logger.info(f"Total Reward (avg):                {total_reward_avg:.3f}")
    logger.info("=" * 50)

    honesty_metrics_dict = {
        'rollout/format_success_rate': format_success_rate,
        'rollout/consistency_success_rate': consistency_success_rate,
        'rollout/insufficient_abstention_rate': insufficient_abstention_rate,
        'rollout/well_defined_correct_rate': well_defined_correct_rate,
        'rollout/insufficient_abstention_rate_raw': insufficient_abstention_rate_raw,
        'rollout/well_defined_correct_rate_raw': well_defined_correct_rate_raw,
        'rollout/total_reward': total_reward_avg,
    }
    logger.info(f"rollout {rollout_id}: {honesty_metrics_dict}")

    if rollout_extra_metrics is not None:
        rollout_extra_metrics.update(honesty_metrics_dict)

    return False
