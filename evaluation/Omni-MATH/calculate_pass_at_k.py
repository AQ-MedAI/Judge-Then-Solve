"""Calculate pass@k metrics for Omni-MATH evaluation results"""
import json
import argparse
from collections import defaultdict


def parse_judge_result(judge_output):
    """Extract correctness from judge output"""
    if not judge_output:
        return False

    parts = judge_output.split("## ")
    for part in parts[1:]:
        lines = part.strip().split("\n")
        if lines[0].strip() == "Equivalence Judgement":
            return lines[1].strip() == "TRUE" if len(lines) > 1 else False
    return False


def calculate_pass_at_k(results, k_values=None):
    """
    Calculate pass@k metrics

    Args:
        results: List of problems with trials
        k_values: List of k values to calculate

    Returns:
        dict: Statistics including pass@k metrics
    """
    if not results or not results[0].get('trials'):
        return {'total': 0}

    num_trials = len(results[0]['trials'])

    if k_values is None:
        k_values = [1, 2, 4, 8, 16]

    valid_k_values = [k for k in k_values if num_trials % k == 0 and k <= num_trials]

    if not valid_k_values:
        return {'total': len(results), 'num_trials': num_trials}

    stats = {
        'total': len(results),
        'num_trials': num_trials,
        'k_values': valid_k_values
    }

    for k in valid_k_values:
        stats[f'pass@{k}'] = 0.0

    by_difficulty = defaultdict(lambda: {'total': 0, **{f'pass@{k}_sum': 0.0 for k in valid_k_values}})
    by_domain = defaultdict(lambda: {'total': 0, **{f'pass@{k}_sum': 0.0 for k in valid_k_values}})

    for r in results:
        difficulty = r.get('difficulty', 'unknown')
        domain = r.get('domain', ['unknown'])[0] if isinstance(r.get('domain'), list) else r.get('domain', 'unknown')

        by_difficulty[difficulty]['total'] += 1
        by_domain[domain]['total'] += 1

        trials = r.get('trials', [])

        for k in valid_k_values:
            num_groups = num_trials // k
            group_results = []

            for group_idx in range(num_groups):
                start_idx = group_idx * k
                end_idx = start_idx + k
                group_trials = trials[start_idx:end_idx]

                has_correct = any(trial.get('is_correct', False) for trial in group_trials)
                group_results.append(1.0 if has_correct else 0.0)

            problem_pass_at_k = sum(group_results) / num_groups if num_groups > 0 else 0.0

            stats[f'pass@{k}'] += problem_pass_at_k
            by_difficulty[difficulty][f'pass@{k}_sum'] += problem_pass_at_k
            by_domain[domain][f'pass@{k}_sum'] += problem_pass_at_k

    for k in valid_k_values:
        stats[f'pass@{k}'] = stats[f'pass@{k}'] / len(results) if results else 0.0

    for difficulty in by_difficulty:
        total = by_difficulty[difficulty]['total']
        for k in valid_k_values:
            by_difficulty[difficulty][f'pass@{k}'] = by_difficulty[difficulty][f'pass@{k}_sum'] / total if total > 0 else 0.0
            del by_difficulty[difficulty][f'pass@{k}_sum']

    for domain in by_domain:
        total = by_domain[domain]['total']
        for k in valid_k_values:
            by_domain[domain][f'pass@{k}'] = by_domain[domain][f'pass@{k}_sum'] / total if total > 0 else 0.0
            del by_domain[domain][f'pass@{k}_sum']

    stats['by_difficulty'] = dict(by_difficulty)
    stats['by_domain'] = dict(by_domain)

    return stats


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input", required=True)
    parser.add_argument("-o", "--output", required=True)
    args = parser.parse_args()

    with open(args.input) as f:
        results = [json.loads(line) for line in f]

    for r in results:
        for trial in r.get('trials', []):
            trial['is_correct'] = parse_judge_result(trial.get('omni_judge', ''))

    stats = calculate_pass_at_k(results)

    with open(args.output, 'w') as f:
        json.dump(stats, f, indent=2)

    print(f"Pass@k statistics saved to {args.output}")


if __name__ == "__main__":
    main()
