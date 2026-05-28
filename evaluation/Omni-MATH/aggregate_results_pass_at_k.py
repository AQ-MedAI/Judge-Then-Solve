#!/usr/bin/env python3
"""Aggregate pass@k results for Omni-MATH evaluation"""
import json
import os
import csv
import argparse
from collections import defaultdict
import matplotlib.pyplot as plt
import numpy as np


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
    """Calculate pass@k metrics"""
    if not results or not results[0].get('trials'):
        return {'total': 0}

    num_trials = len(results[0]['trials'])
    if k_values is None:
        k_values = [1, 2, 4, 8]

    valid_k_values = [k for k in k_values if num_trials % k == 0 and k <= num_trials]

    stats = {'total': len(results), 'num_trials': num_trials, 'k_values': valid_k_values, 'avg_length': 0.0}
    for k in valid_k_values:
        stats[f'pass@{k}'] = 0.0

    by_difficulty = defaultdict(lambda: {'total': 0, 'avg_length_sum': 0.0, **{f'pass@{k}_sum': 0.0 for k in valid_k_values}})

    for r in results:
        difficulty = categorize_difficulty(r.get('difficulty', ''))
        by_difficulty[difficulty]['total'] += 1
        trials = r.get('trials', [])

        # Calculate average length
        lengths = [len(t.get('model_generation', '')) for t in trials]
        avg_len = sum(lengths) / len(lengths) if lengths else 0.0
        stats['avg_length'] += avg_len
        by_difficulty[difficulty]['avg_length_sum'] += avg_len

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

    for k in valid_k_values:
        stats[f'pass@{k}'] = stats[f'pass@{k}'] / len(results) if results else 0.0

    stats['avg_length'] = stats['avg_length'] / len(results) if results else 0.0

    for difficulty in by_difficulty:
        total = by_difficulty[difficulty]['total']
        by_difficulty[difficulty]['avg_length'] = by_difficulty[difficulty]['avg_length_sum'] / total if total > 0 else 0.0
        del by_difficulty[difficulty]['avg_length_sum']
        for k in valid_k_values:
            by_difficulty[difficulty][f'pass@{k}'] = by_difficulty[difficulty][f'pass@{k}_sum'] / total if total > 0 else 0.0
            del by_difficulty[difficulty][f'pass@{k}_sum']

    stats['by_difficulty'] = dict(by_difficulty)
    return stats


def categorize_difficulty(difficulty):
    """Categorize difficulty into ranges"""
    try:
        diff = float(difficulty)
        if 1 <= diff < 2:
            return "1-2"
        elif 2 <= diff < 3:
            return "2-3"
        elif 3 <= diff < 4:
            return "3-4"
        elif 4 <= diff < 5:
            return "4-5"
        elif 5 <= diff < 6:
            return "5-6"
        elif 6 <= diff < 7:
            return "6-7"
        elif 7 <= diff < 8:
            return "7-8"
        elif 8 <= diff < 9:
            return "8-9"
        elif 9 <= diff <= 10:
            return "9-10"
    except:
        pass
    return "other"


def load_evaluation_results(eval_file):
    """Load evaluation results with trials"""
    results = []
    with open(eval_file, 'r', encoding='utf-8') as f:
        for line in f:
            entry = json.loads(line.strip())
            original_json = json.loads(entry['original_json'])

            if 'trials' in original_json:
                for trial in original_json.get('trials', []):
                    trial['is_correct'] = parse_judge_result(trial.get('omni_judge', ''))
                results.append(original_json)

    return results


def aggregate_all_methods(results_base_dir):
    """Aggregate results from all methods"""
    all_method_stats = {}

    if not os.path.exists(results_base_dir):
        return all_method_stats

    method_dirs = [d for d in os.listdir(results_base_dir) if os.path.isdir(os.path.join(results_base_dir, d))]

    for method_name in method_dirs:
        method_path = os.path.join(results_base_dir, method_name)
        subdirs = [d for d in os.listdir(method_path) if os.path.isdir(os.path.join(method_path, d))]
        if not subdirs:
            continue

        subdirs.sort(reverse=True)
        latest_dir = os.path.join(method_path, subdirs[0])
        eval_file = os.path.join(latest_dir, "gpt4o_evaluations.jsonl")

        if not os.path.exists(eval_file):
            continue

        print(f"Processing {method_name}: {eval_file}")
        results = load_evaluation_results(eval_file)
        stats = calculate_pass_at_k(results)
        all_method_stats[method_name] = stats

    return all_method_stats


def save_to_csv(all_method_stats, output_csv):
    """Save pass@k statistics to CSV"""
    if not all_method_stats:
        return

    difficulty_ranges = ["1-2", "2-3", "3-4", "4-5", "5-6", "6-7", "7-8", "8-9", "9-10"]
    rows = []

    for method_name, stats in all_method_stats.items():
        k_values = stats.get('k_values', [])
        for diff_range in difficulty_ranges + ['overall']:
            if diff_range == 'overall':
                row = {'method': method_name, 'difficulty_range': diff_range, 'total': stats.get('total', 0), 'avg_length': f"{stats.get('avg_length', 0.0):.1f}"}
                for k in k_values:
                    row[f'pass@{k}'] = f"{stats.get(f'pass@{k}', 0.0):.4f}"
                rows.append(row)
            elif diff_range in stats.get('by_difficulty', {}):
                data = stats['by_difficulty'][diff_range]
                row = {'method': method_name, 'difficulty_range': diff_range, 'total': data['total'], 'avg_length': f"{data.get('avg_length', 0.0):.1f}"}
                for k in k_values:
                    row[f'pass@{k}'] = f"{data.get(f'pass@{k}', 0.0):.4f}"
                rows.append(row)

    if rows:
        k_values = all_method_stats[list(all_method_stats.keys())[0]].get('k_values', [])
        fieldnames = ['method', 'difficulty_range', 'total', 'avg_length'] + [f'pass@{k}' for k in k_values]

        with open(output_csv, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)

        print(f"\nResults saved to: {output_csv}")


def plot_pass_at_k(all_method_stats, output_dir):
    """Generate plots for pass@k metrics"""
    if not all_method_stats:
        return

    difficulty_ranges = ["1-2", "2-3", "3-4", "4-5", "5-6", "6-7", "7-8", "8-9", "9-10", "overall"]
    methods = list(all_method_stats.keys())
    k_values = all_method_stats[methods[0]].get('k_values', [])

    fig, axes = plt.subplots(1, len(k_values) + 1, figsize=(6 * (len(k_values) + 1), 6))

    x = np.arange(len(difficulty_ranges))
    width = 0.8 / len(methods) if methods else 0.8

    for k_idx, k in enumerate(k_values):
        ax = axes[k_idx]
        for i, method_name in enumerate(methods):
            stats = all_method_stats[method_name]
            values = []
            for dr in difficulty_ranges:
                if dr == 'overall':
                    values.append(stats.get(f'pass@{k}', 0.0))
                else:
                    values.append(stats.get('by_difficulty', {}).get(dr, {}).get(f'pass@{k}', 0.0))
            offset = (i - len(methods)/2 + 0.5) * width
            ax.bar(x + offset, values, width, label=method_name, alpha=0.8)

        ax.set_xlabel('Difficulty Range', fontsize=12)
        ax.set_ylabel(f'pass@{k}', fontsize=12)
        ax.set_title(f'pass@{k} by Difficulty', fontsize=14)
        ax.set_xticks(x)
        ax.set_xticklabels(difficulty_ranges, rotation=45)
        ax.legend()
        ax.grid(axis='y', alpha=0.3)
        ax.set_ylim([0, 1.0])

    # Length plot
    ax = axes[-1]
    for i, method_name in enumerate(methods):
        stats = all_method_stats[method_name]
        values = []
        for dr in difficulty_ranges:
            if dr == 'overall':
                values.append(stats.get('avg_length', 0.0))
            else:
                values.append(stats.get('by_difficulty', {}).get(dr, {}).get('avg_length', 0.0))
        offset = (i - len(methods)/2 + 0.5) * width
        ax.bar(x + offset, values, width, label=method_name, alpha=0.8)

    ax.set_xlabel('Difficulty Range', fontsize=12)
    ax.set_ylabel('Avg Length (tokens)', fontsize=12)
    ax.set_title('Average Response Length', fontsize=14)
    ax.set_xticks(x)
    ax.set_xticklabels(difficulty_ranges, rotation=45)
    ax.legend()
    ax.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    os.makedirs(output_dir, exist_ok=True)
    plot_file = os.path.join(output_dir, "pass_at_k_statistics.png")
    plt.savefig(plot_file, dpi=300, bbox_inches='tight')
    print(f"Plot saved to: {plot_file}")
    plt.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results_dir", type=str, default="results")
    parser.add_argument("--output_csv", type=str, default="evaluation_pass_at_k.csv")
    parser.add_argument("--output_plot_dir", type=str, default="plots")
    args = parser.parse_args()

    all_method_stats = aggregate_all_methods(args.results_dir)
    save_to_csv(all_method_stats, args.output_csv)
    plot_pass_at_k(all_method_stats, args.output_plot_dir)


if __name__ == "__main__":
    main()
