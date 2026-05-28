#!/usr/bin/env python3
"""
Detailed aggregate evaluation results with fine-grained difficulty analysis
Generates CSV summary and visualization plots
"""
import json
import argparse
import os
import csv
from collections import defaultdict
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np


def parse_report(report):
    """Parse GPT-4o evaluation report"""
    parts = report.split("## ")
    data = {}

    for part in parts[1:]:
        lines = part.strip().split("\n")
        title = lines[0].strip()
        content = "\n".join(lines[1:]).strip()

        if title == "Justification":
            data[title] = content
        else:
            data[title] = lines[1].strip() if len(lines) > 1 else ''

    return data


def load_evaluation_results(eval_file):
    """Load and parse evaluation results from JSONL file"""
    results = []

    with open(eval_file, 'r', encoding='utf-8') as f:
        for line in f:
            entry = json.loads(line.strip())
            original_json = json.loads(entry['original_json'])
            gpt4_eval = entry['gen']

            # Check if model generation is empty (exceeded max tokens)
            model_generation = original_json.get('model_generation', '')
            exceeded_max_tokens = (not model_generation or model_generation.strip() == '')

            info = parse_report(gpt4_eval)

            # Determine correctness
            is_correct = False
            if info and 'Equivalence Judgement' in info:
                correctness = info['Equivalence Judgement']
                is_correct = (correctness == 'TRUE')

            results.append({
                'domain': original_json.get('domain', ''),
                'difficulty': original_json.get('difficulty', ''),
                'source': original_json.get('source', ''),
                'correctness': is_correct,
                'exceeded_max_tokens': exceeded_max_tokens
            })

    return results


def categorize_difficulty_fine_grained(difficulty):
    """Categorize difficulty into fine-grained ranges [1-2), [2-3), ..., [9-10]"""
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
        else:
            return "other"
    except:
        return "other"


def calculate_detailed_statistics(results):
    """Calculate detailed statistics by difficulty ranges"""
    difficulty_ranges = ["1-2", "2-3", "3-4", "4-5", "5-6", "6-7", "7-8", "8-9", "9-10"]

    grouped = defaultdict(lambda: {'total': 0, 'not_exceeded': 0, 'correct': 0})

    for result in results:
        diff_range = categorize_difficulty_fine_grained(result['difficulty'])
        if diff_range != "other":
            grouped[diff_range]['total'] += 1
            if not result['exceeded_max_tokens']:
                grouped[diff_range]['not_exceeded'] += 1
            if result['correctness']:
                grouped[diff_range]['correct'] += 1

    # Calculate statistics for each range
    stats = {}
    for diff_range in difficulty_ranges:
        data = grouped[diff_range]
        total = data['total']
        not_exceeded = data['not_exceeded']
        correct = data['correct']

        stats[diff_range] = {
            'total': total,
            'not_exceeded': not_exceeded,
            'correct': correct,
            'not_exceeded_ratio': not_exceeded / total if total > 0 else 0.0,
            'correct_ratio': correct / total if total > 0 else 0.0
        }

    # Calculate overall statistics
    total_all = sum(r['total'] for r in grouped.values())
    not_exceeded_all = sum(r['not_exceeded'] for r in grouped.values())
    correct_all = sum(r['correct'] for r in grouped.values())

    stats['overall'] = {
        'total': total_all,
        'not_exceeded': not_exceeded_all,
        'correct': correct_all,
        'not_exceeded_ratio': not_exceeded_all / total_all if total_all > 0 else 0.0,
        'correct_ratio': correct_all / total_all if total_all > 0 else 0.0
    }

    return stats


def find_latest_result_dir(method_dir):
    """Find the latest timestamped result directory for a method"""
    if not os.path.exists(method_dir):
        return None

    subdirs = [d for d in os.listdir(method_dir)
               if os.path.isdir(os.path.join(method_dir, d))]

    if not subdirs:
        return None

    subdirs.sort(reverse=True)
    return os.path.join(method_dir, subdirs[0])


def aggregate_all_methods(results_base_dir):
    """Aggregate results from all methods"""
    all_method_stats = {}

    if not os.path.exists(results_base_dir):
        print(f"Results directory not found: {results_base_dir}")
        return all_method_stats

    method_dirs = [d for d in os.listdir(results_base_dir)
                   if os.path.isdir(os.path.join(results_base_dir, d))]

    print(f"Found {len(method_dirs)} method directories")

    for method_name in method_dirs:
        method_path = os.path.join(results_base_dir, method_name)
        latest_dir = find_latest_result_dir(method_path)

        if not latest_dir:
            print(f"No results found for method: {method_name}")
            continue

        eval_file = os.path.join(latest_dir, "gpt4o_evaluations.jsonl")

        if not os.path.exists(eval_file):
            print(f"Evaluation file not found: {eval_file}")
            continue

        print(f"Processing {method_name}: {eval_file}")

        results = load_evaluation_results(eval_file)
        stats = calculate_detailed_statistics(results)
        all_method_stats[method_name] = stats

    return all_method_stats


def save_to_csv(all_method_stats, output_csv):
    """Save detailed statistics to CSV file"""
    if not all_method_stats:
        print("No results to save")
        return

    difficulty_ranges = ["1-2", "2-3", "3-4", "4-5", "5-6", "6-7", "7-8", "8-9", "9-10", "overall"]

    # Prepare CSV data
    rows = []
    for method_name, stats in all_method_stats.items():
        for diff_range in difficulty_ranges:
            if diff_range in stats:
                data = stats[diff_range]
                rows.append({
                    'method': method_name,
                    'difficulty_range': diff_range,
                    'total': data['total'],
                    'not_exceeded': data['not_exceeded'],
                    'correct': data['correct'],
                    'not_exceeded_ratio': f"{data['not_exceeded_ratio']:.4f}",
                    'correct_ratio': f"{data['correct_ratio']:.4f}"
                })

    # Write to CSV
    fieldnames = ['method', 'difficulty_range', 'total', 'not_exceeded', 'correct',
                  'not_exceeded_ratio', 'correct_ratio']

    with open(output_csv, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"\nDetailed results saved to: {output_csv}")


def plot_statistics(all_method_stats, output_dir):
    """Generate bar plots for statistics"""
    if not all_method_stats:
        print("No data to plot")
        return

    difficulty_ranges = ["1-2", "2-3", "3-4", "4-5", "5-6", "6-7", "7-8", "8-9", "9-10"]

    # Set up the plot style
    plt.rcParams['font.size'] = 10
    plt.rcParams['figure.figsize'] = (14, 6)

    methods = list(all_method_stats.keys())
    n_methods = len(methods)

    # Create two subplots: one for not_exceeded_ratio, one for correct_ratio
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 6))

    x = np.arange(len(difficulty_ranges))
    width = 0.8 / n_methods if n_methods > 0 else 0.8

    # Plot 1: Not Exceeded Ratio
    for i, method_name in enumerate(methods):
        stats = all_method_stats[method_name]
        not_exceeded_ratios = [stats[dr]['not_exceeded_ratio'] for dr in difficulty_ranges]
        offset = (i - n_methods/2 + 0.5) * width
        ax1.bar(x + offset, not_exceeded_ratios, width, label=method_name, alpha=0.8)

    ax1.set_xlabel('Difficulty Range', fontsize=12)
    ax1.set_ylabel('Not Exceeded Max Token Ratio', fontsize=12)
    ax1.set_title('Ratio of Problems Not Exceeding Max Tokens by Difficulty', fontsize=14)
    ax1.set_xticks(x)
    ax1.set_xticklabels(difficulty_ranges)
    ax1.legend()
    ax1.grid(axis='y', alpha=0.3)
    ax1.set_ylim([0, 1.0])

    # Plot 2: Correct Ratio
    for i, method_name in enumerate(methods):
        stats = all_method_stats[method_name]
        correct_ratios = [stats[dr]['correct_ratio'] for dr in difficulty_ranges]
        offset = (i - n_methods/2 + 0.5) * width
        ax2.bar(x + offset, correct_ratios, width, label=method_name, alpha=0.8)

    ax2.set_xlabel('Difficulty Range', fontsize=12)
    ax2.set_ylabel('Correct Ratio', fontsize=12)
    ax2.set_title('Accuracy by Difficulty Range', fontsize=14)
    ax2.set_xticks(x)
    ax2.set_xticklabels(difficulty_ranges)
    ax2.legend()
    ax2.grid(axis='y', alpha=0.3)
    ax2.set_ylim([0, 1.0])

    plt.tight_layout()

    # Save plot
    os.makedirs(output_dir, exist_ok=True)
    plot_file = os.path.join(output_dir, "detailed_statistics.png")
    plt.savefig(plot_file, dpi=300, bbox_inches='tight')
    print(f"Plot saved to: {plot_file}")
    plt.close()


def print_summary(all_method_stats):
    """Print summary statistics"""
    print("\n" + "="*80)
    print("SUMMARY STATISTICS")
    print("="*80)

    for method_name, stats in all_method_stats.items():
        print(f"\n{method_name}:")
        print("-" * 60)

        # Print overall stats
        overall = stats['overall']
        print(f"Overall ({overall['total']} problems):")
        print(f"  Not Exceeded Max Token: {overall['not_exceeded']}/{overall['total']} ({overall['not_exceeded_ratio']:.2%})")
        print(f"  Correct: {overall['correct']}/{overall['total']} ({overall['correct_ratio']:.2%})")

        print("\nBy Difficulty Range:")
        difficulty_ranges = ["1-2", "2-3", "3-4", "4-5", "5-6", "6-7", "7-8", "8-9", "9-10"]
        for dr in difficulty_ranges:
            if dr in stats:
                data = stats[dr]
                print(f"  {dr}: {data['total']} problems | "
                      f"Not Exceeded: {data['not_exceeded_ratio']:.2%} | "
                      f"Correct: {data['correct_ratio']:.2%}")


def main():
    parser = argparse.ArgumentParser(description="Detailed aggregate evaluation results")
    parser.add_argument("--results_dir", type=str, default="results",
                        help="Base directory containing method result folders")
    parser.add_argument("--output_csv", type=str, default="evaluation_detailed.csv",
                        help="Output CSV file path")
    parser.add_argument("--output_plot_dir", type=str, default="plots",
                        help="Output directory for plots")

    args = parser.parse_args()

    print(f"Results directory: {args.results_dir}")
    print(f"Output CSV: {args.output_csv}")
    print(f"Output plot directory: {args.output_plot_dir}")

    # Aggregate results from all methods
    all_method_stats = aggregate_all_methods(args.results_dir)

    if not all_method_stats:
        print("No results found to aggregate")
        return

    # Save to CSV
    save_to_csv(all_method_stats, args.output_csv)

    # Generate plots
    plot_statistics(all_method_stats, args.output_plot_dir)

    # Print summary
    print_summary(all_method_stats)


if __name__ == "__main__":
    main()