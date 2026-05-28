#!/usr/bin/env python3
"""
Aggregate evaluation results across multiple methods and generate CSV summary
Groups results by difficulty level and calculates accuracy
"""
import json
import argparse
import os
import csv
from collections import defaultdict
from pathlib import Path


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

            info = parse_report(gpt4_eval)
            if not info:
                continue

            try:
                correctness = info['Equivalence Judgement']
                is_correct = (correctness == 'TRUE')

                results.append({
                    'domain': original_json.get('domain', ''),
                    'difficulty': original_json.get('difficulty', ''),
                    'source': original_json.get('source', ''),
                    'correctness': is_correct
                })
            except:
                continue

    return results


def categorize_difficulty(difficulty):
    """Categorize difficulty into ranges"""
    try:
        diff = float(difficulty)
        if 1 <= diff <= 3:
            return "1-3"
        elif 3 < diff <= 5:
            return "3-5"
        elif 5 < diff <= 8:
            return "5-8"
        elif 8 < diff <= 10:
            return "8-10"
        else:
            return "other"
    except:
        return "other"


def calculate_accuracy_by_difficulty(results):
    """Calculate accuracy grouped by difficulty ranges"""
    grouped = defaultdict(list)

    for result in results:
        difficulty_range = categorize_difficulty(result['difficulty'])
        grouped[difficulty_range].append(result['correctness'])

    accuracy_stats = {}
    for difficulty_range, correctness_list in grouped.items():
        total = len(correctness_list)
        correct = sum(correctness_list)
        accuracy = correct / total if total > 0 else 0.0
        accuracy_stats[difficulty_range] = {
            'accuracy': accuracy,
            'correct': correct,
            'total': total
        }

    return accuracy_stats


def find_latest_result_dir(method_dir):
    """Find the latest timestamped result directory for a method"""
    if not os.path.exists(method_dir):
        return None

    subdirs = [d for d in os.listdir(method_dir)
               if os.path.isdir(os.path.join(method_dir, d))]

    if not subdirs:
        return None

    # Sort by timestamp (assuming format YYYYMMDD_HHMMSS)
    subdirs.sort(reverse=True)
    return os.path.join(method_dir, subdirs[0])


def aggregate_all_methods(results_base_dir, output_csv):
    """Aggregate results from all methods and generate CSV"""
    all_results = []

    # Find all method directories
    if not os.path.exists(results_base_dir):
        print(f"Results directory not found: {results_base_dir}")
        return

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

        # Load results
        results = load_evaluation_results(eval_file)
        accuracy_stats = calculate_accuracy_by_difficulty(results)

        # Calculate overall accuracy
        total_correct = sum(r['correctness'] for r in results)
        total_count = len(results)
        overall_accuracy = total_correct / total_count if total_count > 0 else 0.0

        # Add to all_results
        result_entry = {
            'method': method_name,
            'overall_accuracy': overall_accuracy,
            'overall_correct': total_correct,
            'overall_total': total_count
        }

        # Add difficulty-specific accuracies
        for diff_range in ['1-3', '3-5', '5-8', '8-10']:
            if diff_range in accuracy_stats:
                stats = accuracy_stats[diff_range]
                result_entry[f'difficulty_{diff_range}_accuracy'] = stats['accuracy']
                result_entry[f'difficulty_{diff_range}_correct'] = stats['correct']
                result_entry[f'difficulty_{diff_range}_total'] = stats['total']
            else:
                result_entry[f'difficulty_{diff_range}_accuracy'] = 0.0
                result_entry[f'difficulty_{diff_range}_correct'] = 0
                result_entry[f'difficulty_{diff_range}_total'] = 0

        all_results.append(result_entry)

    return all_results


def save_to_csv(all_results, output_csv):
    """Save aggregated results to CSV file"""
    if not all_results:
        print("No results to save")
        return

    # Define CSV columns
    fieldnames = [
        'method',
        'overall_accuracy',
        'overall_correct',
        'overall_total',
        'difficulty_1-3_accuracy',
        'difficulty_1-3_correct',
        'difficulty_1-3_total',
        'difficulty_3-5_accuracy',
        'difficulty_3-5_correct',
        'difficulty_3-5_total',
        'difficulty_5-8_accuracy',
        'difficulty_5-8_correct',
        'difficulty_5-8_total',
        'difficulty_8-10_accuracy',
        'difficulty_8-10_correct',
        'difficulty_8-10_total'
    ]

    with open(output_csv, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(all_results)

    print(f"\nResults saved to: {output_csv}")
    print("\nSummary:")
    for result in all_results:
        print(f"\n{result['method']}:")
        print(f"  Overall: {result['overall_accuracy']:.4f} ({result['overall_correct']}/{result['overall_total']})")
        print(f"  Difficulty 1-3: {result['difficulty_1-3_accuracy']:.4f}")
        print(f"  Difficulty 3-5: {result['difficulty_3-5_accuracy']:.4f}")
        print(f"  Difficulty 5-8: {result['difficulty_5-8_accuracy']:.4f}")
        print(f"  Difficulty 8-10: {result['difficulty_8-10_accuracy']:.4f}")


def main():
    parser = argparse.ArgumentParser(description="Aggregate evaluation results and generate CSV summary")
    parser.add_argument("--results_dir", type=str, default="results",
                        help="Base directory containing method result folders")
    parser.add_argument("--output", type=str, default="evaluation_summary.csv",
                        help="Output CSV file path")

    args = parser.parse_args()

    print(f"Results directory: {args.results_dir}")
    print(f"Output CSV: {args.output}")

    # Aggregate results from all methods
    all_results = aggregate_all_methods(args.results_dir, args.output)

    # Save to CSV
    if all_results:
        save_to_csv(all_results, args.output)
    else:
        print("No results found to aggregate")


if __name__ == "__main__":
    main()
