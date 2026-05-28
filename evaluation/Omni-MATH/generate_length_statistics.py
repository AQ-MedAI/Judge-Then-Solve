#!/usr/bin/env python3
"""
Generate length statistics for Omni-MATH evaluation results.
Calculates average token length for each method across difficulty ranges and overall.
Only counts valid (non-empty) responses.
"""
import json
import argparse
import os
import csv
from collections import defaultdict
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# 默认 Qwen3 tokenizer 路径
TOKENIZER_PATH_QWEN3 = os.getenv("TOKENIZER_PATH_QWEN3", os.getenv("TOKENIZER_PATH", "Qwen/Qwen3-30B-A3B"))
# DeepSeek-R1-Distill-Qwen-14B tokenizer 路径
TOKENIZER_PATH_DEEPSEEK = os.getenv("TOKENIZER_PATH_DEEPSEEK", "deepseek-ai/DeepSeek-R1-Distill-Qwen-14B")


class LengthStatistics:
    """Generate length statistics for evaluation results."""

    def __init__(self, results_dir: str = "results"):
        self.results_dir = Path(results_dir)
        self.tokenizer = None
        # 当前使用的 tokenizer 路径
        self.tokenizer_path = TOKENIZER_PATH_QWEN3

    def load_tokenizer(self):
        """Load tokenizer for calculating response length."""
        logger.info(f"Loading tokenizer from {self.tokenizer_path}")
        try:
            from transformers import AutoTokenizer
            import os

            if os.path.exists(self.tokenizer_path):
                self.tokenizer = AutoTokenizer.from_pretrained(
                    self.tokenizer_path,
                    trust_remote_code=True,
                    local_files_only=True
                )
                logger.info("Tokenizer loaded successfully from local path")
            else:
                logger.warning(f"Local path not found: {self.tokenizer_path}")
                logger.info("Trying to load from HuggingFace: Qwen/Qwen2.5-Math-7B-Instruct")
                self.tokenizer = AutoTokenizer.from_pretrained(
                    "Qwen/Qwen2.5-Math-7B-Instruct",
                    trust_remote_code=True
                )
                logger.info("Tokenizer loaded successfully from HuggingFace")
        except Exception as e:
            logger.error(f"Failed to load tokenizer: {e}")
            logger.warning("Length calculation will be skipped")
            self.tokenizer = None

    def set_tokenizer_for_method(self, method_name: str):
        """
        根据方法名选择合适的 tokenizer。
        - 如果方法名中包含 deepseek（不区分大小写），则使用 DeepSeek-R1-Distill-Qwen-14B 的 tokenizer。
        - 否则默认使用 Qwen3-30B-A3B-Thinking-2507 的 tokenizer。
        """
        lower_name = method_name.lower()
        if "deepseek" in lower_name or "deepseek-r1-distill-qwen-14b" in lower_name:
            new_path = TOKENIZER_PATH_DEEPSEEK
        else:
            new_path = TOKENIZER_PATH_QWEN3

        # 只有在路径变化或 tokenizer 还没加载时才重新加载，避免重复初始化
        if self.tokenizer is None or self.tokenizer_path != new_path:
            self.tokenizer_path = new_path
            self.load_tokenizer()

    def categorize_difficulty(self, difficulty):
        """Categorize difficulty into ranges [1-2), [2-3), ..., [9-10]"""
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

    def calculate_token_length(self, text):
        """Calculate token length of text."""
        if not text or not self.tokenizer:
            return 0
        try:
            tokens = self.tokenizer.encode(text, add_special_tokens=False)
            return len(tokens)
        except Exception as e:
            logger.warning(f"Failed to tokenize text: {e}")
            return 0

    def load_method_results(self, eval_file, gen_file):
        """Load evaluation and generation results for a method."""
        results = []

        # Load generation file to get model responses
        generations = {}
        with open(gen_file, 'r', encoding='utf-8') as f:
            for line in f:
                entry = json.loads(line.strip())
                problem = entry.get('problem', '')
                generations[problem] = entry.get('model_generation', '')

        # Load evaluation file
        with open(eval_file, 'r', encoding='utf-8') as f:
            for line in f:
                entry = json.loads(line.strip())
                original_json = json.loads(entry['original_json'])

                problem = original_json.get('problem', '')
                model_generation = original_json.get('model_generation', '')
                difficulty = original_json.get('difficulty', '')

                # Calculate token length only for valid responses
                token_length = 0
                if model_generation and model_generation.strip():
                    token_length = self.calculate_token_length(model_generation)

                results.append({
                    'difficulty': difficulty,
                    'model_generation': model_generation,
                    'token_length': token_length,
                    'is_valid': bool(model_generation and model_generation.strip())
                })

        return results

    def calculate_length_statistics(self, results):
        """Calculate average length statistics by difficulty ranges."""
        difficulty_ranges = ["1-2", "2-3", "3-4", "4-5", "5-6", "6-7", "7-8", "8-9", "9-10"]

        grouped = defaultdict(lambda: {'total_length': 0, 'valid_count': 0})

        for result in results:
            if not result['is_valid']:
                continue

            diff_range = self.categorize_difficulty(result['difficulty'])
            if diff_range != "other":
                grouped[diff_range]['total_length'] += result['token_length']
                grouped[diff_range]['valid_count'] += 1

        # Calculate average length for each range
        stats = {}
        for diff_range in difficulty_ranges:
            data = grouped[diff_range]
            valid_count = data['valid_count']
            total_length = data['total_length']

            stats[diff_range] = {
                'valid_count': valid_count,
                'total_length': total_length,
                'avg_length': total_length / valid_count if valid_count > 0 else 0.0
            }

        # Calculate overall statistics
        total_length_all = sum(r['total_length'] for r in grouped.values())
        valid_count_all = sum(r['valid_count'] for r in grouped.values())

        stats['overall'] = {
            'valid_count': valid_count_all,
            'total_length': total_length_all,
            'avg_length': total_length_all / valid_count_all if valid_count_all > 0 else 0.0
        }

        return stats

    def find_latest_result_dir(self, method_dir):
        """Find the latest timestamped result directory for a method."""
        if not method_dir.exists():
            return None

        subdirs = [d for d in method_dir.iterdir() if d.is_dir()]
        if not subdirs:
            return None

        subdirs.sort(reverse=True)
        return subdirs[0]

    def aggregate_all_methods(self):
        """Aggregate length statistics from all methods."""
        all_method_stats = {}

        if not self.results_dir.exists():
            logger.error(f"Results directory not found: {self.results_dir}")
            return all_method_stats

        method_dirs = [d for d in self.results_dir.iterdir() if d.is_dir()]
        logger.info(f"Found {len(method_dirs)} method directories")

        for method_dir in method_dirs:
            method_name = method_dir.name

            # 针对每个方法名选择合适的 tokenizer（Qwen3 或 DeepSeek）
            self.set_tokenizer_for_method(method_name)

            latest_dir = self.find_latest_result_dir(method_dir)

            if not latest_dir:
                logger.warning(f"No results found for method: {method_name}")
                continue

            eval_file = latest_dir / "gpt4o_evaluations.jsonl"
            gen_file = latest_dir / "model_generations.jsonl"

            if not eval_file.exists():
                logger.warning(f"Evaluation file not found: {eval_file}")
                continue

            if not gen_file.exists():
                logger.warning(f"Generation file not found: {gen_file}")
                continue

            logger.info(f"Processing {method_name}: {eval_file}")

            results = self.load_method_results(eval_file, gen_file)
            stats = self.calculate_length_statistics(results)
            all_method_stats[method_name] = stats

        return all_method_stats

    def save_to_csv(self, all_method_stats, output_csv):
        """Save length statistics to CSV file."""
        if not all_method_stats:
            logger.warning("No results to save")
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
                        'valid_count': data['valid_count'],
                        'total_length': data['total_length'],
                        'avg_length': f"{data['avg_length']:.2f}"
                    })

        # Write to CSV
        fieldnames = ['method', 'difficulty_range', 'valid_count', 'total_length', 'avg_length']

        with open(output_csv, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)

        logger.info(f"Length statistics saved to: {output_csv}")

    def print_summary(self, all_method_stats):
        """Print summary of length statistics."""
        logger.info("\n" + "="*80)
        logger.info("LENGTH STATISTICS SUMMARY")
        logger.info("="*80)

        for method_name, stats in all_method_stats.items():
            logger.info(f"\n{method_name}:")
            logger.info("-" * 60)

            # Print overall stats
            overall = stats['overall']
            logger.info(f"Overall ({overall['valid_count']} valid responses):")
            logger.info(f"  Average Length: {overall['avg_length']:.2f} tokens")

            logger.info("\nBy Difficulty Range:")
            difficulty_ranges = ["1-2", "2-3", "3-4", "4-5", "5-6", "6-7", "7-8", "8-9", "9-10"]
            for dr in difficulty_ranges:
                if dr in stats:
                    data = stats[dr]
                    logger.info(f"  {dr}: {data['valid_count']} valid | "
                              f"Avg Length: {data['avg_length']:.2f} tokens")


def main():
    """Main function to generate length statistics."""
    parser = argparse.ArgumentParser(description="Generate length statistics for Omni-MATH results")
    parser.add_argument("--results_dir", type=str, default="results",
                        help="Base directory containing method result folders")
    parser.add_argument("--output_csv", type=str, default="length_statistics.csv",
                        help="Output CSV file path")

    args = parser.parse_args()

    logger.info("="*80)
    logger.info("Starting Length Statistics Generation")
    logger.info("="*80)

    stats_generator = LengthStatistics(results_dir=args.results_dir)

    # Aggregate results from all methods（在内部会根据方法名自动选择 tokenizer）
    all_method_stats = stats_generator.aggregate_all_methods()

    if not all_method_stats:
        logger.warning("No results found to aggregate")
        return

    # Save to CSV
    stats_generator.save_to_csv(all_method_stats, args.output_csv)

    # Print summary
    stats_generator.print_summary(all_method_stats)

    logger.info("\n" + "="*80)
    logger.info("Length Statistics Generation Finished!")
    logger.info("="*80)
    logger.info(f"Output file: {args.output_csv}")


if __name__ == "__main__":
    main()
