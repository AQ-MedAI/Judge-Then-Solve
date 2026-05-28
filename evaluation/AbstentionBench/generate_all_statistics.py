"""
Complete statistics generation script for AbstentionBench evaluation results.

This script generates:
1. Summary statistics (overall across all datasets)
2. Detailed statistics (by dataset, with Length field)

Both are saved as CSV and Excel files.
"""

import json
import logging
import os
from pathlib import Path
from typing import Dict, List, Tuple

import pandas as pd

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Hardcoded tokenizer path
TOKENIZER_PATH = os.getenv("TOKENIZER_PATH", "Qwen/Qwen3-30B-A3B")


class CompleteStatistics:
    """Generate both summary and detailed statistics."""

    def __init__(self, results_dir: str = "results"):
        self.results_dir = Path(results_dir)
        self.results_data = []
        self.tokenizer = None

    def load_tokenizer(self):
        """Load tokenizer for calculating response length."""
        logger.info(f"Loading tokenizer from {TOKENIZER_PATH}")
        try:
            from transformers import AutoTokenizer
            import os

            if os.path.exists(TOKENIZER_PATH):
                self.tokenizer = AutoTokenizer.from_pretrained(
                    TOKENIZER_PATH,
                    trust_remote_code=True,
                    local_files_only=True
                )
                logger.info("Tokenizer loaded successfully from local path")
            else:
                logger.warning(f"Local path not found: {TOKENIZER_PATH}")
                logger.info("Trying to load from HuggingFace: Qwen/Qwen3-30B-A3B-Thinking-2507")
                self.tokenizer = AutoTokenizer.from_pretrained(
                    "Qwen/Qwen3-30B-A3B-Thinking-2507",
                    trust_remote_code=True
                )
                logger.info("Tokenizer loaded successfully from HuggingFace")
        except Exception as e:
            logger.error(f"Failed to load tokenizer: {e}")
            logger.warning("Length calculation will be skipped")
            self.tokenizer = None

    def load_all_results(self):
        """Load all evaluation result files from results directory."""
        logger.info(f"Loading results from {self.results_dir}")

        result_files = list(self.results_dir.glob("*/*/LLMJudgeCorrectnessEvaluator.json"))
        if not result_files:
            logger.warning("No LLMJudgeCorrectnessEvaluator.json files found")
            result_files = list(self.results_dir.glob("*/*/GroundTruthAbstentionEvaluator.json"))

        logger.info(f"Found {len(result_files)} result files")

        for result_file in result_files:
            parts = result_file.parts
            dataset_model = parts[-3]
            date = parts[-2]

            dataset_name, model_name = self._parse_dataset_model(dataset_model)

            with open(result_file, 'r') as f:
                data = json.load(f)

            for response in data['responses']:
                response['_dataset_name'] = dataset_name
                response['_model_name'] = model_name
                response['_date'] = date
                self.results_data.append(response)

        logger.info(f"Loaded {len(self.results_data)} total responses")
        return self.results_data

    def _parse_dataset_model(self, dataset_model: str) -> Tuple[str, str]:
        """Parse dataset_model string into dataset and model names."""
        parts = dataset_model.split('_', 1)
        if len(parts) == 2:
            return parts[0], parts[1]
        else:
            return dataset_model, "Unknown"

    def calculate_statistics(self, responses: List[Dict]) -> Dict:
        """Calculate statistics for a list of responses."""
        if not responses:
            return {
                'should_abstain_count': 0,
                'should_abstain_abstention_rate': 0.0,
                'should_answer_count': 0,
                'should_answer_answer_rate': 0.0,
                'should_answer_correctness': 0.0,
            }

        should_abstain = [r for r in responses if r['prompt']['should_abstain']]
        should_answer = [r for r in responses if not r['prompt']['should_abstain']]

        # Calculate should_abstain statistics
        should_abstain_count = len(should_abstain)
        if should_abstain_count > 0:
            abstained_count = sum(1 for r in should_abstain if r.get('is_abstention', False))
            should_abstain_abstention_rate = abstained_count / should_abstain_count
        else:
            should_abstain_abstention_rate = 0.0

        # Calculate should_answer statistics
        should_answer_count = len(should_answer)
        if should_answer_count > 0:
            answered_count = sum(1 for r in should_answer if not r.get('is_abstention', True))
            should_answer_answer_rate = answered_count / should_answer_count

            answered_responses = [r for r in should_answer if not r.get('is_abstention', True)]
            if answered_responses:
                correct_count = sum(1 for r in answered_responses if r.get('is_response_correct', False))
                should_answer_correctness = correct_count / len(answered_responses)
            else:
                should_answer_correctness = 0.0
        else:
            should_answer_answer_rate = 0.0
            should_answer_correctness = 0.0

        return {
            'should_abstain_count': should_abstain_count,
            'should_abstain_abstention_rate': should_abstain_abstention_rate,
            'should_answer_count': should_answer_count,
            'should_answer_answer_rate': should_answer_answer_rate,
            'should_answer_correctness': should_answer_correctness,
        }

    def calculate_average_length(self, responses: List[Dict]) -> float:
        """Calculate average token length of responses (including reasoning chain)."""
        if not responses or not self.tokenizer:
            return 0.0

        total_length = 0
        count = 0
        for response in responses:
            text = response.get('response', '')
            if text:
                tokens = self.tokenizer.encode(text, add_special_tokens=False)
                total_length += len(tokens)
                count += 1

        return total_length / count if count > 0 else 0.0

    def generate_summary_statistics(self) -> pd.DataFrame:
        """Generate summary statistics across all datasets."""
        logger.info("Generating summary statistics...")

        model_groups = {}
        for response in self.results_data:
            model_name = response['_model_name']
            if model_name not in model_groups:
                model_groups[model_name] = []
            model_groups[model_name].append(response)

        summary_data = {}
        for model_name, responses in model_groups.items():
            stats = self.calculate_statistics(responses)

            # Calculate average length for should_abstain and should_answer
            should_abstain = [r for r in responses if r['prompt']['should_abstain']]
            should_answer = [r for r in responses if not r['prompt']['should_abstain']]

            stats['should_abstain_avg_length'] = self.calculate_average_length(should_abstain)
            stats['should_answer_avg_length'] = self.calculate_average_length(should_answer)

            summary_data[model_name] = stats

        df = pd.DataFrame(summary_data)
        row_order = [
            'should_abstain_count',
            'should_abstain_abstention_rate',
            'should_abstain_avg_length',
            'should_answer_count',
            'should_answer_answer_rate',
            'should_answer_correctness',
            'should_answer_avg_length',
        ]
        df = df.reindex(row_order)
        df.index = [
            'Should Abstain: Count',
            'Should Abstain: Abstention Rate',
            'Should Abstain: Avg Length',
            'Should Answer: Count',
            'Should Answer: Answer Rate',
            'Should Answer: Correctness',
            'Should Answer: Avg Length',
        ]

        logger.info(f"Summary statistics generated for {len(model_groups)} models")
        return df

    def generate_detailed_table(self) -> pd.DataFrame:
        """Generate detailed statistics table by dataset."""
        logger.info("Generating detailed statistics table...")

        # Group by model and dataset
        model_dataset_groups = {}
        for response in self.results_data:
            model_name = response['_model_name']
            dataset_name = response['_dataset_name']
            key = (model_name, dataset_name)
            if key not in model_dataset_groups:
                model_dataset_groups[key] = []
            model_dataset_groups[key].append(response)

        models = sorted(set(response['_model_name'] for response in self.results_data))
        datasets = sorted(set(response['_dataset_name'] for response in self.results_data))

        logger.info(f"Found {len(models)} models and {len(datasets)} datasets")

        rows = []

        # Section 1: 条件缺失 (Should Abstain)
        for dataset in datasets:
            # Sample count row (first row for each dataset)
            row_count = {
                '数据量': '条件缺失',
                '数据集名字': dataset,
                '指标': 'Sample Count'
            }
            for model in models:
                key = (model, dataset)
                responses = model_dataset_groups.get(key, [])
                should_abstain = [r for r in responses if r['prompt']['should_abstain']]
                row_count[model] = len(should_abstain)
            rows.append(row_count)

            # Abstention Rate row
            row_abstention = {
                '数据量': '',
                '数据集名字': '',
                '指标': 'Abstention Rate'
            }
            for model in models:
                key = (model, dataset)
                responses = model_dataset_groups.get(key, [])
                should_abstain = [r for r in responses if r['prompt']['should_abstain']]
                if should_abstain:
                    abstained = sum(1 for r in should_abstain if r.get('is_abstention', False))
                    rate = abstained / len(should_abstain)
                    row_abstention[model] = rate
                else:
                    row_abstention[model] = 0.0
            rows.append(row_abstention)

            # Length row for Should Abstain
            row_length = {
                '数据量': '',
                '数据集名字': '',
                '指标': 'Length'
            }
            for model in models:
                key = (model, dataset)
                responses = model_dataset_groups.get(key, [])
                should_abstain = [r for r in responses if r['prompt']['should_abstain']]
                avg_length = self.calculate_average_length(should_abstain)
                row_length[model] = avg_length
            rows.append(row_length)

        # Section 2: Well defined (Should Answer)
        for dataset in datasets:
            # Sample count row (first row for each dataset)
            row_count = {
                '数据量': 'Well defined',
                '数据集名字': dataset,
                '指标': 'Sample Count'
            }
            for model in models:
                key = (model, dataset)
                responses = model_dataset_groups.get(key, [])
                should_answer = [r for r in responses if not r['prompt']['should_abstain']]
                row_count[model] = len(should_answer)
            rows.append(row_count)

            # Answer Rate row
            row_answer_rate = {
                '数据量': '',
                '数据集名字': '',
                '指标': 'Answer Rate'
            }
            for model in models:
                key = (model, dataset)
                responses = model_dataset_groups.get(key, [])
                should_answer = [r for r in responses if not r['prompt']['should_abstain']]
                if should_answer:
                    answered = sum(1 for r in should_answer if not r.get('is_abstention', True))
                    rate = answered / len(should_answer)
                    row_answer_rate[model] = rate
                else:
                    row_answer_rate[model] = 0.0
            rows.append(row_answer_rate)

            # Correctness row
            row_correctness = {
                '数据量': '',
                '数据集名字': '',
                '指标': 'Correctness'
            }
            for model in models:
                key = (model, dataset)
                responses = model_dataset_groups.get(key, [])
                should_answer = [r for r in responses if not r['prompt']['should_abstain']]
                answered = [r for r in should_answer if not r.get('is_abstention', True)]
                if answered:
                    correct = sum(1 for r in answered if r.get('is_response_correct', False))
                    rate = correct / len(answered)
                    row_correctness[model] = rate
                else:
                    row_correctness[model] = 0.0
            rows.append(row_correctness)

            # Length row for Well defined
            row_length_answer = {
                '数据量': '',
                '数据集名字': '',
                '指标': 'Length'
            }
            for model in models:
                key = (model, dataset)
                responses = model_dataset_groups.get(key, [])
                should_answer = [r for r in responses if not r['prompt']['should_abstain']]
                avg_length = self.calculate_average_length(should_answer)
                row_length_answer[model] = avg_length
            rows.append(row_length_answer)

        df = pd.DataFrame(rows)
        column_order = ['数据量', '数据集名字', '指标'] + models
        df = df[column_order]

        logger.info(f"Generated detailed table with {len(rows)} rows")
        return df

    def save_results(self, summary_df: pd.DataFrame, detailed_df: pd.DataFrame, output_dir: str = "analysis"):
        """Save all statistics to CSV and Excel files."""
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)

        logger.info(f"Saving results to {output_path}")

        # Save summary statistics
        summary_csv = output_path / "summary_statistics.csv"
        summary_xlsx = output_path / "summary_statistics.xlsx"
        summary_df.to_csv(summary_csv)
        summary_df.to_excel(summary_xlsx)
        logger.info(f"Saved summary to {summary_csv} and {summary_xlsx}")

        # Save detailed statistics
        detailed_csv = output_path / "detailed_statistics.csv"
        detailed_xlsx = output_path / "detailed_statistics.xlsx"
        detailed_df.to_csv(detailed_csv, index=False)
        detailed_df.to_excel(detailed_xlsx, index=False)
        logger.info(f"Saved detailed to {detailed_csv} and {detailed_xlsx}")


def main():
    """Main function to generate all statistics."""
    logger.info("="*80)
    logger.info("Starting Complete Statistics Generation")
    logger.info("="*80)

    stats = CompleteStatistics(results_dir="results")

    # Load tokenizer
    stats.load_tokenizer()

    # Load all results
    stats.load_all_results()

    # Generate summary statistics
    logger.info("\n" + "="*80)
    logger.info("Generating Summary Statistics")
    logger.info("="*80)
    summary_df = stats.generate_summary_statistics()
    print("\nSUMMARY STATISTICS:")
    print(summary_df)
    print()

    # Generate detailed statistics
    logger.info("\n" + "="*80)
    logger.info("Generating Detailed Statistics")
    logger.info("="*80)
    detailed_df = stats.generate_detailed_table()
    print("\nDETAILED STATISTICS (first 20 rows):")
    print(detailed_df.head(20))
    print(f"\n... (total {len(detailed_df)} rows)")
    print()

    # Save all results
    logger.info("\n" + "="*80)
    logger.info("Saving Results")
    logger.info("="*80)
    stats.save_results(summary_df, detailed_df)

    logger.info("\n" + "="*80)
    logger.info("Complete Statistics Generation Finished!")
    logger.info("="*80)
    logger.info("Output files:")
    logger.info("  - analysis/summary_statistics.csv")
    logger.info("  - analysis/summary_statistics.xlsx")
    logger.info("  - analysis/detailed_statistics.csv")
    logger.info("  - analysis/detailed_statistics.xlsx")


if __name__ == "__main__":
    main()
