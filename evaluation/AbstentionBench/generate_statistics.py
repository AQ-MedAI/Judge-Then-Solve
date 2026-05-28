"""
Generate summary and detailed statistics for abstention evaluation results.

This script generates:
1. Summary statistics (overall across all datasets)
2. Detailed statistics (by dataset)

Both are saved as CSV and Excel files.
"""

import json
import logging
from pathlib import Path
from typing import Dict, List, Tuple

import pandas as pd

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class AbstentionStatistics:
    """Calculate abstention statistics from evaluation results."""

    def __init__(self, results_dir: str = "results", use_correctness_file: bool = True):
        self.results_dir = Path(results_dir)
        self.results_data = []
        self.use_correctness_file = use_correctness_file

    def load_all_results(self):
        """Load all evaluation result files from results directory."""
        logger.info(f"Loading results from {self.results_dir}")

        # Use LLMJudgeCorrectnessEvaluator.json if available (contains correctness info)
        # Otherwise fall back to GroundTruthAbstentionEvaluator.json
        if self.use_correctness_file:
            result_files = list(self.results_dir.glob("*/*/LLMJudgeCorrectnessEvaluator.json"))
            if not result_files:
                logger.warning("No LLMJudgeCorrectnessEvaluator.json files found, falling back to GroundTruthAbstentionEvaluator.json")
                result_files = list(self.results_dir.glob("*/*/GroundTruthAbstentionEvaluator.json"))
        else:
            result_files = list(self.results_dir.glob("*/*/GroundTruthAbstentionEvaluator.json"))

        logger.info(f"Found {len(result_files)} result files")

        for result_file in result_files:
            # Parse directory structure: DATASET_MODEL/DATE/file.json
            parts = result_file.parts
            dataset_model = parts[-3]  # e.g., "MMLUMath_Qwen3-30B-A3B-Thinking-2507"
            date = parts[-2]

            # Split dataset and model name
            dataset_name, model_name = self._parse_dataset_model(dataset_model)

            # Load the JSON file
            with open(result_file, 'r') as f:
                data = json.load(f)

            # Store metadata with each response
            for response in data['responses']:
                response['_dataset_name'] = dataset_name
                response['_model_name'] = model_name
                response['_date'] = date
                self.results_data.append(response)

        logger.info(f"Loaded {len(self.results_data)} total responses")
        return self.results_data

    def _parse_dataset_model(self, dataset_model: str) -> Tuple[str, str]:
        """Parse dataset_model string into dataset and model names.

        Examples:
            "MMLUMath_Qwen3-30B-A3B-Thinking-2507" -> ("MMLUMath", "Qwen3-30B-A3B-Thinking-2507")
            "GPQA_Qwen3-30B-A3B-Thinking-2507_ours_prompt" -> ("GPQA", "Qwen3-30B-A3B-Thinking-2507_ours_prompt")
        """
        parts = dataset_model.split('_', 1)
        if len(parts) == 2:
            return parts[0], parts[1]
        else:
            # Fallback
            return dataset_model, "Unknown"

    def calculate_statistics(self, responses: List[Dict]) -> Dict:
        """Calculate statistics for a list of responses.

        Returns:
            Dict with keys:
                - should_abstain_count: number of questions that should be abstained
                - should_abstain_abstention_rate: proportion that abstained among should_abstain
                - should_answer_count: number of questions that should be answered
                - should_answer_answer_rate: proportion that answered (not abstained) among should_answer
                - should_answer_correctness: proportion correct among should_answer questions that were answered
        """
        if not responses:
            return {
                'should_abstain_count': 0,
                'should_abstain_abstention_rate': 0.0,
                'should_answer_count': 0,
                'should_answer_answer_rate': 0.0,
                'should_answer_correctness': 0.0,
            }

        # Separate into should_abstain and should_answer
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

            # Calculate correctness among answered questions
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

    def generate_summary_statistics(self) -> pd.DataFrame:
        """Generate summary statistics across all datasets for each model/method.

        Returns a DataFrame with metrics as rows and models as columns.
        """
        logger.info("Generating summary statistics...")

        # Group by model
        model_groups = {}
        for response in self.results_data:
            model_name = response['_model_name']
            if model_name not in model_groups:
                model_groups[model_name] = []
            model_groups[model_name].append(response)

        # Calculate statistics for each model
        summary_data = {}
        for model_name, responses in model_groups.items():
            stats = self.calculate_statistics(responses)
            summary_data[model_name] = stats

        # Convert to DataFrame with metrics as rows
        df = pd.DataFrame(summary_data)

        # Reorder rows
        row_order = [
            'should_abstain_count',
            'should_abstain_abstention_rate',
            'should_answer_count',
            'should_answer_answer_rate',
            'should_answer_correctness',
        ]
        df = df.reindex(row_order)

        # Rename rows for better readability
        df.index = [
            'Should Abstain: Count',
            'Should Abstain: Abstention Rate',
            'Should Answer: Count',
            'Should Answer: Answer Rate',
            'Should Answer: Correctness',
        ]

        logger.info(f"Summary statistics generated for {len(model_groups)} models")
        return df

    def generate_detailed_statistics(self) -> pd.DataFrame:
        """Generate detailed statistics by dataset for each model/method.

        Returns a DataFrame with (dataset, metric) as rows and models as columns.
        """
        logger.info("Generating detailed statistics by dataset...")

        # Group by model and dataset
        model_dataset_groups = {}
        for response in self.results_data:
            model_name = response['_model_name']
            dataset_name = response['_dataset_name']
            key = (model_name, dataset_name)
            if key not in model_dataset_groups:
                model_dataset_groups[key] = []
            model_dataset_groups[key].append(response)

        # Get unique models and datasets
        models = sorted(set(response['_model_name'] for response in self.results_data))
        datasets = sorted(set(response['_dataset_name'] for response in self.results_data))

        logger.info(f"Found {len(models)} models and {len(datasets)} datasets")

        # Calculate statistics for each model-dataset combination
        detailed_data = []
        for dataset in datasets:
            for model in models:
                key = (model, dataset)
                responses = model_dataset_groups.get(key, [])
                stats = self.calculate_statistics(responses)

                # Add dataset and model info
                stats['dataset'] = dataset
                stats['model'] = model
                detailed_data.append(stats)

        # Convert to DataFrame
        df = pd.DataFrame(detailed_data)

        return df

    def save_results(self, summary_df: pd.DataFrame, detailed_df: pd.DataFrame, output_dir: str = "analysis"):
        """Save summary and detailed statistics to CSV and Excel files."""
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)

        logger.info(f"Saving results to {output_path}")

        # Save summary statistics
        summary_csv = output_path / "summary_statistics.csv"
        summary_xlsx = output_path / "summary_statistics.xlsx"
        summary_df.to_csv(summary_csv)
        summary_df.to_excel(summary_xlsx)
        logger.info(f"Saved summary statistics to {summary_csv} and {summary_xlsx}")

        # Save detailed statistics
        detailed_csv = output_path / "detailed_statistics.csv"
        detailed_xlsx = output_path / "detailed_statistics.xlsx"
        detailed_df.to_csv(detailed_csv, index=False)
        detailed_df.to_excel(detailed_xlsx, index=False)
        logger.info(f"Saved detailed statistics to {detailed_csv} and {detailed_xlsx}")


def main():
    """Main function to generate and save statistics."""
    logger.info("Starting statistics generation...")

    # Initialize statistics calculator
    stats = AbstentionStatistics(results_dir="results")

    # Load all results
    stats.load_all_results()

    # Generate summary statistics
    summary_df = stats.generate_summary_statistics()
    print("\n" + "="*80)
    print("SUMMARY STATISTICS (Overall)")
    print("="*80)
    print(summary_df)
    print()

    # Generate detailed statistics
    detailed_df = stats.generate_detailed_statistics()
    print("\n" + "="*80)
    print("DETAILED STATISTICS (By Dataset)")
    print("="*80)
    print(detailed_df.head(20))
    print(f"\n... (showing first 20 rows, total {len(detailed_df)} rows)")
    print()

    # Save results
    stats.save_results(summary_df, detailed_df)

    logger.info("Statistics generation completed!")


if __name__ == "__main__":
    main()
