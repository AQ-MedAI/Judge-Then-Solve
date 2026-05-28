"""
Generate detailed statistics with Length field in the specified format.

Format:
    数据量    数据集名字        Model1    Model2    Model3
条件缺失：        MMLU_history    Abstention Rate
            Length
        MMLU_Math    Abstention Rate
            Length
...
Well defined        MMLU_history    Answer_rate
            Correctness
            Length
...
"""

import json
import logging
import os
from pathlib import Path
from typing import Dict, List, Tuple

import pandas as pd
from transformers import AutoTokenizer

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Hardcoded tokenizer path
TOKENIZER_PATH = os.getenv("TOKENIZER_PATH", "Qwen/Qwen3-30B-A3B")


class DetailedStatisticsV2:
    """Generate detailed statistics with Length field."""

    def __init__(self, results_dir: str = "results"):
        self.results_dir = Path(results_dir)
        self.results_data = []
        self.tokenizer = None

    def load_tokenizer(self):
        """Load tokenizer for calculating response length."""
        logger.info(f"Loading tokenizer from {TOKENIZER_PATH}")
        try:
            # Load from local path only (for server without HF access)
            import os
            if os.path.exists(TOKENIZER_PATH):
                self.tokenizer = AutoTokenizer.from_pretrained(
                    TOKENIZER_PATH,
                    trust_remote_code=True,
                    local_files_only=True
                )
                logger.info("Tokenizer loaded successfully from local path")
            else:
                # Fallback to HuggingFace (for local development)
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

    def calculate_average_length(self, responses: List[Dict]) -> float:
        """Calculate average token length of responses (including reasoning chain)."""
        if not responses or not self.tokenizer:
            return 0.0

        total_length = 0
        count = 0
        for response in responses:
            # Use response field (full output including reasoning chain)
            text = response.get('response', '')
            if text:
                tokens = self.tokenizer.encode(text, add_special_tokens=False)
                total_length += len(tokens)
                count += 1

        return total_length / count if count > 0 else 0.0

    def generate_detailed_table(self) -> pd.DataFrame:
        """Generate detailed statistics table in the specified format."""
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

        # Get unique models and datasets
        models = sorted(set(response['_model_name'] for response in self.results_data))
        datasets = sorted(set(response['_dataset_name'] for response in self.results_data))

        logger.info(f"Found {len(models)} models and {len(datasets)} datasets")

        # Build table rows
        rows = []

        # Section 1: 条件缺失 (Should Abstain)
        for dataset in datasets:
            # Abstention Rate row
            row_abstention = {
                '数据量': '条件缺失',
                '数据集名字': dataset,
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

            # Length row
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
            # Answer Rate row
            row_answer_rate = {
                '数据量': 'Well defined',
                '数据集名字': dataset,
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

        # Convert to DataFrame
        df = pd.DataFrame(rows)

        # Reorder columns
        column_order = ['数据量', '数据集名字', '指标'] + models
        df = df[column_order]

        logger.info(f"Generated detailed table with {len(rows)} rows")
        return df

    def save_results(self, df: pd.DataFrame, output_dir: str = "analysis"):
        """Save detailed statistics to CSV and Excel files."""
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)

        logger.info(f"Saving results to {output_path}")

        # Save detailed statistics
        detailed_csv = output_path / "detailed_statistics_v2.csv"
        detailed_xlsx = output_path / "detailed_statistics_v2.xlsx"
        df.to_csv(detailed_csv, index=False)
        df.to_excel(detailed_xlsx, index=False)
        logger.info(f"Saved to {detailed_csv} and {detailed_xlsx}")


def main():
    """Main function to generate and save detailed statistics."""
    logger.info("Starting detailed statistics generation (v2)...")

    # Initialize statistics calculator
    stats = DetailedStatisticsV2(results_dir="results")

    # Load tokenizer
    stats.load_tokenizer()

    # Load all results
    stats.load_all_results()

    # Generate detailed table
    detailed_df = stats.generate_detailed_table()

    print("\n" + "="*80)
    print("DETAILED STATISTICS (V2 Format)")
    print("="*80)
    print(detailed_df.head(20))
    print(f"\n... (showing first 20 rows, total {len(detailed_df)} rows)")
    print()

    # Save results
    stats.save_results(detailed_df)

    logger.info("Detailed statistics generation completed!")


if __name__ == "__main__":
    main()
