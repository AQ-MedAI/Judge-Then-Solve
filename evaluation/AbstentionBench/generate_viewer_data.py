#!/usr/bin/env python3
"""
Generate JSON data files for the dataset viewer
"""
import json
import sys

def generate_data_files():
    # Load FreshQA data
    print("Loading FreshQA data...")
    try:
        with open('results/FreshQADataset_CustomOpenAIModelEnv/2026-01-12_18-10-06/LLMJudgeAbstentionDetector.json', 'r') as f:
            freshqa_data = json.load(f)

        print(f"FreshQA: {len(freshqa_data['responses'])} samples")

        # Save to viewer data file
        with open('freshqa_data.json', 'w') as f:
            json.dump(freshqa_data, f)
        print("✓ Generated freshqa_data.json")
    except Exception as e:
        print(f"✗ Error loading FreshQA: {e}")
        sys.exit(1)

    # Load KUQ data
    print("\nLoading KUQ data...")
    try:
        with open('results/KUQDataset_CustomOpenAIModelEnv/2026-01-16_11-49-33/LLMJudgeAbstentionDetector.json', 'r') as f:
            kuq_data = json.load(f)

        print(f"KUQ: {len(kuq_data['responses'])} samples")

        # Save to viewer data file
        with open('kuq_data.json', 'w') as f:
            json.dump(kuq_data, f)
        print("✓ Generated kuq_data.json")
    except Exception as e:
        print(f"✗ Error loading KUQ: {e}")
        sys.exit(1)

    # Print statistics
    print("\n" + "="*60)
    print("Statistics Summary")
    print("="*60)

    # FreshQA stats
    freshqa_should_abstain = sum(1 for r in freshqa_data['responses'] if r['prompt']['should_abstain'])
    freshqa_model_abstained = sum(1 for r in freshqa_data['responses'] if r['is_abstention'])
    print(f"\nFreshQA:")
    print(f"  Total samples: {len(freshqa_data['responses'])}")
    print(f"  Should abstain: {freshqa_should_abstain}")
    print(f"  Should answer: {len(freshqa_data['responses']) - freshqa_should_abstain}")
    print(f"  Model abstained: {freshqa_model_abstained}")
    print(f"  Model answered: {len(freshqa_data['responses']) - freshqa_model_abstained}")

    # KUQ stats
    kuq_should_abstain = sum(1 for r in kuq_data['responses'] if r['prompt']['should_abstain'])
    kuq_model_abstained = sum(1 for r in kuq_data['responses'] if r['is_abstention'])
    print(f"\nKUQ:")
    print(f"  Total samples: {len(kuq_data['responses'])}")
    print(f"  Should abstain: {kuq_should_abstain}")
    print(f"  Should answer: {len(kuq_data['responses']) - kuq_should_abstain}")
    print(f"  Model abstained: {kuq_model_abstained}")
    print(f"  Model answered: {len(kuq_data['responses']) - kuq_model_abstained}")

    print("\n" + "="*60)
    print("✓ All data files generated successfully!")
    print("✓ Open dataset_viewer.html in your browser to view the data")
    print("="*60)

if __name__ == '__main__':
    generate_data_files()
