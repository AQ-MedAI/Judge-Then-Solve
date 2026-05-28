# Test Data Usage Guide

## Overview

The evaluation pipeline now supports both full dataset and test dataset evaluation through the `DATA_DIR` environment variable.

## Directory Structure

```
MiP-Overthinking/
├── data/                    # Full datasets (original)
│   ├── gsm8k.json
│   ├── math.json
│   ├── svamp.json
│   └── formula.json
└── data/test_data/          # Test datasets (20% split)
    ├── _manifest.json       # Split information
    ├── gsm8k.json          # 234 samples (117 insufficient + 117 well_defined)
    ├── math.json           # 22 samples (11 insufficient + 11 well_defined)
    ├── svamp.json          # 60 samples (60 insufficient)
    └── formula.json        # 10 samples (10 insufficient)
```

## Test Dataset Statistics

From `_manifest.json`:
- **Total test samples**: 326
- **Split rule**: Stratified 70% train / 10% eval / 20% test
- **Seed**: 42 (reproducible)

### Per-dataset breakdown:
- **gsm8k**: 234 samples (117 insufficient + 117 well_defined)
- **math**: 22 samples (11 insufficient + 11 well_defined)
- **svamp**: 60 samples (60 insufficient only)
- **formula**: 10 samples (10 insufficient only)

## Usage

### Option 1: Use Full Dataset (Default)

```bash
# Run with full dataset
./run_evaluation.sh
```

Or explicitly:
```bash
DATA_DIR=data ./run_evaluation.sh
```

### Option 2: Use Test Dataset

```bash
# Run with test dataset (faster, for quick testing)
DATA_DIR=data/test_data ./run_evaluation.sh
```

## When to Use Each Dataset

### Use Full Dataset When:
- Running final evaluation for paper/publication
- Need complete statistics
- Comparing with other models' full results

### Use Test Dataset When:
- Quick testing during development
- Debugging evaluation pipeline
- Testing new models before full evaluation
- Saving API costs (GPT-4o evaluation)
- Faster iteration cycles

## Cost Estimation

Assuming GPT-4o evaluation costs:
- **Full dataset**: ~1618 samples × 2 API calls (insufficient check + eval) = ~3236 API calls
- **Test dataset**: ~326 samples × 2 API calls = ~652 API calls
- **Savings**: ~80% reduction in API calls

## Results Directory

Results are saved to the same `results/` directory structure regardless of which dataset is used:

```
results/
├── gsm8k/
│   ├── normal/
│   │   └── qwen3-30b-thinking/
│   │       ├── response.jsonl
│   │       ├── analysis_info.json
│   │       └── per_sample_eval.jsonl
│   └── MiP/
│       └── qwen3-30b-thinking/
│           └── ...
├── math/
├── svamp/
└── formula/
```

**Note**: Be careful not to mix results from full and test datasets. Consider using different model names or clearing results between runs.

## Example Workflow

```bash
# 1. Quick test with test dataset
DATA_DIR=data/test_data ./run_evaluation.sh

# 2. Check results look good
cat results/gsm8k/MiP/qwen3-30b-thinking/analysis_info.json

# 3. If satisfied, run full evaluation
./run_evaluation.sh
```

## Verification

To verify which dataset was used, check the number of samples in results:

```bash
# Count samples in results
wc -l results/gsm8k/MiP/qwen3-30b-thinking/response.jsonl

# Should be:
# - Full dataset: ~582 lines (gsm8k)
# - Test dataset: ~234 lines (gsm8k)
```
