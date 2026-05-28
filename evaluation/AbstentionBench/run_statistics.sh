#!/bin/bash
# Script to run complete statistics generation on server
# This script generates both summary and detailed statistics

echo "=========================================="
echo "AbstentionBench Statistics Generation"
echo "=========================================="
echo ""

# Run the complete statistics generation script
python generate_all_statistics.py

echo ""
echo "=========================================="
echo "Statistics generation completed!"
echo "=========================================="
echo ""
echo "Output files in analysis/ directory:"
echo "  - summary_statistics.csv"
echo "  - summary_statistics.xlsx"
echo "  - detailed_statistics.csv"
echo "  - detailed_statistics.xlsx"
echo ""

