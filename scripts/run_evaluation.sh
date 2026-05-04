#!/usr/bin/env bash
# Evaluate experiment results with LLM-as-a-judge
#
# Usage:
#   ./run_evaluation.sh [type]
#
# Types:
#   general - Evaluate general comparison (default)
#   global  - Evaluate global weight experiments
#   local   - Evaluate local weight experiments
#
# Examples:
#   ./run_evaluation.sh general
#   ./run_evaluation.sh global
#   ./run_evaluation.sh local
#
# Prerequisites:
#   Experiment results must exist in results/response_gen/[type]/

set -e
cd "$(dirname "$0")/.."

TYPE=${1:-general}
PYTHON_BIN="${PYTHON:-python3}"

echo "=== Evaluating Results: $TYPE ==="

"$PYTHON_BIN" evaluation/run_evaluation.py \
    --type $TYPE \
    --data-dir results/response_gen/$TYPE \
    --output-dir results/evaluation \
    --evaluate

echo "✓ Evaluation complete"
echo "  Output: results/evaluation/"
