#!/usr/bin/env bash
# Generate clinical questions from knowledge graph
#
# Usage:
#   ./run_query_generation.sh
#
# Prerequisites:
#   - Knowledge graph must exist in resources/kg/
#   - Run ./run_graph_construction.sh first if needed
#
# What it does:
#   Generates diverse clinical questions with openness scores
#
# Output:
#   resources/query_set/qa_df_*.json

set -e
cd "$(dirname "$0")/.."

PYTHON_BIN="${PYTHON:-python3}"

echo "=== Generating Queries ==="

"$PYTHON_BIN" query_generation/examples/small_example.py

echo "✓ Query generation complete"
echo "  Output: resources/query_set/"
