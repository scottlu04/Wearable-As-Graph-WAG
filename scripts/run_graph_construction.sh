#!/usr/bin/env bash
# Build knowledge graph from health metrics
#
# Usage:
#   ./run_graph_construction.sh
#
# What it does:
#   1. Generates initial graph nodes from health metrics
#   2. Integrates dataset-specific metrics
#
# Output:
#   resources/kg/nodes.json
#   resources/kg/edges.json

set -e
cd "$(dirname "$0")/.."

PYTHON_BIN="${PYTHON:-python3}"

echo "=== Building Knowledge Graph ==="

"$PYTHON_BIN" graph_construction/examples/small_example_initial.py
"$PYTHON_BIN" graph_construction/examples/small_example_dataset.py

echo "✓ Graph construction complete"
echo "  Output: resources/kg/"
