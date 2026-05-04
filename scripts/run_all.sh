#!/usr/bin/env bash
# Run complete WAG pipeline
#
# Usage:
#   ./run_all.sh
#
# What it does:
#   1. Build knowledge graph from health metrics
#   2. Generate clinical questions
#   3. Run experiments (Base vs RAG vs StaticGraph vs FullContext vs WAG)
#   4. Evaluate results with LLM-as-a-judge
#
# Output:
#   resources/kg/           - Knowledge graph
#   resources/query_set/    - Generated queries
#   results/response_gen/   - Experiment results
#   results/evaluation/     - Evaluation results
#   logs/                   - Log files

set -e
cd "$(dirname "$0")/.."

echo "════════════════════════════════"
echo "  WAG Complete Pipeline"
echo "════════════════════════════════"
echo ""

# Step 1: Build graph
echo "Step 1/4: Building knowledge graph..."
./scripts/run_graph_construction.sh
echo ""

# Step 2: Generate queries
echo "Step 2/4: Generating queries..."
./scripts/run_query_generation.sh
echo ""

# Step 3: Run experiments
echo "Step 3/4: Running experiments..."
./scripts/run_experiments.sh general 4
echo ""

# Step 4: Evaluate
echo "Step 4/4: Evaluating results..."
./scripts/run_evaluation.sh general
echo ""

echo "✓ Complete pipeline finished!"
echo "  Results: results/"
echo "  Logs: logs/"
