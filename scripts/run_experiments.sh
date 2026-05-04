#!/usr/bin/env bash
# Run WAG experiments: Base vs RAG vs StaticGraph vs FullContext vs WAG
#
# Usage:
#   ./run_experiments.sh [experiment] [workers] [qid_filter] [generate_response] [methods]
#
# Examples:
#   ./run_experiments.sh general
#   ./run_experiments.sh general 10
#   ./run_experiments.sh global 10 resources/query_set/q_id_global.json
#   ./run_experiments.sh local 10 resources/query_set/q_id_local.json
#   ./run_experiments.sh general 4 filters/general_balanced_2000.json true
#   ./run_experiments.sh general 4 filters/general_balanced_100.json true Base,Rag,StaticGraph,FullContext,Wag

set -e
cd "$(dirname "$0")/.."

EXPERIMENT=${1:-general}
WORKERS=${2:-4}
QID_FILTER=${3:-""}
GENERATE_RESPONSE=${4:-false}
METHODS=${5:-""}
PYTHON_BIN="${PYTHON:-python3}"

echo "=== Running Experiments: $EXPERIMENT ==="
echo "  Workers: $WORKERS"

# Build command
CMD="$PYTHON_BIN run_experiments.py --experiments $EXPERIMENT"

# Add qid filter if provided and file exists
if [ -n "$QID_FILTER" ]; then
    if [ -f "$QID_FILTER" ]; then
        echo "  Query filter: $QID_FILTER"
        CMD="$CMD --qid-filter $QID_FILTER"
    else
        echo "  Warning: Filter file not found: $QID_FILTER"
    fi
fi

CMD="$CMD --max-workers $WORKERS --output-dir results/response_gen/$EXPERIMENT"

case "$GENERATE_RESPONSE" in
    1|true|TRUE|yes|YES|--generate-response)
        echo "  Generate responses: true"
        CMD="$CMD --generate-response"
        ;;
esac

if [ -n "$METHODS" ]; then
    echo "  Methods: $METHODS"
    CMD="$CMD --methods $METHODS"
fi

# Run the command
eval $CMD

echo "✓ Experiments complete"
echo "  Output: results/response_gen/$EXPERIMENT/"
