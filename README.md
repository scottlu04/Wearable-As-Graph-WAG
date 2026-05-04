# WAG: Wearable As Graph

WAG is a research framework for query-conditioned graph retrieval over
longitudinal wearable-health data. It builds a personalized wearable knowledge
graph, retrieves a query-conditioned subgraph, and uses that context to support
LLM reasoning over personal time-series signals.

## What Is Included

- Core graph construction, query generation, response generation, and evaluation code.
- Response-generation processors for `Base`, `RAG`, `StaticGraph`, `FullContext`,
  `WAG`, and the optional `FullHistory` baseline.
- Reproducibility resources under `resources/`, including the wearable knowledge
  graph, generated query set, processed data tables, and relationship dictionary.
- Shell scripts for running the main graph/query/response/evaluation pipeline.

## What Is Not Included

This open-source copy intentionally excludes generated experiment outputs,
runtime logs, local notebooks, virtual environments, API keys, and internal
diagnostic experiments such as cross-model robustness sweeps and timing-only
paper-table scripts.

## Setup

```bash
git clone <your-repo-url>
cd WAG

python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

cp .env.example .env
# Edit .env and add the API keys for the providers you plan to use.
```

Quick environment check:

```bash
python -c "import dotenv, scipy, sklearn, networkx, openai; print('env ok')"
```

## Repository Structure

```text
WAG/
├── graph_construction/      # Knowledge graph construction and extension
├── query_generation/        # Data-grounded wearable query generation
├── response_generation/     # Base/RAG/StaticGraph/FullContext/WAG processors
├── evaluation/              # LLM-as-a-judge evaluation pipeline
├── shared/                  # Shared models, prompts, utilities, retrieval package
├── scripts/                 # Main shell entrypoints and helper scripts
├── resources/               # KG, processed datasets, generated queries
├── run_experiments.py       # Unified response-generation experiment runner
└── requirements.txt
```

## Run The Pipeline

Build or update the wearable knowledge graph:

```bash
./scripts/run_graph_construction.sh
```

Generate data-grounded queries:

```bash
./scripts/run_query_generation.sh
```

Run the general response-generation experiment:

```bash
./scripts/run_experiments.sh general 10
```

Evaluate generated responses:

```bash
./scripts/run_evaluation.sh general
```

Run everything in sequence:

```bash
./scripts/run_all.sh
```

## Response-Generation Methods

- **Base**: uses only the query-associated personal wearable data.
- **RAG**: retrieves graph context directly related to the primary detected metric.
- **StaticGraph**: ranks related nodes using static prior graph edge weights.
- **FullContext**: provides all available wearable metrics within the query time window.
- **WAG**: retrieves related nodes using query-conditioned global/local graph weighting.

Optional methods can be selected explicitly with `--methods`, for example:

```bash
python run_experiments.py \
  --experiments general \
  --generate-response \
  --methods Base Rag StaticGraph FullContext Wag
```

## Data And Resources

The `resources/` directory contains the default assets expected by the scripts:

- `resources/kg/`: graph nodes and edges.
- `resources/query_set/`: generated wearable-health queries.
- `resources/processed_dataset/`: processed dataset tables.
- `resources/relationship_dict.json`: precomputed relationship statistics.

If you redistribute this repository publicly, please verify that the licenses and
terms of the underlying datasets are compatible with your release plan.

## Outputs

Generated outputs are written under ignored directories such as:

- `results/response_gen/`
- `results/evaluation/`
- `logs/`

These are intentionally not part of the clean open-source copy.

## Tests

```bash
python -m unittest discover
```

## License

MIT License. See `LICENSE`.
