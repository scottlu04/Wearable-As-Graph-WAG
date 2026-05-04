# WAG Scripts

Run these commands from the repository root after activating your environment:

```bash
source .venv/bin/activate
```

Build or update the knowledge graph:

```bash
./scripts/run_graph_construction.sh
```

Generate the query set:

```bash
./scripts/run_query_generation.sh
```

Run response-generation experiments:

```bash
./scripts/run_experiments.sh general 10
./scripts/run_experiments.sh global 10
./scripts/run_experiments.sh local 10
```

Generate final LLM answers as well as contexts:

```bash
./scripts/run_experiments.sh general 4 "" true
```

Run selected methods only:

```bash
./scripts/run_experiments.sh general 4 "" true Base,Rag,StaticGraph,FullContext,Wag
```

Evaluate results:

```bash
./scripts/run_evaluation.sh general
./scripts/run_evaluation.sh global
./scripts/run_evaluation.sh local
```

Generated files are written to ignored directories such as `results/`, `logs/`,
and `filters/`.
