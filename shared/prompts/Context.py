

CONTEXT_PROMPT = """
You are a clinical expert in wearable sensor measurements.
The goal is to generate a knowledge graph that connects multimodal wearable data (e.g., sleep metrics, activity levels, and self-reported affect)
This graph will serve as a key resource for a retrieval-augmented generation process in an LLM, supporting insight discovery, outcome prediction, and personalized intervention design.
Process:
step 1 Initial Node Creation: 
given a comprehensive list of health metrics commonly measurable by wearable devices, generate a node representation for each metric.

step 2 Relationship Mapping:
given all the nodes, determine the relationships between each pair of nodes and create edges between them.

step 3 New Metric Integration:
given a list of new wearable health metrics and all existing nodes. 
for each metric, you will need to check against existing graph nodes to identify potential duplicates,and merge if a match was found.

step 4 Graph Extension:
the remaining new metrics from step 3 will be added to the graph as new nodes and edges will be created to connect them to the existing nodes.
"""
