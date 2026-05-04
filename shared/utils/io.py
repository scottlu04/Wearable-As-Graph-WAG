from __future__ import annotations

import json
import networkx as nx
from typing import Dict, Any

def save_graph_to_json(graph: "nx.Graph", filepath: str):
    """
    Save a NetworkX graph to a JSON file
    
    Args:
        graph: NetworkX graph
        filepath: Path to save JSON file
    """
    # Convert graph to dictionary format
    graph_data = {
        "nodes": [],
        "edges": []
    }
    
    # Add nodes and their attributes
    for node, attrs in graph.nodes(data=True):
        node_data = {
            "id": str(node),  # Convert node ID to string for JSON compatibility
            "attributes": attrs
        }
        graph_data["nodes"].append(node_data)
    
    # Add edges and their attributes
    for source, target, attrs in graph.edges(data=True):
        edge_data = {
            "source": str(source),
            "target": str(target),
            "attributes": attrs
        }
        graph_data["edges"].append(edge_data)
    
    # Save to JSON file
    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(graph_data, f, indent=2, ensure_ascii=False)

def load_graph_from_json(filepath: str) -> "nx.Graph":
    """
    Load a NetworkX graph from a JSON file
    
    Args:
        filepath: Path to JSON file
    
    Returns:
        NetworkX graph
    """
    # Load JSON data
    with open(filepath, 'r', encoding='utf-8') as f:
        graph_data = json.load(f)
    
    # Create new graph
    G = nx.Graph()
    
    # Add nodes with attributes
    for node_data in graph_data["nodes"]:
        G.add_node(node_data["id"], **node_data["attributes"])
    
    # Add edges with attributes
    for edge_data in graph_data["edges"]:
        G.add_edge(edge_data["source"], 
                  edge_data["target"], 
                  **edge_data["attributes"])
    
    return G

# Example usage:
"""
# Save graph
save_graph_to_json(graph, filepath='knowledge_graph.json')

# Load graph
loaded_graph = load_graph_from_json(filepath='knowledge_graph.json')
"""
