"""
Core modules for knowledge graph construction.
"""

__all__ = [
    "NodeGenerator",
    "EdgeGenerator",
    "generate_node",
    "generate_edge",
    "process_edges_in_batches"
]


def __getattr__(name):
    """Lazily import graph-construction components with optional web-search deps."""
    if name in {"NodeGenerator", "generate_node"}:
        from .node_generator import NodeGenerator, generate_node

        return {"NodeGenerator": NodeGenerator, "generate_node": generate_node}[name]
    if name in {"EdgeGenerator", "generate_edge", "process_edges_in_batches"}:
        from .edge_generator import EdgeGenerator, generate_edge, process_edges_in_batches

        return {
            "EdgeGenerator": EdgeGenerator,
            "generate_edge": generate_edge,
            "process_edges_in_batches": process_edges_in_batches,
        }[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
