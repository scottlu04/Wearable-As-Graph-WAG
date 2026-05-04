"""
Knowledge Graph Construction Pipeline

A research framework for constructing and managing knowledge graphs
from health metrics and wearable device data.
"""

__version__ = "1.0.0"
__author__ = "Research Team"

__all__ = ["NodeGenerator", "EdgeGenerator"]


def __getattr__(name):
    """Lazily import graph builders so optional web-search deps load only when used."""
    if name == "NodeGenerator":
        from .core.node_generator import NodeGenerator

        return NodeGenerator
    if name == "EdgeGenerator":
        from .core.edge_generator import EdgeGenerator

        return EdgeGenerator
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
