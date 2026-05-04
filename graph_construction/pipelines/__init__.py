"""
Pipeline modules for knowledge graph construction workflows.
"""

__all__ = [
    "NodeConstructionPipeline",
    "EdgeConstructionPipeline",
    "DatasetIntegrationPipeline"
]


def __getattr__(name):
    """Lazily import graph-construction pipelines with optional web-search deps."""
    if name == "NodeConstructionPipeline":
        from .node_construction_pipeline import NodeConstructionPipeline

        return NodeConstructionPipeline
    if name == "EdgeConstructionPipeline":
        from .edge_construction_pipeline import EdgeConstructionPipeline

        return EdgeConstructionPipeline
    if name == "DatasetIntegrationPipeline":
        from .dataset_integration_pipeline import DatasetIntegrationPipeline

        return DatasetIntegrationPipeline
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
