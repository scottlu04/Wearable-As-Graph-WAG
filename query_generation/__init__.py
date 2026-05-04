"""
Query Generation Pipeline

A professional framework for generating diverse, clinically relevant questions
from health knowledge graphs and wearable device data.
"""

__version__ = "1.0.0"
__author__ = "Research Team"

from .core.query_generator import QueryGenerator
from .pipelines.query_generation_pipeline import QueryGenerationPipeline

__all__ = ["QueryGenerator", "QueryGenerationPipeline"]
