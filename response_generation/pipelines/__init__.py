"""
Response Generation Pipelines

This module contains experimental pipelines for comparing query processing methods.

All pipelines inherit from BasePipeline which provides common functionality for:
- Data loading (QA datasets, health data, relationship matrices)
- Query processing loop
- Result storage and serialization
- Error handling and logging

Available Pipelines:
    - GeneralComparisonPipeline: Base vs RAG vs StaticGraph vs FullContext vs WAG
    - GlobalWeightPipeline: Compare weight sources (prior, pop, ind, global)
    - LocalWeightPipeline: Compare local vs global vs final weights

Usage:
    >>> from response_generation.pipelines import GeneralComparisonPipeline
    >>> pipeline = GeneralComparisonPipeline(root_dir="resources")
    >>> results = pipeline.run()
"""

from .base_pipeline import BasePipeline
from .general_comparison_pipeline import GeneralComparisonPipeline, main as run_general
from .global_weight_pipeline import GlobalWeightPipeline, main as run_global
from .local_weight_pipeline import LocalWeightPipeline, main as run_local

__all__ = [
    'BasePipeline',
    'GeneralComparisonPipeline',
    'GlobalWeightPipeline',
    'LocalWeightPipeline',
    'run_general',
    'run_global',
    'run_local',
]
