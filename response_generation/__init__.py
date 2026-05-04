"""
Response Generation Package

This package implements response generation pipelines for comparing different
query processing methods (Base, RAG, StaticGraph, FullContext, WAG)
on health data.

Main Components:
- core/: Query processors (BaseQueryProcessor, RAGProcessor, StaticGraphProcessor,
  FullContextProcessor, WAGProcessor)
- pipelines/: Experiment pipelines for systematic comparison
- config/: Configuration and hyperparameter management

Example Usage:
    from response_generation.pipelines import GeneralComparisonPipeline

    # Initialize pipeline
    pipeline = GeneralComparisonPipeline(
        data_dir="data/kg",
        output_dir="results/general"
    )

    # Run comparison experiment
    results = pipeline.run()
"""

from .core import (
    BaseQueryProcessor,
    FullContextProcessor,
    FullHistoryContextProcessor,
    RAGProcessor,
    StaticGraphProcessor,
    WAGProcessor,
)
from .config import Config, config
from .logging_config import setup_logging, get_logger, setup_pipeline_logging

__version__ = '1.0.0'
__author__ = 'WAG Research Team'

__all__ = [
    'BaseQueryProcessor',
    'FullContextProcessor',
    'FullHistoryContextProcessor',
    'RAGProcessor',
    'StaticGraphProcessor',
    'WAGProcessor',
    'Config',
    'config',
    'setup_logging',
    'get_logger',
    'setup_pipeline_logging',
]
