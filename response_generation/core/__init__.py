"""
Response Generation Core Processors

This module contains the core query processing implementations:
- BaseQueryProcessor: Simple entity matching
- RAGProcessor: Knowledge graph retrieval with entity context
- StaticGraphProcessor: Static edge-weight graph baseline
- FullContextProcessor: All-metric baseline using the query time window
- FullHistoryContextProcessor: All-metric baseline using full history
- WAGProcessor: Weight-augmented graph traversal

Configuration is now in response_generation.config module.
"""

from .base_processor import BaseQueryProcessor
from .full_context_processor import (
    FullContextProcessor,
    FullHistoryContextProcessor,
)
from .rag_processor import RAGProcessor
from .static_graph_processor import StaticGraphProcessor
from .wag_processor import WAGProcessor

__all__ = [
    'BaseQueryProcessor',
    'FullContextProcessor',
    'FullHistoryContextProcessor',
    'RAGProcessor',
    'StaticGraphProcessor',
    'WAGProcessor',
]
