"""
Evaluation Framework

A professional framework for evaluating retrieval-augmented generation (RAG)
and knowledge graph-based systems using LLM-as-a-judge approach.
"""

__version__ = "2.0.0"
__author__ = "WAG Research Team"

from .core.evaluator import (
    BaseEvaluator,
    ContextEvaluator,
    ResponseEvaluator,
    # Legacy aliases for backward compatibility
    Evaluater,
    Evaluater_Context,
    Evaluater_Response
)

__all__ = [
    "BaseEvaluator",
    "ContextEvaluator",
    "ResponseEvaluator",
    # Legacy names
    "Evaluater",
    "Evaluater_Context",
    "Evaluater_Response"
]
