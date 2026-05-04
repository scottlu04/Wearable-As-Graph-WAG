"""
Core evaluation functionality.
"""

from .evaluator import (
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
