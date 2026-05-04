"""
Evaluation Prompt Registry

Maps prompt names to actual prompt templates from shared.prompts.Eval.
This module provides a convenient interface for selecting evaluation prompts.

Usage:
    >>> from evaluation.config.prompts import get_prompt, list_prompts
    >>>
    >>> # Get a specific prompt
    >>> prompt = get_prompt('single_rank_v3')
    >>>
    >>> # List all available prompts
    >>> prompts = list_prompts()
    >>> print(prompts)
    >>>
    >>> # Get prompt with fallback
    >>> prompt = get_prompt('custom_prompt', default='rank')

"""

from typing import Optional, Dict, List
import logging

from shared.prompts.Eval import (
    EVAL_PROMPT_RANK,
    EVAL_PROMPT_SINGLE_RANK,
)



logger = logging.getLogger(__name__)

# Prompt registry mapping names to actual prompts
PROMPT_REGISTRY: Dict[str, str] = {
    # Response evaluation prompts
    "rank": EVAL_PROMPT_RANK,
    "single_rank": EVAL_PROMPT_SINGLE_RANK,
  
}

# Default prompts by evaluation type
DEFAULT_RESPONSE_PROMPT = "rank"
DEFAULT_CONTEXT_PROMPT = "context"

# Prompt descriptions for help/documentation
PROMPT_DESCRIPTIONS: Dict[str, str] = {
    "rank": "Multi-dimensional ranking (insightfulness, relevance, groundedness, etc.)",
    "single_rank": "Single overall ranking without ties",
}


def get_prompt(prompt_name: str, default: Optional[str] = None) -> str:
    """
    Get evaluation prompt by name.

    Args:
        prompt_name: Name of the prompt (e.g., 'rank', 'single_rank_v3')
        default: Default prompt name to use if prompt_name not found

    Returns:
        Prompt string

    Raises:
        ValueError: If prompt_name not found and no default provided

    Example:
        >>> prompt = get_prompt('single_rank_v3')
        >>> prompt = get_prompt('custom', default='rank')
    """
    if prompt_name in PROMPT_REGISTRY:
        logger.debug(f"Using evaluation prompt: {prompt_name}")
        return PROMPT_REGISTRY[prompt_name]

    if default is not None:
        logger.warning(f"Prompt '{prompt_name}' not found, using default: {default}")
        if default in PROMPT_REGISTRY:
            return PROMPT_REGISTRY[default]
        raise ValueError(f"Default prompt '{default}' not found in registry")

    raise ValueError(
        f"Prompt '{prompt_name}' not found in registry. "
        f"Available prompts: {', '.join(PROMPT_REGISTRY.keys())}"
    )


def list_prompts(include_descriptions: bool = False) -> List[str]:
    """
    List all available prompt names.

    Args:
        include_descriptions: If True, return (name, description) tuples

    Returns:
        List of prompt names or (name, description) tuples

    Example:
        >>> prompts = list_prompts()
        >>> print(prompts)
        ['rank', 'single_rank', 'single_rank_v2', ...]

        >>> prompts = list_prompts(include_descriptions=True)
        >>> for name, desc in prompts:
        ...     print(f"{name}: {desc}")
    """
    if include_descriptions:
        return [
            (name, PROMPT_DESCRIPTIONS.get(name, "No description available"))
            for name in sorted(PROMPT_REGISTRY.keys())
        ]
    return sorted(PROMPT_REGISTRY.keys())


def get_prompt_info(prompt_name: str) -> Dict[str, str]:
    """
    Get detailed information about a specific prompt.

    Args:
        prompt_name: Name of the prompt

    Returns:
        Dictionary with prompt information

    Raises:
        ValueError: If prompt not found

    Example:
        >>> info = get_prompt_info('rank')
        >>> print(info['description'])
    """
    if prompt_name not in PROMPT_REGISTRY:
        raise ValueError(f"Prompt '{prompt_name}' not found")

    return {
        "name": prompt_name,
        "description": PROMPT_DESCRIPTIONS.get(prompt_name, "No description available"),
        "prompt_preview": PROMPT_REGISTRY[prompt_name][:200] + "...",
        "is_default_response": prompt_name == DEFAULT_RESPONSE_PROMPT,
        "is_default_context": prompt_name == DEFAULT_CONTEXT_PROMPT,
    }


def validate_prompt_name(prompt_name: str) -> bool:
    """
    Check if a prompt name is valid.

    Args:
        prompt_name: Name to validate

    Returns:
        True if valid, False otherwise

    Example:
        >>> is_valid = validate_prompt_name('rank')
        >>> print(is_valid)
        True
    """
    return prompt_name in PROMPT_REGISTRY


__all__ = [
    "get_prompt",
    "list_prompts",
    "get_prompt_info",
    "validate_prompt_name",
    "PROMPT_REGISTRY",
    "DEFAULT_RESPONSE_PROMPT",
    "DEFAULT_CONTEXT_PROMPT",
    "PROMPT_DESCRIPTIONS",
]
