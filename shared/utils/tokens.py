# Copyright (c) 2024 Microsoft Corporation.
# Licensed under the MIT License

"""Utilities for working with tokens."""

import logging

import tiktoken

DEFAULT_ENCODING_NAME = "cl100k_base"

# Mapping for non-OpenAI models to a tiktoken-compatible encoding
_MODEL_ENCODING_MAP = {
    "deepseek-chat": "cl100k_base",
    "deepseek-reasoner": "cl100k_base",
}

log = logging.getLogger(__name__)


def _fallback_token_estimate(string: str) -> int:
    """
    Provide a rough token estimate when tokenizer assets are unavailable.

    This keeps offline flows working even when `tiktoken` cannot download or
    load model encodings.
    """
    return max(1, len(string) // 4) if string else 0


def num_tokens_from_string(
    string: str, model: str | None = None, encoding_name: str | None = None
) -> int:
    """Return the number of tokens in a text string."""
    if model is not None:
        # Check non-OpenAI model mapping first
        mapped_encoding = _MODEL_ENCODING_MAP.get(model)
        try:
            if mapped_encoding:
                encoding = tiktoken.get_encoding(mapped_encoding)
            else:
                encoding = tiktoken.encoding_for_model(model)
        except Exception:
            msg = (
                f"Failed to get encoding for {model} when getting "
                f"num_tokens_from_string. Fall back to default encoding "
                f"{DEFAULT_ENCODING_NAME}"
            )
            log.warning(msg)
            try:
                encoding = tiktoken.get_encoding(DEFAULT_ENCODING_NAME)
            except Exception:
                log.warning(
                    "Failed to load default encoding for num_tokens_from_string. "
                    "Using heuristic token estimate instead."
                )
                return _fallback_token_estimate(string)
    else:
        try:
            encoding = tiktoken.get_encoding(encoding_name or DEFAULT_ENCODING_NAME)
        except Exception:
            log.warning(
                "Failed to load requested encoding for num_tokens_from_string. "
                "Using heuristic token estimate instead."
            )
            return _fallback_token_estimate(string)
    return len(encoding.encode(string))


def string_from_tokens(
    tokens: list[int], model: str | None = None, encoding_name: str | None = None
) -> str:
    """Return a text string from a list of tokens."""
    if model is not None:
        mapped_encoding = _MODEL_ENCODING_MAP.get(model)
        try:
            if mapped_encoding:
                encoding = tiktoken.get_encoding(mapped_encoding)
            else:
                encoding = tiktoken.encoding_for_model(model)
        except Exception:
            msg = (
                f"Failed to get encoding for {model} when getting "
                f"string_from_tokens. Fall back to default encoding "
                f"{DEFAULT_ENCODING_NAME}"
            )
            log.warning(msg)
            try:
                encoding = tiktoken.get_encoding(DEFAULT_ENCODING_NAME)
            except Exception as exc:
                raise ValueError(
                    "No tokenizer encoding available for string_from_tokens."
                ) from exc
    elif encoding_name is not None:
        try:
            encoding = tiktoken.get_encoding(encoding_name)
        except Exception as exc:
            raise ValueError(
                f"No tokenizer encoding available for string_from_tokens: {encoding_name}"
            ) from exc
    else:
        msg = "Either model or encoding_name must be specified."
        raise ValueError(msg)
    return encoding.decode(tokens)
