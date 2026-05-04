# Copyright (c) 2024 Microsoft Corporation.
# Licensed under the MIT License

"""Utils methods definition."""

from .dicts import dict_has_keys_with_types
from .hashing import gen_md5_hash
from .is_null import is_null
from .clean_str import clean_str
from .tokens import num_tokens_from_string, string_from_tokens
from .topological_sort import topological_sort
from .uuid import gen_uuid
from .parse import parse_llm_response, parse_search_results

try:
    from .load_graph import load_graph
except Exception:  # pragma: no cover - optional dependency guard
    load_graph = None

try:
    from .io import save_graph_to_json
except Exception:  # pragma: no cover - optional dependency guard
    save_graph_to_json = None

try:
    from .rag import cosine_similarity, get_embedding, search_docs
except Exception:  # pragma: no cover - optional dependency guard
    cosine_similarity = None
    get_embedding = None
    search_docs = None

try:
    from .umls import umls
except Exception:  # pragma: no cover - optional dependency guard
    umls = None

try:
    from .usage import usage_tracker, estimate_query_cost, count_tokens
except Exception:  # pragma: no cover - optional dependency guard
    usage_tracker = None
    estimate_query_cost = None
    count_tokens = None

__all__ = [
    "clean_str",
    "dict_has_keys_with_types",
    "gen_md5_hash",
    "gen_uuid",
    "is_null",
    "load_graph",
    "num_tokens_from_string",
    "string_from_tokens",
    "topological_sort",
    "save_graph_to_json",
    "umls",
    "usage_tracker",
    "estimate_query_cost",
    "count_tokens",
    "parse_llm_response",
    "parse_search_results",
    "cosine_similarity",
    "get_embedding",
    "search_docs"
]
