"""
Configuration Settings for WAG Response Generation

This module provides unified configuration for query processing, including:
- API configuration (OpenAI, DeepSeek, OpenRouter)
- LLM model settings
- Retrieval parameters
- Weight computation defaults
- Context building settings
- File paths and constants
"""

import os
from typing import Dict, Any, Optional

# =============================================================================
# API Configuration
# =============================================================================

# API Base URLs
DEEPSEEK_API_URL = "https://api.deepseek.com"
OPENAI_API_URL = "https://api.openai.com/v1"
OPENROUTER_API_URL = "https://openrouter.ai/api/v1"

# Supported Client Types
CLIENT_DEEPSEEK = "deepseek"
CLIENT_OPENAI = "openai"
CLIENT_OPENROUTER = "openrouter"
CLIENT_GEMINI = "gemini"

SUPPORTED_CHAT_CLIENTS = {CLIENT_DEEPSEEK, CLIENT_OPENAI, CLIENT_OPENROUTER, CLIENT_GEMINI}
SUPPORTED_EMBED_CLIENTS = {CLIENT_OPENAI}

# API Keys (from environment)
OPENAI_API_KEY: str = os.getenv('OPENAI_API_KEY', '')
DEEPSEEK_API_KEY: str = os.getenv('DEEPSEEK_API_KEY', '')
OPENROUTER_API_KEY: str = os.getenv('OPENROUTER_API_KEY', '')
OPENROUTER_HTTP_REFERER: str = os.getenv('OPENROUTER_HTTP_REFERER', '')
OPENROUTER_APP_TITLE: str = os.getenv('OPENROUTER_APP_TITLE', 'WAG Response Generation')
GEMINI_API_KEY: str = os.getenv('GEMINI_API_KEY', '')
USE_DEEPSEEK: bool = os.getenv('USE_DEEPSEEK', 'false').lower() == 'true'

# =============================================================================
# Model Names
# =============================================================================

# Chat Models
MODEL_DEEPSEEK_CHAT = "deepseek-chat"
MODEL_DEEPSEEK_REASONER = "deepseek-reasoner"
MODEL_GPT4O = "gpt-4o"
MODEL_GPT4O_MINI = "gpt-4o-mini"
MODEL_OPENROUTER_DEFAULT = os.getenv('OPENROUTER_MODEL', 'openai/gpt-4o-mini')
MODEL_GEMINI_DEFAULT = os.getenv('GEMINI_MODEL', 'gemini-2.5-pro')

# Default Models per Client
DEFAULT_CHAT_MODELS = {
    CLIENT_DEEPSEEK: MODEL_DEEPSEEK_CHAT,
    CLIENT_OPENAI: MODEL_GPT4O,
    CLIENT_OPENROUTER: MODEL_OPENROUTER_DEFAULT,
    CLIENT_GEMINI: MODEL_GEMINI_DEFAULT,
}

DEFAULT_EMBED_MODELS = {
    CLIENT_OPENAI: MODEL_GPT4O_MINI,
}

# =============================================================================
# LLM Configuration
# =============================================================================

# Retry Configuration
DEFAULT_LLM_ATTEMPTS = 2
DEFAULT_TIMEOUT_SECONDS = int(os.getenv('LLM_TIMEOUT_SECONDS', '180'))

# Temperature and Sampling
DEFAULT_TEMPERATURE = 0.1
DEFAULT_MAX_TOKENS = 4096

# =============================================================================
# Graph File Paths (relative to root_dir)
# =============================================================================

NODES_FILE = "nodes_with_embeddings.json"
EDGES_FILE = "edges.json"
PRIOR_MATRIX_FILE = "prior_matrix.json"

# =============================================================================
# Retrieval Configuration
# =============================================================================

TOP_K_ENTITIES = 5  # Number of entities to retrieve
TOP_K_RELATED = 10  # Number of related nodes to retrieve
SIMILARITY_THRESHOLD = 0.7  # Minimum similarity for entity matching

# =============================================================================
# Context Building
# =============================================================================

MAX_CONTEXT_LENGTH = 160000  # Maximum estimated prompt tokens before full-context is infeasible
INCLUDE_DEFINITIONS = True
INCLUDE_RELATIONSHIPS = True
INCLUDE_DATA_SUMMARY = True

# Maximum lengths for context strings
MAX_NODE_DESCRIPTION_LENGTH = 500
MAX_RELATIONSHIP_DESCRIPTION_LENGTH = 300
MAX_DATA_POINTS_PER_METRIC = 100

# =============================================================================
# Weight Retrieval (for WAG)
# =============================================================================

USE_WEIGHT_RETRIEVAL = True
WEIGHT_THRESHOLD = 0.3  # Minimum weight to include related node
MAX_RELATED_NODES = 15

# =============================================================================
# Default Processor Parameters
# =============================================================================

DEFAULT_PROCESSOR_PARAMS: Dict[str, Any] = {
    'max_num_related_nodes': 5,
    'time_granularity': 7,
    'openness': 0.5,
    'weight_sort_by': 'weight_final',
    'rel_type': 'spearman',
    't_global': 0.9,
    't_local': 0.7,
    'alpha_prior': 1.0,
    'alpha_pop': 10**0.94,
    'alpha_ind': 10**-1.31,
    'beta': 0.5,
    'demog_metrics': [],
}

# =============================================================================
# Function Schemas for LLM Tools
# =============================================================================

# Query Parsing Function Schema
QUERY_PARSE_FUNCTION_SCHEMA: Dict[str, Any] = {
    "type": "function",
    "function": {
        "name": "query_parse",
        "description": "Extract important entities from the provided query.",
        "parameters": {
            "type": "object",
            "properties": {
                "metrics": {
                    "type": "array",
                    "description": "List of key metrics found in the query",
                    "items": {"type": "string"}
                },
                "time_range": {
                    "type": "string",
                    "description": "number of days till the start date, [1, 7, 30,..., all time]",
                },
                "openness_score": {
                    "type": "number",
                    "description": "Openness score of the question",
                }
            },
            "required": ["entities", "time_range", "openness_score"],
            "additionalProperties": False
        },
        "strict": True
    }
}

# =============================================================================
# Data Processing
# =============================================================================

DATE_FORMAT = '%Y-%m-%d'

TIME_GRANULARITY_MAP = {
    1: 'day',
    7: 'week',
    14: 'two weeks',
    30: 'month',
    60: 'two months',
    'all': 'all available time'
}

# =============================================================================
# Validation Limits
# =============================================================================

MIN_QUERY_LENGTH = 3
MAX_QUERY_LENGTH = 1000
MIN_TIME_GRANULARITY = 1
MAX_TIME_GRANULARITY = 365
MIN_OPENNESS_SCORE = 0.0
MAX_OPENNESS_SCORE = 1.0

# =============================================================================
# Error Messages
# =============================================================================

ERROR_INVALID_CHAT_CLIENT = "Invalid chat client. Supported clients: {}"
ERROR_INVALID_EMBED_CLIENT = "Invalid embed client. Supported clients: {}"
ERROR_GRAPH_FILE_NOT_FOUND = "Graph file not found: {}"
ERROR_INVALID_JSON = "Invalid JSON in file: {}"
ERROR_LLM_INFERENCE_FAILED = "LLM inference failed after {} attempts"
ERROR_QUERY_TOO_SHORT = "Query too short (minimum {} characters)"
ERROR_QUERY_TOO_LONG = "Query too long (maximum {} characters)"
ERROR_NO_API_KEY = (
    "API key not found. Set OPENAI_API_KEY, DEEPSEEK_API_KEY, or OPENROUTER_API_KEY."
)

# =============================================================================
# Success Messages
# =============================================================================

SUCCESS_GRAPH_LOADED = "Successfully loaded {} nodes and {} edges"
SUCCESS_CLIENT_INITIALIZED = "Initialized {} client with model {}"
SUCCESS_QUERY_PROCESSED = "Query processed successfully in {:.2f}s"

# =============================================================================
# Configuration Class (for organized access)
# =============================================================================

class Config:
    """Unified configuration class for WAG response generation."""

    # API Configuration
    DEEPSEEK_API_URL = DEEPSEEK_API_URL
    OPENAI_API_URL = OPENAI_API_URL
    OPENROUTER_API_URL = OPENROUTER_API_URL
    CLIENT_DEEPSEEK = CLIENT_DEEPSEEK
    CLIENT_OPENAI = CLIENT_OPENAI
    CLIENT_OPENROUTER = CLIENT_OPENROUTER
    CLIENT_GEMINI = CLIENT_GEMINI
    SUPPORTED_CHAT_CLIENTS = SUPPORTED_CHAT_CLIENTS
    SUPPORTED_EMBED_CLIENTS = SUPPORTED_EMBED_CLIENTS

    OPENAI_API_KEY = OPENAI_API_KEY
    DEEPSEEK_API_KEY = DEEPSEEK_API_KEY
    OPENROUTER_API_KEY = OPENROUTER_API_KEY
    OPENROUTER_HTTP_REFERER = OPENROUTER_HTTP_REFERER
    OPENROUTER_APP_TITLE = OPENROUTER_APP_TITLE
    GEMINI_API_KEY = GEMINI_API_KEY
    USE_DEEPSEEK = USE_DEEPSEEK

    # Model Configuration
    MODEL_DEEPSEEK_CHAT = MODEL_DEEPSEEK_CHAT
    MODEL_GPT4O = MODEL_GPT4O
    MODEL_GPT4O_MINI = MODEL_GPT4O_MINI
    MODEL_OPENROUTER_DEFAULT = MODEL_OPENROUTER_DEFAULT
    MODEL_GEMINI_DEFAULT = MODEL_GEMINI_DEFAULT
    DEFAULT_CHAT_MODELS = DEFAULT_CHAT_MODELS
    DEFAULT_EMBED_MODELS = DEFAULT_EMBED_MODELS

    # LLM Settings
    DEFAULT_LLM_ATTEMPTS = DEFAULT_LLM_ATTEMPTS
    DEFAULT_TEMPERATURE = DEFAULT_TEMPERATURE
    DEFAULT_MAX_TOKENS = DEFAULT_MAX_TOKENS

    # File Paths
    NODES_FILE = NODES_FILE
    EDGES_FILE = EDGES_FILE
    PRIOR_MATRIX_FILE = PRIOR_MATRIX_FILE

    # Retrieval
    TOP_K_ENTITIES = TOP_K_ENTITIES
    TOP_K_RELATED = TOP_K_RELATED
    SIMILARITY_THRESHOLD = SIMILARITY_THRESHOLD

    # Context Building
    MAX_CONTEXT_LENGTH = MAX_CONTEXT_LENGTH
    INCLUDE_DEFINITIONS = INCLUDE_DEFINITIONS
    INCLUDE_RELATIONSHIPS = INCLUDE_RELATIONSHIPS
    INCLUDE_DATA_SUMMARY = INCLUDE_DATA_SUMMARY

    # Weight Retrieval
    USE_WEIGHT_RETRIEVAL = USE_WEIGHT_RETRIEVAL
    WEIGHT_THRESHOLD = WEIGHT_THRESHOLD
    MAX_RELATED_NODES = MAX_RELATED_NODES

    # Processor Parameters
    DEFAULT_PROCESSOR_PARAMS = DEFAULT_PROCESSOR_PARAMS

    # Function Schemas
    QUERY_PARSE_FUNCTION_SCHEMA = QUERY_PARSE_FUNCTION_SCHEMA

    # Data Processing
    DATE_FORMAT = DATE_FORMAT
    TIME_GRANULARITY_MAP = TIME_GRANULARITY_MAP

    @classmethod
    def get_llm_model(cls) -> str:
        """Get the appropriate LLM model based on configuration."""
        return cls.MODEL_DEEPSEEK_CHAT if cls.USE_DEEPSEEK else cls.MODEL_GPT4O

    @classmethod
    def get_api_key(cls) -> str:
        """Get the appropriate API key based on configuration."""
        return cls.DEEPSEEK_API_KEY if cls.USE_DEEPSEEK else cls.OPENAI_API_KEY

    @classmethod
    def get_api_key_for_client(cls, client: str) -> str:
        """Get the API key for a specific chat provider."""
        if client == cls.CLIENT_DEEPSEEK:
            return cls.DEEPSEEK_API_KEY
        if client == cls.CLIENT_OPENROUTER:
            return cls.OPENROUTER_API_KEY
        if client == cls.CLIENT_GEMINI:
            return cls.GEMINI_API_KEY
        return cls.OPENAI_API_KEY

    @classmethod
    def validate_config(cls, chat_client: Optional[str] = None) -> None:
        """Validate configuration settings."""
        api_key = cls.get_api_key_for_client(chat_client) if chat_client else cls.get_api_key()
        if not api_key:
            raise ValueError(ERROR_NO_API_KEY)

        if cls.TOP_K_ENTITIES < 1:
            raise ValueError("TOP_K_ENTITIES must be at least 1")

        if not 0 <= cls.SIMILARITY_THRESHOLD <= 1:
            raise ValueError("SIMILARITY_THRESHOLD must be in [0, 1]")

        if cls.WEIGHT_THRESHOLD < 0:
            raise ValueError("WEIGHT_THRESHOLD must be non-negative")


# Create default config instance for backward compatibility
config = Config()
