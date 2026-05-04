"""
Configuration settings for knowledge graph construction pipeline.

This module centralizes all configuration parameters for API clients,
file paths, and model settings.
"""

import os
from typing import Dict, Any
from dotenv import load_dotenv

# Load environment variables
load_dotenv()


class Config:
    """Main configuration class for the graph construction pipeline."""

    # API Configuration
    OPENAI_API_KEY: str = os.getenv('OPENAI_API_KEY', '')
    DEEPSEEK_API_KEY: str = os.getenv('DEEPSEEK_API_KEY', '')
    DEEPSEEK_API_KEY2: str = os.getenv('DEEPSEEK_API_KEY2', '')
    UMLS_API_KEY: str = os.getenv('UMLS_API_KEY', '')
    SERPER_API_KEY: str = os.getenv('SERPER_API_KEY', '')

    # API Endpoints
    DEEPSEEK_OFFICIAL_API: str = "https://api.deepseek.com"
    DEEPSEEK_EXTERNAL_API: str = "https://tbnx.plus7.plus/v1"

    # Model Selection
    USE_DEEPSEEK: bool = True
    DEEPSEEK_MODEL: str = "deepseek-chat"
    OPENAI_MODEL: str = "gpt-4o-mini"
    EMBEDDING_MODEL: str = "text-embedding-3-small"

    # Model Parameters
    TEMPERATURE: float = 0.1  # Low temperature for consistent outputs

    # Resource Paths
    RESOURCES_DIR: str = "./resources"
    VERIFIED_LINKS_PATH: str = "./resources/verified_links.json"
    INITIAL_METRICS_PATH: str = "./resources/initial_health_metrics_small.json"
    FEATURES_MAP_PATH: str = "./resources/features_map_dict_with_descriptionv3.json"

    # Batch Processing
    DEFAULT_BATCH_SIZE: int = 20
    EDGE_BATCH_SIZE: int = 10

    # Logging Configuration
    LOG_LEVEL: str = "INFO"
    LOG_FORMAT: str = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'

    # Web Search Filter Domains
    TRUSTED_DOMAIN_SUFFIXES: list = ['.edu', '.org', '.gov']

    @classmethod
    def get_chat_model(cls) -> str:
        """Get the appropriate chat model based on configuration."""
        return cls.DEEPSEEK_MODEL if cls.USE_DEEPSEEK else cls.OPENAI_MODEL

    @classmethod
    def get_api_base_url(cls) -> str:
        """Get the API base URL for the selected model."""
        if cls.USE_DEEPSEEK:
            return cls.DEEPSEEK_OFFICIAL_API
        return None  # OpenAI uses default

    @classmethod
    def get_api_key(cls) -> str:
        """Get the appropriate API key based on configuration."""
        if cls.USE_DEEPSEEK:
            return cls.DEEPSEEK_API_KEY or cls.DEEPSEEK_API_KEY2
        return cls.OPENAI_API_KEY

    @classmethod
    def validate(cls) -> bool:
        """Validate that required configuration is present."""
        required_keys = []

        if cls.USE_DEEPSEEK:
            required_keys.extend(['DEEPSEEK_API_KEY', 'DEEPSEEK_API_KEY2'])
        else:
            required_keys.append('OPENAI_API_KEY')

        missing = [key for key in required_keys if not getattr(cls, key)]

        if missing:
            raise ValueError(f"Missing required configuration: {', '.join(missing)}")

        return True


# Singleton configuration instance
config = Config()
