"""
Configuration settings for query generation pipeline.
"""

import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()


class Config:
    """Configuration for query generation pipeline."""

    # API Configuration
    OPENAI_API_KEY: str = os.getenv('OPENAI_API_KEY', '')
    DEEPSEEK_API_KEY: str = os.getenv('DEEPSEEK_API_KEY', '')

    # API Endpoints
    DEEPSEEK_API_URL: str = "https://api.deepseek.com"

    # Model Selection
    USE_DEEPSEEK: bool = True
    DEEPSEEK_MODEL: str = "deepseek-chat"
    OPENAI_MODEL: str = "gpt-4o-mini"
    TEMPERATURE: float = 0.1

    # Batch Processing
    DEFAULT_BATCH_SIZE: int = 10
    MAX_WORKERS: int = 5  # For parallel processing

    # Question Generation
    OPENNESS_DISTRIBUTION: dict = {
        'objective': 0.4,    # openness ≤0.4
        'moderate': 0.3,     # 0.4 < openness < 0.7
        'open_ended': 0.3    # openness ≥0.7
    }

    # Logging
    LOG_LEVEL: str = "INFO"
    LOG_FORMAT: str = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'

    @classmethod
    def get_chat_model(cls) -> str:
        """Get the appropriate chat model."""
        return cls.DEEPSEEK_MODEL if cls.USE_DEEPSEEK else cls.OPENAI_MODEL

    @classmethod
    def get_api_key(cls) -> str:
        """Get the appropriate API key."""
        return cls.DEEPSEEK_API_KEY if cls.USE_DEEPSEEK else cls.OPENAI_API_KEY

    @classmethod
    def get_api_url(cls) -> str:
        """Get API URL."""
        return cls.DEEPSEEK_API_URL if cls.USE_DEEPSEEK else None

    @classmethod
    def validate(cls) -> bool:
        """Validate required configuration."""
        if cls.USE_DEEPSEEK and not cls.DEEPSEEK_API_KEY:
            raise ValueError("Missing DEEPSEEK_API_KEY")
        if not cls.USE_DEEPSEEK and not cls.OPENAI_API_KEY:
            raise ValueError("Missing OPENAI_API_KEY")
        return True


# Singleton configuration instance
config = Config()
