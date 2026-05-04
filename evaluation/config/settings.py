"""
Configuration settings for evaluation framework.
"""

import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()


class EvaluationConfig:
    """Configuration for evaluation framework."""

    # Supported chat providers
    CLIENT_DEEPSEEK: str = "deepseek"
    CLIENT_OPENAI: str = "openai"
    CLIENT_OPENROUTER: str = "openrouter"
    CLIENT_GEMINI: str = "gemini"
    SUPPORTED_CHAT_CLIENTS = {
        CLIENT_DEEPSEEK,
        CLIENT_OPENAI,
        CLIENT_OPENROUTER,
        CLIENT_GEMINI,
    }

    # API Configuration
    OPENAI_API_KEY: str = os.getenv('OPENAI_API_KEY', '')
    DEEPSEEK_API_KEY: str = os.getenv('DEEPSEEK_API_KEY', '')
    OPENROUTER_API_KEY: str = os.getenv('OPENROUTER_API_KEY', '')
    OPENROUTER_HTTP_REFERER: str = os.getenv('OPENROUTER_HTTP_REFERER', '')
    OPENROUTER_APP_TITLE: str = os.getenv('OPENROUTER_APP_TITLE', 'WAG Evaluation')
    GEMINI_API_KEY: str = os.getenv('GEMINI_API_KEY', '')

    # API Endpoints
    DEEPSEEK_API_URL: str = "https://api.deepseek.com"
    OPENAI_API_URL: str = "https://api.openai.com/v1"
    OPENROUTER_API_URL: str = "https://openrouter.ai/api/v1"

    # Model Selection
    DEFAULT_CHAT_CLIENT: str = CLIENT_DEEPSEEK

    # Model Names
    DEEPSEEK_MODEL: str = "deepseek-chat"
    OPENAI_MODEL: str = "gpt-4o"
    OPENROUTER_MODEL: str = os.getenv('OPENROUTER_MODEL', 'openai/gpt-4o-mini')
    GEMINI_MODEL: str = "gemini-2.5-pro"

    # Inference Configuration
    TEMPERATURE: float = 0.1
    MAX_INFERENCE_ATTEMPTS: int = int(os.getenv('LLM_MAX_INFERENCE_ATTEMPTS', '2'))
    MAX_GROUP_EVALUATION_ATTEMPTS: int = int(os.getenv('EVAL_GROUP_ATTEMPTS', '3'))
    DEFAULT_TIMEOUT_SECONDS: int = int(os.getenv('LLM_TIMEOUT_SECONDS', '180'))
    REQUEST_DELAY_SECONDS: float = float(os.getenv('LLM_REQUEST_DELAY_SECONDS', '0'))
    RETRY_DELAY_SECONDS: float = float(os.getenv('LLM_RETRY_DELAY_SECONDS', '0'))
    RATE_LIMIT_BACKOFF_SECONDS: float = float(os.getenv('LLM_RATE_LIMIT_BACKOFF_SECONDS', '60'))

    # Evaluation Settings
    DEFAULT_EVAL_PROMPT_TYPE: str = "rank"  # Options: "rank", "context"

    # Logging
    LOG_LEVEL: str = "INFO"
    LOG_FORMAT: str = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'

    @classmethod
    def get_chat_model(cls, client: str = None) -> str:
        """Get the appropriate chat model based on client."""
        client = client or cls.DEFAULT_CHAT_CLIENT

        if client == cls.CLIENT_DEEPSEEK:
            return cls.DEEPSEEK_MODEL
        elif client == cls.CLIENT_OPENAI:
            return cls.OPENAI_MODEL
        elif client == cls.CLIENT_OPENROUTER:
            return cls.OPENROUTER_MODEL
        elif client == cls.CLIENT_GEMINI:
            return cls.GEMINI_MODEL
        else:
            raise ValueError(f"Unknown chat client: {client}")

    @classmethod
    def get_api_key(cls, client: str = None) -> str:
        """Get the appropriate API key based on client."""
        client = client or cls.DEFAULT_CHAT_CLIENT

        if client == cls.CLIENT_DEEPSEEK:
            return cls.DEEPSEEK_API_KEY
        elif client == cls.CLIENT_OPENAI:
            return cls.OPENAI_API_KEY
        elif client == cls.CLIENT_OPENROUTER:
            return cls.OPENROUTER_API_KEY
        elif client == cls.CLIENT_GEMINI:
            return cls.GEMINI_API_KEY
        else:
            raise ValueError(f"Unknown chat client: {client}")

    @classmethod
    def get_api_url(cls, client: str = None) -> str:
        """Get API URL based on client."""
        client = client or cls.DEFAULT_CHAT_CLIENT
        if client == cls.CLIENT_DEEPSEEK:
            return cls.DEEPSEEK_API_URL
        if client == cls.CLIENT_OPENAI:
            return cls.OPENAI_API_URL
        if client == cls.CLIENT_OPENROUTER:
            return cls.OPENROUTER_API_URL
        return None

    @classmethod
    def get_openrouter_headers(cls) -> dict:
        """Get optional OpenRouter headers for attribution."""
        headers = {}
        if cls.OPENROUTER_HTTP_REFERER:
            headers["HTTP-Referer"] = cls.OPENROUTER_HTTP_REFERER
        if cls.OPENROUTER_APP_TITLE:
            headers["X-Title"] = cls.OPENROUTER_APP_TITLE
        return headers

    @classmethod
    def validate(cls, chat_client: str = None) -> bool:
        """Validate required configuration."""
        client = chat_client or cls.DEFAULT_CHAT_CLIENT
        api_key = cls.get_api_key(client)
        if not api_key:
            env_name = {
                cls.CLIENT_DEEPSEEK: "DEEPSEEK_API_KEY",
                cls.CLIENT_OPENAI: "OPENAI_API_KEY",
                cls.CLIENT_OPENROUTER: "OPENROUTER_API_KEY",
                cls.CLIENT_GEMINI: "GEMINI_API_KEY",
            }[client]
            raise ValueError(f"Missing {env_name}")
        return True


# Singleton configuration instance
config = EvaluationConfig()
