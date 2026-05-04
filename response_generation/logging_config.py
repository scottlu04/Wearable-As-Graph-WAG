"""
Logging Configuration for Response Generation

This module provides centralized logging configuration for the response generation
package. It sets up both file and console logging with appropriate formatting.

Usage:
    from response_generation.logging_config import setup_logging

    # At the start of your script
    setup_logging(level=logging.INFO)

    # Then use loggers in your modules
    logger = logging.getLogger(__name__)
    logger.info("Processing started")
"""

import logging
import os
from typing import Optional
from pathlib import Path

NOISY_LOGGER_LEVELS = {
    "httpx": logging.WARNING,
    "httpcore": logging.WARNING,
    "openai": logging.WARNING,
    "urllib3": logging.WARNING,
}


def setup_logging(
    level: int = logging.INFO,
    log_file: Optional[str] = None,
    log_dir: Optional[str] = None,
    format_string: Optional[str] = None
) -> None:
    """
    Set up logging configuration for response generation package.

    Args:
        level: Logging level (e.g., logging.DEBUG, logging.INFO)
        log_file: Path to log file. If None, uses 'response_generation.log'
        log_dir: Directory for log files. If None, uses current directory
        format_string: Custom format string. If None, uses default format

    Example:
        >>> setup_logging(level=logging.DEBUG, log_dir='logs')
        >>> logger = logging.getLogger(__name__)
        >>> logger.debug("Debug message")
    """
    # Set default log file location
    if log_dir:
        log_dir_path = Path(log_dir)
        log_dir_path.mkdir(parents=True, exist_ok=True)
        log_file_path = log_dir_path / (log_file or 'response_generation.log')
    else:
        log_file_path = log_file or 'response_generation.log'

    # Default format
    if format_string is None:
        format_string = (
            '%(asctime)s - %(name)s - %(levelname)s - '
            '%(filename)s:%(lineno)d - %(message)s'
        )

    # Create formatter
    formatter = logging.Formatter(format_string, datefmt='%Y-%m-%d %H:%M:%S')

    # Configure root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(level)

    # Remove existing handlers
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)

    # Add file handler
    file_handler = logging.FileHandler(log_file_path, mode='a')
    file_handler.setLevel(level)
    file_handler.setFormatter(formatter)
    root_logger.addHandler(file_handler)

    # Add console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(level)
    console_handler.setFormatter(formatter)
    root_logger.addHandler(console_handler)

    # Keep third-party transport libraries from spamming one line per request.
    for logger_name, logger_level in NOISY_LOGGER_LEVELS.items():
        noisy_logger = logging.getLogger(logger_name)
        noisy_logger.setLevel(logger_level)
        noisy_logger.propagate = True

    # Log initialization
    logger = logging.getLogger(__name__)
    logger.info("=" * 70)
    logger.info("Logging initialized")
    logger.info(f"Log level: {logging.getLevelName(level)}")
    logger.info(f"Log file: {log_file_path}")
    logger.info("=" * 70)


def get_logger(name: str, level: Optional[int] = None) -> logging.Logger:
    """
    Get a logger with the specified name.

    Args:
        name: Name for the logger (typically __name__)
        level: Optional logging level override

    Returns:
        Logger instance

    Example:
        >>> logger = get_logger(__name__)
        >>> logger.info("Processing started")
    """
    logger = logging.getLogger(name)
    if level is not None:
        logger.setLevel(level)
    return logger


def log_experiment_info(
    logger: logging.Logger,
    experiment_name: str,
    config: dict
) -> None:
    """
    Log experiment configuration information.

    Args:
        logger: Logger instance
        experiment_name: Name of the experiment
        config: Configuration dictionary

    Example:
        >>> logger = get_logger(__name__)
        >>> log_experiment_info(logger, "Global Weight", hyperparameters)
    """
    logger.info("=" * 70)
    logger.info(f"Experiment: {experiment_name}")
    logger.info("=" * 70)
    logger.info("Configuration:")
    for key, value in config.items():
        logger.info(f"  {key}: {value}")
    logger.info("=" * 70)


# Convenience function for pipelines
def setup_pipeline_logging(
    experiment_name: str,
    log_dir: str = "logs",
    level: int = logging.INFO
) -> logging.Logger:
    """
    Set up logging for a pipeline experiment.

    Args:
        experiment_name: Name of the experiment
        log_dir: Directory for log files
        level: Logging level

    Returns:
        Logger instance for the pipeline

    Example:
        >>> logger = setup_pipeline_logging("global_weight_comparison")
        >>> logger.info("Pipeline started")
    """
    # Create experiment-specific log file
    log_file = f"{experiment_name.lower().replace(' ', '_')}.log"

    # Set up logging
    setup_logging(level=level, log_file=log_file, log_dir=log_dir)

    # Return logger for the pipeline
    return get_logger(experiment_name)
