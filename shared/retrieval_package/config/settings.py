"""
Configuration settings for retrieval package.

This module provides centralized configuration for statistical retrieval,
Bayesian posterior computation, and anomaly detection parameters.
"""

import os
from typing import Dict, Optional


class RetrievalConfig:
    """Configuration for statistical retrieval and weight computation."""

    # Hyperparameters for weight computation
    DEFAULT_HYPERPARAMETERS = {
        # Relationship type: 'spearman', 'pearson', or 'mi'
        'rel_type': 'spearman',

        # Minimum samples needed for valid correlation
        'min_samples_needed': 4,

        # Only consider metrics with full relationship data (llm prior, pop, ind)
        'full_rel_only': False,

        # Temperature parameters for sigmoid transformations
        't_global': 1.0,  # Global relationship temperature
        't_local': 0.7,   # Local abnormality temperature

        # Weighting parameter for combining global and local signals
        'beta': 0.5,  # weight_final = beta * weight_local + (1-beta) * weight_global

        # Variance scaling factors for Bayesian updates
        'alpha_prior': 1.0,  # LLM prior variance scaling
        'alpha_pop': 1.0,    # Population variance scaling
        'alpha_ind': 1.0,    # Individual variance scaling
    }

    # Statistical computation parameters
    MUTUAL_INFO_BOOTSTRAP_SAMPLES = 50
    MUTUAL_INFO_BIN_METHOD = 'fd'  # 'fd' (Freedman-Diaconis), 'sqrt', or int
    MUTUAL_INFO_BIN_RANGE = (5, 50)  # Min and max bins

    # Anomaly detection parameters
    ANOMALY_ZSCORE_THRESHOLD = 2.0

    # Random seed for reproducibility
    RANDOM_SEED = 42

    @classmethod
    def get_hyperparameters(
        cls,
        rel_type: str = 'spearman',
        min_samples: int = 4,
        full_rel_only: bool = False,
        beta: float = 0.5,
        t_global: float = 1.0,
        t_local: float = 0.7,
        **kwargs
    ) -> Dict:
        """
        Get hyperparameters for retrieval with custom overrides.

        Args:
            rel_type: Relationship type ('spearman', 'pearson', 'mi')
            min_samples: Minimum samples needed for valid correlation
            full_rel_only: Only consider metrics with full relationship data
            beta: Weight for combining global and local signals (0-1)
            t_global: Temperature for global relationship sigmoid
            t_local: Temperature for local abnormality sigmoid
            **kwargs: Additional hyperparameter overrides

        Returns:
            Dictionary of hyperparameters
        """
        hyperparameters = cls.DEFAULT_HYPERPARAMETERS.copy()
        hyperparameters.update({
            'rel_type': rel_type,
            'min_samples_needed': min_samples,
            'full_rel_only': full_rel_only,
            'beta': beta,
            't_global': t_global,
            't_local': t_local,
        })
        hyperparameters.update(kwargs)
        return hyperparameters

    @classmethod
    def validate_hyperparameters(cls, hyperparameters: Dict) -> None:
        """
        Validate hyperparameter values.

        Args:
            hyperparameters: Dictionary of hyperparameters to validate

        Raises:
            ValueError: If hyperparameters are invalid
        """
        # Validate rel_type
        valid_rel_types = ['spearman', 'pearson', 'mi']
        if hyperparameters.get('rel_type') not in valid_rel_types:
            raise ValueError(
                f"rel_type must be one of {valid_rel_types}, "
                f"got {hyperparameters.get('rel_type')}"
            )

        # Validate beta range
        beta = hyperparameters.get('beta', 0.5)
        if not 0 <= beta <= 1:
            raise ValueError(f"beta must be in [0, 1], got {beta}")

        # Validate minimum samples
        min_samples = hyperparameters.get('min_samples_needed', 4)
        if min_samples < 3:
            raise ValueError(f"min_samples_needed must be >= 3, got {min_samples}")

        # Validate temperature parameters
        for temp_key in ['t_global', 't_local']:
            temp_val = hyperparameters.get(temp_key, 1.0)
            if temp_val <= 0:
                raise ValueError(f"{temp_key} must be positive, got {temp_val}")


# Create a default configuration instance
config = RetrievalConfig()
