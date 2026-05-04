"""
Retrieval Package - Statistical Weight Computation for Graph Traversal.

This package provides sophisticated weight computation for knowledge graph traversal
using Bayesian updating of LLM priors with population and individual statistics,
combined with local abnormality detection.

Main Components:
- WeightRetriever: Bayesian weight computation for graph traversal
- Statistical utilities: Correlation, mutual information, anomaly detection
- Configuration: Centralized hyperparameter management

Example Usage:
    from shared.retrieval_package.core import WeightRetriever
    from shared.retrieval_package.config import RetrievalConfig

    # Get hyperparameters
    hyperparams = RetrievalConfig.get_hyperparameters(
        rel_type='spearman',
        beta=0.5
    )

    # Initialize retriever
    retriever = WeightRetriever(
        metric_to_check='Heart Rate',
        rel_info=relationship_info,
        llm_prior_df=prior_matrix,
        hyperparameter=hyperparams,
        ind_df=individual_data,
        query_info={'query_date': '2024-01-01', 'time_granularity': 7, 'openness': 0.5}
    )

    # Run weight computation
    weights_df, abnormality_df = retriever.run()

    # Get top related metrics
    top_related = weights_df.sort_values(by='weight_final', ascending=False).head(10)
"""

from .core import WeightRetriever
from .config import RetrievalConfig, config
from .utils import (
    fisher_transform,
    compute_correlation_stats,
    compute_mutual_info_stats,
    detect_anomalies,
    get_prior_matrix,
    get_relationship_matrix,
)

__version__ = '1.0.0'
__author__ = 'WAG Research Team'

__all__ = [
    'WeightRetriever',
    'RetrievalConfig',
    'config',
    'fisher_transform',
    'compute_correlation_stats',
    'compute_mutual_info_stats',
    'detect_anomalies',
    'get_prior_matrix',
    'get_relationship_matrix',
]
