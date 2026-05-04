"""
Local Weight Comparison Pipeline

This pipeline compares different strategies for combining local abnormality signals
with global knowledge in the WAG system:
    1. kgrag_local: Uses only local abnormality weights
    2. kgrag_global: Uses only global Bayesian posterior
    3. kgrag_final: Uses blended weight (β*local + (1-β)*global)

The comparison evaluates the trade-off between:
    - Global knowledge (generalizable, stable, based on correlations)
    - Local signals (personalized, sensitive to recent changes)

All methods use the same WAGProcessor, differing only in the weight_sort_by parameter.

Usage:
    python3 -m response_generation.pipelines.local_weight_new
"""

from typing import Dict, Any, List
import logging

# Import base pipeline
from .base_pipeline import BasePipeline

# Import processor
from response_generation.core import WAGProcessor

# Set up logging
logger = logging.getLogger(__name__)


class LocalWeightPipeline(BasePipeline):
    """
    Pipeline for comparing local vs global vs final weights.

    This experiment evaluates whether local abnormality signals improve upon
    global knowledge-based relationship ranking in the WAG system.
    """

    def __init__(
        self,
        root_dir: str,
        output_dir: str = None,
        max_workers: int = 1,
        qid_filter_file: str = None,
        chat_client: str = None,
        chat_model: str = None,
        generate_responses: bool = False,
        methods: List[str] = None,
    ):
        """
        Initialize local weight comparison pipeline.

        Args:
            root_dir: Root directory containing resources
            output_dir: Output directory for results
            max_workers: Maximum number of parallel workers (default: 1 for sequential)
            qid_filter_file: Path to file containing list of q_ids to process (optional)
        """
        if output_dir is None:
            output_dir = f"{root_dir}/experiment_result/local"

        super().__init__(
            root_dir,
            output_dir,
            max_workers,
            qid_filter_file,
            chat_client=chat_client,
            chat_model=chat_model,
            generate_responses=generate_responses,
            methods=methods,
        )

        # Base hyperparameters (shared across all methods)
        self.base_hyperparameters = {
            'rel_type': 'spearman',
            't_global': 0.9,
            't_local': 0.7,
            'alpha_prior': 1.0,
            'alpha_pop': 10**0.94,
            'alpha_ind': 10**-1.31,
            'beta': 0.5,  # Blend factor for local/global combination
            'max_num_related_nodes': 5,
            'min_samples_needed': 10,
            'full_rel_only': False,  # Local experiment can use partial relationships
            'demog_metrics': []
        }

        # Weight strategies - ONLY weight_sort_by differs
        self.weight_strategies = {
            'kgrag_local': {'weight_sort_by': 'weight_local'},    # Local abnormality only
            'kgrag_global': {'weight_sort_by': 'weight_global'},  # Global Bayesian posterior
            'kgrag_final': {'weight_sort_by': 'weight_final'},    # Blended (β*local + (1-β)*global)
        }

        logger.info("Local Weight Comparison")
        logger.info(f"  Strategies: {list(self.weight_strategies.keys())}")
        logger.info(f"  Beta (blend factor): {self.base_hyperparameters['beta']}")

    def get_clients(self) -> Dict[str, Any]:
        """
        Initialize WAG processors with different weight strategies.

        Returns:
            Dictionary of method_name -> WAGProcessor instance
        """
        clients = {}
        processor_kwargs = self.get_processor_init_kwargs()

        selected_methods = self.select_method_names(list(self.weight_strategies.keys()))

        for method_name in selected_methods:
            strategy = self.weight_strategies[method_name]
            # Merge base hyperparameters with strategy-specific parameter
            hyperparams = {**self.base_hyperparameters, **strategy}
            clients[method_name] = WAGProcessor(
                root_dir=f"{self.root_dir}/kg",
                param=hyperparams,
                **processor_kwargs,
            )

        return clients


def main():
    """Main entry point for command-line execution."""
    from response_generation import setup_pipeline_logging

    # Set up logging
    logger = setup_pipeline_logging("local_weight_comparison", log_dir="logs")

    # Create and run pipeline
    pipeline = LocalWeightPipeline(root_dir='resources')
    results = pipeline.run()

    return results


if __name__ == "__main__":
    main()
