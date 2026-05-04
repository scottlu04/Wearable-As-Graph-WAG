"""
Global Weight Comparison Pipeline

This pipeline compares different global weight sources for ranking relationships
in the knowledge graph:
    1. kgrag_prior: LLM prior knowledge only
    2. kgrag_pop: Population-level correlations
    3. kgrag_ind: Individual-level correlations
    4. kgrag_global: Bayesian posterior combining all three

The comparison evaluates which knowledge source is most effective for:
    - Identifying relevant related health factors
    - Generating accurate and helpful responses
    - Generalizing across different query types and users

All methods use the same WAGProcessor, differing only in the weight_sort_by parameter.

Usage:
    python3 -m response_generation.pipelines.global_weight_new
"""

from typing import Dict, Any, List
import logging

# Import base pipeline
from .base_pipeline import BasePipeline

# Import processor
from response_generation.core import WAGProcessor

# Set up logging
logger = logging.getLogger(__name__)


class GlobalWeightPipeline(BasePipeline):
    """
    Pipeline for comparing global weight sources.

    This experiment isolates the contribution of different knowledge sources
    (LLM priors, population data, individual patterns) in the WAG system.
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
        Initialize global weight comparison pipeline.

        Args:
            root_dir: Root directory containing resources
            output_dir: Output directory for results
            max_workers: Maximum number of parallel workers (default: 1 for sequential)
            qid_filter_file: Path to file containing list of q_ids to process (optional)
        """
        if output_dir is None:
            output_dir = f"{root_dir}/experiment_result/global"

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
            'beta': 0.5,
            'max_num_related_nodes': 5,
            'min_samples_needed': 10,
            'full_rel_only': True,  # Global experiment uses full relationships
            'demog_metrics': []
        }

        # Weight strategies - ONLY weight_sort_by differs
        self.weight_strategies = {
            'kgrag_prior': {'weight_sort_by': 'weight_prior'},    # LLM prior only
            'kgrag_pop': {'weight_sort_by': 'weight_pop'},        # Population correlations
            'kgrag_ind': {'weight_sort_by': 'weight_ind'},        # Individual correlations
            'kgrag_global': {'weight_sort_by': 'weight_global'},  # Bayesian posterior
        }
        self.weight_strategies = {
            'kgrag_prior': {'weight_sort_by': 'llm_prior'},    # LLM prior only
            'kgrag_pop': {'weight_sort_by': 'rel_pop'},        # Population correlations
            'kgrag_ind': {'weight_sort_by': 'rel_ind'},        # Individual correlations
            'kgrag_global': {'weight_sort_by': 'weight_global'},  # Bayesian posterior
        }

        # parm_prior = hyperparameter.copy()
        # parm_prior['weight_sort_by'] = 'llm_prior'
        # parm_rel_pop = hyperparameter.copy()
        # parm_rel_pop['weight_sort_by'] = 'rel_pop'
        # parm_rel_ind = hyperparameter.copy()
        # parm_rel_ind['weight_sort_by'] = 'rel_ind'
        # parm_global = hyperparameter.copy()
        # parm_global['weight_sort_by'] = 'weight_global'

        logger.info("Global Weight Comparison")
        logger.info(f"  Strategies: {list(self.weight_strategies.keys())}")
        logger.info(f"  Base hyperparameters: {self.base_hyperparameters}")

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
    logger = setup_pipeline_logging("global_weight_comparison", log_dir="logs")

    # Create and run pipeline
    pipeline = GlobalWeightPipeline(root_dir='resources')
    results = pipeline.run()

    return results


if __name__ == "__main__":
    main()
