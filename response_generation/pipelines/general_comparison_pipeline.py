"""
General Comparison Pipeline: Base vs RAG vs StaticGraph vs FullContext vs WAG

This pipeline compares five default query processing methods on health data:
    1. Base: Simple entity matching without graph traversal or weights
    2. RAG: Retrieval-Augmented Generation with knowledge graph context
    3. StaticGraph: Static edge-weight graph retrieval without dynamic recomputation
    4. FullContext: All available wearable metrics within the query time window
    5. WAG: Weight-Augmented Graph with statistical relationships and abnormality

FullHistory remains available as an optional method through `--methods`, but is
not included in the default comparison.

The comparison evaluates how different approaches affect:
    - Response quality and relevance
    - Context richness and informativeness
    - Computational efficiency
    - Ability to discover related health factors

Usage:
    python3 -m response_generation.pipelines.general_comparison_new
"""

import logging
from typing import Dict, Any, List

# Import base pipeline
from .base_pipeline import BasePipeline

# Import processors
from response_generation.core import (
    BaseQueryProcessor,
    FullContextProcessor,
    FullHistoryContextProcessor,
    RAGProcessor,
    StaticGraphProcessor,
    WAGProcessor,
)

# Set up logging
logger = logging.getLogger(__name__)


class GeneralComparisonPipeline(BasePipeline):
    """
    Pipeline for comparing Base, RAG, StaticGraph, FullContext, and WAG methods.

    This experiment evaluates the core query processing approaches to understand
    the incremental value of knowledge graph retrieval, static graph ranking,
    all-metric prompting over matched time windows, and weight-based selection.
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
        Initialize general comparison pipeline.

        Args:
            root_dir: Root directory containing resources
            output_dir: Output directory for results (default: root_dir/experiment_result/general)
            max_workers: Maximum number of parallel workers (default: 1 for sequential)
            qid_filter_file: Path to file containing list of q_ids to process (optional)
        """
        if output_dir is None:
            output_dir = f"{root_dir}/experiment_result/general"

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

        # Hyperparameters for WAG
        self.hyperparameter = {
            'rel_type': 'spearman',
            't_global': 0.9,
            't_local': 0.7,
            'alpha_prior': 1.0,
            'alpha_pop': 10**0.94,
            'alpha_ind': 10**-1.31,
            'beta': 0.5,
            'max_num_related_nodes': 5,
            'min_samples_needed': 10,
            'full_rel_only': False,
            'weight_sort_by': 'weight_final',
            'demog_metrics': []
        }
        logger.info("General Comparison: Base vs RAG vs StaticGraph vs FullContext vs WAG")
        logger.info(f"  WAG hyperparameters: {self.hyperparameter}")

    def get_clients(self) -> Dict[str, Any]:
        """
        Initialize Base, RAG, StaticGraph, FullContext, and WAG processors.

        Returns:
            Dictionary of method_name -> processor instance
        """
        processor_kwargs = self.get_processor_init_kwargs()
        default_methods = ['Base', 'Rag', 'StaticGraph', 'FullContext', 'Wag']
        available_methods = ['Base', 'Rag', 'StaticGraph', 'FullContext', 'FullHistory', 'Wag']
        selected_methods = self.select_method_names(
            available_methods if self.methods else default_methods
        )
        clients: Dict[str, Any] = {}

        for method_name in selected_methods:
            if method_name == 'Base':
                clients[method_name] = BaseQueryProcessor(
                    root_dir=f"{self.root_dir}/kg",
                    **processor_kwargs,
                )
            elif method_name == 'Rag':
                clients[method_name] = RAGProcessor(
                    root_dir=f"{self.root_dir}/kg",
                    **processor_kwargs,
                )
            elif method_name == 'StaticGraph':
                clients[method_name] = StaticGraphProcessor(
                    root_dir=f"{self.root_dir}/kg",
                    param=self.hyperparameter,
                    **processor_kwargs,
                )
            elif method_name == 'FullContext':
                clients[method_name] = FullContextProcessor(
                    root_dir=f"{self.root_dir}/kg",
                    **processor_kwargs,
                )
            elif method_name == 'FullHistory':
                clients[method_name] = FullHistoryContextProcessor(
                    root_dir=f"{self.root_dir}/kg",
                    **processor_kwargs,
                )
            elif method_name == 'Wag':
                clients[method_name] = WAGProcessor(
                    root_dir=f"{self.root_dir}/kg",
                    param=self.hyperparameter,
                    **processor_kwargs,
                )

        return clients


def main():
    """Main entry point for command-line execution."""
    from response_generation import setup_pipeline_logging

    # Set up logging
    logger = setup_pipeline_logging("general_comparison", log_dir="logs")

    # Create and run pipeline
    pipeline = GeneralComparisonPipeline(root_dir='resources')
    results = pipeline.run()

    return results


if __name__ == "__main__":
    main()
