"""
Query Generation Pipeline

End-to-end pipeline for generating questions from QA dataframes
and knowledge graph nodes.
"""

import os
import json
import logging
from typing import Dict, List, Optional
import pandas as pd
from tqdm import tqdm

from shared.models.entity import Entity
from ..core.query_generator import QueryGenerator
from ..config.settings import config

# Configure logging
logging.basicConfig(
    level=getattr(logging, config.LOG_LEVEL),
    format=config.LOG_FORMAT
)
logger = logging.getLogger(__name__)


class QueryGenerationPipeline:
    """
    Pipeline for generating queries from QA dataframe.

    This pipeline:
    1. Loads knowledge graph nodes
    2. Loads QA dataframe with query specifications
    3. Prepares query inputs from nodes
    4. Generates queries using LLM
    5. Updates dataframe with generated queries
    6. Saves results
    """

    def __init__(
        self,
        data_dir: str,
        qa_dataframe_path: str,
        batch_size: Optional[int] = None,
        max_workers: Optional[int] = None
    ):
        """
        Initialize the query generation pipeline.

        Args:
            data_dir: Directory containing nodes.json from graph construction.
            qa_dataframe_path: Path to QA dataframe JSON file.
            batch_size: Batch size for LLM processing.
            max_workers: Number of parallel workers.
        """
        self.data_dir = data_dir
        self.qa_dataframe_path = qa_dataframe_path
        self.batch_size = batch_size or config.DEFAULT_BATCH_SIZE
        self.max_workers = max_workers or config.MAX_WORKERS

        # Storage
        self.nodes: Dict[str, Entity] = {}
        self.node_lookup: Dict[str, Entity] = {}
        self.qa_df: Optional[pd.DataFrame] = None

        # Generator
        self.generator: Optional[QueryGenerator] = None

        # Tracking
        self.failed_queries: List[Dict] = []

    def run(self, stages: Optional[List[str]] = None) -> Dict[str, any]:
        """
        Run the query generation pipeline.

        Args:
            stages: List of stages to run. If None, runs all stages.
                   Available stages:
                   - 'load': Load nodes and QA dataframe
                   - 'generate': Generate queries
                   - 'validate': Validate generated queries
                   - 'save': Save results

        Returns:
            Dictionary with pipeline statistics.
        """
        all_stages = ['load', 'generate', 'validate', 'save']
        stages = stages or all_stages

        logger.info("Starting query generation pipeline")
        logger.info(f"Data directory: {self.data_dir}")
        logger.info(f"QA dataframe: {self.qa_dataframe_path}")
        logger.info(f"Stages to run: {stages}")

        stats = {
            'total_specifications': 0,
            'queries_needed': 0,
            'queries_generated': 0,
            'failed_queries': 0
        }

        # Stage 1: Load data
        if 'load' in stages:
            self._load_data()
            stats['total_specifications'] = len(self.qa_df)
            stats['queries_needed'] = len(self.qa_df[self.qa_df['query'].isna()])

        # Stage 2: Generate queries
        if 'generate' in stages:
            stats['queries_generated'] = self._stage_generate_queries()

        # Stage 3: Validate
        if 'validate' in stages:
            self._stage_validate()

        # Stage 4: Save
        if 'save' in stages:
            self._save_results()

        stats['failed_queries'] = len(self.failed_queries)

        logger.info("Pipeline completed successfully")
        logger.info(f"Statistics: {stats}")

        return stats

    def _load_data(self):
        """Load knowledge graph nodes and QA dataframe."""
        logger.info("Loading knowledge graph nodes")

        # Load nodes
        nodes_path = os.path.join(self.data_dir, "nodes.json")
        with open(nodes_path, 'r') as f:
            nodes_data = json.load(f)
            for node_id, node_dict in nodes_data.items():
                self.nodes[node_id] = Entity.from_dict(node_dict)

        # Create node lookup by name
        self.node_lookup = {node.name: node for node in self.nodes.values()}

        logger.info(f"Loaded {len(self.nodes)} nodes")

        # Initialize generator
        self.generator = QueryGenerator(self.node_lookup)

        # Load QA dataframe
        logger.info(f"Loading QA dataframe from {self.qa_dataframe_path}")
        self.qa_df = pd.read_json(self.qa_dataframe_path, orient='records')
        logger.info(f"Loaded {len(self.qa_df)} query specifications")

        # Check for existing queries
        existing_queries = self.qa_df[self.qa_df['query'].notna()]
        missing_queries = self.qa_df[self.qa_df['query'].isna()]
        logger.info(f"  {len(existing_queries)} already have queries")
        logger.info(f"  {len(missing_queries)} need queries")

    def _stage_generate_queries(self) -> int:
        """Generate queries for all specifications without queries."""
        logger.info("Stage: Generating queries")

        # Get rows without queries
        qa_without_queries = self.qa_df[self.qa_df['query'].isna()]

        if len(qa_without_queries) == 0:
            logger.info("No queries need generation")
            return 0

        total_generated = 0

        # Process by dataset
        datasets = qa_without_queries['dataset'].unique()
        logger.info(f"Processing {len(datasets)} datasets")

        for dataset_name in datasets:
            logger.info(f"Processing dataset: {dataset_name}")

            dataset_queries = qa_without_queries[
                qa_without_queries['dataset'] == dataset_name
            ]

            # Separate single and multiple entity queries
            single_queries = dataset_queries[
                dataset_queries['single_or_multiple'] == 'Single'
            ]
            multiple_queries = dataset_queries[
                dataset_queries['single_or_multiple'] == 'Multiple'
            ]

            # Generate single entity queries
            if len(single_queries) > 0:
                generated = self._generate_single_queries(
                    single_queries,
                    dataset_name
                )
                total_generated += generated

            # Generate multiple entity queries
            if len(multiple_queries) > 0:
                generated = self._generate_multiple_queries(
                    multiple_queries,
                    dataset_name
                )
                total_generated += generated

        logger.info(f"Generated {total_generated} total queries")
        return total_generated

    def _generate_single_queries(
        self,
        queries_df: pd.DataFrame,
        dataset_name: str
    ) -> int:
        """Generate single entity queries."""
        logger.info(f"Generating {len(queries_df)} single entity queries")

        # Prepare inputs
        query_inputs = []
        for _, row in queries_df.iterrows():
            try:
                entity_name = row['associated_entity(gt)']
                if isinstance(entity_name, list):
                    entity_name = entity_name[0]

                if pd.isna(entity_name) or not entity_name:
                    logger.warning(f"Skipping query {row['q_id']}: missing entity name")
                    continue

                node = self.node_lookup.get(entity_name)
                if not node:
                    logger.warning(f"Entity '{entity_name}' not found for query {row['q_id']}")
                    continue

                query_input = self.generator.prepare_query_input_from_node(
                    query_id=row['q_id'],
                    node=node,
                    dataset_name=dataset_name,
                    date=pd.to_datetime(row['query_date']).isoformat(),
                    time_granularity=str(row['time_granularity']),
                    abnormality_level=row.get('recent_abnormality', 'low')
                )
                query_inputs.append(query_input)

            except Exception as e:
                logger.error(f"Error preparing query {row['q_id']}: {str(e)}")
                self.failed_queries.append({
                    'q_id': row['q_id'],
                    'stage': 'prepare_single',
                    'error': str(e)
                })

        # Generate queries
        if query_inputs:
            generated_queries, errors = self.generator.generate_single_entity_queries(
                query_inputs,
                batch_size=self.batch_size,
                max_workers=self.max_workers,
                verbose=True
            )

            # Update dataframe
            self._update_dataframe(generated_queries)

            # Track errors
            for error in errors:
                self.failed_queries.append({
                    'stage': 'generate_single',
                    'error': error
                })

            return len(generated_queries)

        return 0

    def _generate_multiple_queries(
        self,
        queries_df: pd.DataFrame,
        dataset_name: str
    ) -> int:
        """Generate multiple entity queries."""
        logger.info(f"Generating {len(queries_df)} multiple entity queries")

        # Prepare inputs
        query_inputs = []
        for _, row in queries_df.iterrows():
            try:
                entity_names = row['associated_entity(gt)']
                if not isinstance(entity_names, list):
                    entity_names = [entity_names]

                # Get nodes
                nodes = []
                for entity_name in entity_names:
                    if pd.isna(entity_name) or not entity_name:
                        continue

                    node = self.node_lookup.get(entity_name)
                    if node:
                        nodes.append(node)
                    else:
                        logger.warning(f"Entity '{entity_name}' not found")

                if len(nodes) < 2:
                    logger.warning(f"Insufficient nodes for multi-query {row['q_id']}")
                    continue

                query_input = self.generator.prepare_multi_query_input_from_nodes(
                    query_id=row['q_id'],
                    nodes=nodes,
                    dataset_name=dataset_name,
                    date=pd.to_datetime(row['query_date']).isoformat(),
                    time_granularity=str(row['time_granularity'])
                )
                query_inputs.append(query_input)

            except Exception as e:
                logger.error(f"Error preparing query {row['q_id']}: {str(e)}")
                self.failed_queries.append({
                    'q_id': row['q_id'],
                    'stage': 'prepare_multiple',
                    'error': str(e)
                })

        # Generate queries
        if query_inputs:
            generated_queries, errors = self.generator.generate_multiple_entity_queries(
                query_inputs,
                batch_size=self.batch_size,
                max_workers=self.max_workers,
                verbose=True
            )

            # Update dataframe
            self._update_dataframe(generated_queries)

            # Track errors
            for error in errors:
                self.failed_queries.append({
                    'stage': 'generate_multiple',
                    'error': error
                })

            return len(generated_queries)

        return 0

    def _update_dataframe(self, generated_queries: List[Dict]):
        """Update dataframe with generated queries."""
        # Create ID to index mapping
        id_to_index = {q_id: idx for idx, q_id in enumerate(self.qa_df['q_id'])}

        for query_result in generated_queries:
            try:
                query_id = query_result['id']

                if query_id not in id_to_index:
                    logger.warning(f"Query ID {query_id} not found in dataframe")
                    continue

                idx = id_to_index[query_id]
                self.qa_df.at[idx, 'query'] = query_result['question']
                self.qa_df.at[idx, 'query_type'] = query_result['question_type']
                self.qa_df.at[idx, 'openness(assigned)'] = query_result['openness']

            except Exception as e:
                logger.error(f"Error updating dataframe for query {query_result.get('id')}: {str(e)}")

    def _stage_validate(self):
        """Validate generated queries."""
        logger.info("Stage: Validating queries")

        # Check for queries with NA values
        invalid_queries = self.qa_df[
            self.qa_df['query'].notna() &
            (self.qa_df['query_type'].isna() | self.qa_df['openness(assigned)'].isna())
        ]

        if len(invalid_queries) > 0:
            logger.warning(f"Found {len(invalid_queries)} incomplete query entries")

        # Check openness range
        valid_openness = self.qa_df[
            self.qa_df['openness(assigned)'].notna() &
            (self.qa_df['openness(assigned)'] >= 0) &
            (self.qa_df['openness(assigned)'] <= 1)
        ]

        invalid_openness = len(self.qa_df[self.qa_df['openness(assigned)'].notna()]) - len(valid_openness)
        if invalid_openness > 0:
            logger.warning(f"Found {invalid_openness} queries with invalid openness scores")

        logger.info("Validation completed")

    def _save_results(self):
        """Save updated dataframe and failed queries."""
        logger.info("Saving results")

        # Save updated dataframe
        output_path = self.qa_dataframe_path.replace('.json', '_updated.json')
        self.qa_df.to_json(output_path, orient='records', indent=4)
        logger.info(f"Saved updated dataframe to {output_path}")

        # Save failed queries if any
        if self.failed_queries:
            failed_path = os.path.join(self.data_dir, "failed_query_generation.json")
            with open(failed_path, 'w') as f:
                json.dump(self.failed_queries, f, indent=2)
            logger.warning(f"Saved {len(self.failed_queries)} failed queries to {failed_path}")


def main():
    """Main entry point for command-line execution."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Query Generation Pipeline"
    )
    parser.add_argument(
        '--data-dir',
        type=str,
        required=True,
        help='Directory containing nodes.json'
    )
    parser.add_argument(
        '--qa-dataframe',
        type=str,
        required=True,
        help='Path to QA dataframe JSON file'
    )
    parser.add_argument(
        '--batch-size',
        type=int,
        default=None,
        help='Batch size for processing'
    )
    parser.add_argument(
        '--max-workers',
        type=int,
        default=None,
        help='Number of parallel workers'
    )
    parser.add_argument(
        '--stages',
        nargs='+',
        choices=['load', 'generate', 'validate', 'save'],
        default=None,
        help='Pipeline stages to run (default: all)'
    )

    args = parser.parse_args()

    # Create and run pipeline
    pipeline = QueryGenerationPipeline(
        data_dir=args.data_dir,
        qa_dataframe_path=args.qa_dataframe,
        batch_size=args.batch_size,
        max_workers=args.max_workers
    )

    stats = pipeline.run(stages=args.stages)

    print("\n" + "=" * 50)
    print("QUERY GENERATION STATISTICS")
    print("=" * 50)
    for key, value in stats.items():
        print(f"{key}: {value}")


if __name__ == "__main__":
    main()
