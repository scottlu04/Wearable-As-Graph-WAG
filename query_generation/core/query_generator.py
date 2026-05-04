"""
Query Generator Module

Generates diverse, clinically relevant questions from health knowledge graphs
and wearable device data.
"""

import json
import logging
from typing import Dict, List, Optional, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed

from openai import OpenAI
from tqdm import tqdm

from ..config.settings import config
from shared.prompts.QueryGen import (
    QUERY_GENERATION_JSON_PROMPT_SINGLE_ENTITY,
    QUERY_GENERATION_JSON_PROMPT_MULTIPLE_ENTITIES
)
from shared.prompts.Context import CONTEXT_PROMPT
from shared.utils import parse_llm_response

# Configure logging
logger = logging.getLogger(__name__)


class QueryGenerator:
    """
    Generates health metric queries using LLM.

    This class handles:
    - Single entity queries (7 question types)
    - Multi-entity queries (relationship questions)
    - Batch processing with parallel execution
    - Time-bound question generation
    """

    def __init__(self, node_lookup: Optional[Dict] = None):
        """
        Initialize the QueryGenerator.

        Args:
            node_lookup: Dictionary mapping node names to Entity objects.
                        If None, must be provided in generate methods.
        """
        # Validate configuration
        config.validate()

        # Store node lookup
        self.node_lookup = node_lookup

        # Initialize OpenAI client
        if config.USE_DEEPSEEK:
            self.client = OpenAI(
                api_key=config.get_api_key(),
                base_url=config.get_api_url()
            )
        else:
            self.client = OpenAI(api_key=config.OPENAI_API_KEY)

        self.model = config.get_chat_model()

    def generate_single_entity_queries(
        self,
        query_specs: List[Dict],
        batch_size: Optional[int] = None,
        max_workers: Optional[int] = None,
        verbose: bool = False
    ) -> Tuple[List[Dict], List[str]]:
        """
        Generate questions about single health metrics.

        Args:
            query_specs: List of query specifications with:
                - id: unique identifier
                - name: metric name
                - description: metric description
                - date: query date (ISO format)
                - time_granularity: time period (1/7/14/30/60/all)
                - abnormality_level: deviation level (low/medium/high)
            batch_size: Number of queries per batch.
            max_workers: Number of parallel workers.
            verbose: Enable detailed logging.

        Returns:
            Tuple of (successful_queries, errors)
            Each query contains:
                - id: matching input id
                - question: generated question text
                - question_type: type classification
                - openness: openness score (0.0-1.0)

        Example:
            >>> generator = QueryGenerator(node_lookup)
            >>> queries, errors = generator.generate_single_entity_queries([
            ...     {
            ...         "id": "q1",
            ...         "name": "Heart Rate",
            ...         "description": "Beats per minute...",
            ...         "date": "2024-01-01",
            ...         "time_granularity": "7",
            ...         "abnormality_level": "low"
            ...     }
            ... ])
        """
        batch_size = batch_size or config.DEFAULT_BATCH_SIZE
        max_workers = max_workers or config.MAX_WORKERS

        if verbose:
            logger.info(f"Generating single entity queries for {len(query_specs)} specifications")

        # Split into batches
        batches = self._split_into_batches(query_specs, batch_size)

        # Process in parallel
        return self._batch_generate(
            batches,
            QUERY_GENERATION_JSON_PROMPT_SINGLE_ENTITY,
            "single entity",
            max_workers,
            verbose
        )

    def generate_multiple_entity_queries(
        self,
        query_specs: List[Dict],
        batch_size: Optional[int] = None,
        max_workers: Optional[int] = None,
        verbose: bool = False
    ) -> Tuple[List[Dict], List[str]]:
        """
        Generate questions about metric relationships.

        Args:
            query_specs: List of specifications with:
                - id: unique identifier
                - metrics: list of {name, description} dicts
                - date: query date (ISO format)
                - time_granularity: time period (1/7/14/30/60/all)
            batch_size: Number of queries per batch.
            max_workers: Number of parallel workers.
            verbose: Enable detailed logging.

        Returns:
            Tuple of (successful_queries, errors)

        Example:
            >>> generator = QueryGenerator(node_lookup)
            >>> queries, errors = generator.generate_multiple_entity_queries([
            ...     {
            ...         "id": "q1",
            ...         "metrics": [
            ...             {"name": "Sleep Duration", "description": "..."},
            ...             {"name": "Heart Rate", "description": "..."}
            ...         ],
            ...         "date": "2024-01-01",
            ...         "time_granularity": "30"
            ...     }
            ... ])
        """
        batch_size = batch_size or config.DEFAULT_BATCH_SIZE
        max_workers = max_workers or config.MAX_WORKERS

        if verbose:
            logger.info(f"Generating multi-entity queries for {len(query_specs)} specifications")

        # Split into batches
        batches = self._split_into_batches(query_specs, batch_size)

        # Process in parallel
        return self._batch_generate(
            batches,
            QUERY_GENERATION_JSON_PROMPT_MULTIPLE_ENTITIES,
            "multiple entity",
            max_workers,
            verbose
        )

    def _batch_generate(
        self,
        batches: List[List[Dict]],
        prompt_template: str,
        query_type: str,
        max_workers: int,
        verbose: bool
    ) -> Tuple[List[Dict], List[str]]:
        """
        Process batches in parallel using ThreadPoolExecutor.

        Args:
            batches: List of batches to process.
            prompt_template: Prompt template to use.
            query_type: Type description for logging.
            max_workers: Number of parallel workers.
            verbose: Enable detailed logging.

        Returns:
            Tuple of (successful_results, errors)
        """
        results = []
        errors = []

        def process_batch(batch: List[Dict]) -> Tuple[Optional[List[Dict]], Optional[str]]:
            """Process a single batch."""
            try:
                prompt = (
                    prompt_template +
                    "\nInput:\n" +
                    json.dumps(batch, indent=2) +
                    "\nOutput:"
                )

                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {
                            "role": "system",
                            "content": (
                                CONTEXT_PROMPT +
                                "You have completed generating the knowledge graph. "
                                "Now you will generate queries to test the graph."
                            )
                        },
                        {"role": "user", "content": prompt}
                    ],
                    temperature=config.TEMPERATURE
                )

                output_llm = parse_llm_response(response.choices[0].message.content)
                return output_llm, None

            except Exception as e:
                error_msg = f"Error processing batch: {str(e)}"
                logger.error(error_msg)
                return None, error_msg

        # Process batches in parallel
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = [executor.submit(process_batch, batch) for batch in batches]

            # Collect results with progress bar
            desc = f"Generating {query_type} queries"
            for future in tqdm(as_completed(futures), total=len(futures), desc=desc, disable=not verbose):
                output_llm, error = future.result()

                if output_llm:
                    results.extend(output_llm)
                if error:
                    errors.append(error)

        if verbose:
            logger.info(f"Generated {len(results)} queries, {len(errors)} errors")

        return results, errors

    def _split_into_batches(self, data: List, batch_size: int) -> List[List]:
        """Split data into batches of given size."""
        batches = []
        for i in range(0, len(data), batch_size):
            batch = data[i:i + batch_size]
            batches.append(batch)
        return batches

    @staticmethod
    def prepare_query_input_from_node(
        query_id: str,
        node,
        dataset_name: str,
        date: str,
        time_granularity: str,
        abnormality_level: str = "low"
    ) -> Dict:
        """
        Prepare query input from a knowledge graph node.

        Args:
            query_id: Unique query identifier.
            node: Entity object from knowledge graph.
            dataset_name: Dataset name for dataSource lookup.
            date: Query date (ISO format).
            time_granularity: Time period (1/7/14/30/60/all).
            abnormality_level: Deviation level (low/medium/high).

        Returns:
            Dictionary ready for query generation.
        """
        # Build description
        description_parts = []
        if node.description:
            description_parts.append(node.description)

        # Add dataset-specific description if available
        if hasattr(node, 'dataSource') and node.dataSource:
            dataset_desc = node.dataSource.get(dataset_name, {}).get('description')
            if dataset_desc:
                description_parts.append(f"Extra wearable specific description: {dataset_desc}")

        return {
            "id": query_id,
            "name": node.name,
            "description": "\n".join(description_parts),
            "date": date,
            "time_granularity": str(time_granularity),
            "abnormality_level": abnormality_level
        }

    @staticmethod
    def prepare_multi_query_input_from_nodes(
        query_id: str,
        nodes: List,
        dataset_name: str,
        date: str,
        time_granularity: str
    ) -> Dict:
        """
        Prepare multi-entity query input from knowledge graph nodes.

        Args:
            query_id: Unique query identifier.
            nodes: List of Entity objects.
            dataset_name: Dataset name for dataSource lookup.
            date: Query date (ISO format).
            time_granularity: Time period (1/7/14/30/60/all).

        Returns:
            Dictionary ready for multi-entity query generation.
        """
        metrics = []
        for node in nodes:
            # Build description
            description_parts = []
            if node.description:
                description_parts.append(node.description)

            # Add dataset-specific description
            if hasattr(node, 'dataSource') and node.dataSource:
                dataset_desc = node.dataSource.get(dataset_name, {}).get('description')
                if dataset_desc:
                    description_parts.append(f"Extra wearable specific description: {dataset_desc}")

            metrics.append({
                'name': node.name,
                'description': '\n'.join(description_parts)
            })

        return {
            "id": query_id,
            "metrics": metrics,
            "date": date,
            "time_granularity": str(time_granularity)
        }
