"""
Edge Construction Pipeline

This script implements the edge/relationship construction workflow from
notebook 2_relationship_mapping_batch.ipynb in a structured format.
"""

import os
import json
import logging
from typing import Dict, List, Optional, Tuple
from tqdm import tqdm

from shared.models.entity import Entity
from shared.models.relationship import Relationship
from openai import OpenAI
from ..core.edge_generator import EdgeGenerator
from ..config.settings import config
from shared.prompts.EntityRelationship_batch import ENTITY_RELATIONSHIPS_GENERATION_JSON_PROMPT_BATCH
from shared.prompts.Context import CONTEXT_PROMPT
from shared.utils import parse_llm_response

# Configure logging
logging.basicConfig(
    level=getattr(logging, config.LOG_LEVEL),
    format=config.LOG_FORMAT
)
logger = logging.getLogger(__name__)


class EdgeConstructionPipeline:
    """
    Pipeline for constructing edges/relationships between nodes.

    This pipeline:
    1. Loads existing nodes
    2. Generates all possible entity pairs
    3. Performs web searches for relationships
    4. Uses LLM to determine relationship strength and description
    5. Saves edges and relationship mappings
    """

    def __init__(
        self,
        data_dir: str,
        load_existing_edges: bool = True,
        batch_size: Optional[int] = None
    ):
        """
        Initialize the edge construction pipeline.

        Args:
            data_dir: Directory containing nodes and where edges will be saved.
            load_existing_edges: Whether to load existing edges if available.
            batch_size: Batch size for LLM processing.
        """
        self.data_dir = data_dir
        self.load_existing_edges = load_existing_edges
        self.batch_size = batch_size or config.EDGE_BATCH_SIZE

        # Initialize edge generator
        self.generator = EdgeGenerator()

        # Initialize OpenAI client for LLM relationship mapping
        if config.USE_DEEPSEEK:
            self.llm_client = OpenAI(
                api_key=config.DEEPSEEK_API_KEY or config.DEEPSEEK_API_KEY2,
                base_url=config.DEEPSEEK_OFFICIAL_API
            )
        else:
            self.llm_client = OpenAI(api_key=config.OPENAI_API_KEY)

        # Storage
        self.nodes: Dict[str, Entity] = {}
        self.edges: Dict[str, Relationship] = {}
        self.relationship_id_map: Dict[str, List[str]] = {}
        self.failed_updates: List[Dict] = []
        self.feature_pairs: List[Tuple[Entity, Entity]] = []

    def run(self, stages: Optional[List[str]] = None) -> Dict[str, any]:
        """
        Run the edge construction pipeline.

        Args:
            stages: List of stages to run. If None, runs all stages.
                   Available stages:
                   - 'load': Load nodes and existing edges
                   - 'generate_pairs': Generate entity pairs
                   - 'web_search': Perform web searches
                   - 'llm_mapping': LLM relationship mapping
                   - 'save': Save results

        Returns:
            Dictionary with pipeline statistics.
        """
        all_stages = ['load', 'generate_pairs', 'web_search', 'llm_mapping', 'save']
        stages = stages or all_stages

        logger.info("Starting edge construction pipeline")
        logger.info(f"Data directory: {self.data_dir}")
        logger.info(f"Stages to run: {stages}")

        stats = {
            'total_nodes': 0,
            'total_edges': 0,
            'new_edges': 0,
            'failed_updates': 0
        }

        # Stage 1: Load data
        if 'load' in stages:
            self._load_data()
            stats['total_nodes'] = len(self.nodes)

        # Stage 2: Generate entity pairs
        # feature_pairs = []
        if 'generate_pairs' in stages:
            self.feature_pairs = self._generate_feature_pairs()
            logger.info(f"Generated {len(self.feature_pairs)} entity pairs")
        # Stage 3: Web search for relationships
        if 'web_search' in stages:
            stats['new_edges'] = self._stage_web_search(self.feature_pairs)

        # Stage 4: LLM relationship mapping
        if 'llm_mapping' in stages:
            self._stage_llm_relationship_mapping()

        # Stage 5: Save results
        if 'save' in stages:
            self._save_results()

        stats['total_edges'] = len(self.edges)
        stats['failed_updates'] = len(self.failed_updates)

        logger.info("Pipeline completed successfully")
        logger.info(f"Statistics: {stats}")

        return stats

    def _load_data(self):
        """Load nodes and optionally existing edges."""
        # Load nodes
        nodes_path = os.path.join(self.data_dir, "nodes.json")
        with open(nodes_path, 'r') as f:
            nodes_data = json.load(f)
            for node_id, node_dict in nodes_data.items():
                self.nodes[node_id] = Entity.from_dict(node_dict)
        logger.info(f"Loaded {len(self.nodes)} nodes")

        # Load existing edges if requested
        edges_path = os.path.join(self.data_dir, "edges.json")
        if self.load_existing_edges and os.path.exists(edges_path):
            with open(edges_path, 'r') as f:
                edges_data = json.load(f)
                for edge_id, edge_dict in edges_data.items():
                    self.edges[edge_id] = Relationship.from_dict(edge_dict)

            rel_map_path = os.path.join(self.data_dir, "relationship_id_map.json")
            if os.path.exists(rel_map_path):
                with open(rel_map_path, 'r') as f:
                    self.relationship_id_map = json.load(f)

            logger.info(f"Loaded {len(self.edges)} existing edges")

    def _generate_feature_pairs(self) -> List[Tuple[Entity, Entity]]:
        """
        Generate all possible pairs of entities.

        Returns:
            List of entity pairs (tuples).
        """
        logger.info("Generating entity pairs")

        nodes_list = list(self.nodes.values())
        feature_pairs = []

        for i in range(len(nodes_list)):
            for j in range(i + 1, len(nodes_list)):
                feature_pairs.append((nodes_list[i], nodes_list[j]))

        expected_count = len(nodes_list) * (len(nodes_list) - 1) / 2
        assert len(feature_pairs) == expected_count

        logger.info(f"Generated {len(feature_pairs)} entity pairs")
        return feature_pairs

    def _check_relationship_exists(
        self,
        entity_1_name: str,
        entity_2_name: str
    ) -> bool:
        """
        Check if a relationship already exists between two entities.

        Args:
            entity_1_name: Name of first entity.
            entity_2_name: Name of second entity.

        Returns:
            True if relationship exists, False otherwise.
        """
        for edge in self.edges.values():
            if (edge.entity_1_name == entity_1_name and edge.entity_2_name == entity_2_name):
                return True
            if (edge.entity_1_name == entity_2_name and edge.entity_2_name == entity_1_name):
                return True
        return False

    def _stage_web_search(self, feature_pairs: List[Tuple[Entity, Entity]]) -> int:
        """
        Stage: Perform web searches for all entity pairs.

        Args:
            feature_pairs: List of entity pairs.

        Returns:
            Number of new edges created.
        """
        logger.info("Stage: Web search for relationships")

        new_edges_count = 0

        for pair in tqdm(feature_pairs, desc="Searching relationships"):
            entity_1, entity_2 = pair
            # Check if relationship already exists
            if self._check_relationship_exists(entity_1.name, entity_2.name):
                logger.debug(
                    f"Relationship already exists: {entity_1.name} <-> {entity_2.name}"
                )
                continue

            # Generate edge with web search
            edge, failed_update = self.generator.generate_edge(
                entity_1_name=entity_1.name,
                entity_1_id=entity_1.id,
                entity_1_description=entity_1.description,
                entity_2_name=entity_2.name,
                entity_2_id=entity_2.id,
                entity_2_description=entity_2.description,
                verbose=False
            )

            if failed_update:
                self.failed_updates.append({
                    'entities': [entity_1.id, entity_2.id],
                    'errors': failed_update
                })
            else:
                self.relationship_id_map[edge.id] = [entity_1.id, entity_2.id]
                self.edges[edge.id] = edge
                new_edges_count += 1

        logger.info(f"Created {new_edges_count} new edges")
        return new_edges_count

    def _stage_llm_relationship_mapping(self):
        """Stage: Use LLM to generate relationship descriptions and strengths."""
        logger.info("Stage: LLM relationship mapping")

        # Filter edges that need LLM processing
        edges_to_process = [
            edge for edge in self.edges.values()
            if edge.description is None
        ]

        if not edges_to_process:
            logger.info("No edges need LLM processing")
            return

        logger.info(f"Processing {len(edges_to_process)} edges with LLM")

        # Process in batches
        batched_edges = self.generator.process_edges_in_batches(
            edges_to_process,
            batch_size=self.batch_size,
            ref_filter=True,
            verbose=False
        )

        # Process each batch with LLM
        output = []
        failed_cases = []
        model = config.get_chat_model()

        for batch in tqdm(batched_edges, desc="LLM processing batches"):
            prompt = (
                ENTITY_RELATIONSHIPS_GENERATION_JSON_PROMPT_BATCH +
                '\nInput:\n' +
                json.dumps(batch, indent=2) +
                '\nOutput:\n'
            )

            try:
                response = self.llm_client.chat.completions.create(
                    model=model,
                    messages=[
                        {
                            "role": "system",
                            "content": (
                                CONTEXT_PROMPT +
                                "Now you have finished step 1 Initial Node Creation. "
                                "You are now at step 2 Relationship Mapping. "
                                "You will evaluate a list of paired nodes to determine "
                                "the relationship between them."
                            )
                        },
                        {"role": "user", "content": prompt}
                    ],
                    temperature=config.TEMPERATURE
                )

                output_llm = parse_llm_response(response.choices[0].message.content)
                output.extend(output_llm)

            except Exception as e:
                logger.error(f"LLM processing failed for batch: {str(e)}")
                failed_cases.append(batch)

        # Update edges with LLM output
        for llm_result in output:
            for edge_id, edge in self.edges.items():
                # Match edge by entity names
                if ((edge.entity_1_name == llm_result['entity_1_name'] and
                     edge.entity_2_name == llm_result['entity_2_name']) or
                    (edge.entity_1_name == llm_result['entity_2_name'] and
                     edge.entity_2_name == llm_result['entity_1_name'])):

                    assert edge.id == llm_result['id']
                    edge.description = llm_result['relationship']['description']
                    edge.weight = llm_result['relationship']['strength']
                    break

        logger.info("LLM relationship mapping completed")

        if failed_cases:
            logger.warning(f"{len(failed_cases)} batches failed LLM processing")

    def _save_results(self):
        """Save edges and relationship mappings to JSON files."""
        logger.info("Saving results")

        # Convert edges to dictionaries
        edges_dict = {
            edge_id: edge.to_dict()
            for edge_id, edge in self.edges.items()
        }

        # Save edges
        edges_path = os.path.join(self.data_dir, "edges.json")
        with open(edges_path, 'w') as f:
            json.dump(edges_dict, f, indent=2)
        logger.info(f"Saved edges to {edges_path}")

        # Save relationship ID map
        rel_map_path = os.path.join(self.data_dir, "relationship_id_map.json")
        with open(rel_map_path, 'w') as f:
            json.dump(self.relationship_id_map, f, indent=2)
        logger.info(f"Saved relationship ID map to {rel_map_path}")

        # Save failed updates if any
        if self.failed_updates:
            failed_path = os.path.join(self.data_dir, "failed_edge_updates.json")
            with open(failed_path, 'w') as f:
                json.dump(self.failed_updates, f, indent=2)
            logger.warning(
                f"Saved {len(self.failed_updates)} failed updates to {failed_path}"
            )


def main():
    """Main entry point for command-line execution."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Knowledge Graph Edge Construction Pipeline"
    )
    parser.add_argument(
        '--data-dir',
        type=str,
        required=True,
        help='Directory containing nodes and where edges will be saved'
    )
    parser.add_argument(
        '--batch-size',
        type=int,
        default=None,
        help='Batch size for LLM processing'
    )
    parser.add_argument(
        '--stages',
        nargs='+',
        choices=['load', 'generate_pairs', 'web_search', 'llm_mapping', 'save'],
        default=None,
        help='Pipeline stages to run (default: all)'
    )
    parser.add_argument(
        '--no-load-existing',
        action='store_true',
        help='Do not load existing edges, start fresh'
    )

    args = parser.parse_args()

    # Create and run pipeline
    pipeline = EdgeConstructionPipeline(
        data_dir=args.data_dir,
        load_existing_edges=not args.no_load_existing,
        batch_size=args.batch_size
    )

    stats = pipeline.run(stages=args.stages)

    print("\n" + "=" * 50)
    print("PIPELINE STATISTICS")
    print("=" * 50)
    for key, value in stats.items():
        print(f"{key}: {value}")


if __name__ == "__main__":
    main()
