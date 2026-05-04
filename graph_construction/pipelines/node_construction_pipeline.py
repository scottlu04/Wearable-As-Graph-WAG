"""
Node Construction Pipeline

This script implements the node construction workflow from notebook 1_init_graph_node_gen.ipynb
in a more structured, production-ready format.
"""

import os
import json
import logging
from typing import Dict, List, Optional
from tqdm import tqdm

from shared.models.entity import Entity
from ..core.node_generator import NodeGenerator
from ..config.settings import config

# Configure logging
logging.basicConfig(
    level=getattr(logging, config.LOG_LEVEL),
    format=config.LOG_FORMAT
)
logger = logging.getLogger(__name__)


class NodeConstructionPipeline:
    """
    Pipeline for initial node construction from health metrics.

    This pipeline:
    1. Loads initial health metrics
    2. Creates nodes with UMLS and web search
    3. Generates LLM descriptions
    4. Creates embeddings
    5. Saves nodes and mappings
    """

    def __init__(
        self,
        output_dir: str,
        metrics_file: Optional[str] = None,
        load_existing: bool = True
    ):
        """
        Initialize the node construction pipeline.

        Args:
            output_dir: Directory to save outputs (nodes.json, node_id_map.json).
            metrics_file: Path to initial health metrics JSON file.
            load_existing: Whether to load existing nodes if available.
        """
        self.output_dir = output_dir
        self.metrics_file = metrics_file or config.INITIAL_METRICS_PATH
        self.load_existing = load_existing

        # Ensure output directory exists
        os.makedirs(self.output_dir, exist_ok=True)

        # Initialize node generator
        self.generator = NodeGenerator()

        # Storage
        self.nodes: Dict[str, Entity] = {}
        self.node_id_map: Dict[str, str] = {}
        self.failed_updates: List[Dict] = []

    def run(self, stages: Optional[List[str]] = None) -> Dict[str, any]:
        """
        Run the node construction pipeline.

        Args:
            stages: List of stages to run. If None, runs all stages.
                   Available stages:
                   - 'load': Load existing nodes or create new
                   - 'umls_web': UMLS and web search
                   - 'llm': LLM description generation
                   - 'embeddings': Generate embeddings
                   - 'save': Save results

        Returns:
            Dictionary with pipeline statistics.
        """
        all_stages = ['load', 'umls_web', 'llm', 'embeddings', 'save']
        stages = stages or all_stages

        logger.info("Starting node construction pipeline")
        logger.info(f"Output directory: {self.output_dir}")
        logger.info(f"Stages to run: {stages}")

        stats = {
            'total_nodes': 0,
            'new_nodes': 0,
            'failed_updates': 0
        }

        # Stage 1: Load or initialize
        if 'load' in stages:
            self._load_or_initialize()

        # Stage 2: UMLS and web search
        if 'umls_web' in stages:
            stats['new_nodes'] = self._stage_umls_web_search()

        # Stage 3: LLM description generation
        if 'llm' in stages:
            self._stage_llm_generation()

        # Stage 4: Generate embeddings
        if 'embeddings' in stages:
            self._stage_embeddings()

        # Stage 5: Save results
        if 'save' in stages:
            self._save_results()

        stats['total_nodes'] = len(self.nodes)
        stats['failed_updates'] = len(self.failed_updates)

        logger.info("Pipeline completed successfully")
        logger.info(f"Statistics: {stats}")

        return stats

    def _load_or_initialize(self):
        """Load existing nodes or initialize from metrics file."""
        nodes_path = os.path.join(self.output_dir, "nodes.json")
        id_map_path = os.path.join(self.output_dir, "node_id_map.json")

        # Try to load existing nodes
        if self.load_existing and os.path.exists(nodes_path) and os.path.exists(id_map_path):
            logger.info(f"Loading existing nodes from {self.output_dir}")
            with open(nodes_path, 'r') as f:
                nodes_data = json.load(f)
                for node_id, node_dict in nodes_data.items():
                    self.nodes[node_id] = Entity.from_dict(node_dict)

            with open(id_map_path, 'r') as f:
                self.node_id_map = json.load(f)

            logger.info(f"Loaded {len(self.nodes)} existing nodes")
        else:
            logger.info("Initializing new node set")
            self.nodes = {}
            self.node_id_map = {}

    def _stage_umls_web_search(self) -> int:
        """
        Stage 1: UMLS and web search for all metrics.

        Returns:
            Number of new nodes created.
        """
        logger.info("Stage: UMLS and web search")

        # Load initial metrics
        with open(self.metrics_file, 'r') as f:
            metrics = json.load(f)

        # Count total metrics
        total_count = sum(len(entities) for entities in metrics.values())
        logger.info(f"Processing {total_count} metrics")

        new_nodes_count = 0

        # Process each metric
        for metric_type, entity_names in metrics.items():
            for entity_name in tqdm(entity_names, desc=f"Processing {metric_type}"):
                # Skip if already exists
                if entity_name in self.node_id_map.values():
                    logger.debug(f"Entity {entity_name} already exists, skipping")
                    continue

                # Generate node with UMLS and web search
                new_node, failed_update = self.generator.generate_node(
                    updates=['umls', 'web_search'],
                    verbose=False,
                    name=entity_name,
                    type=metric_type
                )

                if failed_update:
                    self.failed_updates.append({
                        'node_id': new_node.id,
                        'name': entity_name,
                        'errors': failed_update
                    })
                    logger.warning(f"Failed updates for {entity_name}: {failed_update}")

                self.nodes[new_node.id] = new_node
                self.node_id_map[new_node.id] = entity_name
                new_nodes_count += 1

        logger.info(f"Created {new_nodes_count} new nodes")
        return new_nodes_count

    def _stage_llm_generation(self):
        """Stage 2: LLM description generation for all nodes."""
        logger.info("Stage: LLM description generation")

        for node_id in tqdm(self.nodes, desc="Generating LLM descriptions"):
            node = self.nodes[node_id]

            # Generate LLM output
            updated_node, failed_update = self.generator.generate_node(
                node,
                updates=['ref_filter', 'llm_output'],
                verbose=False
            )

            if failed_update:
                self.failed_updates.append({
                    'node_id': node_id,
                    'name': node.name,
                    'errors': failed_update
                })
                logger.warning(f"Failed LLM generation for {node.name}: {failed_update}")

            self.nodes[node_id] = updated_node

        logger.info("LLM description generation completed")

    def _stage_embeddings(self):
        """Stage 3: Generate embeddings for all nodes."""
        logger.info("Stage: Embedding generation")

        for node_id in tqdm(self.nodes, desc="Generating embeddings"):
            node = self.nodes[node_id]

            # Generate embeddings
            updated_node, failed_update = self.generator.generate_node(
                node,
                updates=['name_embedding', 'semantic_embedding'],
                verbose=False
            )

            if failed_update:
                self.failed_updates.append({
                    'node_id': node_id,
                    'name': node.name,
                    'errors': failed_update
                })
                logger.warning(f"Failed embedding generation for {node.name}: {failed_update}")

            self.nodes[node_id] = updated_node

        logger.info("Embedding generation completed")

    def _save_results(self):
        """Save nodes and mappings to JSON files."""
        logger.info("Saving results")

        # Convert nodes to dictionaries
        nodes_dict = {
            node_id: node.to_dict()
            for node_id, node in self.nodes.items()
        }

        # Save nodes
        nodes_path = os.path.join(self.output_dir, "nodes.json")
        with open(nodes_path, 'w') as f:
            json.dump(nodes_dict, f, indent=2)
        logger.info(f"Saved nodes to {nodes_path}")

        # Save ID map
        id_map_path = os.path.join(self.output_dir, "node_id_map.json")
        with open(id_map_path, 'w') as f:
            json.dump(self.node_id_map, f, indent=2)
        logger.info(f"Saved node ID map to {id_map_path}")

        # Save failed updates if any
        if self.failed_updates:
            failed_path = os.path.join(self.output_dir, "failed_node_updates.json")
            with open(failed_path, 'w') as f:
                json.dump(self.failed_updates, f, indent=2)
            logger.warning(f"Saved {len(self.failed_updates)} failed updates to {failed_path}")


def main():
    """Main entry point for command-line execution."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Knowledge Graph Node Construction Pipeline"
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        required=True,
        help='Output directory for nodes and mappings'
    )
    parser.add_argument(
        '--metrics-file',
        type=str,
        default=None,
        help='Path to initial health metrics JSON file'
    )
    parser.add_argument(
        '--stages',
        nargs='+',
        choices=['load', 'umls_web', 'llm', 'embeddings', 'save'],
        default=None,
        help='Pipeline stages to run (default: all)'
    )
    parser.add_argument(
        '--no-load-existing',
        action='store_true',
        help='Do not load existing nodes, start fresh'
    )

    args = parser.parse_args()

    # Create and run pipeline
    pipeline = NodeConstructionPipeline(
        output_dir=args.output_dir,
        metrics_file=args.metrics_file,
        load_existing=not args.no_load_existing
    )

    stats = pipeline.run(stages=args.stages)

    print("\n" + "=" * 50)
    print("PIPELINE STATISTICS")
    print("=" * 50)
    for key, value in stats.items():
        print(f"{key}: {value}")


if __name__ == "__main__":
    main()
