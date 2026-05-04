"""
Complete Knowledge Graph Construction Workflow

This example demonstrates the full pipeline from initial metrics to complete graph.
"""

import os
import sys
import logging
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from graph_construction.pipelines import (
    NodeConstructionPipeline,
    EdgeConstructionPipeline
)
from graph_construction.config import config

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def main():
    """Run complete knowledge graph construction workflow."""

    # Configuration
    EXPERIMENT_NAME = "small_example"
    OUTPUT_DIR = f"./{EXPERIMENT_NAME}"

    logger.info("=" * 60)
    logger.info("KNOWLEDGE GRAPH CONSTRUCTION WORKFLOW")
    logger.info("=" * 60)
    logger.info(f"Experiment: {EXPERIMENT_NAME}")
    logger.info(f"Output directory: {OUTPUT_DIR}")
    logger.info("")

    # =========================================================================
    # PHASE 1: Node Construction
    # =========================================================================
    # logger.info("PHASE 1: Node Construction")
    # logger.info("-" * 60)

    # node_pipeline = NodeConstructionPipeline(
    #     output_dir=OUTPUT_DIR,
    #     metrics_file=config.INITIAL_METRICS_PATH,
    #     load_existing=True
    # )

    # # Run node construction in stages
    # logger.info("Stage 1: Loading/Initializing...")
    # node_pipeline.run(stages=['load'])
    

    # logger.info("Stage 2: UMLS and Web Search...")
    # node_pipeline.run(stages=['umls_web'])
    
    # logger.info("Stage 3: LLM Description Generation...")
    # node_pipeline.run(stages=['llm'])

    # # logger.info("Stage 4: Embedding Generation...")
    # # node_pipeline.run(stages=['embeddings'])

    # logger.info("Stage 5: Saving Nodes...")
    # node_stats = node_pipeline.run(stages=['save'])

    # logger.info("")
    # logger.info("Node Construction Results:")
    # logger.info(f"  Total nodes: {node_stats['total_nodes']}")
    # logger.info(f"  New nodes: {node_stats['new_nodes']}")
    # logger.info(f"  Failed updates: {node_stats['failed_updates']}")
    # logger.info("")

    # =========================================================================
    # PHASE 2: Edge Construction
    # =========================================================================
    logger.info("PHASE 2: Edge Construction")
    logger.info("-" * 60)

    edge_pipeline = EdgeConstructionPipeline(
        data_dir=OUTPUT_DIR,
        load_existing_edges=True,
        batch_size=10
    )

    # Run edge construction in stages
    logger.info("Stage 1: Loading Data...")
    edge_pipeline.run(stages=['load'])

    logger.info("Stage 2: Generating Entity Pairs...")
    edge_pipeline.run(stages=['generate_pairs'])

    logger.info("Stage 3: Web Search for Relationships...")
    edge_pipeline.run(stages=['web_search'])

    logger.info("Stage 4: LLM Relationship Mapping...")
    edge_pipeline.run(stages=['llm_mapping'])

    logger.info("Stage 5: Saving Edges...")
    edge_stats = edge_pipeline.run(stages=['save'])

    logger.info("")
    logger.info("Edge Construction Results:")
    logger.info(f"  Total nodes: {edge_stats['total_nodes']}")
    logger.info(f"  Total edges: {edge_stats['total_edges']}")
    logger.info(f"  New edges: {edge_stats['new_edges']}")
    logger.info(f"  Failed updates: {edge_stats['failed_updates']}")
    logger.info("")

    # =========================================================================
    # Summary
    # =========================================================================
    logger.info("=" * 60)
    logger.info("WORKFLOW COMPLETED SUCCESSFULLY")
    logger.info("=" * 60)
    logger.info(f"Knowledge Graph Statistics:")
    logger.info(f"  Nodes: {node_stats['total_nodes']}")
    logger.info(f"  Edges: {edge_stats['total_edges']}")
    logger.info(f"  Density: {edge_stats['total_edges'] / (node_stats['total_nodes'] * (node_stats['total_nodes'] - 1) / 2):.2%}")
    logger.info("")
    logger.info(f"Output files saved to: {OUTPUT_DIR}/")
    logger.info("  - nodes.json")
    logger.info("  - node_id_map.json")
    logger.info("  - edges.json")
    logger.info("  - relationship_id_map.json")
    logger.info("")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        logger.warning("Workflow interrupted by user")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Workflow failed: {str(e)}", exc_info=True)
        sys.exit(1)
