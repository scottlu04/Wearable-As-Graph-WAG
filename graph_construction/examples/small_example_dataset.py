"""
Dataset Integration Example

This example demonstrates how to integrate new metrics from a dataset
into an existing knowledge graph.
"""

import os
import sys
import logging
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from graph_construction.pipelines import DatasetIntegrationPipeline

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def main():
    """Run dataset integration workflow."""

    # Configuration
    DATA_DIR = "./small_example"  # Directory with existing graph
    FEATURES_MAP = "resources/features_map_dict_small.json"
    SIMILARITY_THRESHOLD = 0.7  # Threshold for considering nodes as duplicates

    logger.info("=" * 60)
    logger.info("DATASET INTEGRATION WORKFLOW")
    logger.info("=" * 60)
    logger.info(f"Data directory: {DATA_DIR}")
    logger.info(f"Features map: {FEATURES_MAP}")
    logger.info(f"Similarity threshold: {SIMILARITY_THRESHOLD}")
    logger.info("")

    # =========================================================================
    # Dataset Integration
    # =========================================================================
    logger.info("Integrating new dataset metrics...")
    logger.info("-" * 60)

    pipeline = DatasetIntegrationPipeline(
        data_dir=DATA_DIR,
        features_map_file=FEATURES_MAP,
        batch_size=10,
        similarity_threshold=SIMILARITY_THRESHOLD
    )

    # OPTION 1: Run all stages at once (fully automatic, no manual review)
    # stats = pipeline.run()

    # OPTION 2: Run with manual review
    # Step 1: Run comparison and generate review file
    # pipeline.run(stages=['load', 'compare', 'review_merges'])
    #
    # Step 2: Manually edit the merge_review.json file in DATA_DIR
    # - Review all proposed merges
    # - Set 'approved': false to reject any merge
    # - Copy a different candidate to 'selected_merge' to choose it
    #
    # Step 3: Continue with reviewed decisions (include 'load' to reload existing nodes)
    stats = pipeline.run(stages=['load', 'review_merges', 'merge', 'create_new', 'create_edges', 'save'])

    # OPTION 3: Run specific stages individually
    # logger.info("Stage 1: Loading data...")
    # pipeline.run(stages=['load'])

    # logger.info("Stage 2: Comparing metrics...")
    # pipeline.run(stages=['compare'])

    # logger.info("Stage 3: Preparing merge review...")
    # pipeline.run(stages=['review_merges'])

    # logger.info("Stage 4: Merging nodes...")
    # pipeline.run(stages=['merge'])

    # logger.info("Stage 5: Creating new nodes...")
    # pipeline.run(stages=['create_new'])

    # logger.info("Stage 6: Creating edges...")
    # pipeline.run(stages=['create_edges'])

    # logger.info("Stage 7: Saving results...")
    # stats = pipeline.run(stages=['save'])

    # =========================================================================
    # Summary
    # =========================================================================
    logger.info("")
    logger.info("=" * 60)
    logger.info("INTEGRATION COMPLETED")
    logger.info("=" * 60)
    logger.info(f"Existing nodes: {stats['existing_nodes']}")
    logger.info(f"New metrics processed: {stats['new_metrics']}")
    logger.info(f"Nodes merged: {stats['merged_nodes']}")
    logger.info(f"New nodes created: {stats['new_nodes_created']}")
    logger.info(f"New edges created: {stats['new_edges_created']}")
    logger.info(f"Failed updates: {stats['failed_updates']}")
    logger.info("")

    if stats['merged_nodes'] > 0:
        logger.info(f"✅ {stats['merged_nodes']} metrics were merged into existing nodes")
        logger.info("   (existing nodes updated with new dataset information)")

    if stats['new_nodes_created'] > 0:
        logger.info(f"✅ {stats['new_nodes_created']} unique metrics created as new nodes")

    logger.info("")
    logger.info("Output files:")
    logger.info(f"  - {DATA_DIR}/nodes.json (updated)")
    logger.info(f"  - {DATA_DIR}/edges.json (updated)")
    logger.info(f"  - {DATA_DIR}/merge_decisions.json (LLM comparison results)")
    logger.info(f"  - {DATA_DIR}/merge_review.json (manual review file)")
    logger.info(f"  - {DATA_DIR}/non_merge_list.json (metrics that created new nodes)")
    logger.info(f"  - {DATA_DIR}/successful_merges.json (tracking for reruns)")
    logger.info(f"  - {DATA_DIR}/successful_new_nodes.json (tracking for reruns)")
    logger.info(f"  - {DATA_DIR}/successful_edges.json (tracking for reruns)")

    if stats['failed_updates'] > 0:
        logger.warning(f"  - {DATA_DIR}/failed_integration_updates.json (check for errors)")

    # =========================================================================
    # What Happened?
    # =========================================================================
    logger.info("")
    logger.info("=" * 60)
    logger.info("WHAT HAPPENED?")
    logger.info("=" * 60)
    logger.info("""
1. COMPARISON PHASE:
   - Each new metric was compared against ALL existing nodes using LLM
   - LLM evaluated semantic similarity and medical concept matching
   - Similarity scores and reasoning were recorded in merge_decisions.json

2. REVIEW PHASE (OPTIONAL):
   - merge_review.json created with all potential merge candidates
   - Shows candidates where same_concept=True (LLM thinks same concept)
   - Shows candidates where same_concept=False (similar but different)
   - Human can review and override LLM decisions if needed
   - Default: best match where same_concept=True

3. MERGE PHASE:
   - Metrics merged into existing nodes based on approved decisions
   - Existing nodes enriched with new dataset information
   - dataSource field updated with dataset-specific metadata
   - Only ONE merge allowed per feature name (validated before merging)

4. NEW NODE CREATION:
   - Metrics without matching existing nodes created as new nodes
   - Full node generation pipeline (UMLS, web search, LLM, embeddings)
   - dataSource field populated with dataset information

5. EDGE CREATION:
   - New nodes connected to existing graph
   - Edges created between new nodes and all existing nodes
   - Relationship strength evaluated via LLM
    """)

    logger.info("")
    logger.info("Next steps:")
    logger.info("1. Review merge_review.json to see all merge candidates and decisions")
    logger.info("2. Check merge_decisions.json to see raw LLM comparison results")
    logger.info("3. Check non_merge_list.json to see which metrics became new nodes")
    logger.info("4. Inspect nodes with dataSource field to see dataset integration")
    logger.info("")
    logger.info("To manually review merges:")
    logger.info("1. Run with stages=['load', 'compare', 'review_merges']")
    logger.info("2. Edit merge_review.json to approve/reject/change merge decisions")
    logger.info("3. Run with stages=['load', 'review_merges', 'merge', 'create_new', 'create_edges', 'save']")
    logger.info("")

    if stats['failed_updates'] > 0:
        logger.info("=" * 60)
        logger.info("HANDLING FAILURES")
        logger.info("=" * 60)
        logger.info(f"⚠️  {stats['failed_updates']} operations failed")
        logger.info("")
        logger.info("The pipeline is IDEMPOTENT - you can safely rerun it:")
        logger.info("1. Fix any issues (API keys, network, etc.)")
        logger.info("2. Simply rerun the same pipeline command")
        logger.info("3. Already-successful operations will be skipped")
        logger.info("4. Only failures will be retried")
        logger.info("")
        logger.info("Tracking files ensure no duplicates:")
        logger.info("  - successful_merges.json: Already merged metrics")
        logger.info("  - successful_new_nodes.json: Already created nodes")
        logger.info("  - successful_edges.json: Already created edges")
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
