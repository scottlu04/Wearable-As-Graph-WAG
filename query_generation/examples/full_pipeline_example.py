"""
Full Pipeline Example

Demonstrates the complete query generation workflow using QueryGenerationPipeline.
"""

import sys
import logging
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from query_generation.pipelines import QueryGenerationPipeline

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def main():
    """Demonstrate full pipeline workflow."""

    # Configuration
    DATA_DIR = "experiment_1"  # Directory with nodes.json
    QA_DATAFRAME = "experiment_1/qa_df.json"  # QA dataframe path

    logger.info("=" * 60)
    logger.info("FULL PIPELINE EXAMPLE")
    logger.info("=" * 60)
    logger.info(f"Data directory: {DATA_DIR}")
    logger.info(f"QA dataframe: {QA_DATAFRAME}")
    logger.info("")

    # =========================================================================
    # Run Complete Pipeline
    # =========================================================================
    logger.info("Running complete pipeline...")
    logger.info("-" * 60)

    pipeline = QueryGenerationPipeline(
        data_dir=DATA_DIR,
        qa_dataframe_path=QA_DATAFRAME,
        batch_size=10,
        max_workers=5
    )

    # Run all stages
    stats = pipeline.run()

    # =========================================================================
    # Run Specific Stages (Alternative)
    # =========================================================================
    # You can also run specific stages:
    #
    # logger.info("Stage 1: Loading data...")
    # pipeline.run(stages=['load'])
    #
    # logger.info("Stage 2: Generating queries...")
    # pipeline.run(stages=['generate'])
    #
    # logger.info("Stage 3: Validating...")
    # pipeline.run(stages=['validate'])
    #
    # logger.info("Stage 4: Saving results...")
    # stats = pipeline.run(stages=['save'])

    # =========================================================================
    # Summary
    # =========================================================================
    logger.info("")
    logger.info("=" * 60)
    logger.info("PIPELINE COMPLETED")
    logger.info("=" * 60)
    logger.info(f"Total specifications: {stats['total_specifications']}")
    logger.info(f"Queries needed: {stats['queries_needed']}")
    logger.info(f"Queries generated: {stats['queries_generated']}")
    logger.info(f"Failed queries: {stats['failed_queries']}")
    logger.info("")

    success_rate = (
        stats['queries_generated'] / stats['queries_needed'] * 100
        if stats['queries_needed'] > 0 else 0
    )
    logger.info(f"Success rate: {success_rate:.1f}%")
    logger.info("")

    logger.info("Output files:")
    logger.info(f"  - {QA_DATAFRAME.replace('.json', '_updated.json')} (updated dataframe)")
    if stats['failed_queries'] > 0:
        logger.info(f"  - {DATA_DIR}/failed_query_generation.json (failed queries)")
    logger.info("")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        logger.warning("Pipeline interrupted by user")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Pipeline failed: {str(e)}", exc_info=True)
        sys.exit(1)
