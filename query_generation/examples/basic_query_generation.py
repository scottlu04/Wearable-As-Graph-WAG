"""
Basic Query Generation Example

Demonstrates how to generate queries using the QueryGenerator class directly.
"""

import sys
import logging
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from query_generation.core import QueryGenerator

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def main():
    """Demonstrate basic query generation."""

    logger.info("=" * 60)
    logger.info("BASIC QUERY GENERATION EXAMPLE")
    logger.info("=" * 60)

    # Initialize generator
    generator = QueryGenerator()

    # =========================================================================
    # Example 1: Single Entity Queries
    # =========================================================================
    logger.info("\nExample 1: Single Entity Queries")
    logger.info("-" * 60)

    single_query_specs = [
        {
            "id": "q1",
            "name": "Heart Rate",
            "description": "Beats per minute at rest",
            "date": "2024-01-15",
            "time_granularity": "7",
            "abnormality_level": "low"
        },
        {
            "id": "q2",
            "name": "Sleep Duration",
            "description": "Total minutes of sleep per night",
            "date": "2024-01-15",
            "time_granularity": "30",
            "abnormality_level": "high"
        },
        {
            "id": "q3",
            "name": "Steps Taken",
            "description": "Total daily step count",
            "date": "2024-01-15",
            "time_granularity": "1",
            "abnormality_level": "medium"
        }
    ]

    queries, errors = generator.generate_single_entity_queries(
        single_query_specs,
        batch_size=10,
        verbose=True
    )

    logger.info(f"\nGenerated {len(queries)} single entity queries:")
    for query in queries:
        logger.info(f"  ID: {query['id']}")
        logger.info(f"  Question: {query['question']}")
        logger.info(f"  Type: {query['question_type']}")
        logger.info(f"  Openness: {query['openness']}")
        logger.info("")

    if errors:
        logger.warning(f"Encountered {len(errors)} errors")

    # =========================================================================
    # Example 2: Multiple Entity Queries
    # =========================================================================
    logger.info("\nExample 2: Multiple Entity Queries (Relationships)")
    logger.info("-" * 60)

    multi_query_specs = [
        {
            "id": "q4",
            "metrics": [
                {
                    "name": "Heart Rate",
                    "description": "Beats per minute at rest"
                },
                {
                    "name": "Sleep Duration",
                    "description": "Total minutes of sleep per night"
                }
            ],
            "date": "2024-01-15",
            "time_granularity": "30"
        },
        {
            "id": "q5",
            "metrics": [
                {
                    "name": "Steps Taken",
                    "description": "Total daily step count"
                },
                {
                    "name": "Energy Expenditure",
                    "description": "Calories burned per day"
                },
                {
                    "name": "Active Minutes",
                    "description": "Minutes of moderate to vigorous activity"
                }
            ],
            "date": "2024-01-15",
            "time_granularity": "14"
        }
    ]

    queries, errors = generator.generate_multiple_entity_queries(
        multi_query_specs,
        batch_size=10,
        verbose=True
    )

    logger.info(f"\nGenerated {len(queries)} multi-entity queries:")
    for query in queries:
        logger.info(f"  ID: {query['id']}")
        logger.info(f"  Question: {query['question']}")
        logger.info(f"  Type: {query['question_type']}")
        logger.info(f"  Openness: {query['openness']}")
        logger.info("")

    if errors:
        logger.warning(f"Encountered {len(errors)} errors")

    # =========================================================================
    # Summary
    # =========================================================================
    logger.info("=" * 60)
    logger.info("EXAMPLE COMPLETED")
    logger.info("=" * 60)
    logger.info(f"Total queries generated: {len(queries) + len(single_query_specs)}")
    logger.info("")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        logger.warning("Example interrupted by user")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Example failed: {str(e)}", exc_info=True)
        sys.exit(1)
