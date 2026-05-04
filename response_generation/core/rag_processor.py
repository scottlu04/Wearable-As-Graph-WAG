"""
RAG (Retrieval-Augmented Generation) Processor Module

This module implements the RAG query processing approach, which extends the base
processor with enhanced context building specifically designed for retrieval-augmented
generation workflows. RAGProcessor focuses on providing rich entity context with
detailed health data and relationships, optimized for LLM consumption.

Key Differences from BaseQueryProcessor:
    - Enhanced node context building with sensor-specific information
    - Detailed data formatting with markdown tables
    - Abnormality reporting in context strings
    - No weight computation (unlike WAGProcessor)
    - Optimized for LLM-based response generation

The RAGProcessor is ideal for:
    - Scenarios requiring detailed entity information
    - LLM-based question answering over health data
    - Applications where explainability through context is important
    - Experiments comparing retrieval strategies

Architecture:
    1. Inherits base entity matching and graph traversal from BaseQueryProcessor
    2. Overrides build_node_context() to provide RAG-optimized formatting
    3. Uses edges from knowledge graph without weight-based ranking
    4. Includes sensor-specific metadata and data tables in context

Usage Example:
    >>> from response_generation.core import RAGProcessor
    >>> processor = RAGProcessor(root_dir="data/kg")
    >>>
    >>> output, df = processor.query_quick(
    ...     user_query="How has my sleep been?",
    ...     par_df=user_health_data,
    ...     gt_nodes=["Sleep Duration"],
    ...     query_info={'query_date': '2024-01-15', 'time_granularity': 7, 'openness': 0.5},
    ...     response_gen=True
    ... )
    >>>
    >>> print(output['context'])
    >>> # Sleep Duration:
    >>> # description: Average hours of sleep per night
    >>> # range: 0-12 hours
    >>> # ...
    >>> # Data:
    >>> # | date       | Sleep Duration |
    >>> # |------------|----------------|
    >>> # | 2024-01-08 | 7.2            |
    >>> # ...

Context Format:
    [Entity Name]:
    description: [Natural language description]
    range: [Expected value range]
    recommendation: [Health recommendations]
    sensor specific information:
        1. description: [Dataset-specific description]
        2. range: [Dataset-specific range]
        3. unit: [Measurement unit]
    Data:
    [Markdown table with date and values]
    Abnormality: [Abnormality score if available]

"""

from typing import Dict, Tuple, Optional, Union, Any
import logging

from shared.models.entity import Entity
from response_generation.core.base_processor import BaseQueryProcessor
import pandas as pd

# Set up logging
logger = logging.getLogger(__name__)


class RAGProcessor(BaseQueryProcessor):
    """
    RAG-enhanced query processor with detailed context building.

    Extends BaseQueryProcessor with RAG-specific context formatting that includes
    comprehensive node information, sensor metadata, and abnormality scores.
    """

    def __init__(
        self,
        max_hops: int = 2,
        min_confidence: float = 0.7,
        **kwargs
    ) -> None:
        """
        Initialize RAG processor.

        Args:
            max_hops: Maximum hops for graph traversal (reserved for future use)
            min_confidence: Minimum confidence threshold (reserved for future use)
            **kwargs: Additional arguments passed to BaseQueryProcessor
        """
        super().__init__(**kwargs)
        self.max_hops = max_hops
        self.min_confidence = min_confidence

    def build_node_context(
        self,
        node: Entity,
        start_date: str,
        time_range: Union[str, int],
        par_df: pd.DataFrame,
        dataset_name: str,
        abnormality: Optional[float] = None
    ) -> Tuple[str, Dict[str, Any]]:
        """
        Build enhanced context for a node with full details.

        RAG-specific implementation that includes:
        - Node description, range, and recommendations
        - Sensor-specific information from dataset
        - Full data table in markdown format
        - Abnormality information if available

        Args:
            node: Entity node to build context for
            start_date: Query date
            time_range: Time range in days
            par_df: User's health data DataFrame
            dataset_name: Dataset name
            abnormality: Abnormality score (optional)

        Returns:
            Tuple of (context_string, data_dict)
        """
        data = {}

        if dataset_name not in node.dataSource:
            return f"data: No data\n", data

        sensor_info = node.dataSource[dataset_name]

        # Build sensor-specific information
        sensor_specific = (
            f"1. description: {sensor_info['description']}\n"
            f"2. range: {sensor_info['range']}\n"
            f"3. unit: {sensor_info['unit']}\n"
        )

        # Build main context with node information
        context = (
            f"{node.name}:\n"
            f"description: {node.description}\n"
            f"range: {node.range}\n"
            f"recommendation: {node.recommendation}\n"
            f"sensor specific information:\n{sensor_specific}"
            f"Data:\n"
        )

        # Get data and build trace
        df = self.get_data(par_df.copy(), start_date, time_range, node.name)
        if not df.empty:
            mark_data = self._format_dataframe(df)
            context += f"{mark_data}\n"

            data[node.name] = {
                "x": df['date'].tolist(),
                "y": df[node.name].tolist(),
            }

        # Add abnormality information
        if abnormality is not None:
            context += (
                f"Abnormality of recent {time_range} days compared to "
                f"individual's average: {abnormality}\n"
            )
        else:
            context += (
                f"Abnormality of recent {time_range} days compared to "
                f"individual's average: No abnormality level\n"
            )

        return context, data
