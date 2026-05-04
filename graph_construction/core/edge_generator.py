"""
Edge Generator Module

This module handles the generation of knowledge graph edges (relationships)
between entities, including web search and batch processing capabilities.
"""

import uuid
import json
import logging
from typing import Optional, List, Dict, Any, Tuple

from openai import OpenAI
from langchain_community.utilities import GoogleSerperAPIWrapper
from tqdm import tqdm

from shared.models.relationship import Relationship
from shared.utils import parse_search_results
from ..config.settings import config

# Configure logging
logger = logging.getLogger(__name__)


class EdgeGenerator:
    """
    Generates and enriches knowledge graph edges (relationships between entities).

    This class handles:
    - Web search for relationship information
    - Batch processing of multiple edges
    - Reference filtering for trusted sources
    """

    def __init__(self, verified_links_path: Optional[str] = None):
        """
        Initialize the EdgeGenerator with required API clients.

        Args:
            verified_links_path: Path to JSON file containing verified web sources.
                                Defaults to config setting if not provided.

        Raises:
            FileNotFoundError: If verified links file doesn't exist.
            ValueError: If required API keys are missing.
        """
        # Validate configuration
        config.validate()

        # Initialize API clients
        self.search = GoogleSerperAPIWrapper()

        # Initialize OpenAI client
        if config.USE_DEEPSEEK:
            self.client = OpenAI(
                api_key=config.DEEPSEEK_API_KEY,
                base_url=config.DEEPSEEK_OFFICIAL_API
            )
        else:
            self.client = OpenAI(api_key=config.OPENAI_API_KEY)

        # Load verified links for web filtering
        verified_links_path = verified_links_path or config.VERIFIED_LINKS_PATH
        try:
            with open(verified_links_path, 'r') as f:
                self.verified_links = json.load(f)
            logger.info(f"Loaded {len(self.verified_links)} verified links")
        except FileNotFoundError:
            logger.warning(f"Verified links file not found: {verified_links_path}")
            self.verified_links = []

    def generate_edge(
        self,
        edge: Optional[Relationship] = None,
        updates: Optional[List[str]] = None,
        verbose: bool = False,
        **attributes
    ) -> Tuple[Relationship, List[str]]:
        """
        Generate or update a Relationship edge with enriched attributes.

        Args:
            edge: Existing Relationship to update, or None to create new edge.
            updates: List of update operations to perform.
            verbose: Enable detailed logging output.
            **attributes: Additional relationship attributes to set including:
                        - entity_1_name: Name of first entity
                        - entity_1_id: ID of first entity
                        - entity_1_description: Description of first entity
                        - entity_2_name: Name of second entity
                        - entity_2_id: ID of second entity
                        - entity_2_description: Description of second entity

        Returns:
            Tuple containing:
                - Relationship: The generated/updated relationship
                - List[str]: List of errors encountered during processing

        Example:
            >>> generator = EdgeGenerator()
            >>> edge, errors = generator.generate_edge(
            ...     entity_1_name="Heart Rate",
            ...     entity_1_id="uuid1",
            ...     entity_2_name="Blood Pressure",
            ...     entity_2_id="uuid2"
            ... )
        """
        updates = updates or []
        failed_updates = []
        edge = edge or Relationship()

        # Define all possible edge fields
        fields = [
            'id', 'entity_1_name', 'entity_1_id', 'entity_1_description',
            'entity_2_name', 'entity_2_id', 'entity_2_description',
            'web_search_results'
        ]

        # Set attributes that are not already defined or marked for update
        for field in fields:
            if getattr(edge, field, None) is None or field in updates:
                if verbose:
                    logger.info(f"Setting {field} to {attributes.get(field)}")
                setattr(edge, field, attributes.get(field))

        # Generate UUID if not present
        if edge.id is None:
            edge.id = str(uuid.uuid4())
            if verbose:
                logger.info(f"Generated edge ID: {edge.id}")

        # Perform web search if needed
        if edge.web_search_results is None:
            edge, search_errors = self._perform_web_search(edge, verbose)
            failed_updates.extend(search_errors)

        return edge, failed_updates

    def _perform_web_search(
        self,
        edge: Relationship,
        verbose: bool
    ) -> Tuple[Relationship, List[str]]:
        """
        Perform web search to find relationship information between entities.

        Args:
            edge: Edge with entity information.
            verbose: Enable logging.

        Returns:
            Tuple of (updated edge, list of errors)
        """
        errors = []
        try:
            if verbose:
                logger.info(
                    f"Web search for relationship: "
                    f"{edge.entity_1_name} <-> {edge.entity_2_name}"
                )

            # Search for relationship
            relationship_web = self.search.results(
                f"relationship between {edge.entity_1_name} and {edge.entity_2_name}"
            )

            edge.web_search_results = relationship_web

            if verbose:
                logger.info(
                    f"Completed web search for: "
                    f"{edge.entity_1_name} <-> {edge.entity_2_name}"
                )

        except Exception as e:
            error_msg = (
                f"Web search failed for edge {edge.entity_1_name} "
                f"and {edge.entity_2_name}: {str(e)}"
            )
            logger.error(error_msg)
            errors.append(error_msg)

        return edge, errors

    def process_edges_in_batches(
        self,
        edges: List[Relationship],
        batch_size: int = None,
        ref_filter: bool = False,
        verbose: bool = False
    ) -> List[List[Dict[str, Any]]]:
        """
        Process edges in batches and prepare input data for relationship mapping.

        This method is useful for batch LLM processing of edge descriptions.

        Args:
            edges: List of Relationship objects to process.
            batch_size: Number of edges per batch. Defaults to config setting.
            ref_filter: Whether to filter web results to trusted sources.
            verbose: Enable detailed logging.

        Returns:
            List of batches, where each batch is a list of edge dictionaries
            ready for LLM processing.

        Example:
            >>> generator = EdgeGenerator()
            >>> batches = generator.process_edges_in_batches(
            ...     edges=my_edges,
            ...     batch_size=20,
            ...     ref_filter=True
            ... )
        """
        batch_size = batch_size or config.DEFAULT_BATCH_SIZE
        batched_edges = []

        # Process edges with progress bar
        with tqdm(total=len(edges), desc="Processing edges") as pbar:
            for i in range(0, len(edges), batch_size):
                batch = edges[i:i + batch_size]
                input_data = []

                for edge in batch:
                    try:
                        # Apply reference filter if requested
                        if ref_filter:
                            filtered_web_result = self._filter_web_results(edge, verbose)
                        else:
                            filtered_web_result = edge.web_search_results

                        # Parse search results
                        try:
                            parsed_relationship = parse_search_results(filtered_web_result)
                        except Exception as e:
                            logger.error(
                                f"Failed to parse search results for edge {edge.id}: {str(e)}"
                            )
                            continue

                        # Validate required fields
                        if not all([edge.entity_1_name, edge.entity_2_name]):
                            logger.warning(
                                f"Skipping edge {edge.id}: missing required entity names"
                            )
                            continue

                        # Prepare edge data for LLM
                        input_data.append({
                            "entity_1_name": edge.entity_1_name,
                            "entity_1_description": edge.entity_1_description or "",
                            "entity_2_name": edge.entity_2_name,
                            "entity_2_description": edge.entity_2_description or "",
                            "web_search_results": parsed_relationship,
                            "id": edge.id
                        })

                    except Exception as e:
                        logger.error(f"Error processing edge {edge.id}: {str(e)}")
                        continue

                # Only append batch if it contains data
                if input_data:
                    batched_edges.append(input_data)

                pbar.update(len(batch))

        logger.info(
            f"Processed {len(edges)} edges into {len(batched_edges)} batches"
        )
        return batched_edges

    def _filter_web_results(
        self,
        edge: Relationship,
        verbose: bool
    ) -> Dict[str, Any]:
        """
        Filter web search results to only include trusted sources.

        Args:
            edge: Edge with web search results.
            verbose: Enable logging.

        Returns:
            Filtered web results dictionary.
        """
        if verbose:
            logger.info(
                f"Filtering web result for "
                f"{edge.entity_1_name} and {edge.entity_2_name}"
            )

        edge_dict = edge.to_dict()
        web_results = edge_dict.get('web_search_results', {})
        organic_links = web_results.get('organic', [])

        # Filter links to trusted sources
        filtered_links = []
        for item in organic_links:
            link = item.get('link', '')

            # Extract domain
            if 'https://' in link:
                domain = link.split('https://')[1].split('/')[0]
            elif 'http://' in link:
                domain = link.split('http://')[1].split('/')[0]
            else:
                continue

            # Check if domain is trusted
            is_verified = domain in self.verified_links
            is_trusted_tld = any(
                domain.endswith(suffix)
                for suffix in config.TRUSTED_DOMAIN_SUFFIXES
            )

            if is_verified or is_trusted_tld:
                filtered_links.append(item)

        web_results['organic'] = filtered_links

        if verbose:
            logger.info(
                f"{len(filtered_links)}/{len(organic_links)} links retained"
            )

        return web_results


# Backward compatibility functions
def generate_edge(
    edge: Optional[Relationship] = None,
    updates: Optional[List[str]] = None,
    verbose: bool = False,
    **attributes
) -> Tuple[Relationship, List[str]]:
    """
    Legacy function for backward compatibility.

    Creates an EdgeGenerator instance and calls generate_edge.
    For production use, instantiate EdgeGenerator directly to reuse API clients.

    Args:
        edge: Relationship to update or None for new edge.
        updates: List of update operations.
        verbose: Enable verbose logging.
        **attributes: Relationship attributes.

    Returns:
        Tuple of (Relationship, errors list)
    """
    generator = EdgeGenerator()
    return generator.generate_edge(edge, updates, verbose, **attributes)


def process_edges_in_batches(
    edges: List[Relationship],
    batch_size: int = 20,
    ref_filter: bool = False,
    verbose: bool = False
) -> List[List[Dict[str, Any]]]:
    """
    Legacy function for backward compatibility.

    Creates an EdgeGenerator instance and calls process_edges_in_batches.

    Args:
        edges: List of Relationship objects.
        batch_size: Edges per batch.
        ref_filter: Filter to trusted sources.
        verbose: Enable verbose logging.
 
    Returns:
        List of batched edge data.
    """
    generator = EdgeGenerator()
    return generator.process_edges_in_batches(edges, batch_size, ref_filter, verbose)
