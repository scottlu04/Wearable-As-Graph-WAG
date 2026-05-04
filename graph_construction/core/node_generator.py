"""
Node Generator Module

This module handles the generation and enrichment of knowledge graph nodes (entities)
with data from multiple sources including UMLS, web search, and LLM processing.
"""

import uuid
import json
import logging
from typing import Optional, List, Dict, Any, Tuple

from openai import OpenAI
from langchain_community.utilities import GoogleSerperAPIWrapper

from shared.models.entity import Entity
from shared.utils.umls import umls
from shared.utils import parse_llm_response, parse_search_results
from shared.prompts.Entity import ENTITY_GENERATION_JSON_PROMPT_2
from shared.prompts.Context import CONTEXT_PROMPT
from ..config.settings import config

# Configure logging
logger = logging.getLogger(__name__)


class NodeGenerator:
    """
    Generates and enriches knowledge graph nodes with semantic information.

    This class orchestrates the node generation pipeline including:
    - UMLS medical ontology integration
    - Web search for contextual information
    - LLM-based description generation
    - Embedding generation for semantic similarity
    """

    def __init__(self, verified_links_path: Optional[str] = None):
        """
        Initialize the NodeGenerator with required API clients.

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
        self.umls_api = umls(apikey=config.UMLS_API_KEY)
        self.search = GoogleSerperAPIWrapper()

        # Initialize OpenAI clients
        if config.USE_DEEPSEEK:
            self.chat_client = OpenAI(
                api_key=config.get_api_key(),
                base_url=config.get_api_base_url()
            )
        else:
            self.chat_client = OpenAI(api_key=config.OPENAI_API_KEY)

        self.embed_client = OpenAI(api_key=config.OPENAI_API_KEY)

        # Load verified links for web filtering
        verified_links_path = verified_links_path or config.VERIFIED_LINKS_PATH
        try:
            with open(verified_links_path, 'r') as f:
                self.verified_links = json.load(f)
            logger.info(f"Loaded {len(self.verified_links)} verified links")
        except FileNotFoundError:
            logger.warning(f"Verified links file not found: {verified_links_path}")
            self.verified_links = []

    def generate_node(
        self,
        entity: Optional[Entity] = None,
        updates: Optional[List[str]] = None,
        verbose: bool = False,
        **attributes
    ) -> Tuple[Entity, List[str]]:
        """
        Generate or update an Entity node with enriched attributes.

        This method supports incremental updates through the 'updates' parameter,
        allowing specific processing steps to be executed selectively.

        Args:
            entity: Existing Entity to update, or None to create new entity.
            updates: List of update operations to perform. Supported operations:
                    - 'umls': Fetch UMLS ontology data
                    - 'web_search': Perform web searches for context
                    - 'ref_filter': Filter web results to trusted sources
                    - 'llm_output': Generate descriptions via LLM
                    - 'name_embedding': Generate name embedding
                    - 'semantic_embedding': Generate semantic embedding
            verbose: Enable detailed logging output.
            **attributes: Additional entity attributes to set.

        Returns:
            Tuple containing:
                - Entity: The generated/updated entity
                - List[str]: List of errors encountered during processing

        Raises:
            AssertionError: If entity name is not provided.

        Example:
            >>> generator = NodeGenerator()
            >>> entity, errors = generator.generate_node(
            ...     name="Heart Rate",
            ...     type="vital_sign",
            ...     updates=['umls', 'web_search', 'llm_output']
            ... )
        """
        updates = updates or []
        failed_updates = []
        entity = entity or Entity()

        # Define all possible entity fields
        fields = [
            'id', 'name', 'type', 'description', 'range', 'recommendation',
            'data', 'weight', 'if_data_associated',
            'semantic_embedding', 'name_embedding', 'graph_embedding',
            'provided_name', 'provided_description', 'provided_range',
            'umls_name', 'cui', 'umls_definition', 'raw_web_result'
        ]

        # Set attributes that are not already defined or marked for update
        for field in fields:
            if getattr(entity, field, None) is None or field in updates:
                if verbose:
                    logger.info(f"Setting {field} to {attributes.get(field)}")
                setattr(entity, field, attributes.get(field))

        # Validate required fields
        assert entity.name is not None, "Entity name is required"
        entity_name = entity.name

        # Generate UUID if not present
        if entity.id is None:
            entity.id = str(uuid.uuid4())
            if verbose:
                logger.info(f"Generated ID: {entity.id}")

        # UMLS Integration
        if entity.umls_name is None and 'umls' in updates:
            entity, umls_errors = self._fetch_umls_data(entity, entity_name, verbose)
            failed_updates.extend(umls_errors)

        # Web Search
        if entity.raw_web_result is None and 'web_search' in updates:
            entity, search_errors = self._perform_web_search(entity, entity_name, verbose)
            failed_updates.extend(search_errors)

        # Filter Web Results
        if 'ref_filter' in updates and entity.raw_web_result is not None:
            filtered_web_result = self._filter_web_results(entity, verbose)
        else:
            filtered_web_result = entity.raw_web_result

        # LLM Generation
        if 'llm_output' in updates:
            entity, llm_errors = self._generate_llm_output(
                entity,
                entity_name,
                filtered_web_result,
                verbose
            )
            failed_updates.extend(llm_errors)

        # Name Embedding Generation
        if entity.name_embedding is None and 'name_embedding' in updates:
            entity, embed_errors = self._generate_name_embedding(entity, entity_name, verbose)
            failed_updates.extend(embed_errors)

        # Semantic Embedding Generation
        if entity.semantic_embedding is None and 'semantic_embedding' in updates:
            entity, sem_errors = self._generate_semantic_embedding(entity, entity_name, verbose)
            failed_updates.extend(sem_errors)

        return entity, failed_updates

    def _fetch_umls_data(
        self,
        entity: Entity,
        entity_name: str,
        verbose: bool
    ) -> Tuple[Entity, List[str]]:
        """
        Fetch UMLS (Unified Medical Language System) ontology data.

        Args:
            entity: Entity to update.
            entity_name: Name to search in UMLS.
            verbose: Enable logging.

        Returns:
            Tuple of (updated entity, list of errors)
        """
        errors = []
        try:
            if verbose:
                logger.info(f"UMLS search for: {entity_name}")

            cuis = self.umls_api.search_cui(entity_name)

            if len(cuis) != 0:
                cui = cuis[0][0]
                umls_name = cuis[0][1]
                umls_definition = self.umls_api.get_definitions(cui)
            else:
                cui = None
                umls_name = None
                umls_definition = None

            entity.cui = cui
            entity.umls_name = umls_name
            entity.umls_definition = umls_definition

            if verbose:
                logger.info(f"Found UMLS CUI: {cui}")

        except Exception as e:
            error_msg = f"UMLS search failed for {entity_name}: {str(e)}"
            logger.error(error_msg)
            errors.append(error_msg)

        return entity, errors

    def _perform_web_search(
        self,
        entity: Entity,
        entity_name: str,
        verbose: bool
    ) -> Tuple[Entity, List[str]]:
        """
        Perform web searches to gather contextual information about the entity.

        Searches for:
        - Description: "What is {entity_name}?"
        - Range: "What is the range of {entity_name}?"
        - Recommendations: "How to improve {entity_name}"

        Args:
            entity: Entity to update.
            entity_name: Entity name for search queries.
            verbose: Enable logging.

        Returns:
            Tuple of (updated entity, list of errors)
        """
        errors = []
        try:
            if verbose:
                logger.info(f"Performing web search for: {entity_name}")

            # Search for description
            description_web = self.search.results(f"What is {entity_name}?")

            # Search for range
            range_web = self.search.results(f"What is the range of {entity_name}?")

            # Search for recommendations
            recommendation_web = self.search.results(f"How to improve {entity_name}")

            raw_web_result = {
                "description": description_web,
                "range": range_web,
                "recommendation": recommendation_web,
            }

            entity.raw_web_result = raw_web_result

            if verbose:
                logger.info(f"Completed web searches for: {entity_name}")

        except Exception as e:
            error_msg = f"Web search failed for {entity_name}: {str(e)}"
            logger.error(error_msg)
            errors.append(error_msg)

        return entity, errors

    def _filter_web_results(self, entity: Entity, verbose: bool) -> Dict[str, Any]:
        """
        Filter web search results to only include trusted sources.

        Filters based on:
        - Verified links list
        - Trusted domain suffixes (.edu, .org, .gov)

        Args:
            entity: Entity with raw web results.
            verbose: Enable logging.

        Returns:
            Filtered web results dictionary.
        """
        if verbose:
            logger.info(f"Filtering web results for: {entity.name}")

        filtered_web_result = {}
        entity_dict = entity.to_dict()

        # Filter each result type
        for result_type in ['description', 'range', 'recommendation']:
            result_data = entity_dict['raw_web_result'][result_type]
            organic_links = result_data.get('organic', [])

            # Filter links
            filtered_links = self._filter_links(organic_links)

            if verbose:
                logger.info(
                    f"{len(filtered_links)}/{len(organic_links)} links "
                    f"retained for {result_type}"
                )

            result_data['organic'] = filtered_links
            filtered_web_result[result_type] = result_data

        return filtered_web_result

    def _filter_links(self, links: List[Dict]) -> List[Dict]:
        """
        Filter a list of web links to trusted sources.

        Args:
            links: List of link dictionaries containing 'link' field.

        Returns:
            Filtered list of links.
        """
        filtered = []
        for item in links:
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
                filtered.append(item)

        return filtered

    def _generate_llm_output(
        self,
        entity: Entity,
        entity_name: str,
        filtered_web_result: Dict[str, Any],
        verbose: bool
    ) -> Tuple[Entity, List[str]]:
        """
        Generate entity descriptions using LLM based on gathered data.

        Args:
            entity: Entity to update.
            entity_name: Entity name.
            filtered_web_result: Filtered web search results.
            verbose: Enable logging.

        Returns:
            Tuple of (updated entity, list of errors)
        """
        errors = []
        try:
            if verbose:
                logger.info(f"Generating LLM output for: {entity_name}")

            # Parse web search results
            parsed_description = parse_search_results(filtered_web_result['description'])
            parsed_range = parse_search_results(filtered_web_result['range'])
            parsed_recommendation = parse_search_results(filtered_web_result['recommendation'])

            # Format UMLS definitions
            umls_definition = entity.to_dict().get('umls_definition', [])
            formatted_umls_definitions = ''
            if umls_definition and len(umls_definition) > 0:
                for definition in umls_definition:
                    formatted_umls_definitions += f"{definition.get('value', '')}\n"

            # Prepare input for LLM
            input_data = {
                "entity_name": entity_name,
                "provided_description": entity.provided_description,
                "provided_range": entity.provided_range,
                "web_description": parsed_description,
                "umls_description": formatted_umls_definitions,
                "value_range": parsed_range,
                "recommendations": parsed_recommendation
            }

            prompt = (
                ENTITY_GENERATION_JSON_PROMPT_2 +
                "\nInput:\n" +
                json.dumps(input_data, indent=2) +
                "\nOutput:"
            )

            # Call LLM
            model = config.get_chat_model()
            response = self.chat_client.chat.completions.create(
                model=model,
                messages=[
                    {
                        "role": "system",
                        "content": (
                            CONTEXT_PROMPT +
                            "The current task is at step 1 node generation. "
                            "You will generate one node from one metric at a time."
                        )
                    },
                    {"role": "user", "content": prompt}
                ],
                temperature=config.TEMPERATURE
            )

            # Parse LLM response
            output_llm = parse_llm_response(response.choices[0].message.content)

            # Validate response
            required_fields = ['description', 'range', 'recommendations']
            if not all(key in output_llm for key in required_fields):
                error_msg = f"LLM response missing required fields for {entity_name}"
                raise ValueError(error_msg)

            # Update entity
            entity.description = output_llm['description']
            entity.range = output_llm['range']
            entity.recommendation = output_llm['recommendations']

            if verbose:
                logger.info(f"Successfully generated LLM output for: {entity_name}")

        except Exception as e:
            error_msg = f"Error generating LLM output for {entity_name}: {str(e)}"
            logger.error(error_msg)
            errors.append(error_msg)

        return entity, errors

    def _generate_name_embedding(
        self,
        entity: Entity,
        entity_name: str,
        verbose: bool
    ) -> Tuple[Entity, List[str]]:
        """
        Generate embedding vector for entity name.

        Args:
            entity: Entity to update.
            entity_name: Entity name to embed.
            verbose: Enable logging.

        Returns:
            Tuple of (updated entity, list of errors)
        """
        errors = []
        try:
            if verbose:
                logger.info(f"Generating name embedding for: {entity_name}")

            response = self.embed_client.embeddings.create(
                model=config.EMBEDDING_MODEL,
                input=entity_name
            )

            entity.name_embedding = response.data[0].embedding

            if verbose:
                logger.info(f"Generated name embedding for: {entity_name}")

        except Exception as e:
            error_msg = f"Error generating name embedding for {entity_name}: {str(e)}"
            logger.error(error_msg)
            errors.append(error_msg)

        return entity, errors

    def _generate_semantic_embedding(
        self,
        entity: Entity,
        entity_name: str,
        verbose: bool
    ) -> Tuple[Entity, List[str]]:
        """
        Generate semantic embedding combining name, description, range, and recommendation.

        Args:
            entity: Entity to update (must have description, range, recommendation).
            entity_name: Entity name.
            verbose: Enable logging.

        Returns:
            Tuple of (updated entity, list of errors)
        """
        errors = []
        try:
            if verbose:
                logger.info(f"Generating semantic embedding for: {entity_name}")

            # Combine all semantic information
            input_text = (
                f"{entity.name}\n"
                f"{entity.description}\n"
                f"{entity.range}\n"
                f"{entity.recommendation}"
            )

            response = self.embed_client.embeddings.create(
                model=config.EMBEDDING_MODEL,
                input=input_text
            )

            entity.semantic_embedding = response.data[0].embedding

            if verbose:
                logger.info(f"Generated semantic embedding for: {entity_name}")

        except Exception as e:
            error_msg = f"Error generating semantic embedding for {entity_name}: {str(e)}"
            logger.error(error_msg)
            errors.append(error_msg)

        return entity, errors


# Backward compatibility function
def generate_node(
    entity: Optional[Entity] = None,
    updates: Optional[List[str]] = None,
    verbose: bool = False,
    **attributes
) -> Tuple[Entity, List[str]]:
    """
    Legacy function for backward compatibility.

    Creates a NodeGenerator instance and calls generate_node.
    For production use, instantiate NodeGenerator directly to reuse API clients.

    Args:
        entity: Entity to update or None for new entity.
        updates: List of update operations.
        verbose: Enable verbose logging.
        **attributes: Entity attributes.

    Returns:
        Tuple of (Entity, errors list)
    """
    generator = NodeGenerator()
    return generator.generate_node(entity, updates, verbose, **attributes)
