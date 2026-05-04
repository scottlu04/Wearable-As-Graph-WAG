"""
Base Query Processor Module

This module implements the foundational query processing functionality for the WAG
(Weight-Augmented Graph) system. It provides core capabilities for entity matching,
context building, data retrieval, and LLM-based response generation.

The BaseQueryProcessor serves as the foundation for more specialized processors
(RAGProcessor, WAGProcessor) and handles:
- Loading and managing knowledge graph structure (nodes and edges)
- Entity matching using semantic similarity
- Building context with health metrics and relationships
- Extracting time-series data for matched entities
- LLM inference with retry logic
- Response generation from retrieved context

Key Components:
    BaseQueryProcessor: Main class providing query processing functionality
        - __init__(): Load graph, prior matrix, configure LLM clients
        - query(): Full pipeline with LLM entity parsing
        - query_quick(): Streamlined processing with ground truth entities
        - search_knowledge_graph(): Entity matching and relationship traversal
        - build_context(): Format retrieved information for LLM
        - get_data(): Extract time-series health data
        - llm_inference(): Make LLM calls with retry logic

Architecture:
    1. Entity Matching: Use embeddings to find relevant health metrics
    2. Graph Traversal: Follow relationships to find related entities
    3. Context Building: Format matched entities with data and relationships
    4. Response Generation: Use LLM to generate natural language response

Usage Example:
    >>> from response_generation.core import BaseQueryProcessor
    >>> processor = BaseQueryProcessor(root_dir="data/kg")
    >>>
    >>> # Process query with ground truth entities (for experiments)
    >>> output, df = processor.query_quick(
    ...     user_query="Why am I feeling tired lately?",
    ...     par_df=user_health_data,
    ...     gt_nodes=["Sleep Duration", "Energy Level"],
    ...     query_info={
    ...         'query_date': '2024-01-15',
    ...         'time_granularity': 7,
    ...         'openness': 0.5
    ...     },
    ...     response_gen=True
    ... )
    >>>
    >>> print(output['response'])
    >>> # "Based on your sleep data from the past week..."

Output Structure:
    output = {
        'result_dict': {
            'entity_name': {
                'similarity': 1.0,
                'primary_node': [Entity, abnormality_score],
                'related_nodes': [(Entity, Relationship, weight, abnormality), ...],
                'demog_nodes': [(Entity, Relationship, weight), ...]
            }
        },
        'context': "Formatted context string for LLM",
        'data': {'metric_name': {'x': [dates], 'y': [values]}, ...},
        'response': "LLM generated natural language response"
    }

Dependencies:
    - wag.shared.models: Entity and Relationship models
    - wag.shared.prompts: LLM prompt templates
    - wag.shared.utils: RAG utilities
    - retriever.wag_retriever: Weight computation utilities
    - OpenAI/DeepSeek: LLM inference

"""

from typing import List, Dict, Optional, Tuple, Any, Union
import json
import os
import logging
import time
from types import SimpleNamespace

from shared.models.entity import Entity
from shared.models.relationship import Relationship
from shared.prompts.Query import QUERY_PROMPT_BASE
from openai import OpenAI
import requests
import pandas as pd
from shared.utils.tokens import num_tokens_from_string

# Import configuration
from ..config.settings import (
    CLIENT_DEEPSEEK, CLIENT_OPENAI, CLIENT_OPENROUTER, CLIENT_GEMINI,
    SUPPORTED_CHAT_CLIENTS, SUPPORTED_EMBED_CLIENTS,
    DEEPSEEK_API_URL, OPENAI_API_URL, OPENROUTER_API_URL,
    DEFAULT_CHAT_MODELS, DEFAULT_EMBED_MODELS,
    NODES_FILE, EDGES_FILE,
    QUERY_PARSE_FUNCTION_SCHEMA,
    DEFAULT_LLM_ATTEMPTS, DEFAULT_TIMEOUT_SECONDS,
    ERROR_INVALID_CHAT_CLIENT, ERROR_INVALID_EMBED_CLIENT,
    ERROR_GRAPH_FILE_NOT_FOUND, ERROR_INVALID_JSON,
    SUCCESS_GRAPH_LOADED, SUCCESS_CLIENT_INITIALIZED,
)

# Set up logging
logger = logging.getLogger(__name__)

# Constants
DEFAULT_LLM_TEMPERATURE = 0.1
PERFECT_SIMILARITY_SCORE = 1.0


class BaseQueryProcessor:
    """
    Base class for query processing operations.

    Handles graph loading, client initialization, and core query processing.
    Thread-safe for parallel query execution.
    """

    def __init__(
        self,
        root_dir: str = "resources/kg",
        chat_client: str = CLIENT_DEEPSEEK,
        chat_model: Optional[str] = None,
        embed_client: str = CLIENT_OPENAI,
        query_prompt: str = QUERY_PROMPT_BASE,
        param: Optional[Dict[str, Any]] = None,
    ) -> None:
        """
        Initialize the query processor.

        Args:
            root_dir: Directory containing graph data files
            chat_client: Chat LLM client type ('deepseek', 'openai', or 'openrouter')
            chat_model: Optional explicit chat model override for the provider
            embed_client: Embedding client type ('openai')
            query_prompt: Prompt template for query generation
            param: Additional parameters for query processing

        Raises:
            ValueError: If invalid client types are specified
            FileNotFoundError: If graph files are not found
            json.JSONDecodeError: If graph files contain invalid JSON
        """
        self.root_dir = root_dir
        self.param = param or {}
        self.prompt = query_prompt

        # Initialize components
        self._load_graph_data()
        self._load_prior_matrix()
        self._initialize_chat_client(chat_client, chat_model=chat_model)
        self._initialize_embed_client(embed_client)
        self._setup_function_schemas()

        logger.debug(f"{self.__class__.__name__} initialized successfully")

    # =========================================================================
    # Initialization Methods
    # =========================================================================

    def _load_graph_data(self) -> None:
        """
        Load knowledge graph nodes and edges from JSON files.

        Raises:
            FileNotFoundError: If graph files are not found
            json.JSONDecodeError: If files contain invalid JSON
        """
        nodes_path = os.path.join(self.root_dir, NODES_FILE)
        edges_path = os.path.join(self.root_dir, EDGES_FILE)

        try:
            # Load and convert nodes
            self._validate_file_exists(nodes_path)
            with open(nodes_path, "r") as f:
                nodes_data = json.load(f)

            self.nodes_with_embeddings = {
                node_id: Entity.from_dict(node_dict)
                for node_id, node_dict in nodes_data.items()
            }

            # Load and convert edges
            self._validate_file_exists(edges_path)
            with open(edges_path, "r") as f:
                edges_data = json.load(f)

            self.edges = {
                edge_id: Relationship.from_dict(edge_dict)
                for edge_id, edge_dict in edges_data.items()
            }

            logger.debug(SUCCESS_GRAPH_LOADED.format(
                len(self.nodes_with_embeddings),
                len(self.edges)
            ))

        except json.JSONDecodeError as e:
            logger.error(ERROR_INVALID_JSON.format(str(e)))
            raise
        except Exception as e:
            logger.error(f"Unexpected error loading graph data: {e}")
            raise

    def _validate_file_exists(self, file_path: str) -> None:
        """Validate that a file exists, raise error if not."""
        if not os.path.exists(file_path):
            raise FileNotFoundError(ERROR_GRAPH_FILE_NOT_FOUND.format(file_path))

    def _load_prior_matrix(self) -> None:
        """Load prior knowledge matrix for relationship weights."""
        try:
            from shared.retrieval_package import get_prior_matrix
            self.prior_matrix = get_prior_matrix(self.root_dir)
            logger.debug("Prior matrix loaded successfully")
        except Exception as e:
            logger.warning(f"Could not load prior matrix: {e}")
            self.prior_matrix = None

    def _initialize_chat_client(
        self,
        chat_client: str,
        chat_model: Optional[str] = None,
    ) -> None:
        """
        Initialize the chat LLM client.

        Args:
            chat_client: Client type ('deepseek', 'openai', or 'openrouter')
            chat_model: Optional explicit model override

        Raises:
            ValueError: If invalid client type is specified
        """
        if chat_client not in SUPPORTED_CHAT_CLIENTS:
            raise ValueError(ERROR_INVALID_CHAT_CLIENT.format(SUPPORTED_CHAT_CLIENTS))

        self.chat_provider = chat_client
        self.chat_api_base_url = None

        if chat_client == CLIENT_DEEPSEEK:
            api_key = os.getenv('DEEPSEEK_API_KEY')
            self.chat_api_base_url = DEEPSEEK_API_URL
            self.chat_client = OpenAI(api_key=api_key, base_url=self.chat_api_base_url)
        elif chat_client == CLIENT_GEMINI:
            api_key = os.getenv('GEMINI_API_KEY') or os.getenv('GOOGLE_API_KEY')
            self.chat_api_key = api_key
            self.chat_client = None
        elif chat_client == CLIENT_OPENROUTER:
            api_key = os.getenv('OPENROUTER_API_KEY')
            self.chat_api_base_url = OPENROUTER_API_URL
            default_headers = {}
            http_referer = os.getenv('OPENROUTER_HTTP_REFERER')
            app_title = os.getenv('OPENROUTER_APP_TITLE')
            if http_referer:
                default_headers['HTTP-Referer'] = http_referer
            if app_title:
                default_headers['X-Title'] = app_title
            self.chat_client = OpenAI(
                api_key=api_key,
                base_url=self.chat_api_base_url,
                default_headers=default_headers or None,
            )
        else:  # CLIENT_OPENAI
            self.chat_api_base_url = OPENAI_API_URL
            self.chat_client = OpenAI()

        self.chat_model = chat_model or DEFAULT_CHAT_MODELS[chat_client]

        logger.debug(SUCCESS_CLIENT_INITIALIZED.format(chat_client, self.chat_model))

    def _initialize_embed_client(self, embed_client: str) -> None:
        """
        Initialize the embedding client.

        Args:
            embed_client: Client type ('openai')

        Raises:
            ValueError: If invalid client type is specified
        """
        if embed_client not in SUPPORTED_EMBED_CLIENTS:
            raise ValueError(ERROR_INVALID_EMBED_CLIENT.format(SUPPORTED_EMBED_CLIENTS))

        self.embed_model = DEFAULT_EMBED_MODELS[CLIENT_OPENAI]
        openai_api_key = os.getenv('OPENAI_API_KEY')

        if not openai_api_key:
            # query_quick paths in our experiments use ground-truth entities and do
            # not require embeddings, so we degrade gracefully when only a chat
            # provider such as OpenRouter is configured.
            self.embed_client = None
            logger.warning(
                "OPENAI_API_KEY is not set; embedding client was not initialized. "
                "query_quick experiments will still work, but embedding-based query parsing will not."
            )
            return

        self.embed_client = OpenAI(api_key=openai_api_key)
        logger.debug(f"Initialized embed client: {embed_client} with model {self.embed_model}")

    def _setup_function_schemas(self) -> None:
        """Set up LLM function calling schemas."""
        self.functions = [QUERY_PARSE_FUNCTION_SCHEMA]

    # =========================================================================
    # LLM Inference Methods
    # =========================================================================

    def llm_inference(
        self,
        messages: List[Dict[str, str]],
        tools: Optional[List[Dict[str, Any]]] = None,
        attempts: int = DEFAULT_LLM_ATTEMPTS,
        temperature: float = DEFAULT_LLM_TEMPERATURE
    ) -> Any:
        """
        Make LLM API call with retry logic.

        Args:
            messages: List of message dicts with role and content
            tools: List of function schemas for tool calling (None for no tools)
            attempts: Number of retry attempts
            temperature: Sampling temperature for LLM

        Returns:
            LLM API response object

        Raises:
            RuntimeError: If all attempts fail
        """
        for attempt in range(attempts):
            try:
                if self.chat_provider == CLIENT_GEMINI:
                    response = self._gemini_inference(
                        messages=messages,
                        tools=tools,
                        temperature=temperature,
                    )
                    logger.debug(f"LLM inference succeeded on attempt {attempt + 1}")
                    return response

                kwargs = {
                    "model": self.chat_model,
                    "messages": messages,
                    "temperature": temperature,
                    "timeout": DEFAULT_TIMEOUT_SECONDS,
                }

                # Add tool calling parameters if tools provided
                if tools:
                    kwargs["tools"] = tools
                    kwargs["tool_choice"] = {
                        "type": "function",
                        "function": {"name": "query_parse"}
                    }

                response = self.chat_client.chat.completions.create(**kwargs)

                if response:
                    logger.debug(f"LLM inference succeeded on attempt {attempt + 1}")
                    return response

                logger.warning(f"Attempt {attempt + 1}/{attempts} failed: Empty response")

            except Exception as e:
                logger.error(f"Attempt {attempt + 1}/{attempts} failed: {e}")
                if attempt == attempts - 1:
                    raise RuntimeError(f"LLM inference failed after {attempts} attempts: {e}")

        raise RuntimeError(f"LLM inference failed after {attempts} attempts")

    @staticmethod
    def _prepare_gemini_contents(messages: List[Dict[str, str]]) -> Dict[str, Any]:
        """Convert OpenAI-style chat messages into Gemini REST payload."""
        parts = []
        for message in messages:
            content = message.get("content", "")
            if not content:
                continue
            role = message.get("role")
            prefix = "System" if role == "system" else "User"
            parts.append({"text": f"{prefix}: {content}"})
        return {"contents": [{"parts": parts}]}

    @staticmethod
    def _extract_gemini_text(data: Dict[str, Any]) -> str:
        """Extract generated text from a Gemini REST response."""
        candidates = data.get("candidates", [])
        if not candidates:
            return ""
        content = candidates[0].get("content", {})
        chunks = []
        for part in content.get("parts", []):
            if isinstance(part, dict) and "text" in part:
                chunks.append(part["text"])
        return "".join(chunks)

    @staticmethod
    def _extract_gemini_usage(data: Dict[str, Any]) -> Dict[str, Optional[int]]:
        """Normalize Gemini usage metadata to OpenAI-style token fields."""
        usage = data.get("usageMetadata") or {}
        prompt_tokens = usage.get("promptTokenCount")
        completion_tokens = usage.get("candidatesTokenCount")
        total_tokens = usage.get("totalTokenCount")
        return {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": total_tokens,
        }

    def _gemini_inference(
        self,
        messages: List[Dict[str, str]],
        tools: Optional[List[Dict[str, Any]]] = None,
        temperature: float = DEFAULT_LLM_TEMPERATURE,
    ) -> Any:
        """Call Google Gemini/Gemma REST API and return an OpenAI-like object."""
        if tools:
            raise ValueError("Gemini query parsing tools are not supported in response generation")
        if not getattr(self, "chat_api_key", None):
            raise RuntimeError("GEMINI_API_KEY or GOOGLE_API_KEY environment variable not set")

        payload = self._prepare_gemini_contents(messages)
        payload["generationConfig"] = {"temperature": temperature}
        response = requests.post(
            f"https://generativelanguage.googleapis.com/v1beta/models/{self.chat_model}:generateContent",
            params={"key": self.chat_api_key},
            json=payload,
            timeout=DEFAULT_TIMEOUT_SECONDS,
        )
        response.raise_for_status()
        data = response.json()
        text = self._extract_gemini_text(data)
        if not text:
            raise RuntimeError("Empty Gemini response")
        return SimpleNamespace(
            choices=[
                SimpleNamespace(
                    message=SimpleNamespace(content=text, tool_calls=None)
                )
            ],
            usage=self._extract_gemini_usage(data),
        )

    # =========================================================================
    # Query Processing Methods
    # =========================================================================

    def query_quick(
        self,
        user_query: str,
        par_df: pd.DataFrame,
        gt_nodes: List[str],
        query_info: Dict[str, Any],
        **kwargs
    ) -> Tuple[Dict[str, Any], Optional[pd.DataFrame]]:
        """
        Process query with ground truth entities (for experiments).

        Thread-safe query processing with optimized data loading.

        Args:
            user_query: User's natural language query
            par_df: User's health data DataFrame
            gt_nodes: Ground truth entity names
            query_info: Query metadata (query_date, time_granularity, openness)
            **kwargs: Additional parameters (dataset_name, response_gen)

        Returns:
            Tuple of (output_dict, dataframe) where output_dict contains:
                - result_dict: Matched entities and relationships
                - context: Formatted context string
                - data: Time-series data for visualization
                - response: LLM-generated response (if response_gen=True)
                - status: Processing status
                - infeasible_reason: Reason for infeasibility, if any
                - context_tokens: Estimated token count for context only
                - estimated_prompt_tokens: Estimated prompt token count
                - input_tokens: Prompt tokens used for method comparison
                - output_tokens: Completion tokens used for method comparison
                - retrieval_time_seconds: Retrieval/context construction time
                - e2e_time_seconds: End-to-end query time
                - primary_nodes: Flattened list of primary node names
                - related_nodes: Flattened list of related node names
                - generation_called: Whether response generation was executed
        """
        e2e_start_time = time.perf_counter()
        # Extract parameters (no instance state modification for thread safety)
        dataset_name = kwargs.pop('dataset_name')

        retrieval_start_time = time.perf_counter()

        # Search knowledge graph for entities
        result_dict, out_df = self.search_knowledge_graph(
            gt_nodes,
            query_info,
            par_df=par_df,
            dataset_name=dataset_name,
            **kwargs
        )

        # Build context from retrieved entities
        context, data = self.build_context(
            result_dict,
            query_info['query_date'],
            query_info['time_granularity'],
            par_df=par_df,
            dataset_name=dataset_name
        )

        primary_nodes, related_nodes = self._extract_node_names(result_dict)
        context_tokens = num_tokens_from_string(context, model=self.chat_model)
        estimated_prompt_tokens = self._estimate_prompt_tokens(user_query, context)
        retrieval_time_seconds = time.perf_counter() - retrieval_start_time

        # Generate response if requested
        final_response = None
        generation_called = False
        generation_time_seconds = 0.0
        llm_usage = {
            'prompt_tokens': None,
            'completion_tokens': None,
            'total_tokens': None,
        }
        if kwargs.get('response_gen', False):
            generation_start_time = time.perf_counter()
            final_response, llm_usage = self._generate_response_with_usage(
                user_query,
                context
            )
            generation_time_seconds = time.perf_counter() - generation_start_time
            generation_called = True

        input_tokens = (
            llm_usage.get('prompt_tokens')
            if llm_usage.get('prompt_tokens') is not None
            else estimated_prompt_tokens
        )
        output_tokens = (
            llm_usage.get('completion_tokens')
            if llm_usage.get('completion_tokens') is not None
            else (
                num_tokens_from_string(final_response, model=self.chat_model)
                if final_response
                else 0
            )
        )
        total_tokens = (
            llm_usage.get('total_tokens')
            if llm_usage.get('total_tokens') is not None
            else (
                input_tokens + output_tokens
                if input_tokens is not None and output_tokens is not None
                else None
            )
        )
        e2e_time_seconds = time.perf_counter() - e2e_start_time

        output = {
            'result_dict': result_dict,
            'context': context,
            'data': data,
            'response': final_response,
            'status': 'success',
            'infeasible_reason': None,
            'context_tokens': context_tokens,
            'estimated_prompt_tokens': estimated_prompt_tokens,
            'input_tokens': input_tokens,
            'output_tokens': output_tokens,
            'total_tokens': total_tokens,
            'llm_prompt_tokens': llm_usage.get('prompt_tokens'),
            'llm_completion_tokens': llm_usage.get('completion_tokens'),
            'llm_total_tokens': llm_usage.get('total_tokens'),
            'retrieval_time_seconds': retrieval_time_seconds,
            'generation_time_seconds': generation_time_seconds,
            'e2e_time_seconds': e2e_time_seconds,
            'primary_nodes': primary_nodes,
            'related_nodes': related_nodes,
            'generation_called': generation_called,
        }

        return output, out_df

    def query_parse(
        self,
        user_query: str,
        **kwargs
    ) -> List[Tuple[str, float]]:
        """
        Parse user query to extract entities using LLM.

        Args:
            user_query: User's natural language query
            **kwargs: Additional parameters

        Returns:
            List of (entity_name, similarity_score) tuples

        Raises:
            ValueError: If LLM doesn't return function call
        """
        messages = [
            {"role": "system", "content": self.prompt},
            {"role": "user", "content": user_query}
        ]

        response = self.llm_inference(messages=messages, tools=self.functions)

        if not response.choices[0].message.tool_calls:
            raise ValueError("No function call in LLM response")

        tool_call = response.choices[0].message.tool_calls[0]
        args = json.loads(tool_call.function.arguments)
        logger.debug(f"LLM parsed arguments: {args}")

        # Match extracted entities to knowledge graph nodes
        matched_entities = []
        from shared.utils.rag import search_docs

        for entity in args['metrics']:
            results = search_docs(
                self.nodes_with_embeddings,
                entity.capitalize(),
                top_n=1
            )
            if results:
                node = results[0]['node']
                similarity = results[0]['similarity']
                matched_entities.append((node.name, similarity))

        return matched_entities

    def query(
        self,
        user_query: str,
        par_df: pd.DataFrame,
        query_date: str,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Full query processing pipeline with LLM entity extraction.

        Args:
            user_query: User's natural language query
            par_df: User's health data DataFrame
            query_date: Query date for context
            **kwargs: Additional parameters (must include dataset_name)

        Returns:
            Dictionary containing:
                - result_dict: Matched entities and relationships
                - context: Formatted context string
                - data: Time-series data
                - response: LLM-generated response
        """
        dataset_name = kwargs.pop('dataset_name')

        # Extract entities and query info from LLM
        prompt = self.prompt.format(today_date=query_date)
        messages = [
            {"role": "system", "content": prompt},
            {"role": "user", "content": user_query}
        ]

        response = self.llm_inference(messages=messages, tools=self.functions)

        if not response.choices[0].message.tool_calls:
            raise ValueError("No function call in LLM response")

        tool_call = response.choices[0].message.tool_calls[0]
        args = json.loads(tool_call.function.arguments)

        # Build query info from extracted parameters
        query_info = {
            'query_date': args['start_date'],
            'time_granularity': args['time_range'],
            'openness': args['openness_score']
        }

        # Search knowledge graph
        result_dict = self.search_knowledge_graph(
            args['entities'],
            query_info,
            par_df=par_df,
            dataset_name=dataset_name,
            **kwargs
        )

        # Build context
        context, data = self.build_context(
            result_dict,
            args['start_date'],
            args['time_range'],
            par_df=par_df,
            dataset_name=dataset_name
        )

        output = {
            'entities': args['entities'],
            'start_date': args['start_date'],
            'time_range': args['time_range'],
            'openness_score': args['openness_score'],
            'result_dict': result_dict,
            'context': context,
            'data': data,
            'response': None  # Response generation can be added here
        }

        return output

    def _generate_response(self, user_query: str, context: str) -> str:
        """
        Generate LLM response given query and context.

        Args:
            user_query: User's query
            context: Retrieved context

        Returns:
            Generated response string
        """
        messages = [
            {"role": "system", "content": self.prompt},
            {"role": "user", "content": f"Query: {user_query}\nContext: {context}\nAnswer:"}
        ]

        response = self.llm_inference(messages=messages, tools=None)
        return response.choices[0].message.content

    def _generate_response_with_usage(
        self,
        user_query: str,
        context: str
    ) -> Tuple[str, Dict[str, Optional[int]]]:
        """
        Generate an LLM response and return provider usage when available.

        Args:
            user_query: User's query
            context: Retrieved context

        Returns:
            Tuple of (response_text, usage_dict)
        """
        messages = [
            {"role": "system", "content": self.prompt},
            {"role": "user", "content": f"Query: {user_query}\nContext: {context}\nAnswer:"}
        ]

        response = self.llm_inference(messages=messages, tools=None)
        usage = self._extract_usage_tokens(response)
        return response.choices[0].message.content, usage

    def _extract_usage_tokens(self, response: Any) -> Dict[str, Optional[int]]:
        """
        Extract token usage from an OpenAI-compatible response object.

        Args:
            response: Chat completion response

        Returns:
            Dict with prompt/completion/total token counts when available
        """
        usage = getattr(response, 'usage', None)
        if usage is None:
            return {
                'prompt_tokens': None,
                'completion_tokens': None,
                'total_tokens': None,
            }

        if isinstance(usage, dict):
            return {
                'prompt_tokens': usage.get('prompt_tokens'),
                'completion_tokens': usage.get('completion_tokens'),
                'total_tokens': usage.get('total_tokens'),
            }

        return {
            'prompt_tokens': getattr(usage, 'prompt_tokens', None),
            'completion_tokens': getattr(usage, 'completion_tokens', None),
            'total_tokens': getattr(usage, 'total_tokens', None),
        }

    def _estimate_prompt_tokens(self, user_query: str, context: str) -> int:
        """
        Estimate the token count of the prompt sent to the chat model.

        Args:
            user_query: User's natural language query
            context: Retrieved context string

        Returns:
            Estimated prompt token count
        """
        system_tokens = num_tokens_from_string(self.prompt, model=self.chat_model)
        user_tokens = num_tokens_from_string(
            f"Query: {user_query}\nContext: {context}\nAnswer:",
            model=self.chat_model
        )
        return system_tokens + user_tokens

    def _extract_node_names(
        self,
        result_dict: Dict[str, Any]
    ) -> Tuple[List[str], List[str]]:
        """
        Extract flattened primary and related node name lists from a result dict.

        Args:
            result_dict: Query result dictionary

        Returns:
            Tuple of (primary_nodes, related_nodes)
        """
        primary_nodes: List[str] = []
        related_nodes: List[str] = []

        for entity_info in result_dict.values():
            primary_node = entity_info['primary_node'][0]
            primary_nodes.append(primary_node.name)

            for related_info in entity_info.get('related_nodes', []):
                related_node = related_info[0]
                related_nodes.append(related_node.name)

        return primary_nodes, related_nodes

    def _format_dataframe(self, df: pd.DataFrame) -> str:
        """
        Format a DataFrame for inclusion in context.

        Falls back to plain-text formatting when the optional `tabulate`
        dependency required by `DataFrame.to_markdown()` is unavailable.
        """
        try:
            return df.to_markdown(index=False)
        except ImportError:
            logger.warning(
                "tabulate is unavailable; falling back to DataFrame.to_string() "
                "for context formatting"
            )
            return df.to_string(index=False)

    # =========================================================================
    # Knowledge Graph Search Methods
    # =========================================================================

    def search_knowledge_graph(
        self,
        entities: List[str],
        query_info: Dict[str, Any],
        par_df: Optional[pd.DataFrame] = None,
        dataset_name: Optional[str] = None,
        **kwargs
    ) -> Tuple[Dict[str, Any], Optional[pd.DataFrame]]:
        """
        Search the knowledge graph for given entities.

        Args:
            entities: List of entity names to search for
            query_info: Query metadata
            par_df: User's health data DataFrame (passed through for subclasses)
            dataset_name: Dataset name (passed through for subclasses)
            **kwargs: Additional parameters

        Returns:
            Tuple of (result_dict, dataframe)
        """
        result_dict = {}

        for entity in entities:
            # Find matching node by exact name match
            matched_node = self._find_node_by_name(entity)

            if matched_node:
                result_dict[entity] = {
                    'similarity': PERFECT_SIMILARITY_SCORE,
                    'primary_node': [matched_node, None],
                    'related_nodes': [],
                    'demog_nodes': []
                }

        return result_dict, None

    def _find_node_by_name(self, entity_name: str) -> Optional[Entity]:
        """
        Find a node by exact name match.

        Args:
            entity_name: Name of entity to find

        Returns:
            Matched Entity or None if not found
        """
        for node in self.nodes_with_embeddings.values():
            if node.name == entity_name:
                return node
        return None

    # =========================================================================
    # Context Building Methods
    # =========================================================================

    def build_context(
        self,
        result_dict: Dict[str, Any],
        start_date: str,
        time_range: Union[str, int],
        par_df: pd.DataFrame,
        dataset_name: str
    ) -> Tuple[str, Dict[str, Any]]:
        """
        Build formatted context string from search results.

        Args:
            result_dict: Search results dictionary
            start_date: Query date
            time_range: Time range in days
            par_df: User's health data DataFrame
            dataset_name: Dataset name

        Returns:
            Tuple of (context_string, data_dict)
        """
        # Organize nodes by type
        primary_nodes = self._extract_primary_nodes(result_dict)
        related_nodes = self._extract_related_nodes(result_dict)
        demog_nodes = self._extract_demog_nodes(result_dict)

        # Build context sections
        primary_context, data = self._build_primary_context(
            primary_nodes, start_date, time_range, par_df, dataset_name
        )

        related_context, related_data = self._build_related_context(
            related_nodes, start_date, time_range, par_df, dataset_name
        )
        data.update(related_data)

        demog_context = self._build_demog_context(demog_nodes, par_df, dataset_name)

        # Combine contexts
        full_context = f"Matched nodes:\n\n{primary_context}"
        if related_context:
            full_context += f"\n\nNodes related to matched nodes which might be helpful:\n\n{related_context}"

        return full_context, data

    def _extract_primary_nodes(
        self,
        result_dict: Dict[str, Any]
    ) -> Dict[str, Tuple[Entity, Any]]:
        """Extract primary nodes from result dictionary."""
        primary_nodes = {}
        for entity, inner_dict in result_dict.items():
            primary_node, abnormality = inner_dict['primary_node']
            primary_nodes[primary_node.name] = (primary_node, abnormality)
        return primary_nodes

    def _extract_related_nodes(
        self,
        result_dict: Dict[str, Any]
    ) -> Dict[str, Tuple[Entity, List[Tuple]]]:
        """Extract related nodes with their edge information."""
        related_nodes = {}
        for entity, inner_dict in result_dict.items():
            primary_node = inner_dict['primary_node'][0]

            for related_node, edge, weight, abnormality in inner_dict['related_nodes']:
                edge_info = (edge, weight, abnormality, primary_node.name)

                if related_node.name not in related_nodes:
                    related_nodes[related_node.name] = (related_node, [edge_info])
                else:
                    related_nodes[related_node.name][1].append(edge_info)

        return related_nodes

    def _extract_demog_nodes(
        self,
        result_dict: Dict[str, Any]
    ) -> Dict[str, Tuple[Entity, List[Tuple]]]:
        """Extract demographic nodes with their edge information."""
        demog_nodes = {}
        for entity, inner_dict in result_dict.items():
            primary_node = inner_dict['primary_node'][0]

            for demog_node, edge, weight in inner_dict['demog_nodes']:
                edge_info = (edge, weight, primary_node.name)

                if demog_node.name not in demog_nodes:
                    demog_nodes[demog_node.name] = (demog_node, [edge_info])
                else:
                    demog_nodes[demog_node.name][1].append(edge_info)

        return demog_nodes

    def _build_primary_context(
        self,
        primary_nodes: Dict[str, Tuple[Entity, Any]],
        start_date: str,
        time_range: Union[str, int],
        par_df: pd.DataFrame,
        dataset_name: str
    ) -> Tuple[str, Dict[str, Any]]:
        """Build context for primary nodes."""
        context_parts = []
        data = {}

        for primary_node, abnormality in primary_nodes.values():
            node_context, node_data = self.build_node_context(
                primary_node, start_date, time_range, par_df, dataset_name, abnormality
            )
            context_parts.append(node_context)
            data.update(node_data)

        return '\n'.join(context_parts), data

    def _build_related_context(
        self,
        related_nodes: Dict[str, Tuple[Entity, List[Tuple]]],
        start_date: str,
        time_range: Union[str, int],
        par_df: pd.DataFrame,
        dataset_name: str
    ) -> Tuple[str, Dict[str, Any]]:
        """Build context for related nodes."""
        context_parts = []
        data = {}

        for related_node_name, (related_node, edges) in related_nodes.items():
            # Add edge contexts
            for edge, weight, abnormality, primary_node_name in edges:
                edge_context = self.build_edge_context(
                    edge, related_node_name, primary_node_name, weight
                )
                context_parts.append(edge_context)

            # Add node context
            node_context, node_data = self.build_node_context(
                related_node, start_date, time_range, par_df, dataset_name, abnormality
            )
            context_parts.append(node_context)
            data.update(node_data)

        return '\n'.join(context_parts), data

    def _build_demog_context(
        self,
        demog_nodes: Dict[str, Tuple[Entity, List[Tuple]]],
        par_df: pd.DataFrame,
        dataset_name: str
    ) -> str:
        """Build context for demographic nodes."""
        context_parts = []

        for demog_node_name, (demog_node, edges) in demog_nodes.items():
            demog_node_context = self.build_demog_node_context(demog_node, par_df, dataset_name)

            if demog_node_context:
                for edge, weight, primary_node_name in edges:
                    edge_context = self.build_edge_context(
                        edge, demog_node_name, primary_node_name, weight
                    )
                    context_parts.append(edge_context)

                context_parts.append(demog_node_context)

        return '\n'.join(context_parts)

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
        Build context string for a single node.

        Args:
            node: Entity node
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
            return f"{node.name}:\ndata: No data\n", data

        sensor_info = node.dataSource[dataset_name]

        # Build sensor information
        context = (
            f"{node.name}:\n"
            f"sensor specific information if available:\n"
            f"1. description: {sensor_info['description']}\n"
            f"2. range: {sensor_info.get('range', '')}\n"
            f"3. unit: {sensor_info.get('unit', '')}\n"
            f"Data:\n"
        )

        # Get and format data
        df = self.get_data(par_df.copy(), start_date, time_range, node.name)
        if not df.empty:
            context += f"{self._format_dataframe(df)}\n"
            data[node.name] = {
                'x': df['date'].tolist(),
                'y': df[node.name].tolist()
            }

        # Add abnormality information
        if abnormality is not None:
            context += (
                f"Recent {time_range}-day value deviates from the individual's average by "
                f"{float(abnormality):.2f} standard deviations.\n"
            )
        else:
            context += f"No deviation from baseline recorded for the recent {time_range}-day period.\n\n"

        return context, data

    def build_edge_context(
        self,
        edge: Relationship,
        related_node_name: str,
        primary_node_name: str,
        weight: float
    ) -> str:
        """
        Build context string for a relationship edge.

        Args:
            edge: Relationship object
            related_node_name: Name of related node
            primary_node_name: Name of primary node
            weight: Relationship weight

        Returns:
            Context string
        """
        return (
            f"{related_node_name} is related to {primary_node_name}:\n"
            f"{edge.description}\n"
        )

    def build_demog_node_context(
        self,
        node: Entity,
        par_df: pd.DataFrame,
        dataset_name: str
    ) -> str:
        """
        Build context for demographic node.

        Args:
            node: Demographic node
            par_df: User's health data DataFrame
            dataset_name: Dataset name

        Returns:
            Context string
        """
        if dataset_name not in node.dataSource:
            return ''

        df = self.get_demog_data(par_df.copy(), node.name)
        if df.empty:
            return ''

        sensor_info = node.dataSource[dataset_name]

        context = (
            f"{node.name}:\n"
            f"description: {node.description}\n"
            f"range: {node.range}\n"
            f"recommendation: {node.recommendation}\n"
            f"sensor specific information if available:\n"
            f"1. description: {sensor_info['description']}\n"
            f"2. range: {sensor_info.get('range', '')}\n"
            f"3. unit: {sensor_info.get('unit', '')}\n"
            f"{self._format_dataframe(df)}\n"
        )

        return context

    # =========================================================================
    # Data Retrieval Methods
    # =========================================================================

    def get_data(
        self,
        par_df: pd.DataFrame,
        start_date: str,
        time_range: Union[str, int],
        feat_name: str,
        date_column: str = "date"
    ) -> pd.DataFrame:
        """
        Extract time-series data for a feature.

        Args:
            par_df: User's health data DataFrame
            start_date: End date for data extraction
            time_range: Number of days to retrieve or 'all'
            feat_name: Feature name to extract
            date_column: Name of date column

        Returns:
            DataFrame with date and feature columns
        """
        if feat_name not in par_df.columns:
            logger.warning(f"Feature {feat_name} not found in the dataset")
            return pd.DataFrame()

        # Select relevant columns and convert dates
        selected_df = par_df[[date_column, feat_name]].copy()
        selected_df[date_column] = pd.to_datetime(selected_df[date_column])

        # Filter data up to start_date
        temp_df = selected_df[
            selected_df[date_column] <= start_date
        ].sort_values(by=date_column).copy()

        # Determine actual time range
        if time_range == 'all' or len(temp_df) < int(time_range) + 1:
            time_range = len(temp_df)

        # Format dates and select rows
        temp_df[date_column] = pd.to_datetime(temp_df[date_column]).dt.date
        selected_rows = temp_df.iloc[-int(time_range):]

        if selected_rows.empty:
            logger.warning(
                f"No data found between {start_date} and {time_range} days ago"
            )

        return selected_rows

    def get_demog_data(
        self,
        par_df: pd.DataFrame,
        feat_name: str
    ) -> pd.DataFrame:
        """
        Get latest demographic data for a feature.

        Args:
            par_df: User's health data DataFrame
            feat_name: Feature name

        Returns:
            DataFrame with latest demographic value
        """
        if feat_name not in par_df.columns:
            logger.warning(f"Feature {feat_name} not found in the dataset")
            return pd.DataFrame()

        # Get latest non-null value
        selected_df = par_df[['date', feat_name]].copy()
        filtered_df = selected_df[
            selected_df[feat_name].notnull()
        ].sort_values(by='date', ascending=False).copy()

        if len(filtered_df) > 0:
            return filtered_df.iloc[0:1]

        return pd.DataFrame()
