"""
WAG (Weight-Augmented Graph) Processor Module

This module implements the Weight-Augmented Graph (WAG) query processing approach,
which extends the base processor with sophisticated weight computation and abnormality
detection for personalized health data analysis.

WAG combines three knowledge sources to rank relationships in the knowledge graph:
    1. LLM Prior Knowledge: General medical knowledge from large language models
    2. Population Correlations: Statistical relationships from cohort studies
    3. Individual Patterns: Personal health metric correlations

The WAGProcessor is the most advanced query processor in the system, providing:
    - Bayesian weight computation combining prior, population, and individual evidence
    - Abnormality detection comparing recent vs. baseline patterns
    - Dynamic related node selection based on query openness
    - Multiple weight sorting strategies (prior, pop, ind, global, local, final)
    - Support for demographic nodes (age, gender, etc.)

Key Weight Types:
    - weight_prior: LLM prior knowledge only
    - weight_pop: Population-level correlations only
    - weight_ind: Individual-level correlations only
    - weight_global: Bayesian posterior (prior + pop + ind)
    - weight_local: Local abnormality-based weights
    - weight_final: Combined global and local (β*local + (1-β)*global)

Algorithm Overview:
    1. Match primary entities from query
    2. For each primary entity:
        a. Run weight_recomputation() to compute relationship weights
        b. Calculate abnormality score for the primary entity
        c. Select top-N related nodes based on weight_sort_by strategy
        d. Add demographic nodes if specified
    3. Build context with weighted relationships and abnormality information
    4. Generate LLM response using enriched context

Usage Example:
    >>> from response_generation.core import WAGProcessor
    >>>
    >>> # Define hyperparameters
    >>> hyperparameters = {
    ...     'rel_type': 'spearman',
    ...     't_global': 0.9,  # Threshold for global weights
    ...     't_local': 0.7,   # Threshold for local weights
    ...     'alpha_prior': 1.0,  # LLM prior confidence
    ...     'alpha_pop': 8.71,   # Population evidence weight (10^0.94)
    ...     'alpha_ind': 0.049,  # Individual evidence weight (10^-1.31)
    ...     'beta': 0.5,  # Blend factor for local/global combination
    ...     'max_num_related_nodes': 5,
    ...     'weight_sort_by': 'weight_final',  # Which weight to use for ranking
    ...     'demog_metrics': ['Age', 'Gender', 'BMI']
    ... }
    >>>
    >>> processor = WAGProcessor(root_dir="data/kg", param=hyperparameters)
    >>>
    >>> output, df = processor.query_quick(
    ...     user_query="Why am I feeling tired?",
    ...     par_df=user_health_data,
    ...     gt_nodes=["Energy Level", "Sleep Duration"],
    ...     query_info={
    ...         'query_date': '2024-01-15',
    ...         'time_granularity': 7,
    ...         'openness': 0.5  # 0-1, controls number of related nodes
    ...     },
    ...     rel_info=relationship_matrices,
    ...     response_gen=True
    ... )
    >>>
    >>> # df contains weight computations for all relationships
    >>> print(df[['target_node', 'weight_final', 'abnormality']].head())
    >>>
    >>> # output contains primary nodes, related nodes with weights, and response
    >>> print(output['response'])

Weight Computation Details:
    The weight_recomputation() method calls the wag_retriever which implements:
    - Bayesian updating: P(relationship | evidence) ∝ P(evidence | relationship) * P(relationship)
    - Alpha parameters control confidence in each evidence source
    - T thresholds filter weak relationships
    - Beta blends global knowledge with local abnormality signals

Abnormality Detection:
    For each entity, compares recent values (time_granularity period) to baseline:
    - Baseline: Individual's historical average
    - Recent: Average over recent time window
    - Abnormality score: Standardized deviation from baseline
    - Used to prioritize entities showing unusual patterns

Dynamic Related Node Selection:
    Number of related nodes = openness * max_num_related_nodes
    - openness=0.0: No related nodes (closed question)
    - openness=0.5: Half of max related nodes
    - openness=1.0: All available related nodes (open question)

Experiment Configurations:
    Different experiments vary the weight_sort_by parameter:
    - Global experiments: Compare prior vs pop vs ind vs global
    - Local experiments: Compare local vs global vs final
    - General experiments: Compare Base vs RAG vs StaticGraph vs FullContext vs WAG

References:
    [1] WAG Paper: Weight-Augmented Graph for Health Query Processing
    [2] Bayesian Network Meta-Analysis for Relationship Confidence

"""

from typing import List, Dict, Optional, Tuple, Any, Union
import logging

from shared.models.entity import Entity
from response_generation.core.base_processor import BaseQueryProcessor
from shared.prompts.Query import QUERY_PROMPT_KGRAG_v3
import pandas as pd

# Set up logging
logger = logging.getLogger(__name__)

# Constants
PERFECT_SIMILARITY_SCORE = 1.0


class WAGProcessor(BaseQueryProcessor):
    """
    Weight-Augmented Graph processor with Bayesian weight computation.

    Extends BaseQueryProcessor with sophisticated weight-based relationship ranking
    combining LLM priors, population statistics, and individual patterns.
    """

    def __init__(
        self,
        max_hops: int = 2,
        min_confidence: float = 0.7,
        **kwargs
    ) -> None:
        """
        Initialize WAG processor.

        Args:
            max_hops: Maximum hops for graph traversal (reserved for future use)
            min_confidence: Minimum confidence threshold (reserved for future use)
            **kwargs: Additional arguments passed to BaseQueryProcessor,
                     must include 'param' dict with WAG hyperparameters
        """
        super().__init__(
            query_prompt=QUERY_PROMPT_KGRAG_v3,
            **kwargs
        )
        self.max_hops = max_hops
        self.min_confidence = min_confidence

    # =========================================================================
    # Knowledge Graph Search with Weight Computation
    # =========================================================================

    def search_knowledge_graph(
        self,
        entities: List[str],
        query_info: Dict[str, Any],
        par_df: Optional[pd.DataFrame] = None,
        dataset_name: Optional[str] = None,
        **kwargs
    ) -> Tuple[Dict[str, Any], pd.DataFrame]:
        """
        Search knowledge graph with weight-based relationship ranking.

        Extends base search with:
        - Weight computation for relationships
        - Abnormality detection for nodes
        - Dynamic related node selection based on openness
        - Demographic node inclusion

        Args:
            entities: List of entity names to search
            query_info: Query metadata including openness score
            par_df: User's health data DataFrame
            dataset_name: Dataset name (not used in this method but part of signature)
            **kwargs: Must include 'rel_info' with relationship matrices

        Returns:
            Tuple of (result_dict, weights_dataframe) where result_dict contains:
                - similarity: Match confidence score
                - primary_node: [Entity, abnormality_score]
                - related_nodes: [(Entity, Relationship, weight, abnormality), ...]
                - demog_nodes: [(Entity, Relationship, weight), ...]
        """
        # Build node name mapping for fast lookup
        node_name_map = {
            node.name: node
            for node in self.nodes_with_embeddings.values()
        }

        # Find primary entities
        result_dict = self._match_primary_entities(entities, node_name_map)

        # Calculate number of related nodes based on openness
        num_related_nodes = int(
            query_info['openness'] * self.param['max_num_related_nodes']
        )
        # num_related_nodes_per_node = max(1, num_related_nodes // len(result_dict))
        num_related_nodes_per_node = num_related_nodes // len(result_dict)
        # Get relationship info
        rel_info = kwargs['rel_info']

        # Process each primary entity
        for entity, inner_dict in result_dict.items():
            primary_node = inner_dict['primary_node'][0]

            # Compute weights and abnormalities
            out_df, abnormality_df = self.weight_recomputation(
                primary_node.name,
                query_info,
                rel_info,
                self.prior_matrix,
                par_df.copy(),
                self.param
            )

            # Update primary node abnormality
            inner_dict['primary_node'][1] = abnormality_df.loc[
                primary_node.name, 'abnormality'
            ]

            # Select top related nodes by weight
            top_metrics = self._select_top_related_nodes(
                out_df,
                self.param['weight_sort_by'],
                num_related_nodes_per_node
            )

            # Build edge mapping for this node
            edge_dict = self._build_edge_dict(primary_node.name)

            # Add related nodes with weights and abnormalities
            self._add_related_nodes(
                result_dict[entity],
                top_metrics,
                node_name_map,
                edge_dict,
                abnormality_df
            )

            # Add demographic nodes if specified
            self._add_demographic_nodes(
                result_dict[entity],
                node_name_map,
                edge_dict
            )

        return result_dict, out_df

    def _match_primary_entities(
        self,
        entities: List[str],
        node_name_map: Dict[str, Entity]
    ) -> Dict[str, Dict[str, Any]]:
        """
        Match entities to nodes in the knowledge graph.

        Args:
            entities: Entity names to match
            node_name_map: Mapping of node names to Entity objects

        Returns:
            Dictionary with matched entity information
        """
        result_dict = {}

        for entity in entities:
            # Find node by exact name match
            if entity in node_name_map:
                node = node_name_map[entity]

                if entity not in result_dict:
                    result_dict[entity] = {
                        'similarity': PERFECT_SIMILARITY_SCORE,
                        'primary_node': [node, None],
                        'related_nodes': [],
                        'demog_nodes': []
                    }
                else:
                    # Update if higher similarity (always 1.0 for exact match)
                    if PERFECT_SIMILARITY_SCORE > result_dict[entity]['similarity']:
                        result_dict[entity]['similarity'] = PERFECT_SIMILARITY_SCORE
                        result_dict[entity]['primary_node'] = [node, None]

        return result_dict

    def _select_top_related_nodes(
        self,
        out_df: pd.DataFrame,
        weight_sort_by: str,
        num_nodes: int
    ) -> pd.DataFrame:
        """
        Select top N related nodes by weight.

        Args:
            out_df: DataFrame with computed weights
            weight_sort_by: Column name to sort by (e.g., 'weight_final')
            num_nodes: Number of nodes to select

        Returns:
            DataFrame with top N nodes
        """
        return (
            out_df
            .reset_index()
            .sort_values(
                by=[weight_sort_by, 'index'],
                ascending=[False, True]
            )
            .head(num_nodes)
            .set_index('index')
        )

    def _build_edge_dict(self, node_name: str) -> Dict[str, Any]:
        """
        Build dictionary mapping related node names to edges.

        Args:
            node_name: Name of primary node

        Returns:
            Dictionary mapping related node names to Relationship objects
        """
        return {
            (edge.entity_2_name if edge.entity_1_name == node_name
             else edge.entity_1_name): edge
            for edge in self.edges.values()
            if node_name in (edge.entity_1_name, edge.entity_2_name)
        }

    def _add_related_nodes(
        self,
        entity_dict: Dict[str, Any],
        top_metrics: pd.DataFrame,
        node_name_map: Dict[str, Entity],
        edge_dict: Dict[str, Any],
        abnormality_df: pd.DataFrame
    ) -> None:
        """
        Add related nodes to entity dictionary.

        Args:
            entity_dict: Entity result dictionary to update
            top_metrics: DataFrame with top related metrics
            node_name_map: Node name to Entity mapping
            edge_dict: Node name to Relationship mapping
            abnormality_df: DataFrame with abnormality scores
        """
        ordered_metrics = top_metrics.index.tolist()
        edge_weights = top_metrics[self.param['weight_sort_by']].tolist()

        entity_dict['related_nodes'].extend(
            (
                node_name_map[metric],
                edge_dict[metric],
                weight,
                abnormality_df.loc[metric, 'abnormality']
            )
            for metric, weight in zip(ordered_metrics, edge_weights)
            if metric in edge_dict and metric in node_name_map
        )

    def _add_demographic_nodes(
        self,
        entity_dict: Dict[str, Any],
        node_name_map: Dict[str, Entity],
        edge_dict: Dict[str, Any]
    ) -> None:
        """
        Add demographic nodes to entity dictionary.

        Args:
            entity_dict: Entity result dictionary to update
            node_name_map: Node name to Entity mapping
            edge_dict: Node name to Relationship mapping
        """
        for demog_metric in self.param.get('demog_metrics', []):
            if demog_metric in node_name_map and demog_metric in edge_dict:
                edge = edge_dict[demog_metric]
                entity_dict['demog_nodes'].append(
                    (node_name_map[demog_metric], edge, edge.weight)
                )

    # =========================================================================
    # Weight Computation
    # =========================================================================

    def weight_recomputation(
        self,
        node_name: str,
        query_info: Dict[str, Any],
        rel_info: Dict[str, Any],
        prior_matrix: Any,
        par_df: pd.DataFrame,
        hyperparameter: Dict[str, Any]
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Compute relationship weights using Bayesian approach.

        Combines three knowledge sources:
        1. LLM prior knowledge
        2. Population-level correlations
        3. Individual-level patterns

        Args:
            node_name: Primary node name
            query_info: Query metadata
            rel_info: Relationship information (pop/ind correlations)
            prior_matrix: LLM prior knowledge matrix
            par_df: User's health data
            hyperparameter: Weight computation parameters

        Returns:
            Tuple of (weights_df, abnormality_df)
        """
        numeric_metrics = rel_info['numeric_metrics']
        query_rel_info = rel_info.copy()

        # Handle non-numeric nodes
        if node_name not in numeric_metrics:
            query_rel_info['rel_pop_all'] = None
            query_rel_info['rel_var_pop_all'] = None
            query_rel_info['rel_pop_sample_size_all'] = None
            query_rel_info['rel_ind_all'] = None
            query_rel_info['rel_var_ind_all'] = None
            query_rel_info['rel_ind_sample_size_all'] = None

        # Run WAG retriever
        from shared.retrieval_package import WeightRetriever

        graph_searcher = WeightRetriever(
            node_name,
            query_rel_info,
            prior_matrix,
            hyperparameter,
            par_df,
            query_info
        )
        out_df, abnormality_df = graph_searcher.run()

        return out_df, abnormality_df

    # =========================================================================
    # Context Building (WAG-specific)
    # =========================================================================

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
        Build context for a node with WAG-specific formatting.

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
            return "data: No data\n", data

        sensor_info = node.dataSource[dataset_name]

        # Build context with sensor information
        context = (
            f"{node.name}:\n"
            f"description: {node.description}\n"
            f"range: {node.range}\n"
            f"recommendation: {node.recommendation}\n"
            f"sensor specific information:\n"
            f"1. description: {sensor_info['description']}\n"
            f"2. range: {sensor_info['range']}\n"
            f"3. unit: {sensor_info['unit']}\n"
            f"Data:\n"
        )

        # Get data
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
                f"Recent {time_range}-day value deviates from the individual's average by "
                f"{float(abnormality):.2f} standard deviations.\n"
            )
        else:
            context += (
                f"No deviation from baseline recorded for the recent {time_range}-day period.\n\n"
            )

        return context, data
