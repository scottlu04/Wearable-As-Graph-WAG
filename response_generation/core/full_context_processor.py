"""
Full-context baseline processors for general response generation experiments.

Two closely related baselines are defined here:

1. FullContextProcessor:
   Includes all available wearable metrics for the current user and dataset,
   but constrains each metric to the query's requested time window so it is
   directly comparable to Base/RAG/StaticGraph/WAG.

2. FullHistoryContextProcessor:
   Preserves the earlier full-history behavior, concatenating every available
   metric's complete history up to the query date.

If the estimated prompt would exceed the configured context limit, the query is
marked as infeasible and LLM generation is skipped.
"""

from typing import Any, Dict, List, Optional, Tuple, Union
import logging
import time

import pandas as pd

from response_generation.core.base_processor import BaseQueryProcessor
from shared.models.entity import Entity
from ..config.settings import MAX_CONTEXT_LENGTH
from shared.utils.tokens import num_tokens_from_string

logger = logging.getLogger(__name__)

INFEASIBLE_RESPONSE_TEXT = (
    "Full-context baseline was not executed because the estimated prompt length "
    "exceeds the configured context limit."
)


class FullContextProcessor(BaseQueryProcessor):
    """
    Full-context baseline that concatenates all available metric windows.

    Ground-truth nodes remain the primary nodes for schema compatibility, while
    every additional available wearable metric is appended to the context using
    the same query period as the other response-generation baselines.
    """

    full_context_time_scope = 'query_period'
    related_nodes_heading = 'Nodes related to matched nodes which might be helpful'

    def query_quick(
        self,
        user_query: str,
        par_df: pd.DataFrame,
        gt_nodes: List[str],
        query_info: Dict[str, Any],
        **kwargs
    ) -> Tuple[Dict[str, Any], Optional[pd.DataFrame]]:
        """
        Build a full-context prompt for the current user and dataset.

        Unlike graph-based processors, this baseline does not rank or filter
        metrics beyond basic availability checks.
        """
        e2e_start_time = time.perf_counter()
        dataset_name = kwargs.pop('dataset_name')
        response_gen = kwargs.get('response_gen', False)
        rel_info = kwargs.get('rel_info', {})
        allowed_metric_names = rel_info.get('data_associated_metrics')

        retrieval_start_time = time.perf_counter()

        primary_nodes = self._get_primary_nodes(gt_nodes)
        related_nodes = self._get_available_metric_nodes(
            par_df=par_df,
            dataset_name=dataset_name,
            query_date=query_info['query_date'],
            excluded_node_names={node.name for node in primary_nodes},
            allowed_metric_names=set(allowed_metric_names) if allowed_metric_names else None,
        )

        result_dict = self._build_result_dict(primary_nodes)
        time_range = self._get_context_time_range(query_info)
        context, data = self._build_full_context(
            primary_nodes=primary_nodes,
            related_nodes=related_nodes,
            query_date=query_info['query_date'],
            time_range=time_range,
            par_df=par_df,
            dataset_name=dataset_name,
        )

        context_tokens = num_tokens_from_string(context, model=self.chat_model)
        estimated_prompt_tokens = self._estimate_prompt_tokens(user_query, context)
        retrieval_time_seconds = time.perf_counter() - retrieval_start_time
        generation_called = False
        generation_time_seconds = 0.0
        status = 'success'
        infeasible_reason = None
        response = None
        llm_usage = {
            'prompt_tokens': None,
            'completion_tokens': None,
            'total_tokens': None,
        }

        if estimated_prompt_tokens > MAX_CONTEXT_LENGTH:
            status = 'infeasible'
            infeasible_reason = 'context_limit_exceeded'
            response = INFEASIBLE_RESPONSE_TEXT
        elif response_gen:
            generation_start_time = time.perf_counter()
            response, llm_usage = self._generate_response_with_usage(
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
                num_tokens_from_string(response, model=self.chat_model)
                if response and generation_called
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
            'response': response,
            'status': status,
            'infeasible_reason': infeasible_reason,
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
            'primary_nodes': [node.name for node in primary_nodes],
            'related_nodes': [node.name for node in related_nodes],
            'full_context_metric_count': len(primary_nodes) + len(related_nodes),
            'full_context_history_span_days': self._get_history_span_days(
                par_df,
                query_info['query_date']
            ),
            'full_context_time_scope': self.full_context_time_scope,
            'full_context_used_time_range': time_range,
            'generation_called': generation_called,
        }

        return output, None

    def _get_context_time_range(
        self,
        query_info: Dict[str, Any],
    ) -> Any:
        """Return the time range used to build metric context."""
        return query_info['time_granularity']

    def _get_primary_nodes(self, gt_nodes: List[str]) -> List[Entity]:
        """Resolve ground-truth nodes to graph entities, preserving order."""
        primary_nodes: List[Entity] = []
        seen = set()

        for node_name in gt_nodes:
            node = self._find_node_by_name(node_name)
            if node is None or node.name in seen:
                continue
            primary_nodes.append(node)
            seen.add(node.name)

        return primary_nodes

    def _get_available_metric_nodes(
        self,
        par_df: pd.DataFrame,
        dataset_name: str,
        query_date: str,
        excluded_node_names: set[str],
        allowed_metric_names: Optional[set[str]] = None,
    ) -> List[Entity]:
        """Collect all dataset-backed wearable metrics with historical data."""
        available_nodes: List[Entity] = []
        query_ts = pd.to_datetime(query_date)

        for node in self.nodes_with_embeddings.values():
            if node.name in excluded_node_names:
                continue
            if allowed_metric_names is not None and node.name not in allowed_metric_names:
                continue
            if dataset_name not in node.dataSource:
                continue
            if node.name not in par_df.columns:
                continue

            history = par_df.loc[
                pd.to_datetime(par_df['date']) <= query_ts,
                node.name
            ]
            if history.notna().any():
                available_nodes.append(node)

        return sorted(available_nodes, key=lambda node: node.name)

    def _build_result_dict(
        self,
        primary_nodes: List[Entity]
    ) -> Dict[str, Dict[str, Any]]:
        """Build a minimal result_dict aligned with the existing schema."""
        return {
            node.name: {
                'similarity': 1.0,
                'primary_node': [node, None],
                'related_nodes': [],
                'demog_nodes': [],
            }
            for node in primary_nodes
        }

    def _build_full_context(
        self,
        primary_nodes: List[Entity],
        related_nodes: List[Entity],
        query_date: str,
        time_range: Any,
        par_df: pd.DataFrame,
        dataset_name: str,
    ) -> Tuple[str, Dict[str, Any]]:
        """Build full-context sections using the configured time scope."""
        data: Dict[str, Any] = {}

        primary_parts = self._build_node_context_parts(
            primary_nodes,
            query_date,
            time_range,
            par_df,
            dataset_name,
            data,
        )
        related_parts = self._build_node_context_parts(
            related_nodes,
            query_date,
            time_range,
            par_df,
            dataset_name,
            data,
            primary_nodes=primary_nodes,
        )

        primary_section = '\n'.join(primary_parts) if primary_parts else 'No matched nodes.\n'
        full_context = f"Matched nodes:\n\n{primary_section}"
        if related_parts:
            full_context += (
                f"\n\n{self.related_nodes_heading}:\n\n"
                + '\n'.join(related_parts)
            )

        return full_context, data

    def _build_node_context_parts(
        self,
        nodes: List[Entity],
        query_date: str,
        time_range: Any,
        par_df: pd.DataFrame,
        dataset_name: str,
        data: Dict[str, Any],
        primary_nodes: Optional[List[Entity]] = None,
    ) -> List[str]:
        """Build context strings for a list of nodes."""
        context_parts: List[str] = []

        for node in nodes:
            if primary_nodes:
                context_parts.extend(
                    self._build_edge_context_parts(node, primary_nodes)
                )

            node_context, node_data = self.build_node_context(
                node=node,
                start_date=query_date,
                time_range=time_range,
                par_df=par_df,
                dataset_name=dataset_name,
                abnormality=None,
            )
            context_parts.append(node_context)
            data.update(node_data)

        return context_parts

    def _build_edge_context_parts(
        self,
        related_node: Entity,
        primary_nodes: List[Entity],
    ) -> List[str]:
        """Build WAG-style relationship text between a metric and primary nodes."""
        edge_contexts: List[str] = []
        for primary_node in primary_nodes:
            edge = self._find_edge_between(related_node.name, primary_node.name)
            if edge is None:
                continue
            edge_contexts.append(
                self.build_edge_context(
                    edge=edge,
                    related_node_name=related_node.name,
                    primary_node_name=primary_node.name,
                    weight=edge.weight or 0,
                )
            )

        return edge_contexts

    def _find_edge_between(
        self,
        node_name_a: str,
        node_name_b: str,
    ) -> Optional[Any]:
        """Return the direct graph edge connecting two node names, if present."""
        for edge in self.edges.values():
            edge_nodes = {edge.entity_1_name, edge.entity_2_name}
            if {node_name_a, node_name_b} == edge_nodes:
                return edge
        return None

    def build_node_context(
        self,
        node: Entity,
        start_date: str,
        time_range: Union[str, int],
        par_df: pd.DataFrame,
        dataset_name: str,
        abnormality: Optional[float] = None,
    ) -> Tuple[str, Dict[str, Any]]:
        """
        Build context using the same node format as WAG and StaticGraph.

        FullContext differs only in breadth: it includes all available metrics
        rather than ranked related metrics. The textual format stays aligned so
        method comparisons are not confounded by missing descriptions.
        """
        data: Dict[str, Any] = {}

        if dataset_name not in node.dataSource:
            return "data: No data\n", data

        sensor_info = node.dataSource[dataset_name]

        context = (
            f"{node.name}:\n"
            f"description: {node.description}\n"
            f"range: {node.range}\n"
            f"recommendation: {node.recommendation}\n"
            f"sensor specific information:\n"
            f"1. description: {sensor_info.get('description', '')}\n"
            f"2. range: {sensor_info.get('range', '')}\n"
            f"3. unit: {sensor_info.get('unit', '')}\n"
            f"Data:\n"
        )

        df = self.get_data(par_df.copy(), start_date, time_range, node.name)
        if not df.empty:
            context += f"{self._format_dataframe(df)}\n"
            data[node.name] = {
                'x': df['date'].tolist(),
                'y': df[node.name].tolist(),
            }

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

    def _get_history_span_days(
        self,
        par_df: pd.DataFrame,
        query_date: str,
    ) -> int:
        """Compute the available history span in days up to the query date."""
        if par_df.empty or 'date' not in par_df.columns:
            return 0

        filtered_dates = pd.to_datetime(par_df['date'])
        filtered_dates = filtered_dates[filtered_dates <= pd.to_datetime(query_date)]
        if filtered_dates.empty:
            return 0

        return int((filtered_dates.max() - filtered_dates.min()).days)


class FullHistoryContextProcessor(FullContextProcessor):
    """
    Full-history variant of the full-context baseline.

    This preserves the previous behavior of including all available metrics and
    their entire history up to the query date.
    """

    full_context_time_scope = 'all_history'
    related_nodes_heading = 'Nodes related to matched nodes which might be helpful'

    def _get_context_time_range(
        self,
        query_info: Dict[str, Any],
    ) -> str:
        """Always use all available history up to the query date."""
        return 'all'
