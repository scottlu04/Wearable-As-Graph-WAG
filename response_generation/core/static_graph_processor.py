"""
Static-graph baseline processor for general response generation experiments.

This baseline uses the knowledge graph's stored edge weights directly to rank
related nodes. It preserves WAG's per-query openness budget and context format,
but does not run dynamic relationship recomputation.
"""

from typing import Any, Dict, List, Optional, Tuple
import logging

import pandas as pd

from response_generation.core.wag_processor import WAGProcessor

logger = logging.getLogger(__name__)


class StaticGraphProcessor(WAGProcessor):
    """
    Static graph baseline using precomputed edge weights only.

    Compared with WAG, this method:
    - keeps exact-match primary node resolution
    - keeps the same openness-driven related-node budget
    - keeps the same context formatting as WAG
    - replaces dynamic score recomputation with static edge.weight ranking
    """

    def query_quick(
        self,
        user_query: str,
        par_df: pd.DataFrame,
        gt_nodes: List[str],
        query_info: Dict[str, Any],
        **kwargs
    ) -> Tuple[Dict[str, Any], Optional[pd.DataFrame]]:
        """Run the standard query flow and append StaticGraph metadata."""
        output, out_df = super().query_quick(
            user_query=user_query,
            par_df=par_df,
            gt_nodes=gt_nodes,
            query_info=query_info,
            **kwargs,
        )

        metadata = dict(getattr(out_df, 'attrs', {})) if out_df is not None else {}
        output.update({
            'candidate_pool_size': metadata.get('candidate_pool_size', 0),
            'requested_related_node_count': metadata.get('requested_related_node_count', 0),
            'actual_related_node_count': metadata.get('actual_related_node_count', 0),
        })

        return output, out_df

    def search_knowledge_graph(
        self,
        entities: List[str],
        query_info: Dict[str, Any],
        par_df: Optional[pd.DataFrame] = None,
        dataset_name: Optional[str] = None,
        **kwargs
    ) -> Tuple[Dict[str, Any], pd.DataFrame]:
        """Search the graph using static edge weights for ranking."""
        node_name_map = {
            node.name: node
            for node in self.nodes_with_embeddings.values()
        }
        result_dict = self._match_primary_entities(entities, node_name_map)

        requested_related_node_count = int(
            query_info['openness'] * self.param['max_num_related_nodes']
        )
        if not result_dict:
            empty_df = pd.DataFrame()
            empty_df.attrs.update({
                'candidate_pool_size': 0,
                'requested_related_node_count': requested_related_node_count,
                'actual_related_node_count': 0,
            })
            return result_dict, empty_df

        num_related_nodes_per_node = requested_related_node_count // len(result_dict)
        primary_node_names = {
            inner_dict['primary_node'][0].name
            for inner_dict in result_dict.values()
        }
        rel_info = kwargs.get('rel_info', {})
        allowed_metric_names = rel_info.get('data_associated_metrics')
        allowed_metric_names = (
            set(allowed_metric_names)
            if allowed_metric_names is not None
            else None
        )

        total_candidate_pool_size = 0
        total_actual_related_nodes = 0
        last_out_df = pd.DataFrame()

        for entity, inner_dict in result_dict.items():
            primary_node = inner_dict['primary_node'][0]
            edge_dict = self._build_edge_dict(primary_node.name)
            candidate_pool = self._build_candidate_pool(
                primary_node_name=primary_node.name,
                all_primary_node_names=primary_node_names,
                node_name_map=node_name_map,
                edge_dict=edge_dict,
                dataset_name=dataset_name,
                par_df=par_df,
                query_date=query_info['query_date'],
                allowed_metric_names=allowed_metric_names,
            )
            total_candidate_pool_size += len(candidate_pool)

            selected_df = (
                candidate_pool.sort_values(
                    by=['static_edge_weight', 'metric_name'],
                    ascending=[False, True],
                )
                .head(num_related_nodes_per_node)
            )
            total_actual_related_nodes += len(selected_df)
            last_out_df = selected_df

            self._add_static_related_nodes(
                entity_dict=inner_dict,
                selected_df=selected_df,
                node_name_map=node_name_map,
                edge_dict=edge_dict,
            )
            self._add_demographic_nodes(
                inner_dict,
                node_name_map,
                edge_dict,
            )

        last_out_df.attrs.update({
            'candidate_pool_size': total_candidate_pool_size,
            'requested_related_node_count': requested_related_node_count,
            'actual_related_node_count': total_actual_related_nodes,
        })

        return result_dict, last_out_df

    def _build_candidate_pool(
        self,
        primary_node_name: str,
        all_primary_node_names: set[str],
        node_name_map: Dict[str, Any],
        edge_dict: Dict[str, Any],
        dataset_name: Optional[str],
        par_df: Optional[pd.DataFrame],
        query_date: Any,
        allowed_metric_names: Optional[set[str]] = None,
    ) -> pd.DataFrame:
        """Build the static candidate pool from graph neighbors with data."""
        rows: List[Dict[str, Any]] = []
        query_ts = pd.to_datetime(query_date)

        for metric_name, edge in edge_dict.items():
            node = node_name_map.get(metric_name)
            if node is None:
                continue
            if metric_name == primary_node_name or metric_name in all_primary_node_names:
                continue
            if allowed_metric_names is not None and metric_name not in allowed_metric_names:
                continue
            if dataset_name is not None and dataset_name not in node.dataSource:
                continue
            if par_df is not None:
                if metric_name not in par_df.columns:
                    continue
                history = par_df.loc[
                    pd.to_datetime(par_df['date']) <= query_ts,
                    metric_name,
                ]
                if not history.notna().any():
                    continue

            rows.append({
                'metric_name': metric_name,
                'static_edge_weight': float(edge.weight or 0.0),
            })

        if not rows:
            return pd.DataFrame(columns=['metric_name', 'static_edge_weight']).set_index('metric_name')

        return pd.DataFrame(rows).set_index('metric_name')

    def _add_static_related_nodes(
        self,
        entity_dict: Dict[str, Any],
        selected_df: pd.DataFrame,
        node_name_map: Dict[str, Any],
        edge_dict: Dict[str, Any],
    ) -> None:
        """Append selected related nodes using static edge weights."""
        for metric_name, row in selected_df.iterrows():
            if metric_name not in node_name_map or metric_name not in edge_dict:
                continue
            entity_dict['related_nodes'].append(
                (
                    node_name_map[metric_name],
                    edge_dict[metric_name],
                    float(row['static_edge_weight']),
                    None,
                )
            )
