"""
Base Pipeline for Experiment Execution

This module provides the BasePipeline class that consolidates common functionality
across all experiment pipelines (general, global weight, local weight). It eliminates
code duplication by extracting shared operations like data loading, query processing,
and result saving.

The BasePipeline implements the Template Method pattern where child classes only need
to specify their unique configurations and processor initialization logic.

Key Features:
    - Data loading (QA datasets, health data, relationship matrices)
    - Common query processing loop
    - Result storage and serialization
    - Error handling and logging
    - Configurable output directories

Architecture:
    BasePipeline (abstract)
        ├─> get_clients() - Abstract method for processor initialization
        ├─> run() - Main execution template method
        └─> Helper methods (load_data, process_queries, save_results)

    Child Classes:
        - GeneralComparisonPipeline: Compares Base, RAG, StaticGraph, FullContext, and WAG
        - GlobalWeightPipeline: Compares weight sources (prior/pop/ind/global)
        - LocalWeightPipeline: Compares local/global/final weights

Usage Example:
    >>> # Implement a custom pipeline
    >>> class MyPipeline(BasePipeline):
    ...     def get_clients(self):
    ...         return {
    ...             'method1': Processor1(...),
    ...             'method2': Processor2(...)
    ...         }
    >>>
    >>> pipeline = MyPipeline(root_dir="resources", output_dir="results/my_exp")
    >>> results_df = pipeline.run()

"""

import os
import json
import pandas as pd
from typing import Dict, List, Any, Tuple
from tqdm import tqdm
import traceback
from abc import ABC, abstractmethod
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from collections import OrderedDict

# Set up logging
logger = logging.getLogger(__name__)


def convert_to_serializable(obj: Any) -> Any:
    """
    Convert objects to JSON-serializable format.

    Args:
        obj: Object to convert

    Returns:
        JSON-serializable representation
    """
    if isinstance(obj, pd.Timestamp):
        return obj.isoformat()
    elif hasattr(obj, '__dict__'):
        return {k: convert_to_serializable(v) for k, v in obj.__dict__.items()}
    elif isinstance(obj, dict):
        return {k: convert_to_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [convert_to_serializable(x) for x in obj]
    return obj


class BasePipeline(ABC):
    """
    Base class for all experiment pipelines.

    This abstract class provides common functionality for running experiments
    that compare different query processing methods. Child classes must implement
    get_clients() to define which processors to compare.

    Attributes:
        root_dir: Root directory containing data files
        output_dir: Directory for saving experiment results
        path_map: Mapping of dataset names to file names
        result_dict: Dictionary storing query metadata
        results_df: DataFrame storing all experimental results
    """

    def __init__(
        self,
        root_dir: str,
        output_dir: str = None,
        max_workers: int = 1,
        qid_filter_file: str = None,
        chat_client: str = None,
        chat_model: str = None,
        generate_responses: bool = False,
        methods: List[str] = None,
    ):
        """
        Initialize the pipeline.

        Args:
            root_dir: Root directory containing resources
            output_dir: Output directory for results (optional)
            max_workers: Maximum number of parallel workers for query processing (default: 1 for sequential)
                        Set to > 1 for parallel processing (e.g., 4, 8)
            qid_filter_file: Path to file containing list of q_ids to process (optional)
                            File should contain one q_id per line, or a JSON list
                            If None, all queries will be processed
            chat_client: LLM provider for response generation processors
            chat_model: Optional explicit model override for the chat provider
            generate_responses: Whether to call the chat model to generate
                               final answers during experiments
            methods: Optional ordered list of method names to run. If omitted,
                     each pipeline runs all of its methods.
        """
        self.root_dir = root_dir
        self.output_dir = output_dir or f"{root_dir}/experiment_result"
        self.max_workers = max_workers
        self.qid_filter_file = qid_filter_file
        self.qid_filter = None  # Will be loaded if qid_filter_file is provided
        self.chat_client = chat_client
        self.chat_model = chat_model
        self.generate_responses = generate_responses
        self.methods = methods
        self.generation_method_attempts = max(
            1,
            int(os.getenv('GENERATION_METHOD_ATTEMPTS', '3')),
        )

        # Dataset mappings
        self.path_map = {
            'pmdata': 'pmdata_df',
            'ifh_affect': 'ifh_df',
            'globem': 'globem_df',
            'lifesnap': 'lifesnap_df',
        }

        # Results storage
        self.result_dict = {}
        self.results_df = pd.DataFrame()

        logger.info(f"Initialized {self.__class__.__name__}")
        logger.info(f"  Root directory: {root_dir}")
        logger.info(f"  Output directory: {self.output_dir}")
        logger.info(f"  Max workers: {max_workers} ({'parallel' if max_workers > 1 else 'sequential'})")
        logger.info(f"  Chat client: {chat_client or 'default'}")
        logger.info(f"  Chat model: {chat_model or 'default'}")
        logger.info(f"  Generate responses: {generate_responses}")
        logger.info(f"  Methods: {methods or 'all'}")
        logger.info(f"  Generation attempts per method: {self.generation_method_attempts}")

        # Load q_id filter if provided
        if qid_filter_file:
            self.qid_filter = self.load_qid_filter(qid_filter_file)
            logger.info(f"  Q_ID filter: {len(self.qid_filter)} queries loaded from {qid_filter_file}")
        else:
            logger.info(f"  Q_ID filter: None (processing all queries)")

    def get_processor_init_kwargs(self) -> Dict[str, Any]:
        """
        Return shared processor initialization kwargs for child pipelines.

        Child pipelines can pass this through when constructing processors so
        model/provider selection remains consistent across all methods.
        """
        kwargs: Dict[str, Any] = {}
        chat_client = getattr(self, 'chat_client', None)
        chat_model = getattr(self, 'chat_model', None)

        if chat_client is not None:
            kwargs['chat_client'] = chat_client
        if chat_model is not None:
            kwargs['chat_model'] = chat_model

        return kwargs

    def select_method_names(self, available_methods: List[str]) -> List[str]:
        """
        Resolve requested method names against a pipeline's available methods.

        Args:
            available_methods: Ordered list of methods supported by the pipeline

        Returns:
            Ordered list of method names to run

        Raises:
            ValueError: If a requested method is not available for the pipeline
        """
        if not self.methods:
            return list(available_methods)

        requested = list(OrderedDict.fromkeys(self.methods))
        unknown_methods = [method for method in requested if method not in available_methods]
        if unknown_methods:
            raise ValueError(
                "Unknown method(s) for this experiment: "
                f"{unknown_methods}. Available methods: {available_methods}"
            )

        return requested

    def load_qid_filter(self, filter_file: str) -> set:
        """
        Load q_id filter from file.

        Supports two formats:
        1. Text file with one q_id per line
        2. JSON file with a list of q_ids

        Args:
            filter_file: Path to file containing q_ids to process

        Returns:
            Set of q_ids to process

        Raises:
            FileNotFoundError: If filter file doesn't exist
            ValueError: If file format is invalid
        """
        if not os.path.exists(filter_file):
            raise FileNotFoundError(f"Q_ID filter file not found: {filter_file}")

        logger.info(f"Loading q_id filter from {filter_file}")

        # Try to load as JSON first
        try:
            with open(filter_file, 'r') as f:
                data = json.load(f)

            # Handle different JSON structures
            if isinstance(data, list):
                qids = set(data)
            elif isinstance(data, dict):
                # Support dict with 'q_ids' or 'qids' key
                qids = set(data.get('q_ids', data.get('qids', [])))
            else:
                raise ValueError(f"Invalid JSON format in {filter_file}. Expected list or dict.")

        except json.JSONDecodeError:
            # Not JSON, try as text file (one q_id per line)
            with open(filter_file, 'r') as f:
                qids = set(line.strip() for line in f if line.strip())

        if not qids:
            raise ValueError(f"No q_ids found in filter file: {filter_file}")

        logger.info(f"  Loaded {len(qids)} q_ids from filter")
        return qids

    @abstractmethod
    def get_clients(self) -> Dict[str, Any]:
        """
        Get processor clients for this experiment.

        Child classes must implement this to return a dictionary of
        method names to processor instances.

        Returns:
            Dictionary mapping method_name -> processor instance

        Example:
            {
                'Base': BaseQueryProcessor(root_dir=...),
                'Rag': RAGProcessor(root_dir=...),
                'Wag': WAGProcessor(root_dir=..., param=...)
            }
        """
        pass

    def load_qa_dataset(self) -> pd.DataFrame:
        """
        Load the QA dataset.

        Returns:
            DataFrame with queries
        """
        logger.debug("Loading QA dataset...")
        qa_path = f"{self.root_dir}/query_set/qa_df_full.json"

        if not os.path.exists(qa_path):
            raise FileNotFoundError(f"QA dataset not found: {qa_path}")

        with open(qa_path, 'r') as f:
            qa_data = json.load(f)

        qa_df = pd.DataFrame(qa_data)
        logger.debug(f"Loaded {len(qa_df)} queries from {qa_path}")

        return qa_df

    def load_health_data(self, dataset_name: str) -> pd.DataFrame:
        """
        Load health data for a specific dataset.

        Args:
            dataset_name: Name of the dataset

        Returns:
            DataFrame with health data
        """
        dataset_file = f"{self.root_dir}/processed_dataset/{self.path_map[dataset_name]}.csv"

        if not os.path.exists(dataset_file):
            logger.warning(f"Dataset file not found: {dataset_file}")
            return None

        logger.debug(f"Loading health data from {dataset_file}")
        dataset_df = pd.read_csv(dataset_file)
        dataset_df['date'] = pd.to_datetime(dataset_df['date'])

        logger.debug(f"  Loaded {len(dataset_df)} rows, {len(dataset_df.columns)} columns")

        return dataset_df

    def load_relationship_info(self, dataset_name: str = None) -> Dict:
        """
        Load relationship matrices for population and individual correlations.

        Args:
            dataset_name: Name of the dataset to load relationships for.
                         If None, loads the entire relationship_dict.

        Returns:
            Dictionary with relationship information or the full relationship_dict
        """
        rel_dict_path = f"{self.root_dir}/relationship_dict.json"

        if not os.path.exists(rel_dict_path):
            logger.warning(f"Relationship dict not found: {rel_dict_path}")
            return None

        with open(rel_dict_path, 'r') as f:
            relationship_dict = json.load(f)

        # If no dataset specified, return the full dict for per-dataset processing
        if dataset_name is None:
            return relationship_dict

        # Check if dataset exists in relationship_dict
        if dataset_name not in relationship_dict:
            logger.warning(f"Dataset {dataset_name} not found in relationship_dict")
            return None

        logger.debug(f"Computing relationship matrices for {dataset_name}...")

        from shared.retrieval_package.utils import get_relationship_matrix

        # Get relationship matrices for this specific dataset
        rel_pop, sample_pop, p_pop, metrics = get_relationship_matrix(
            relationship_dict[dataset_name],
            rel_type='spearman',
            data_type='pop'
        )

        # Note: Individual relationships are loaded per-user, not here

        rel_info = {
            'rel_pop_all': rel_pop,
            'rel_var_pop_all': None,
            'rel_pop_sample_size_all': sample_pop,
            'rel_ind_all': None,  # Will be set per-user
            'rel_var_ind_all': None,
            'rel_ind_sample_size_all': None,  # Will be set per-user
            'numeric_metrics': metrics,
            'data_associated_metrics': metrics,
            'dataset_relationships': relationship_dict[dataset_name],  # Store for per-user loading
        }

        logger.debug(f"  Population correlations: {rel_pop.shape if rel_pop is not None else 'None'}")

        return rel_info

    def load_individual_relationships(
        self,
        dataset_relationships: Dict,
        user_id: str,
        rel_type: str = 'spearman'
    ) -> Tuple:
        """
        Load individual relationship matrices for a specific user.

        Args:
            dataset_relationships: The dataset-specific relationship dict
            user_id: User ID to load relationships for
            rel_type: Type of correlation ('spearman' or 'pearson')

        Returns:
            Tuple of (rel_ind, sample_ind, p_ind, metrics)
        """
        from shared.retrieval_package.utils import get_relationship_matrix

        if 'users' not in dataset_relationships or user_id not in dataset_relationships['users']:
            logger.debug(f"  No individual relationships found for user {user_id}")
            return None, None, None, None

        rel_ind, sample_ind, p_ind, _ = get_relationship_matrix(
            dataset_relationships['users'][user_id],
            rel_type=rel_type,
            data_type='ind'
        )

        return rel_ind, sample_ind, p_ind, None

    def _run_method_with_retries(
        self,
        method_name: str,
        client: Any,
        user_query: str,
        par_df: pd.DataFrame,
        gt_nodes: List[str],
        query_info: Dict[str, Any],
        dataset_name: str,
        rel_info: Dict,
        q_id: str,
    ) -> Tuple[Dict[str, Any], Any]:
        """Run one method for one query, retrying transient method failures."""
        last_error = None

        for attempt in range(1, self.generation_method_attempts + 1):
            try:
                method_rel_info = rel_info.copy() if isinstance(rel_info, dict) else rel_info
                output, out_df = client.query_quick(
                    user_query=user_query,
                    par_df=par_df,
                    gt_nodes=gt_nodes,
                    query_info=query_info,
                    dataset_name=dataset_name,
                    rel_info=method_rel_info,
                    response_gen=self.generate_responses,
                )
                output['generation_attempts'] = attempt
                output['generation_retry_count'] = attempt - 1
                if attempt > 1:
                    logger.info(
                        "Generation succeeded for query %s method %s on attempt %s/%s",
                        q_id,
                        method_name,
                        attempt,
                        self.generation_method_attempts,
                    )
                return output, out_df
            except Exception as e:
                last_error = e
                if attempt < self.generation_method_attempts:
                    logger.warning(
                        "Generation attempt %s/%s failed for query %s method %s: %s",
                        attempt,
                        self.generation_method_attempts,
                        q_id,
                        method_name,
                        e,
                    )
                else:
                    logger.error(
                        "All %s generation attempts failed for query %s method %s: %s",
                        self.generation_method_attempts,
                        q_id,
                        method_name,
                        e,
                    )

        raise last_error or RuntimeError(f"Generation failed for query {q_id} method {method_name}")

    def process_single_row(
        self,
        row: pd.Series,
        clients: Dict[str, Any],
        par_df: pd.DataFrame,
        dataset_name: str,
        rel_info: Dict,
        user_id: str
    ) -> Tuple[Dict, List[Dict]]:
        """
        Process a single query row with multiple methods.

        Args:
            row: Query row from DataFrame
            clients: Dictionary of method_name -> processor
            par_df: User's health data
            dataset_name: Dataset identifier
            rel_info: Relationship information
            user_id: User identifier

        Returns:
            Tuple of (query_dict, results_rows)
        """
        q_id = row['q_id']
        user_query = row['query']
        gt_nodes = row['associated_entity(gt)']
        query_info = {
            'query_date': row['query_date'],
            'time_granularity': row['time_granularity'],
            'openness': row['openness(assigned)']
        }

        query_dict = {
            'q_id': q_id,
            'dataset': dataset_name,
            'user_id': user_id,
            'query': user_query,
            'query_date': query_info['query_date'],
            'time_granularity': query_info['time_granularity'],
            'openness(assigned)': query_info['openness'],
            'associated_entity(gt)': gt_nodes,
        }

        results_rows = []

        for method_name, client in clients.items():
            try:
                output, out_df = self._run_method_with_retries(
                    method_name=method_name,
                    client=client,
                    user_query=user_query,
                    par_df=par_df,
                    gt_nodes=gt_nodes,
                    query_info=query_info,
                    dataset_name=dataset_name,
                    rel_info=rel_info,
                    q_id=q_id,
                )

                # Extract results
                result_dict = output['result_dict']
                context = output['context']
                data = output['data']
                response = output['response']
                primary_nodes = output.get('primary_nodes')
                related_nodes = output.get('related_nodes')

                if primary_nodes is None or related_nodes is None:
                    primary_nodes = []
                    related_nodes = []

                    for entity_info in result_dict.values():
                        primary_node = entity_info['primary_node'][0]
                        primary_nodes.append(primary_node.name)

                        for related_info in entity_info.get('related_nodes', []):
                            related_node = related_info[0]
                            related_nodes.append(related_node.name)

                results_rows.append({
                    'q_id': q_id,
                    'dataset': dataset_name,
                    'user_id': user_id,
                    'query': user_query,
                    'query_date': str(query_info['query_date']),
                    'time_granularity': query_info['time_granularity'],
                    'openness(assigned)': query_info['openness'],
                    'associated_entity(gt)': gt_nodes,
                    'method': method_name,
                    'chat_client': getattr(client, 'chat_provider', None),
                    'chat_model': getattr(client, 'chat_model', None),
                    'status': output.get('status', 'success'),
                    'infeasible_reason': output.get('infeasible_reason'),
                    'context_tokens': output.get('context_tokens'),
                    'estimated_prompt_tokens': output.get('estimated_prompt_tokens'),
                    'input_tokens': output.get('input_tokens'),
                    'output_tokens': output.get('output_tokens'),
                    'total_tokens': output.get('total_tokens'),
                    'llm_prompt_tokens': output.get('llm_prompt_tokens'),
                    'llm_completion_tokens': output.get('llm_completion_tokens'),
                    'llm_total_tokens': output.get('llm_total_tokens'),
                    'retrieval_time_seconds': output.get('retrieval_time_seconds'),
                    'generation_time_seconds': output.get('generation_time_seconds'),
                    'e2e_time_seconds': output.get('e2e_time_seconds'),
                    'primary_nodes': primary_nodes,
                    'related_nodes': related_nodes,
                    'full_context_metric_count': output.get('full_context_metric_count'),
                    'full_context_history_span_days': output.get('full_context_history_span_days'),
                    'full_context_time_scope': output.get('full_context_time_scope'),
                    'full_context_used_time_range': output.get('full_context_used_time_range'),
                    'generation_called': output.get('generation_called'),
                    'generation_attempts': output.get('generation_attempts'),
                    'generation_retry_count': output.get('generation_retry_count'),
                    'context': context,
                    'response': response,
                    'data': convert_to_serializable(data)
                })

                logger.debug(f"  [{method_name}] Processed query {q_id}")

            except Exception as e:
                logger.error(f"Error processing {q_id} with {method_name}: {e}")
                logger.debug(traceback.format_exc())

                results_rows.append({
                    'q_id': q_id,
                    'dataset': dataset_name,
                    'user_id': user_id,
                    'query': user_query,
                    'query_date': str(query_info['query_date']),
                    'time_granularity': query_info['time_granularity'],
                    'openness(assigned)': query_info['openness'],
                    'associated_entity(gt)': gt_nodes,
                    'method': method_name,
                    'chat_client': getattr(client, 'chat_provider', None),
                    'chat_model': getattr(client, 'chat_model', None),
                    'status': 'error',
                    'infeasible_reason': None,
                    'context_tokens': None,
                    'estimated_prompt_tokens': None,
                    'input_tokens': None,
                    'output_tokens': None,
                    'total_tokens': None,
                    'llm_prompt_tokens': None,
                    'llm_completion_tokens': None,
                    'llm_total_tokens': None,
                    'retrieval_time_seconds': None,
                    'generation_time_seconds': None,
                    'e2e_time_seconds': None,
                    'primary_nodes': None,
                    'related_nodes': None,
                    'full_context_metric_count': None,
                    'full_context_history_span_days': None,
                    'full_context_time_scope': None,
                    'full_context_used_time_range': None,
                    'generation_called': None,
                    'generation_attempts': self.generation_method_attempts,
                    'generation_retry_count': max(0, self.generation_method_attempts - 1),
                    'context': None,
                    'response': None,
                    'data': None,
                    'error': str(e)
                })

        return query_dict, results_rows

    @staticmethod
    def _sorted_results_df(results_df: pd.DataFrame) -> pd.DataFrame:
        """Return results sorted by stable query/method identifiers."""
        if results_df.empty:
            return results_df

        sort_columns = [
            column
            for column in ['dataset', 'user_id', 'q_id', 'method']
            if column in results_df.columns
        ]
        if not sort_columns:
            return results_df.reset_index(drop=True)

        return (
            results_df
            .sort_values(by=sort_columns, kind='stable', na_position='last')
            .reset_index(drop=True)
        )

    @staticmethod
    def _sorted_result_dict(result_dict: Dict[str, Any]) -> "OrderedDict[str, Any]":
        """Return query metadata sorted by dataset/user/query for stable serialization."""
        def sort_key(item: Tuple[str, Any]) -> Tuple[str, str, str]:
            q_id, payload = item
            payload = payload if isinstance(payload, dict) else {}
            return (
                str(payload.get('dataset', '')),
                str(payload.get('user_id', '')),
                str(payload.get('q_id', q_id)),
            )

        return OrderedDict(sorted(result_dict.items(), key=sort_key))

    def save_results(self):
        """Save experiment results to JSON files."""
        logger.info("Saving results...")

        os.makedirs(self.output_dir, exist_ok=True)

        self.results_df = self._sorted_results_df(self.results_df)
        self.result_dict = self._sorted_result_dict(self.result_dict)

        # Save results DataFrame
        results_df_json = self.results_df.to_json(orient='records', indent=2)
        results_path = f"{self.output_dir}/results_df.json"
        with open(results_path, 'w') as f:
            f.write(results_df_json)
        logger.info(f"  Saved results to {results_path}")

        # Save per-method token/time summary for quick experiment comparison
        metrics_summary = self.build_method_metrics_summary(self.results_df)
        summary_path = f"{self.output_dir}/method_metrics_summary.json"
        with open(summary_path, 'w') as f:
            json.dump(metrics_summary, f, indent=2)
        logger.info(f"  Saved method metrics summary to {summary_path}")

        # Save result dictionary
        result_dict_json = json.dumps(self.result_dict, indent=2, default=str)
        dict_path = f"{self.output_dir}/result_dict.json"
        with open(dict_path, 'w') as f:
            f.write(result_dict_json)
        logger.info(f"  Saved result dict to {dict_path}")

        logger.info(f"✓ Results saved to {self.output_dir}/")

    def build_method_metrics_summary(self, results_df: pd.DataFrame) -> Dict[str, Any]:
        """
        Build per-method token and timing summaries.

        Args:
            results_df: Result rows from the experiment

        Returns:
            Dictionary keyed by method name with count and metric statistics
        """
        if results_df.empty or 'method' not in results_df.columns:
            return {}

        metric_columns = [
            'input_tokens',
            'output_tokens',
            'context_tokens',
            'estimated_prompt_tokens',
            'retrieval_time_seconds',
            'generation_time_seconds',
            'e2e_time_seconds',
        ]
        summary: Dict[str, Any] = {}

        for method_name, method_df in results_df.groupby('method', sort=False):
            status_series = (
                method_df['status']
                if 'status' in method_df.columns
                else pd.Series([], dtype=object)
            )
            method_summary: Dict[str, Any] = {
                'count': int(len(method_df)),
                'success_count': int((status_series == 'success').sum()),
                'infeasible_count': int((status_series == 'infeasible').sum()),
                'error_count': int((status_series == 'error').sum()),
            }

            for metric in metric_columns:
                if metric not in method_df.columns:
                    method_summary[metric] = {
                        'mean': None,
                        'max': None,
                        'min': None,
                    }
                    continue

                values = pd.to_numeric(method_df[metric], errors='coerce').dropna()
                if values.empty:
                    method_summary[metric] = {
                        'mean': None,
                        'max': None,
                        'min': None,
                    }
                    continue

                method_summary[metric] = {
                    'mean': float(values.mean()),
                    'max': float(values.max()),
                    'min': float(values.min()),
                }

            summary[method_name] = method_summary

        return summary

    def run(self) -> pd.DataFrame:
        """
        Run the experiment pipeline.

        This is the main template method that orchestrates the entire experiment.

        Returns:
            DataFrame with all results
        """
        logger.info("=" * 70)
        logger.info(f"Starting {self.__class__.__name__}")
        logger.info("=" * 70)

        # Load QA dataset
        QA_df = self.load_qa_dataset()

        # Apply q_id filter if specified
        if self.qid_filter is not None:
            original_count = len(QA_df)
            QA_df = QA_df[QA_df['q_id'].isin(self.qid_filter)]
            filtered_count = len(QA_df)
            logger.info(f"Q_ID filter applied: {filtered_count}/{original_count} queries selected")

            if filtered_count == 0:
                logger.warning("No queries match the q_id filter! Check your filter file.")
                return pd.DataFrame()
        # Process each dataset
        for dataset_name in tqdm(self.path_map.keys(), desc="Datasets"):
            logger.debug(f"Processing dataset: {dataset_name}")

            # Filter queries for this dataset
            QA_df_dataset = QA_df[QA_df['dataset'] == dataset_name]
            logger.debug(f"  Queries: {len(QA_df_dataset)}")

            if len(QA_df_dataset) == 0:
                logger.debug(f"  No queries for {dataset_name}, skipping")
                continue

            # Load dataset health data
            dataset_df = self.load_health_data(dataset_name)
            if dataset_df is None:
                logger.warning(f"  Skipping {dataset_name} - no health data")
                continue

            # Load relationship info for this dataset
            rel_info = self.load_relationship_info(dataset_name)
            if rel_info is None:
                logger.warning(f"  Skipping {dataset_name} - no relationship info")
                continue

            # Process each user
            for user_id in tqdm(QA_df_dataset['user_id'].unique(), desc="Users"):
                # Get user data (health data uses 'participant_id', not 'user_id')
                par_df = dataset_df[dataset_df['participant_id'] == user_id].copy()
                if len(par_df) == 0:
                    logger.debug(f"  No data for user {user_id}, skipping")
                    continue

                # Get queries for this user
                queries_per_user = QA_df_dataset[QA_df_dataset['user_id'] == user_id]

                # Load individual relationships for this user
                rel_ind, sample_ind, _, _ = self.load_individual_relationships(
                    rel_info['dataset_relationships'],
                    user_id,
                    rel_type='spearman'
                )

                # Update rel_info with user-specific relationships
                user_rel_info = rel_info.copy()
                user_rel_info['rel_ind_all'] = rel_ind
                user_rel_info['rel_ind_sample_size_all'] = sample_ind

                logger.debug(f"  Individual correlations for user {user_id}: {rel_ind.shape if rel_ind is not None else 'None'}")

                # Initialize processors for this user
                clients = self.get_clients()
                logger.debug(f"  Initialized {len(clients)} clients: {list(clients.keys())}")

                # Process queries (in parallel if max_workers > 1)
                with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
                    # Submit all queries to the executor
                    future_to_row = {
                        executor.submit(
                            self.process_single_row,
                            row, clients, par_df, dataset_name, user_rel_info.copy(), user_id
                        ): row
                        for _, row in queries_per_user.iterrows()
                    }

                    # Process results as they complete
                    for future in tqdm(
                        as_completed(future_to_row),
                        total=len(future_to_row),
                        desc="Queries",
                        leave=False
                    ):
                        try:
                            q_dict, new_rows = future.result()

                            if q_dict:
                                self.result_dict[q_dict['q_id']] = q_dict

                                for row_data in new_rows:
                                    row_df = pd.DataFrame([row_data])
                                    # Align columns
                                    for col in self.results_df.columns:
                                        if col not in row_df.columns:
                                            row_df[col] = pd.NA
                                    self.results_df = pd.concat([self.results_df, row_df], ignore_index=True)

                        except Exception as e:
                            row = future_to_row[future]
                            logger.error(f"Error processing query {row.get('q_id', 'unknown')}: {e}")
                            logger.debug(traceback.format_exc())

        # Save results
        self.save_results()

        # Print summary
        logger.info("\n" + "=" * 70)
        logger.info("EXPERIMENT COMPLETE")
        logger.info("=" * 70)
        logger.info(f"  Total queries processed: {len(self.result_dict)}")
        logger.info(f"  Total result rows: {len(self.results_df)}")
        if len(self.results_df) > 0 and 'method' in self.results_df.columns:
            logger.info(f"  Methods compared: {self.results_df['method'].unique().tolist()}")
        logger.info("=" * 70)

        return self.results_df
