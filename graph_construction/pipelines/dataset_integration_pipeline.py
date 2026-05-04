"""
Dataset Integration Pipeline

This script implements the dataset-specific metric incorporation workflow from
notebook 3_dataset_specific_incorperation_all_at_once.ipynb.

This pipeline:
1. Loads new metrics from a dataset feature map
2. Compares new metrics against existing nodes using LLM
3. Prepares merge decisions for optional manual review
4. Merges similar nodes or creates new ones
5. Updates dataSource information for merged nodes
6. Creates edges between new and existing nodes

Features:
- Optional manual review of merge decisions
- Idempotent reruns: safely rerun after failures without duplicates
- Tracks successful operations to skip them on reruns
- Only retries failed operations
"""

import os
import json
import logging
from typing import Dict, List, Optional, Tuple, Set
from tqdm import tqdm

from shared.models.entity import Entity
from shared.models.relationship import Relationship
from openai import OpenAI
from ..core.node_generator import NodeGenerator
from ..core.edge_generator import EdgeGenerator
from ..config.settings import config
from shared.prompts.EntityMerge import ENTITY_MERGE_JSON_PROMPT_ALL_V2
from shared.prompts.Context import CONTEXT_PROMPT
from shared.utils import parse_llm_response

# Configure logging
logging.basicConfig(
    level=getattr(logging, config.LOG_LEVEL),
    format=config.LOG_FORMAT
)
logger = logging.getLogger(__name__)


class DatasetIntegrationPipeline:
    """
    Pipeline for integrating new dataset metrics into existing knowledge graph.

    This pipeline handles:
    - Loading new metrics from dataset feature maps
    - LLM-based entity matching and merging
    - Updating existing nodes with new dataset information
    - Creating new nodes for unique metrics
    - Generating edges between new and existing nodes
    """

    def __init__(
        self,
        data_dir: str,
        features_map_file: str,
        batch_size: int = 10,
        similarity_threshold: float = 0.7
    ):
        """
        Initialize the dataset integration pipeline.

        Args:
            data_dir: Directory containing existing graph data.
            features_map_file: Path to features map JSON with new dataset metrics.
            batch_size: Batch size for LLM comparison.
            similarity_threshold: Threshold for considering nodes as duplicates (0-1).
        """
        self.data_dir = data_dir
        self.features_map_file = features_map_file
        self.batch_size = batch_size
        self.similarity_threshold = similarity_threshold

        # Initialize generators
        self.node_gen = NodeGenerator()
        self.edge_gen = EdgeGenerator()

        # Initialize OpenAI client for LLM merge decisions
        if config.USE_DEEPSEEK:
            self.llm_client = OpenAI(
                api_key=config.DEEPSEEK_API_KEY,
                base_url=config.DEEPSEEK_OFFICIAL_API
            )
        else:
            self.llm_client = OpenAI(api_key=config.OPENAI_API_KEY)

        # Storage
        self.nodes: Dict[str, Entity] = {}
        self.node_id_map: Dict[str, str] = {}
        self.edges: Dict[str, Relationship] = {}
        self.relationship_id_map: Dict[str, List[str]] = {}
        self.new_metrics: Dict = {}

        # Tracking
        self.merge_decisions: Dict = {}
        self.approved_merges: Dict = {}  # Manually reviewed and approved merges
        self.non_merge_list: List[str] = []
        self.failed_updates: List[Dict] = []

        # Success tracking for idempotent reruns
        self.successful_merges: Set[str] = set()  # Metrics successfully merged
        self.successful_new_nodes: Set[str] = set()  # Metrics that became new nodes
        self.successful_edges: Set[Tuple[str, str]] = set()  # Edge pairs created

    def run(self, stages: Optional[List[str]] = None) -> Dict[str, any]:
        """
        Run the dataset integration pipeline.

        Args:
            stages: List of stages to run. If None, runs all stages.
                   Available stages:
                   - 'load': Load existing graph and new metrics
                   - 'compare': Compare new metrics with existing nodes
                   - 'review_merges': Prepare merge decisions for manual review
                   - 'merge': Merge similar nodes
                   - 'create_new': Create new nodes for non-merged metrics
                   - 'create_edges': Create edges between new and existing nodes
                   - 'save': Save updated graph

        Returns:
            Dictionary with pipeline statistics.
        """
        all_stages = ['load', 'compare', 'review_merges', 'merge', 'create_new', 'create_edges', 'save']
        stages = stages or all_stages

        logger.info("Starting dataset integration pipeline")
        logger.info(f"Data directory: {self.data_dir}")
        logger.info(f"Features map: {self.features_map_file}")
        logger.info(f"Stages to run: {stages}")

        stats = {
            'existing_nodes': 0,
            'new_metrics': 0,
            'merged_nodes': 0,
            'new_nodes_created': 0,
            'new_edges_created': 0,
            'failed_updates': 0
        }

        # Stage 1: Load data
        if 'load' in stages:
            self._load_data()
            stats['existing_nodes'] = len(self.nodes)
            stats['new_metrics'] = len(self.new_metrics)

        # Stage 2: Compare new metrics with existing nodes
        if 'compare' in stages:
            self._stage_compare_metrics()

        # Stage 3: Prepare merge decisions for manual review
        if 'review_merges' in stages:
            self._stage_review_merges()

        # Stage 4: Merge similar nodes
        if 'merge' in stages:
            stats['merged_nodes'] = self._stage_merge_nodes()

        # Stage 5: Create new nodes for non-merged metrics
        if 'create_new' in stages:
            stats['new_nodes_created'] = self._stage_create_new_nodes()

        # Stage 6: Create edges
        if 'create_edges' in stages:
            stats['new_edges_created'] = self._stage_create_edges()

        # Stage 7: Save results
        if 'save' in stages:
            self._save_results()

        stats['failed_updates'] = len(self.failed_updates)

        logger.info("Pipeline completed successfully")
        logger.info(f"Statistics: {stats}")

        return stats

    def _load_data(self):
        """Load existing graph data and new metrics."""
        logger.info("Loading existing graph data")

        # Load existing nodes
        nodes_path = os.path.join(self.data_dir, "nodes.json")
        with open(nodes_path, 'r') as f:
            nodes_data = json.load(f)
            for node_id, node_dict in nodes_data.items():
                self.nodes[node_id] = Entity.from_dict(node_dict)

        with open(os.path.join(self.data_dir, "node_id_map.json"), 'r') as f:
            self.node_id_map = json.load(f)

        logger.info(f"Loaded {len(self.nodes)} existing nodes")

        # Load previous run tracking for idempotent reruns
        self._load_previous_run_state()

        # Load existing edges if available
        edges_path = os.path.join(self.data_dir, "edges.json")
        if os.path.exists(edges_path):
            with open(edges_path, 'r') as f:
                edges_data = json.load(f)
                for edge_id, edge_dict in edges_data.items():
                    self.edges[edge_id] = Relationship.from_dict(edge_dict)

            rel_map_path = os.path.join(self.data_dir, "relationship_id_map.json")
            if os.path.exists(rel_map_path):
                with open(rel_map_path, 'r') as f:
                    self.relationship_id_map = json.load(f)

            logger.info(f"Loaded {len(self.edges)} existing edges")

        # Load new metrics from features map
        logger.info(f"Loading new metrics from {self.features_map_file}")
        with open(self.features_map_file, 'r') as f:
            self.new_metrics = json.load(f)

        logger.info(f"Loaded {len(self.new_metrics)} new metrics")

    def _stage_compare_metrics(self):
        """
        Compare new metrics with existing nodes using LLM.

        Uses batch processing to efficiently compare each new metric
        against all existing nodes.
        """
        logger.info("Stage: Comparing new metrics with existing nodes")

        # Prepare existing nodes for comparison
        existing_nodes_cropped = []
        for node in self.nodes.values():
            existing_nodes_cropped.append({
                "name": node.name,
                "description": node.description
            })

        # Split into batches
        batched_existing_nodes = self._split_into_batches(
            existing_nodes_cropped,
            self.batch_size
        )

        model = config.get_chat_model()

        # Compare each new metric
        for feature_name in tqdm(self.new_metrics, desc="Comparing metrics"):
            try:
                metric_data = self.new_metrics[feature_name]

                # Prepare description from all datasets
                description_str = ''
                for dataset_name in metric_data['description']:
                    dataset_desc = metric_data['description'][dataset_name]['description']
                    if dataset_desc:
                        description_str += f"{dataset_desc}\n"

                # Compare against all existing nodes in batches
                all_comparisons = []
                for batch in batched_existing_nodes:
                    input_data = {
                        "input_name": feature_name,
                        "input_description": description_str.strip(),
                        "references": batch
                    }

                    prompt = (
                        ENTITY_MERGE_JSON_PROMPT_ALL_V2 +
                        "\nInput:\n" +
                        json.dumps(input_data, indent=2) +
                        "\nOutput:"
                    )
    

                    response = self.llm_client.chat.completions.create(
                        model=model,
                        messages=[
                            {
                                "role": "system",
                                "content": (
                                    CONTEXT_PROMPT +
                                    "You are now at step 3 New Metric Integration. "
                                    "You have finished step 1 and 2, and the initial graph is created."
                                )
                            },
                            {"role": "user", "content": prompt}
                        ],
                        temperature=config.TEMPERATURE
                    )

                    batch_results = parse_llm_response(response.choices[0].message.content)
                    all_comparisons.extend(batch_results)

                # Store all comparison results for this metric
                self.merge_decisions[feature_name] = all_comparisons

            except Exception as e:
                error_msg = f"Error comparing metric {feature_name}: {str(e)}"
                logger.error(error_msg)
                self.failed_updates.append({
                    'metric': feature_name,
                    'stage': 'compare',
                    'error': error_msg
                })

        logger.info(f"Completed comparison for {len(self.merge_decisions)} metrics")

    def _stage_review_merges(self):
        """
        Prepare merge decisions for manual review and load reviewed decisions.

        This stage creates a human-readable review file with ALL potential candidates.
        The user can optionally modify this file to select specific merges or reject them.
        If no review file exists, the pipeline will use default behavior (best match where same_concept=True).

        Note: When running this stage after manual review (step 3), make sure to include
        'load' stage to reload existing nodes: stages=['load', 'review_merges', 'merge', ...]
        """
        logger.info("Stage: Preparing merge decisions for manual review")

        review_file_path = os.path.join(self.data_dir, "merge_review.json")

        # Check if reviewed file already exists
        if os.path.exists(review_file_path):
            logger.info(f"Found existing review file: {review_file_path}")
            logger.info("Loading reviewed merge decisions...")

            with open(review_file_path, 'r') as f:
                review_data = json.load(f)

            # Process reviewed decisions
            for metric_name, review_info in review_data.items():
                if review_info.get('approved', False) and review_info.get('selected_merge'):
                    self.approved_merges[metric_name] = review_info['selected_merge']
                    logger.info(f"Approved merge: '{metric_name}' -> '{review_info['selected_merge']['reference_name']}'")

            logger.info(f"Loaded {len(self.approved_merges)} approved merges from review file")

        else:
            # Prepare merge decisions for review
            logger.info("Preparing merge decisions for manual review...")

            review_data = {}

            for feature_name, comparisons in self.merge_decisions.items():
                # Get ALL candidates above similarity threshold, regardless of same_concept
                all_candidates_above_threshold = []
                candidates_same_concept = []

                for comparison in comparisons:
                    if comparison.get('similarity_score', 0) >= self.similarity_threshold:
                        all_candidates_above_threshold.append(comparison)
                        if comparison.get('same_concept', False):
                            candidates_same_concept.append(comparison)

                # Sort by similarity score (highest first)
                all_candidates_above_threshold.sort(key=lambda x: x.get('similarity_score', 0), reverse=True)
                candidates_same_concept.sort(key=lambda x: x.get('similarity_score', 0), reverse=True)

                # Get metric info
                metric_data = self.new_metrics.get(feature_name, {})
                description_str = ''
                for dataset_name in metric_data.get('description', {}):
                    dataset_desc = metric_data['description'][dataset_name]['description']
                    if dataset_desc:
                        description_str += f"{dataset_desc}\n"

                # Default selection: best match where same_concept=True
                default_merge = candidates_same_concept[0] if candidates_same_concept else None

                review_data[feature_name] = {
                    "new_metric_name": feature_name,
                    "new_metric_description": description_str.strip(),
                    "candidates_same_concept_true": candidates_same_concept,
                    "candidates_same_concept_false": [c for c in all_candidates_above_threshold if not c.get('same_concept', False)],
                    "all_candidates_above_threshold": all_candidates_above_threshold,
                    "selected_merge": default_merge,
                    "approved": True if default_merge else False,
                    "note": "Review all candidates. Set 'approved' to false to create new node. Copy any candidate to 'selected_merge' to choose it."
                }

            # Save review file
            with open(review_file_path, 'w') as f:
                json.dump(review_data, f, indent=2)

            logger.info(f"Saved merge review file to: {review_file_path}")
            logger.info("\n" + "=" * 70)
            logger.info("OPTIONAL MANUAL REVIEW")
            logger.info("=" * 70)
            logger.info(f"Merge review file created at: {review_file_path}")
            logger.info("")
            logger.info("You can optionally review and modify merge decisions:")
            logger.info("1. 'candidates_same_concept_true': LLM thinks these are the same concept")
            logger.info("2. 'candidates_same_concept_false': LLM thinks different but similar")
            logger.info("3. 'all_candidates_above_threshold': All candidates above threshold")
            logger.info("4. Default: best match where same_concept=True (if any)")
            logger.info("5. To use different candidate, copy it to 'selected_merge'")
            logger.info("6. Set 'approved': false to reject merge and create new node")
            logger.info("7. After editing, re-run with:")
            logger.info("   stages=['load', 'review_merges', 'merge', 'create_new', 'create_edges', 'save']")
            logger.info("   (Include 'load' to reload existing nodes)")
            logger.info("")
            logger.info("Pipeline will continue with default selections if not modified.")
            logger.info("=" * 70)
            logger.info("")

            # Load the defaults we just created so pipeline can continue
            for metric_name, review_info in review_data.items():
                if review_info.get('approved', False) and review_info.get('selected_merge'):
                    self.approved_merges[metric_name] = review_info['selected_merge']

    def _stage_merge_nodes(self) -> int:
        """
        Merge new metrics into existing nodes based on approved merge decisions.

        Uses approved_merges if available (from review stage), otherwise falls back
        to automatic selection. Validates that only one merge is selected per feature.

        Returns:
            Number of nodes merged.
        """
        logger.info("Stage: Merging similar nodes")

        merged_count = 0

        # Use approved merges if available, otherwise use automatic selection
        if self.approved_merges:
            logger.info(f"Using {len(self.approved_merges)} approved merges from review")
            metrics_to_merge = self.approved_merges
        else:
            logger.info("No approved merges found, using automatic selection")
            # Build automatic selection - take only the FIRST same_concept=True candidate
            metrics_to_merge = {}
            for feature_name, comparisons in self.merge_decisions.items():
                merge_candidate = None
                for comparison in comparisons:
                    if comparison.get('same_concept', False):
                        if comparison.get('similarity_score', 0) >= self.similarity_threshold:
                            merge_candidate = comparison
                            break
                if merge_candidate:
                    metrics_to_merge[feature_name] = merge_candidate

        # Validate: ensure only one merge per feature name
        for feature_name, merge_candidate in tqdm(
            metrics_to_merge.items(),
            desc="Merging nodes"
        ):
            # Skip if already successfully merged in previous run
            if feature_name in self.successful_merges:
                logger.debug(f"Skipping '{feature_name}' - already merged in previous run")
                merged_count += 1
                continue

            if not merge_candidate:
                logger.warning(f"No merge candidate for '{feature_name}', skipping")
                self.non_merge_list.append(feature_name)
                continue

            # Validate it's a single candidate, not a list
            if isinstance(merge_candidate, list):
                logger.error(f"Multiple merge candidates found for '{feature_name}'. Only one allowed.")
                logger.error("Please review the merge_review.json file and select only one candidate.")
                self.failed_updates.append({
                    'metric': feature_name,
                    'stage': 'merge',
                    'error': 'Multiple merge candidates selected - only one allowed per feature'
                })
                self.non_merge_list.append(feature_name)
                continue

            if merge_candidate:
                # Find the existing node to merge into
                reference_name = merge_candidate['reference_name']
                existing_node = None

                for node in self.nodes.values():
                    if node.name == reference_name:
                        existing_node = node
                        break

                if existing_node:
                    # Update existing node with new dataset information
                    metric_data = self.new_metrics[feature_name]

                    # Add dataSource information
                    if not hasattr(existing_node, 'dataSource') or existing_node.dataSource is None:
                        existing_node.dataSource = {}

                    for dataset_name in metric_data['description']:
                        dataset_info = metric_data['description'][dataset_name]
                        if dataset_info['description']:
                            new_dict = {
                                'feature_name': feature_name,
                                'description': dataset_info['description'],
                                'range': dataset_info.get('range', ''),
                                'unit': dataset_info.get('unit', ''),
                                'type': 'date'  # or timestamp based on data
                            }
                            existing_node.dataSource[dataset_name] = new_dict

                    merged_count += 1
                    self.successful_merges.add(feature_name)
                    logger.info(
                        f"Merged '{feature_name}' into existing node '{reference_name}'"
                    )
                else:
                    logger.warning(
                        f"Could not find existing node '{reference_name}' for merge"
                    )
                    self.non_merge_list.append(feature_name)
            else:
                # No merge candidate found
                self.non_merge_list.append(feature_name)

        # Add metrics that were not in merge candidates to non_merge_list
        # These are metrics that had no merge candidates at all or were rejected during review
        metrics_with_candidates = set(metrics_to_merge.keys())
        all_metrics = set(self.new_metrics.keys())

        for metric_name in all_metrics:
            if metric_name not in metrics_with_candidates:
                # This metric had no merge candidate or was rejected
                if metric_name not in self.non_merge_list:
                    self.non_merge_list.append(metric_name)
                    logger.debug(f"No merge candidate for '{metric_name}', will create new node")

        logger.info(f"Merged {merged_count} metrics into existing nodes")
        logger.info(f"{len(self.non_merge_list)} metrics need new nodes")

        return merged_count

    def _stage_create_new_nodes(self) -> int:
        """
        Create new nodes for metrics that don't match existing nodes.

        Returns:
            Number of new nodes created.
        """
        logger.info("Stage: Creating new nodes")

        new_nodes_count = 0

        for feature_name in tqdm(self.non_merge_list, desc="Creating new nodes"):
            # Skip if already successfully created in previous run
            if feature_name in self.successful_new_nodes:
                logger.debug(f"Skipping '{feature_name}' - already created in previous run")
                new_nodes_count += 1
                continue

            # Check if already exists
            if feature_name in self.node_id_map.values():
                logger.debug(f"Node '{feature_name}' already exists, skipping")
                continue

            metric_data = self.new_metrics[feature_name]

            # Prepare descriptions from all datasets
            description_str = ''
            range_str = ''
            datasource = {}

            for dataset_name in metric_data['description']:
                dataset_info = metric_data['description'][dataset_name]
                if dataset_info['description']:
                    description_str += f"{dataset_info['description']}\n"

                    # Build dataSource
                    new_dict = {
                        'feature_name': feature_name,
                        'description': dataset_info['description'],
                        'range': dataset_info.get('range', ''),
                        'unit': dataset_info.get('unit', ''),
                        'type': 'date'
                    }
                    datasource[dataset_name] = new_dict

            # Generate new node
            try:
                new_node, failed_update = self.node_gen.generate_node(
                    name=feature_name,
                    type=metric_data.get('type', 'unknown'),
                    provided_name=feature_name,
                    provided_description=description_str.strip(),
                    provided_range=range_str,
                    verbose=False,
                    updates=['umls', 'web_search', 'ref_filter', 'llm_output']
                )

                if failed_update:
                    self.failed_updates.append({
                        'metric': feature_name,
                        'stage': 'create_new',
                        'errors': failed_update
                    })
                else:
                    # Add dataSource
                    new_node.dataSource = datasource

                    # Add to nodes
                    self.nodes[new_node.id] = new_node
                    self.node_id_map[new_node.id] = feature_name
                    new_nodes_count += 1
                    self.successful_new_nodes.add(feature_name)

                    logger.info(f"Created new node: '{feature_name}'")

            except Exception as e:
                error_msg = f"Error creating node for {feature_name}: {str(e)}"
                logger.error(error_msg)
                self.failed_updates.append({
                    'metric': feature_name,
                    'stage': 'create_new',
                    'error': error_msg
                })

        logger.info(f"Created {new_nodes_count} new nodes")
        return new_nodes_count

    def _stage_create_edges(self) -> int:
        """
        Create edges between new nodes and existing nodes, and between new nodes.

        Returns:
            Number of new edges created.
        """
        logger.info("Stage: Creating edges for new nodes")

        # Identify which nodes are new (in non_merge_list)
        new_node_ids = []
        for node_id, node_name in self.node_id_map.items():
            if node_name in self.non_merge_list:
                new_node_ids.append(node_id)

        if not new_node_ids:
            logger.info("No new nodes to create edges for")
            return 0

        # Generate pairs: (existing_node_id, new_node_id) AND (new_node_id, new_node_id)
        feature_pairs = []

        # Edges between existing and new nodes
        for existing_id in self.nodes.keys():
            if existing_id not in new_node_ids:
                for new_id in new_node_ids:
                    feature_pairs.append((existing_id, new_id))

        # Edges between new nodes (all combinations)
        for i, new_id_1 in enumerate(new_node_ids):
            for new_id_2 in new_node_ids[i+1:]:  # Avoid duplicates and self-loops
                feature_pairs.append((new_id_1, new_id_2))

        logger.info(f"Generating {len(feature_pairs)} new edges")

        new_edges_count = 0

        for node_id_1, node_id_2 in tqdm(feature_pairs, desc="Creating edges"):
            # Skip if already successfully created in previous run
            edge_pair = tuple(sorted([node_id_1, node_id_2]))
            if edge_pair in self.successful_edges:
                logger.debug(f"Skipping edge {node_id_1} <-> {node_id_2} - already created in previous run")
                new_edges_count += 1
                continue

            # Check if edge already exists
            if self._edge_exists(node_id_1, node_id_2):
                continue

            node_1 = self.nodes[node_id_1]
            node_2 = self.nodes[node_id_2]

            try:
                edge, failed_update = self.edge_gen.generate_edge(
                    entity_1_name=node_1.name,
                    entity_1_id=node_1.id,
                    entity_1_description=node_1.description,
                    entity_2_name=node_2.name,
                    entity_2_id=node_2.id,
                    entity_2_description=node_2.description,
                    verbose=False
                )

                if failed_update:
                    self.failed_updates.append({
                        'entities': [node_id_1, node_id_2],
                        'stage': 'create_edges',
                        'errors': failed_update
                    })
                else:
                    self.edges[edge.id] = edge
                    self.relationship_id_map[edge.id] = [node_id_1, node_id_2]
                    new_edges_count += 1
                    edge_pair = tuple(sorted([node_id_1, node_id_2]))
                    self.successful_edges.add(edge_pair)

            except Exception as e:
                error_msg = f"Error creating edge between {node_1.name} and {node_2.name}: {str(e)}"
                logger.error(error_msg)
                self.failed_updates.append({
                    'entities': [node_id_1, node_id_2],
                    'stage': 'create_edges',
                    'error': error_msg
                })

        logger.info(f"Created {new_edges_count} new edges")
        return new_edges_count

    def _edge_exists(self, node_id_1: str, node_id_2: str) -> bool:
        """Check if an edge already exists between two nodes."""
        for edge in self.edges.values():
            if ((edge.entity_1_id == node_id_1 and edge.entity_2_id == node_id_2) or
                (edge.entity_1_id == node_id_2 and edge.entity_2_id == node_id_1)):
                return True
        return False

    def _save_results(self):
        """Save updated nodes, edges, and tracking information."""
        logger.info("Saving results")

        # Save nodes
        nodes_dict = {
            node_id: node.to_dict()
            for node_id, node in self.nodes.items()
        }
        nodes_path = os.path.join(self.data_dir, "nodes.json")
        with open(nodes_path, 'w') as f:
            json.dump(nodes_dict, f, indent=2)
        logger.info(f"Saved {len(nodes_dict)} nodes to {nodes_path}")

        # Save node ID map
        id_map_path = os.path.join(self.data_dir, "node_id_map.json")
        with open(id_map_path, 'w') as f:
            json.dump(self.node_id_map, f, indent=2)
        logger.info(f"Saved node ID map to {id_map_path}")

        # Save edges
        edges_dict = {
            edge_id: edge.to_dict()
            for edge_id, edge in self.edges.items()
        }
        edges_path = os.path.join(self.data_dir, "edges.json")
        with open(edges_path, 'w') as f:
            json.dump(edges_dict, f, indent=2)
        logger.info(f"Saved {len(edges_dict)} edges to {edges_path}")

        # Save relationship ID map
        rel_map_path = os.path.join(self.data_dir, "relationship_id_map.json")
        with open(rel_map_path, 'w') as f:
            json.dump(self.relationship_id_map, f, indent=2)
        logger.info(f"Saved relationship ID map to {rel_map_path}")

        # Save merge decisions
        merge_path = os.path.join(self.data_dir, "merge_decisions.json")
        with open(merge_path, 'w') as f:
            json.dump(self.merge_decisions, f, indent=2)
        logger.info(f"Saved merge decisions to {merge_path}")

        # Save non-merge list
        non_merge_path = os.path.join(self.data_dir, "non_merge_list.json")
        with open(non_merge_path, 'w') as f:
            json.dump(self.non_merge_list, f, indent=2)
        logger.info(f"Saved non-merge list to {non_merge_path}")

        # Save failed updates if any
        if self.failed_updates:
            failed_path = os.path.join(self.data_dir, "failed_integration_updates.json")
            with open(failed_path, 'w') as f:
                json.dump(self.failed_updates, f, indent=2)
            logger.warning(f"Saved {len(self.failed_updates)} failed updates to {failed_path}")

        # Save success tracking for idempotent reruns
        if self.successful_merges:
            success_merge_path = os.path.join(self.data_dir, "successful_merges.json")
            with open(success_merge_path, 'w') as f:
                json.dump(list(self.successful_merges), f, indent=2)
            logger.info(f"Saved {len(self.successful_merges)} successful merges")

        if self.successful_new_nodes:
            success_nodes_path = os.path.join(self.data_dir, "successful_new_nodes.json")
            with open(success_nodes_path, 'w') as f:
                json.dump(list(self.successful_new_nodes), f, indent=2)
            logger.info(f"Saved {len(self.successful_new_nodes)} successful new nodes")

        if self.successful_edges:
            success_edges_path = os.path.join(self.data_dir, "successful_edges.json")
            with open(success_edges_path, 'w') as f:
                json.dump([list(edge) for edge in self.successful_edges], f, indent=2)
            logger.info(f"Saved {len(self.successful_edges)} successful edges")

    def _split_into_batches(self, data: List, batch_size: int) -> List[List]:
        """Split data into batches of given size."""
        batches = []
        for i in range(0, len(data), batch_size):
            batch = data[i:i + batch_size]
            batches.append(batch)
        return batches

    def _load_previous_run_state(self):
        """
        Load tracking data from previous pipeline runs for idempotent reruns.

        This allows the pipeline to skip already-successful operations and only
        retry failures when rerun.
        """
        # Load successful merges
        success_merge_path = os.path.join(self.data_dir, "successful_merges.json")
        if os.path.exists(success_merge_path):
            with open(success_merge_path, 'r') as f:
                merge_list = json.load(f)
                self.successful_merges = set(merge_list)
            logger.info(f"Found {len(self.successful_merges)} previously successful merges")

        # Load successful new nodes
        success_nodes_path = os.path.join(self.data_dir, "successful_new_nodes.json")
        if os.path.exists(success_nodes_path):
            with open(success_nodes_path, 'r') as f:
                nodes_list = json.load(f)
                self.successful_new_nodes = set(nodes_list)
            logger.info(f"Found {len(self.successful_new_nodes)} previously created new nodes")

        # Load successful edges
        success_edges_path = os.path.join(self.data_dir, "successful_edges.json")
        if os.path.exists(success_edges_path):
            with open(success_edges_path, 'r') as f:
                edges_list = json.load(f)
                self.successful_edges = set(tuple(edge) for edge in edges_list)
            logger.info(f"Found {len(self.successful_edges)} previously created edges")

        if self.successful_merges or self.successful_new_nodes or self.successful_edges:
            logger.info("Pipeline will skip previously successful operations and retry only failures")


def main():
    """Main entry point for command-line execution."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Dataset Integration Pipeline - Integrate new dataset metrics into existing graph"
    )
    parser.add_argument(
        '--data-dir',
        type=str,
        required=True,
        help='Directory containing existing graph data'
    )
    parser.add_argument(
        '--features-map',
        type=str,
        required=True,
        help='Path to features map JSON with new dataset metrics'
    )
    parser.add_argument(
        '--batch-size',
        type=int,
        default=10,
        help='Batch size for LLM comparison (default: 10)'
    )
    parser.add_argument(
        '--similarity-threshold',
        type=float,
        default=0.7,
        help='Similarity threshold for merging nodes (0-1, default: 0.7)'
    )
    parser.add_argument(
        '--stages',
        nargs='+',
        choices=['load', 'compare', 'review_merges', 'merge', 'create_new', 'create_edges', 'save'],
        default=None,
        help='Pipeline stages to run (default: all)'
    )

    args = parser.parse_args()

    # Create and run pipeline
    pipeline = DatasetIntegrationPipeline(
        data_dir=args.data_dir,
        features_map_file=args.features_map,
        batch_size=args.batch_size,
        similarity_threshold=args.similarity_threshold
    )

    stats = pipeline.run(stages=args.stages)

    print("\n" + "=" * 50)
    print("DATASET INTEGRATION STATISTICS")
    print("=" * 50)
    for key, value in stats.items():
        print(f"{key}: {value}")


if __name__ == "__main__":
    main()
