"""
Evaluation Pipeline

End-to-end pipeline for running evaluations and generating analysis reports.
This pipeline coordinates the evaluation of RAG/KGRAG systems using LLM-as-a-judge.

Key Features:
    - Load and validate evaluation results
    - Run parallel LLM-based evaluations
    - Calculate statistical metrics and win rates
    - Generate visualizations and reports
    - Export results in multiple formats

Usage Example:
    >>> from evaluation.pipelines import EvaluationPipeline
    >>> from evaluation.config.prompts import get_prompt
    >>>
    >>> # Initialize pipeline with default prompt
    >>> pipeline = EvaluationPipeline(
    ...     results_path="data/results.json",
    ...     output_dir="output/analysis",
    ...     chat_client="deepseek",
    ...     run_evaluation=True
    ... )
    >>>
    >>> # Initialize with custom prompt from registry
    >>> prompt = get_prompt('single_rank_v3')
    >>> pipeline = EvaluationPipeline(
    ...     results_path="data/results.json",
    ...     output_dir="output/analysis",
    ...     eval_prompt=prompt,
    ...     run_evaluation=True
    ... )
    >>>
    >>> # Run all stages
    >>> stats = pipeline.run()
    >>>
    >>> # Or run specific stages
    >>> stats = pipeline.run(stages=['load', 'analyze', 'report'])

"""

import os
import json
import logging
from typing import Dict, List, Optional, Any
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from functools import partial

import pandas as pd
from tqdm import tqdm

from ..core.evaluator import BaseEvaluator, ContextEvaluator, ResponseEvaluator
from ..utils.evaluation_utils import calculate_wr, calculate_wr_by, generate_latex_table
from ..config.settings import config

# Configure logging
logger = logging.getLogger(__name__)


class EvaluationPipeline:
    """
    Pipeline for evaluating RAG and knowledge graph-based systems.

    This pipeline orchestrates the complete evaluation workflow:
    1. Load evaluation results data
    2. Run LLM-based evaluations (optional)
    3. Calculate statistics and win rates
    4. Generate LaTeX tables and reports
    5. Save processed results

    Attributes:
        results_path (str): Path to input results file
        output_dir (Path): Output directory for results
        chat_client (str): LLM provider name
        run_evaluation (bool): Whether to run evaluations
        eval_prompt (Optional[str]): Custom evaluation prompt
        results (List[Dict]): Loaded results data
        results_df (pd.DataFrame): Results as DataFrame
        stats (Dict): Pipeline statistics

    Example:
        >>> pipeline = EvaluationPipeline(
        ...     results_path="results/data.json",
        ...     output_dir="results/analysis"
        ... )
        >>> stats = pipeline.run()
        >>> print(f"Processed {stats['total_evaluations']} evaluations")
    """

    def __init__(
        self,
        results_path: str,
        output_dir: str,
        chat_client: str = "deepseek",
        chat_model: Optional[str] = None,
        run_evaluation: bool = True,
        max_workers: int = 10,
        eval_prompt: Optional[str] = None
    ):
        """
        Initialize the evaluation pipeline.

        Args:
            results_path: Path to results JSON file
            output_dir: Directory for output files
            chat_client: LLM provider ("deepseek", "openai", "openrouter", "gemini")
            chat_model: Optional explicit model override for the provider
            run_evaluation: If True, run evaluations; if False, use existing scores
            max_workers: Number of parallel workers for evaluation
            eval_prompt: Custom evaluation prompt string or name from prompt registry.
                        If None, uses default prompt for ResponseEvaluator.

        Raises:
            FileNotFoundError: If results_path doesn't exist
        """
        self.results_path = results_path
        self.output_dir = Path(output_dir)
        self.chat_client = chat_client
        self.chat_model = chat_model
        self.run_evaluation = run_evaluation
        self.max_workers = max_workers
        self.eval_prompt = eval_prompt
        self.max_group_attempts = max(1, config.MAX_GROUP_EVALUATION_ATTEMPTS)

        # Create output directory
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Storage
        self.results: List[Dict] = []
        self.results_df: Optional[pd.DataFrame] = None

        # Evaluators
        self.evaluator: Optional[ResponseEvaluator] = None

        # Statistics
        self.stats: Dict[str, Any] = {}

        logger.info(f"Initialized EvaluationPipeline")
        logger.info(f"  Results: {results_path}")
        logger.info(f"  Output: {output_dir}")
        logger.info(f"  Client: {chat_client}")
        logger.info(f"  Model: {chat_model or 'default'}")
        logger.info(f"  Evaluation attempts per query: {self.max_group_attempts}")

    def run(self, stages: Optional[List[str]] = None) -> Dict[str, Any]:
        """
        Run the evaluation pipeline.

        Args:
            stages: List of stages to run. Options:
                   - 'load': Load results data
                   - 'evaluate': Run LLM evaluations
                   - 'analyze': Calculate statistics
                   - 'report': Generate reports
                   - 'save': Save results
                   If None, runs all stages.

        Returns:
            Dictionary with pipeline statistics including:
                - total_evaluations: Number of evaluations
                - evaluations_run: Number of LLM evaluations performed
                - failed_evaluations: Number of failed evaluations
                - methods_analyzed: Number of methods analyzed

        Example:
            >>> # Run all stages
            >>> stats = pipeline.run()
            >>>
            >>> # Run specific stages
            >>> stats = pipeline.run(stages=['load', 'analyze'])
        """
        all_stages = ['load', 'evaluate', 'analyze', 'report', 'save']
        stages = stages or all_stages

        logger.info("=" * 70)
        logger.info("Starting Evaluation Pipeline")
        logger.info("=" * 70)
        logger.info(f"Stages to run: {stages}")

        # Initialize statistics
        self.stats = {
            'total_evaluations': 0,
            'evaluations_run': 0,
            'failed_evaluations': 0,
            'methods_analyzed': 0
        }

        try:
            # Stage 1: Load data
            if 'load' in stages:
                logger.info("Stage 1: Loading data")
                self._load_data()
                self.stats['total_evaluations'] = len(self.results)

            # Stage 2: Run evaluations (if needed)
            if 'evaluate' in stages and self.run_evaluation:
                logger.info("Stage 2: Running evaluations")
                self._run_evaluations()

            # Stage 3: Analyze results
            if 'analyze' in stages:
                logger.info("Stage 3: Analyzing results")
                self._analyze_results()

            # Stage 4: Generate reports
            if 'report' in stages:
                logger.info("Stage 4: Generating reports")
                self._generate_reports()

            # Stage 5: Save results
            if 'save' in stages:
                logger.info("Stage 5: Saving results")
                self._save_results()

            logger.info("=" * 70)
            logger.info("Pipeline completed successfully")
            logger.info(f"Statistics: {self.stats}")
            logger.info("=" * 70)

        except Exception as e:
            logger.error(f"Pipeline failed: {e}")
            raise

        return self.stats

    def _load_data(self) -> None:
        """
        Load evaluation results from JSON file.

        Raises:
            FileNotFoundError: If results file doesn't exist
            json.JSONDecodeError: If file is not valid JSON
        """
        logger.info(f"Loading results from: {self.results_path}")

        if not os.path.exists(self.results_path):
            raise FileNotFoundError(f"Results file not found: {self.results_path}")

        try:
            with open(self.results_path, 'r') as f:
                data = json.load(f)

            # Handle both list and dict formats
            if isinstance(data, list):
                self.results = data
            elif isinstance(data, dict):
                # If it's a dict, convert to list
                self.results = list(data.values()) if data else []
            else:
                raise ValueError(f"Unexpected data format: {type(data)}")

            logger.info(f"✓ Loaded {len(self.results)} evaluation results")

        except json.JSONDecodeError as e:
            logger.error(f"Invalid JSON file: {e}")
            raise

    def _run_evaluations(self) -> None:
        """
        Run LLM-based evaluations on results in parallel.

        This method processes results grouped by query ID and evaluates
        responses from different methods using the ResponseEvaluator.
        """
        logger.info("Initializing ResponseEvaluator")
        if self.eval_prompt:
            logger.info(f"Using custom evaluation prompt")
            self.evaluator = ResponseEvaluator(
                chat_client=self.chat_client,
                chat_model=self.chat_model,
                eval_prompt=self.eval_prompt
            )
        else:
            self.evaluator = ResponseEvaluator(
                chat_client=self.chat_client,
                chat_model=self.chat_model,
            )

        # Convert results to DataFrame for processing
        if not self.results:
            logger.warning("No results to evaluate")
            return

        # Assuming results are in format with q_id, method, query, response
        logger.info(f"Processing {len(self.results)} items")

        # Group by query ID
        grouped_results = {}
        for item in self.results:
            q_id = item.get('q_id', item.get('id', len(grouped_results)))
            if q_id not in grouped_results:
                grouped_results[q_id] = []
            grouped_results[q_id].append(item)

        evaluated_results = []
        failed_count = 0

        # Process groups in parallel
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            futures = {}

            for q_id, items in grouped_results.items():
                future = executor.submit(self._evaluate_group_with_retries, q_id, items)
                futures[future] = (q_id, items)

            # Collect results with progress bar
            for future in tqdm(as_completed(futures), total=len(futures), desc="Evaluating"):
                q_id, items = futures[future]
                try:
                    results = future.result()
                    evaluated_results.extend(results)
                    self.stats['evaluations_run'] += len(results)
                except Exception as e:
                    logger.error(f"Evaluation failed for query {q_id}: {e}")
                    failed_count += 1
                    evaluated_results.extend(
                        self._build_failed_group_results(
                            items,
                            e,
                            attempts=self.max_group_attempts,
                        )
                    )

        self.results = self._sort_result_items(evaluated_results)
        self.stats['failed_evaluations'] = failed_count

        logger.info(f"✓ Completed {self.stats['evaluations_run']} evaluations")
        logger.info(f"Failed: {failed_count}")

    def _evaluate_group_with_retries(self, q_id: str, items: List[Dict]) -> List[Dict]:
        """Evaluate one query group, retrying transient judge or parse failures."""
        last_error: Optional[Exception] = None

        for attempt in range(1, self.max_group_attempts + 1):
            try:
                evaluated_items = self._evaluate_group(
                    q_id,
                    [dict(item) for item in items],
                )
                for item in evaluated_items:
                    item['eval_attempts'] = attempt
                    item['eval_retry_count'] = attempt - 1
                if attempt > 1:
                    logger.info(
                        "Evaluation succeeded for query %s on attempt %s/%s",
                        q_id,
                        attempt,
                        self.max_group_attempts,
                    )
                return evaluated_items
            except Exception as e:
                last_error = e
                if attempt < self.max_group_attempts:
                    logger.warning(
                        "Evaluation attempt %s/%s failed for query %s: %s",
                        attempt,
                        self.max_group_attempts,
                        q_id,
                        e,
                    )
                else:
                    logger.error(
                        "All %s evaluation attempts failed for query %s: %s",
                        self.max_group_attempts,
                        q_id,
                        e,
                    )

        raise last_error or RuntimeError(f"Evaluation failed for query {q_id}")

    def _evaluate_group(self, q_id: str, items: List[Dict]) -> List[Dict]:
        """
        Evaluate a single query group.

        Args:
            q_id: Query identifier
            items: List of result items for this query

        Returns:
            List of items with evaluation scores added

        Raises:
            Exception: If evaluation fails
        """
        try:
            # Build response dictionary
            response_dict = {}
            query = None

            for item in items:
                method = item.get('method', 'unknown')
                response = item.get('response', '')
                response_dict[method] = response

                if query is None:
                    query = item.get('query', item.get('question', ''))

            if not query or not response_dict:
                logger.warning(f"Skipping query {q_id}: missing query or responses")
                skipped_items = []
                for item in items:
                    skipped_item = dict(item)
                    skipped_item['eval_status'] = 'skipped'
                    skipped_item['eval_error'] = 'missing_query_or_responses'
                    skipped_item['eval_scores'] = None
                    skipped_item['overall_quality'] = None
                    skipped_item['eval_chat_client'] = getattr(self.evaluator, 'chat_client', None)
                    skipped_item['eval_chat_model'] = getattr(self.evaluator, 'chat_model', None)
                    skipped_items.append(skipped_item)
                return skipped_items

            # Evaluate
            scores = self.evaluator.evaluate(query, response_dict)

            # Add scores to items
            for item in items:
                method = item.get('method', 'unknown')
                if method in scores:
                    score_value = scores[method]

                    # Handle different score formats
                    if isinstance(score_value, dict):
                        # Dictionary format (e.g., from 'rank' prompt)
                        item['eval_scores'] = score_value
                        item['overall_quality'] = score_value.get('overall_quality', score_value.get('Overall_quality'))
                    elif isinstance(score_value, (int, float)):
                        # Simple numeric format (e.g., from 'single_rank' prompt)
                        item['eval_scores'] = {'overall_quality': score_value}
                        item['overall_quality'] = score_value
                    elif isinstance(score_value, list) and len(score_value) >= 1:
                        # List format [score, rationale] (e.g., from 'single_rank_v3')
                        item['eval_scores'] = {'overall_quality': score_value[0], 'rationale': score_value[1] if len(score_value) > 1 else None}
                        item['overall_quality'] = score_value[0]
                    else:
                        # Unknown format, store as-is
                        logger.warning(f"Unexpected score format for {method}: {type(score_value)}")
                        item['eval_scores'] = score_value
                        item['overall_quality'] = score_value
                else:
                    item['eval_scores'] = None
                    item['overall_quality'] = None

                item['eval_status'] = 'success'
                item['eval_error'] = None
                item['eval_chat_client'] = getattr(self.evaluator, 'chat_client', None)
                item['eval_chat_model'] = getattr(self.evaluator, 'chat_model', None)

            return items

        except Exception as e:
            logger.error(f"Error evaluating group {q_id}: {e}")
            raise

    def _build_failed_group_results(
        self,
        items: List[Dict],
        error: Exception,
        attempts: Optional[int] = None,
    ) -> List[Dict]:
        """Preserve failed evaluation groups in the output with explicit error metadata."""
        failed_items = []
        error_text = str(error)
        attempts = attempts or 1

        for item in items:
            failed_item = dict(item)
            failed_item['eval_status'] = 'error'
            failed_item['eval_error'] = error_text
            failed_item['eval_scores'] = None
            failed_item['overall_quality'] = None
            failed_item['eval_chat_client'] = getattr(self.evaluator, 'chat_client', None)
            failed_item['eval_chat_model'] = getattr(self.evaluator, 'chat_model', None)
            failed_item['eval_attempts'] = attempts
            failed_item['eval_retry_count'] = max(0, attempts - 1)
            failed_items.append(failed_item)

        return failed_items

    @staticmethod
    def _sort_result_items(items: List[Dict]) -> List[Dict]:
        """Return evaluation items sorted by stable query/method identifiers."""
        return sorted(
            items,
            key=lambda item: (
                str(item.get('dataset', '')),
                str(item.get('user_id', '')),
                str(item.get('q_id', item.get('id', ''))),
                str(item.get('method', '')),
            ),
        )

    def _analyze_results(self) -> None:
        """
        Analyze evaluation results and calculate statistics.

        Converts results to DataFrame format and calculates:
        - Method-level statistics
        - Win rates
        - Distribution metrics
        """
        logger.info("Analyzing evaluation results")

        if not self.results:
            logger.warning("No results to analyze")
            return

        # Convert to DataFrame
        rows = []
        for item in self.results:
            row = {
                'q_id': item.get('q_id', item.get('id', '')),
                'query': item.get('query', item.get('question', '')),
                'method': item.get('method', 'unknown'),
                'overall_quality': item.get('overall_quality'),
                'dataset': item.get('dataset', 'default'),
                'eval_chat_client': item.get('eval_chat_client'),
                'eval_chat_model': item.get('eval_chat_model'),
            }

            # Add evaluation metrics if available
            if 'eval_scores' in item:
                for key, value in item['eval_scores'].items():
                    row[key] = value

            rows.append(row)

        self.results_df = pd.DataFrame(rows)
        sort_columns = [
            column
            for column in ['dataset', 'q_id', 'method']
            if column in self.results_df.columns
        ]
        if sort_columns:
            self.results_df = (
                self.results_df
                .sort_values(by=sort_columns, kind='stable', na_position='last')
                .reset_index(drop=True)
            )
        self.stats['methods_analyzed'] = self.results_df['method'].nunique()

        logger.info(f"✓ Created analysis DataFrame with {len(self.results_df)} rows")
        logger.info(f"Methods: {list(self.results_df['method'].unique())}")

        # Calculate win rates
        if 'overall_quality' in self.results_df.columns:
            win_rates = calculate_wr(self.results_df)
            logger.info(f"Win rates calculated:\n{win_rates}")

    def _generate_reports(self) -> None:
        """
        Generate LaTeX tables and statistical reports.

        Creates:
        - Win rate tables
        - Mean score tables
        - LaTeX formatted output
        """
        if self.results_df is None or len(self.results_df) == 0:
            logger.warning("No results DataFrame available for reporting")
            return

        logger.info("Generating statistical reports")

        try:
            # Calculate win rates
            win_rates = calculate_wr(self.results_df)
            win_rates_path = self.output_dir / "win_rates.csv"
            win_rates.to_csv(win_rates_path)
            logger.info(f"✓ Win rates saved to: {win_rates_path}")

            # Calculate mean scores
            if 'overall_quality' in self.results_df.columns:
                mean_scores = self.results_df.groupby('method')['overall_quality'].mean()
                logger.info(f"Mean scores:\n{mean_scores}")

                # Generate LaTeX table
                self._generate_latex_table(mean_scores, win_rates)

        except Exception as e:
            logger.error(f"Error generating reports: {e}")
            raise

    def _generate_latex_table(
        self,
        mean_scores: pd.Series,
        win_rates: pd.DataFrame
    ) -> None:
        """
        Generate LaTeX formatted table.

        Args:
            mean_scores: Series of mean scores by method
            win_rates: DataFrame of win rates
        """
        logger.info("Generating LaTeX table")

        latex_output_path = self.output_dir / "evaluation_table.tex"

        # Prepare data
        mean_df = pd.DataFrame([mean_scores], index=['all'])
        wr_df = pd.DataFrame([win_rates['win_rate']], index=['all'])

        # Generate table
        import sys
        from io import StringIO

        old_stdout = sys.stdout
        sys.stdout = buffer = StringIO()

        try:
            generate_latex_table(
                mean_df,
                wr_df,
                caption="Evaluation Results",
                label="tab:evaluation_results"
            )
            latex_output = buffer.getvalue()
        finally:
            sys.stdout = old_stdout

        # Save to file
        with open(latex_output_path, 'w') as f:
            f.write(latex_output)

        logger.info(f"✓ LaTeX table saved to: {latex_output_path}")

    def _save_results(self) -> None:
        """
        Save processed results to files.

        Saves:
        - Evaluated results as JSON
        - Results DataFrame as CSV
        - Pipeline statistics as JSON
        """
        logger.info("Saving processed results")

        try:
            # Save evaluated results
            results_path = self.output_dir / "evaluated_results.json"
            with open(results_path, 'w') as f:
                json.dump(self.results, f, indent=2)
            logger.info(f"✓ Evaluated results saved to: {results_path}")

            # Save DataFrame
            if self.results_df is not None and len(self.results_df) > 0:
                df_path = self.output_dir / "results_dataframe.csv"
                self.results_df.to_csv(df_path, index=False)
                logger.info(f"✓ Results DataFrame saved to: {df_path}")

            # Save statistics
            stats_path = self.output_dir / "pipeline_stats.json"
            with open(stats_path, 'w') as f:
                json.dump(self.stats, f, indent=2)
            logger.info(f"✓ Pipeline statistics saved to: {stats_path}")

            logger.info("✓ All results saved successfully")

        except Exception as e:
            logger.error(f"Error saving results: {e}")
            raise
