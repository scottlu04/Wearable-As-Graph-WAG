#!/usr/bin/env python3
"""
WAG Response Generation Experiment Runner

This script provides a unified interface for running all WAG experiments:
    1. General Comparison: Base vs RAG vs StaticGraph vs FullContext vs WAG
       (FullHistory is available as an optional method via --methods)
    2. Global Weight Comparison: Prior vs Pop vs Ind vs Global
    3. Local Weight Comparison: Local vs Global vs Final

The script integrates with the new pipeline infrastructure and provides:
    - Professional logging to files and console
    - Progress tracking with detailed statistics
    - Error handling and recovery
    - Flexible experiment selection
    - Comprehensive result summaries

Usage:
    # Run all experiments
    python3 run_experiments.py

    # Run specific experiments
    python3 run_experiments.py --experiments general global

    # Custom configuration
    python3 run_experiments.py --data-dir resources --output-dir results --verbose

    # Parallel processing with 8 workers
    python3 run_experiments.py --max-workers 8

    # Retry failed queries
    python3 run_experiments.py --qid-filter failed_queries.txt

    # Dry run (check setup without running)
    python3 run_experiments.py --dry-run

Examples:
    # Quick test on small dataset
    python3 run_experiments.py --experiments general --data-dir resources/small

    # Full run with debug logging and parallel processing
    python3 run_experiments.py --verbose --log-dir logs --max-workers 16

    # Run only weight comparisons
    python3 run_experiments.py --experiments global local

    # Retry failed queries with 4 parallel workers
    python3 run_experiments.py --qid-filter filters/failed_run1.json --max-workers 4 --output-dir results/retry

"""

import os
import sys
import argparse
import time
import logging
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Optional
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Add project root to path
# Add both current directory (for local response_generation) and parent (for wag.shared imports)
PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))  # Priority: local imports first
sys.path.insert(1, str(PROJECT_ROOT.parent))  # Then parent for wag.shared imports

# Import pipeline infrastructure
from response_generation import setup_logging
from response_generation.config import SUPPORTED_CHAT_CLIENTS
from response_generation.pipelines import (
    GeneralComparisonPipeline,
    GlobalWeightPipeline,
    LocalWeightPipeline
)

# Set up module logger
logger = logging.getLogger(__name__)


def normalize_methods_arg(raw_methods: Optional[List[str]]) -> Optional[List[str]]:
    """
    Normalize method CLI input.

    Supports both `--methods Base Rag Wag` and `--methods Base,Rag,Wag`.
    """
    if not raw_methods:
        return None

    methods: List[str] = []
    for item in raw_methods:
        methods.extend(
            method.strip()
            for method in item.split(',')
            if method.strip()
        )

    return methods or None


class ExperimentRunner:
    """
    Manages execution of WAG experiments.

    This class orchestrates running multiple experiments with proper logging,
    error handling, and result tracking.
    """

    def __init__(
        self,
        data_dir: str,
        output_dir: str,
        log_dir: str = "logs",
        verbose: bool = False,
        qid_filter_file: Optional[str] = None,
        max_workers: int = 1,
        chat_client: Optional[str] = None,
        chat_model: Optional[str] = None,
        generate_responses: bool = False,
        methods: Optional[List[str]] = None,
    ):
        """
        Initialize the experiment runner.

        Args:
            data_dir: Root directory containing data files
            output_dir: Base directory for experiment results
            log_dir: Directory for log files
            verbose: Enable verbose logging (DEBUG level)
            qid_filter_file: Path to file containing q_ids to process (optional)
            max_workers: Number of parallel workers for query processing (default: 1)
            chat_client: LLM provider for response generation processors
            chat_model: Optional explicit chat model override
            generate_responses: Whether to call the chat model and store
                                generated answers in experiment outputs
            methods: Optional ordered list of method names to run
        """
        self.data_dir = Path(data_dir)
        self.output_dir = Path(output_dir)
        self.log_dir = Path(log_dir)
        self.verbose = verbose
        self.qid_filter_file = qid_filter_file
        self.max_workers = max_workers
        self.chat_client = chat_client
        self.chat_model = chat_model
        self.generate_responses = generate_responses
        self.methods = methods

        # Experiment definitions
        self.experiments = {
            'general': {
                'name': 'General Comparison',
                'description': 'Base vs RAG vs StaticGraph vs FullContext vs WAG',
                'pipeline_class': GeneralComparisonPipeline,
                'output_subdir': 'general'
            },
            'global': {
                'name': 'Global Weight Comparison',
                'description': 'Prior vs Pop vs Ind vs Global',
                'pipeline_class': GlobalWeightPipeline,
                'output_subdir': 'global'
            },
            'local': {
                'name': 'Local Weight Comparison',
                'description': 'Local vs Global vs Final',
                'pipeline_class': LocalWeightPipeline,
                'output_subdir': 'local'
            }
        }

        # Results tracking
        self.results = {}
        self.start_time = None
        self.end_time = None

    def setup(self) -> bool:
        """
        Set up directories and logging.

        Returns:
            True if setup successful, False otherwise
        """
        logger.info("Setting up experiment runner...")

        # Check data directory exists
        if not self.data_dir.exists():
            logger.error(f"Data directory not found: {self.data_dir}")
            logger.error("Please create the directory or specify a valid path with --data-dir")
            return False

        # Create output directory
        self.output_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"Output directory: {self.output_dir}")

        # Create log directory
        self.log_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"Log directory: {self.log_dir}")

        # Check required files/subdirectories
        required_paths = [
            self.data_dir / "query_set",
            self.data_dir / "processed_dataset",
            self.data_dir / "relationship_dict.json",
            self.data_dir / "kg"
        ]

        missing = [p for p in required_paths if not p.exists()]
        if missing:
            logger.warning("Some required paths are missing:")
            for path in missing:
                logger.warning(f"  - {path}")
            logger.warning("Experiments may fail if these are needed")

        # Check q_id filter file if specified
        if self.qid_filter_file:
            filter_path = Path(self.qid_filter_file)
            if not filter_path.exists():
                logger.error(f"Q_ID filter file not found: {self.qid_filter_file}")
                logger.error("Please create the file or specify a valid path with --qid-filter")
                return False
            logger.info(f"Q_ID filter file: {self.qid_filter_file}")
        else:
            logger.info("Q_ID filter: None (processing all queries)")

        logger.info("Setup complete")
        return True

    def run_experiment(
        self,
        exp_key: str,
    ) -> bool:
        """
        Run a single experiment.

        Args:
            exp_key: Experiment key ('general', 'global', or 'local')

        Returns:
            True if successful, False otherwise
        """
        exp_config = self.experiments[exp_key]

        logger.info("=" * 70)
        logger.info(f"EXPERIMENT: {exp_config['name']}")
        logger.info(f"Description: {exp_config['description']}")
        logger.info("=" * 70)



        try:
            # Set up experiment-specific output directory
            exp_output_dir = self.output_dir / exp_config['output_subdir']
            exp_output_dir.mkdir(parents=True, exist_ok=True)

            # Initialize pipeline
            pipeline_class = exp_config['pipeline_class']
            logger.info(f"Initializing {pipeline_class.__name__}...")

            pipeline = pipeline_class(
                root_dir=str(self.data_dir),
                output_dir=str(exp_output_dir),
                max_workers=self.max_workers,
                qid_filter_file=self.qid_filter_file,
                chat_client=self.chat_client,
                chat_model=self.chat_model,
                generate_responses=self.generate_responses,
                methods=self.methods,
            )

            # Run pipeline
            logger.info("Starting pipeline execution...")
            exp_start = time.time()

            results_df = pipeline.run()

            exp_elapsed = time.time() - exp_start

            # Store results
            self.results[exp_key] = {
                'success': True,
                'elapsed_time': exp_elapsed,
                'num_queries': len(results_df) if results_df is not None else 0,
                'output_dir': exp_output_dir
            }

            logger.info("=" * 70)
            logger.info(f"✓ {exp_config['name']} COMPLETED")
            logger.info(f"  Time: {exp_elapsed:.2f}s ({exp_elapsed/60:.2f} min)")
            logger.info(f"  Queries: {self.results[exp_key]['num_queries']}")
            logger.info(f"  Results: {exp_output_dir}")
            logger.info("=" * 70)

            return True

        except Exception as e:
            exp_elapsed = time.time() - exp_start if 'exp_start' in locals() else 0

            logger.error(f"✗ {exp_config['name']} FAILED")
            logger.error(f"  Error: {e}")
            logger.error(f"  Time before failure: {exp_elapsed:.2f}s")
            logger.exception("Exception details:")

            self.results[exp_key] = {
                'success': False,
                'elapsed_time': exp_elapsed,
                'error': str(e)
            }

            return False

    def run_all(
        self,
        experiment_keys: List[str],
    ) -> bool:
        """
        Run multiple experiments.

        Args:
            experiment_keys: List of experiment keys to run
            dry_run: If True, validate but don't run

        Returns:
            True if all successful, False if any failed
        """
        self.start_time = datetime.now()

        logger.info("=" * 70)
        logger.info("WAG RESPONSE GENERATION EXPERIMENTS")
        logger.info("=" * 70)
        logger.info(f"Start time: {self.start_time.strftime('%Y-%m-%d %H:%M:%S')}")
        logger.info(f"Data directory: {self.data_dir}")
        logger.info(f"Output directory: {self.output_dir}")
        logger.info(f"Experiments: {', '.join(experiment_keys)}")
        logger.info(f"Max workers: {self.max_workers} ({'parallel' if self.max_workers > 1 else 'sequential'})")
        logger.info(f"Chat client: {self.chat_client or 'default'}")
        logger.info(f"Chat model: {self.chat_model or 'default'}")
        logger.info(f"Generate responses: {self.generate_responses}")
        logger.info(f"Methods: {self.methods or 'all'}")
        if self.qid_filter_file:
            logger.info(f"Q_ID filter: {self.qid_filter_file}")
        logger.info("=" * 70)

        # Run each experiment
        all_success = True
        for exp_key in experiment_keys:
            if exp_key not in self.experiments:
                logger.error(f"Unknown experiment: {exp_key}")
                all_success = False
                continue

            success = self.run_experiment(exp_key)
            if not success:
                all_success = False

        self.end_time = datetime.now()

        # Print summary
        self._print_summary()

        return all_success

    def _print_summary(self):
        """Print experiment summary."""
        logger.info("")
        logger.info("=" * 70)
        logger.info("EXPERIMENT SUMMARY")
        logger.info("=" * 70)

        if self.end_time and self.start_time:
            total_time = (self.end_time - self.start_time).total_seconds()
            logger.info(f"End time: {self.end_time.strftime('%Y-%m-%d %H:%M:%S')}")
            logger.info(f"Total time: {total_time:.2f}s ({total_time/60:.2f} min)")
            logger.info("")

        # Print individual experiment results
        for exp_key in ['general', 'global', 'local']:
            if exp_key in self.results:
                result = self.results[exp_key]
                exp_name = self.experiments[exp_key]['name']

                if result['success']:
                    status = "✓ PASS"
                    details = f"{result['elapsed_time']:.1f}s, {result['num_queries']} queries"
                else:
                    status = "✗ FAIL"
                    details = result.get('error', 'Unknown error')

                logger.info(f"{exp_name:.<40} {status}")
                logger.info(f"  └─ {details}")

        logger.info("=" * 70)

        # Print final status
        all_success = all(r.get('success', False) for r in self.results.values())

        if all_success:
            logger.info("")
            logger.info("✓ ALL EXPERIMENTS COMPLETED SUCCESSFULLY!")
            logger.info("")
            logger.info(f"Results saved to: {self.output_dir}/")
            for exp_key, result in self.results.items():
                if result['success']:
                    subdir = self.experiments[exp_key]['output_subdir']
                    logger.info(f"  - {subdir}/results_df.json")
            logger.info("")
        else:
            logger.error("")
            logger.error("✗ SOME EXPERIMENTS FAILED")
            logger.error("")
            logger.error("Check the logs above for error details")
            logger.error("")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Run WAG response generation experiments",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run all experiments
  python3 run_experiments.py

  # Run specific experiments
  python3 run_experiments.py --experiments general global

  # Custom directories with verbose logging
  python3 run_experiments.py --data-dir resources --output-dir results --verbose

  # Parallel processing with 8 workers
  python3 run_experiments.py --max-workers 8

  # Run with OpenRouter and an explicit model
  python3 run_experiments.py --experiments general --chat-client openrouter --chat-model anthropic/claude-3.5-sonnet

  # Generate final answers with the selected model
  python3 run_experiments.py --experiments general --generate-response

  # Compare selected methods only
  python3 run_experiments.py --experiments general --methods Base Rag StaticGraph FullContext Wag

  # Retry failed queries using filter file
  python3 run_experiments.py --qid-filter failed_queries.txt --max-workers 4

  # Dry run to check setup
  python3 run_experiments.py --dry-run

  # Debug mode with detailed logs
  python3 run_experiments.py --verbose --log-dir logs/debug

  # Full retry workflow with parallel processing
  python3 run_experiments.py --experiments general --qid-filter filters/retry_run1.json --max-workers 16 --output-dir results/retry
        """
    )

    parser.add_argument(
        '--data-dir',
        type=str,
        default='resources',
        help='Root data directory containing QA data, KG, etc. (default: resources)'
    )

    parser.add_argument(
        '--output-dir',
        type=str,
        default='results',
        help='Output directory for experiment results (default: results)'
    )

    parser.add_argument(
        '--log-dir',
        type=str,
        default='logs',
        help='Directory for log files (default: logs)'
    )

    parser.add_argument(
        '--experiments',
        nargs='+',
        choices=['general', 'global', 'local', 'all'],
        default=['all'],
        help='Which experiments to run (default: all)'
    )

    parser.add_argument(
        '--verbose',
        action='store_true',
        help='Enable verbose logging (DEBUG level)'
    )

    parser.add_argument(
        '--dry-run',
        action='store_true',
        help='Validate setup without running experiments'
    )

    parser.add_argument(
        '--log-to-console',
        action='store_true',
        default=True,
        help='Log to console (default: True)'
    )

    parser.add_argument(
        '--qid-filter',
        type=str,
        default=None,
        help='Path to file containing q_ids to process (for experiment retry). '
             'Supports text files (one q_id per line) or JSON files (list or dict with "q_ids" key)'
    )

    parser.add_argument(
        '--max-workers',
        type=int,
        default=1,
        help='Maximum number of parallel workers for query processing (default: 1 for sequential). '
             'Use higher values (e.g., 4, 8, 16) for parallel processing'
    )

    parser.add_argument(
        '--chat-client',
        type=str,
        choices=sorted(SUPPORTED_CHAT_CLIENTS),
        default=None,
        help='LLM provider for response generation processors (default: provider default in code)'
    )

    parser.add_argument(
        '--chat-model',
        type=str,
        default=None,
        help='Optional explicit chat model override, e.g. gpt-4o, deepseek-chat, or openai/gpt-4o-mini via OpenRouter'
    )

    parser.add_argument(
        '--generate-response',
        action='store_true',
        help='Call the chat model to generate final answers during experiments. '
             'By default, experiments only build context/retrieval outputs.'
    )

    parser.add_argument(
        '--methods',
        nargs='+',
        default=None,
        help='Optional method names to run for the selected experiment. '
             'Accepts space-separated or comma-separated names, e.g. '
             '`--methods Base Rag Wag` or `--methods Base,Rag,Wag`.'
    )

    args = parser.parse_args()
    methods = normalize_methods_arg(args.methods)

    # Set up logging
    log_level = logging.DEBUG if args.verbose else logging.INFO

    # Create log directory
    log_dir = Path(args.log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)

    # Set up logging with timestamp
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_filename = f"experiments_{timestamp}.log"

    setup_logging(
        level=log_level,
        log_file=log_filename,
        log_dir=str(log_dir)
    )

    log_file = log_dir / log_filename

    logger.info("=" * 70)
    logger.info("WAG Experiment Runner v2.0")
    logger.info("=" * 70)
    logger.info(f"Log file: {log_file}")
    logger.info(f"Log level: {logging.getLevelName(log_level)}")
    logger.info("")

    # Determine which experiments to run
    experiments_to_run = set(args.experiments)
    if 'all' in experiments_to_run:
        experiments_to_run = {'general', 'global', 'local'}
    experiments_to_run = sorted(experiments_to_run)

    # Create experiment runner
    runner = ExperimentRunner(
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        log_dir=args.log_dir,
        verbose=args.verbose,
        qid_filter_file=args.qid_filter,
        max_workers=args.max_workers,
        chat_client=args.chat_client,
        chat_model=args.chat_model,
        generate_responses=args.generate_response,
        methods=methods,
    )

    # Setup
    if not runner.setup():
        logger.error("Setup failed. Exiting.")
        return 1

    # Run experiments
    success = runner.run_all(experiments_to_run)

    # Exit with appropriate code
    return 0 if success else 1


if __name__ == "__main__":
    sys.exit(main())
