"""
Run Evaluation Pipeline

Execute evaluation pipelines for different analysis types:
- general: Comprehensive evaluation across all methods and datasets
- global: Dataset-level analysis with cross-dataset comparison
- local: Query-level analysis with pattern identification

Usage:
    # List available evaluation prompts
    python run_evaluation.py --list-prompts

    # Run all evaluation types
    python run_evaluation.py --all

    # Run specific evaluation type
    python run_evaluation.py --type general
    python run_evaluation.py --type global
    python run_evaluation.py --type local

    # Run with evaluation (uses LLM-as-a-judge)
    python run_evaluation.py --type general --evaluate

    # Use custom evaluation prompt
    python run_evaluation.py --type general --evaluate --eval-prompt single_rank_v3

    # Specify custom paths
    python run_evaluation.py --type general --results data/my_results.json

"""

import argparse
import logging
import sys
from pathlib import Path
from typing import Dict, Any, Optional

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from evaluation.pipelines.evaluation_pipeline import EvaluationPipeline
from evaluation.logging_config import setup_logging
from evaluation.config.settings import config
from evaluation.config.prompts import get_prompt, list_prompts, validate_prompt_name

# Set up logging
setup_logging(level=logging.INFO, log_dir='logs')
logger = logging.getLogger(__name__)


class EvaluationRunner:
    """
    Runner for executing different types of evaluation pipelines.

    Coordinates the execution of general, global, and local evaluations
    with appropriate configurations for each analysis type.

    Attributes:
        base_dir (Path): Base directory for evaluation data
        output_base (Path): Base output directory
        chat_client (str): LLM provider for evaluations
        run_evaluation (bool): Whether to run LLM evaluations
        max_workers (int): Number of parallel workers
        eval_prompt (Optional[str]): Evaluation prompt string or name
    """

    def __init__(
        self,
        base_dir: str = "data",
        output_base: str = "results",
        chat_client: str = "deepseek",
        chat_model: Optional[str] = None,
        run_evaluation: bool = True,
        max_workers: int = 10,
        eval_prompt: Optional[str] = None
    ):
        """
        Initialize the evaluation runner.

        Args:
            base_dir: Directory containing evaluation data
            output_base: Base directory for outputs
            chat_client: LLM provider (deepseek, openai, openrouter, gemini)
            chat_model: Optional explicit model override for the provider
            run_evaluation: Whether to run LLM-based evaluations
            max_workers: Number of parallel workers
            eval_prompt: Evaluation prompt name from registry or custom prompt string
        """
        self.base_dir = Path(base_dir)
        self.output_base = Path(output_base)
        self.chat_client = chat_client
        self.chat_model = chat_model
        self.run_evaluation = run_evaluation
        self.max_workers = max_workers

        # Resolve eval_prompt if it's a name in the registry
        if eval_prompt and validate_prompt_name(eval_prompt):
            logger.info(f"Using prompt from registry: {eval_prompt}")
            self.eval_prompt = get_prompt(eval_prompt)
        else:
            self.eval_prompt = eval_prompt

        logger.info("="*70)
        logger.info("Evaluation Runner Initialized")
        logger.info("="*70)
        logger.info(f"Data directory: {self.base_dir}")
        logger.info(f"Output directory: {self.output_base}")
        logger.info(f"Chat client: {self.chat_client}")
        logger.info(f"Chat model: {self.chat_model or 'default'}")
        logger.info(f"Run evaluation: {self.run_evaluation}")
        logger.info("="*70)

    def run_general_evaluation(
        self,
        results_file: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Run general evaluation (comprehensive analysis across all methods).

        This evaluation provides:
        - Overall method comparison
        - Mean scores and win rates
        - Comprehensive visualizations
        - Statistical summaries

        Args:
            results_file: Custom results file path (optional)

        Returns:
            Dictionary with pipeline statistics
        """
        logger.info("\n" + "="*70)
        logger.info("RUNNING GENERAL EVALUATION")
        logger.info("="*70)

        # Configuration
        results_path = results_file or str(self.base_dir / "results_df.json")
        output_dir = str(self.output_base / "general")

        logger.info(f"Input: {results_path}")
        logger.info(f"Output: {output_dir}")

        # Create and run pipeline
        pipeline = EvaluationPipeline(
            results_path=results_path,
            output_dir=output_dir,
            chat_client=self.chat_client,
            chat_model=self.chat_model,
            run_evaluation=self.run_evaluation,
            max_workers=self.max_workers,
            eval_prompt=self.eval_prompt
        )


        stages = ['load','save']#, 'analyze', 'report', 'save']
        if self.run_evaluation:
            stages.insert(1, 'evaluate')

        stats = pipeline.run(stages=stages)

        logger.info("\n" + "="*70)
        logger.info("GENERAL EVALUATION COMPLETE")
        logger.info("="*70)
        logger.info(f"Total evaluations: {stats.get('total_evaluations', 0)}")
        logger.info(f"Methods analyzed: {stats.get('methods_analyzed', 0)}")
        logger.info(f"Results saved to: {output_dir}")
        logger.info("="*70 + "\n")

        return stats

    def run_global_evaluation(
        self,
        results_file: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Run global evaluation (dataset-level analysis).

        This evaluation provides:
        - Performance by dataset
        - Cross-dataset comparison
        - Dataset-specific win rates
        - Statistical significance testing

        Args:
            results_file: Custom results file path (optional)

        Returns:
            Dictionary with pipeline statistics
        """
        logger.info("\n" + "="*70)
        logger.info("RUNNING GLOBAL EVALUATION (Dataset-Level)")
        logger.info("="*70)

        # Configuration
        results_path = results_file or str(self.base_dir / "results_df.json")
        output_dir = str(self.output_base / "global")

        logger.info(f"Input: {results_path}")
        logger.info(f"Output: {output_dir}")

        # Create and run pipeline
        pipeline = EvaluationPipeline(
            results_path=results_path,
            output_dir=output_dir,
            chat_client=self.chat_client,
            chat_model=self.chat_model,
            run_evaluation=self.run_evaluation,
            max_workers=self.max_workers,
            eval_prompt=self.eval_prompt
        )

        # Run analysis stages (skip evaluation if data is pre-evaluated)
        stages = ['load', 'save']
        if self.run_evaluation:
            stages.insert(1, 'evaluate')

        stats = pipeline.run(stages=stages)

        logger.info("\n" + "="*70)
        logger.info("GLOBAL EVALUATION COMPLETE")
        logger.info("="*70)
        logger.info(f"Total evaluations: {stats.get('total_evaluations', 0)}")
        logger.info(f"Methods analyzed: {stats.get('methods_analyzed', 0)}")
        logger.info(f"Results saved to: {output_dir}")
        logger.info("="*70 + "\n")

        return stats

    def run_local_evaluation(
        self,
        results_file: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Run local evaluation (query-level analysis).

        This evaluation provides:
        - Individual query performance
        - Query characteristic analysis
        - Winner distribution
        - Edge case identification

        Args:
            results_file: Custom results file path (optional)

        Returns:
            Dictionary with pipeline statistics
        """
        logger.info("\n" + "="*70)
        logger.info("RUNNING LOCAL EVALUATION (Query-Level)")
        logger.info("="*70)

        # Configuration
        results_path = results_file or str(self.base_dir / "results_df.json")
        output_dir = str(self.output_base / "local")

        logger.info(f"Input: {results_path}")
        logger.info(f"Output: {output_dir}")

        # Create and run pipeline
        pipeline = EvaluationPipeline(
            results_path=results_path,
            output_dir=output_dir,
            chat_client=self.chat_client,
            chat_model=self.chat_model,
            run_evaluation=self.run_evaluation,
            max_workers=self.max_workers,
            eval_prompt=self.eval_prompt
        )

        # Run analysis stages
        stages = ['load', 'save']
        if self.run_evaluation:
            stages.insert(1, 'evaluate')

        stats = pipeline.run(stages=stages)

        logger.info("\n" + "="*70)
        logger.info("LOCAL EVALUATION COMPLETE")
        logger.info("="*70)
        logger.info(f"Total evaluations: {stats.get('total_evaluations', 0)}")
        logger.info(f"Methods analyzed: {stats.get('methods_analyzed', 0)}")
        logger.info(f"Results saved to: {output_dir}")
        logger.info("="*70 + "\n")

        return stats

    def run_all(
        self,
        general_results: Optional[str] = None,
        global_results: Optional[str] = None,
        local_results: Optional[str] = None
    ) -> Dict[str, Dict[str, Any]]:
        """
        Run all three evaluation types in sequence.

        Args:
            general_results: Custom results file for general evaluation
            global_results: Custom results file for global evaluation
            local_results: Custom results file for local evaluation

        Returns:
            Dictionary with statistics for all three evaluations
        """
        logger.info("\n" + "="*70)
        logger.info("RUNNING ALL EVALUATIONS")
        logger.info("="*70 + "\n")

        all_stats = {}

        try:
            # Run general evaluation
            all_stats['general'] = self.run_general_evaluation(general_results)

            # Run global evaluation
            all_stats['global'] = self.run_global_evaluation(global_results)

            # Run local evaluation
            all_stats['local'] = self.run_local_evaluation(local_results)

            logger.info("\n" + "="*70)
            logger.info("ALL EVALUATIONS COMPLETE")
            logger.info("="*70)
            logger.info("Summary:")
            for eval_type, stats in all_stats.items():
                logger.info(f"  {eval_type.upper()}: {stats.get('total_evaluations', 0)} evaluations")
            logger.info("="*70 + "\n")

        except Exception as e:
            logger.error(f"Error during evaluation: {e}")
            raise

        return all_stats


def main():
    """Main entry point for the evaluation runner."""
    parser = argparse.ArgumentParser(
        description="Run evaluation pipelines for RAG/KGRAG systems",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # List available prompts
  python run_evaluation.py --list-prompts

  # Run all evaluation types
  python run_evaluation.py --all

  # Run specific evaluation
  python run_evaluation.py --type general
  python run_evaluation.py --type global
  python run_evaluation.py --type local

  # Run with LLM evaluation
  python run_evaluation.py --type general --evaluate

  # Use custom evaluation prompt
  python run_evaluation.py --type general --evaluate --eval-prompt single_rank_v3
  python run_evaluation.py --type general --evaluate --eval-prompt val_v2

  # Run with OpenRouter and an explicit model
  python run_evaluation.py --type general --evaluate --chat-client openrouter --chat-model anthropic/claude-3.5-sonnet

  # Custom paths
  python run_evaluation.py --type general --results data/my_results.json
  python run_evaluation.py --type general --output results/my_analysis
        """
    )

    # Evaluation type
    parser.add_argument(
        '--type',
        choices=['general', 'global', 'local'],
        help='Type of evaluation to run'
    )
    parser.add_argument(
        '--all',
        action='store_true',
        help='Run all evaluation types'
    )

    # Paths
    parser.add_argument(
        '--data-dir',
        default='data',
        help='Directory containing evaluation data (default: data)'
    )
    parser.add_argument(
        '--output-dir',
        default='results',
        help='Base output directory (default: results)'
    )
    parser.add_argument(
        '--results',
        help='Custom results file path'
    )

    # Evaluation settings
    parser.add_argument(
        '--evaluate',
        action='store_true',
        help='Run LLM-based evaluations (requires API keys)'
    )
    parser.add_argument(
        '--chat-client',
        choices=sorted(config.SUPPORTED_CHAT_CLIENTS),
        default=config.DEFAULT_CHAT_CLIENT,
        help=f'LLM provider for evaluations (default: {config.DEFAULT_CHAT_CLIENT})'
    )
    parser.add_argument(
        '--chat-model',
        type=str,
        default=None,
        help='Optional explicit chat model override, e.g. gpt-4o, deepseek-chat, openai/gpt-4o-mini, anthropic/claude-3.5-sonnet'
    )
    parser.add_argument(
        '--max-workers',
        type=int,
        default=10,
        help='Number of parallel workers (default: 10)'
    )
    parser.add_argument(
        '--eval-prompt',
        type=str,
        help='Evaluation prompt name from registry (e.g., "single_rank_v3") or custom prompt string'
    )
    parser.add_argument(
        '--list-prompts',
        action='store_true',
        help='List all available evaluation prompts and exit'
    )

    args = parser.parse_args()

    # Handle --list-prompts
    if args.list_prompts:
        print("Available Evaluation Prompts:")
        print("=" * 70)
        prompts = list_prompts(include_descriptions=True)
        for name, desc in prompts:
            print(f"\n{name}:")
            print(f"  {desc}")
        print("\n" + "=" * 70)
        print(f"\nDefault response prompt: rank")
        print(f"Default context prompt: context")
        print("\nUsage: --eval-prompt <prompt_name>")
        sys.exit(0)

    # Validate arguments
    if not args.all and not args.type:
        parser.error("Must specify either --all or --type")

    # Create runner
    runner = EvaluationRunner(
        base_dir=args.data_dir,
        output_base=args.output_dir,
        chat_client=args.chat_client,
        chat_model=args.chat_model,
        run_evaluation=args.evaluate,
        max_workers=args.max_workers,
        eval_prompt=args.eval_prompt
    )

    try:
        # Run evaluations
        if args.all:
            runner.run_all()
        elif args.type == 'general':
            runner.run_general_evaluation(args.results)
        elif args.type == 'global':
            runner.run_global_evaluation(args.results)
        elif args.type == 'local':
            runner.run_local_evaluation(args.results)

        logger.info("\n✓ Evaluation completed successfully!\n")

    except FileNotFoundError as e:
        logger.error(f"File not found: {e}")
        logger.error("Please check that your data files exist in the specified directory")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Evaluation failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
