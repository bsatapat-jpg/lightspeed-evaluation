"""LightSpeed Evaluation Framework - Metric Suggestion Runner.

This module provides a CLI for suggesting evaluation metrics based on
user's use case descriptions.
"""

import argparse
import sys
from typing import Optional

from lightspeed_evaluation.core.suggest.suggester import run_metric_suggestion
from lightspeed_evaluation.core.system import ConfigLoader


def run_suggestion(args: argparse.Namespace) -> Optional[dict[str, str]]:
    """Run the metric suggestion workflow.

    Args:
        args: Parsed command line arguments

    Returns:
        Dictionary with paths to generated files, or None on failure
    """
    print("🔍 LightSpeed Metric Suggester")
    print("=" * 50)

    try:
        # Load system configuration
        print("🔧 Loading Configuration...")
        loader = ConfigLoader()
        system_config = loader.load_system_config(args.system_config)

        # Override suggest config with CLI arguments if provided
        if args.prompt:
            system_config.suggest.enabled = True
            system_config.suggest.prompt = args.prompt

        if args.output:
            system_config.suggest.output_file = args.output

        # Check if suggestion is enabled
        if not system_config.suggest.enabled:
            print("\n⚠️ Metric suggestion is not enabled.")
            print("   Enable it in your system config with 'suggest.enabled: true'")
            print("   Or provide a prompt with --prompt argument")
            return None

        if not system_config.suggest.prompt:
            print("\n⚠️ No use case prompt provided.")
            print("   Set 'suggest.prompt' in your config or use --prompt argument")
            return None

        # Run the suggestion workflow
        run_metric_suggestion(system_config.suggest, system_config.llm)

        return {
            "config_file": system_config.suggest.output_file,
            "eval_data_file": system_config.suggest.output_file.replace(
                ".yaml", "_eval_data.yaml"
            ),
        }

    except (FileNotFoundError, ValueError, RuntimeError) as e:
        print(f"\n❌ Metric suggestion failed: {e}")
        return None


def main() -> int:
    """Command line interface for metric suggestion."""
    parser = argparse.ArgumentParser(
        description=(
            "LightSpeed Metric Suggester - "
            "Identify appropriate metrics for your evaluation"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Using configuration file
  python -m lightspeed_evaluation.runner.suggest --system-config config/system.yaml

  # Using CLI prompt
  python -m lightspeed_evaluation.runner.suggest --prompt "I want to evaluate RAG application responses for accuracy and relevance"

  # Specify output file
  python -m lightspeed_evaluation.runner.suggest --prompt "Testing AI agent tool calls" --output my_config.yaml

Use Case Prompt Guidelines:
  Your prompt should describe:
  - What type of AI/LLM application you're evaluating (chatbot, RAG, agent, etc.)
  - What aspects you want to evaluate (accuracy, relevance, tool usage, etc.)
  - What data you have available (expected responses, contexts, etc.)
  - Any specific requirements or constraints

Example prompts:
  "I'm evaluating a RAG-based Q&A system that retrieves context from documentation.
   I need to verify that responses are accurate, grounded in the context, and
   relevant to user questions. I have expected answers for each query."

  "Testing an AI agent that uses function calling to interact with Kubernetes.
   I need to verify correct tool selection and argument values. Responses should
   also be technically accurate."
        """,
    )
    parser.add_argument(
        "--system-config",
        default="config/system.yaml",
        help="Path to system configuration file (default: config/system.yaml)",
    )
    parser.add_argument(
        "--prompt",
        "-p",
        type=str,
        default=None,
        help="Use case description for metric suggestion (overrides config)",
    )
    parser.add_argument(
        "--output",
        "-o",
        type=str,
        default=None,
        help="Output file path for generated configuration (default: sample_system.yaml)",
    )

    args = parser.parse_args()

    result = run_suggestion(args)
    return 0 if result is not None else 1


if __name__ == "__main__":
    sys.exit(main())
