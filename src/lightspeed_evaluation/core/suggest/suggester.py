"""Metric suggester module for LightSpeed Evaluation Framework.

This module provides LLM-based metric suggestion functionality that analyzes
user's use case descriptions and recommends appropriate evaluation metrics
from the available frameworks (ragas, deepeval, custom, nlp, geval).
"""

import json
import logging
from pathlib import Path
from typing import Any, Optional

import yaml

from lightspeed_evaluation.core.constants import (
    DEFAULT_API_BASE,
    DEFAULT_API_CACHE_DIR,
    DEFAULT_API_VERSION,
    DEFAULT_EMBEDDING_CACHE_DIR,
    DEFAULT_EMBEDDING_MODEL,
    DEFAULT_EMBEDDING_PROVIDER,
    DEFAULT_ENDPOINT_TYPE,
    DEFAULT_LLM_CACHE_DIR,
    DEFAULT_LLM_MAX_TOKENS,
    DEFAULT_LLM_MODEL,
    DEFAULT_LLM_PROVIDER,
    DEFAULT_LLM_RETRIES,
    DEFAULT_LLM_TEMPERATURE,
    DEFAULT_OUTPUT_DIR,
)
from lightspeed_evaluation.core.llm.custom import BaseCustomLLM
from lightspeed_evaluation.core.llm.manager import LLMManager
from lightspeed_evaluation.core.models.system import LLMConfig, SuggestConfig
from lightspeed_evaluation.core.suggest.prompts import (
    AVAILABLE_METRICS,
    METRIC_SUGGESTION_PROMPT,
)
from lightspeed_evaluation.core.system.exceptions import LLMError

logger = logging.getLogger(__name__)

# Default thresholds for metrics
DEFAULT_TURN_THRESHOLD = 0.8
DEFAULT_CONVERSATION_THRESHOLD = 0.7


def parse_json_from_llm_response(response: str) -> dict[str, Any]:
    """Parse JSON from LLM response, handling markdown code blocks.

    This is a utility function that can be reused across modules.

    Args:
        response: Raw LLM response string that may contain markdown formatting

    Returns:
        Parsed JSON dictionary

    Raises:
        json.JSONDecodeError: If the response cannot be parsed as JSON
    """
    cleaned = response.strip()

    # Remove markdown code block markers
    if cleaned.startswith("```json"):
        cleaned = cleaned[7:]
    elif cleaned.startswith("```"):
        cleaned = cleaned[3:]
    if cleaned.endswith("```"):
        cleaned = cleaned[:-3]

    return json.loads(cleaned.strip())


def write_yaml_with_header(
    file_path: Path,
    data: dict[str, Any],
    header_lines: list[str],
) -> None:
    """Write YAML file with header comments.

    Args:
        file_path: Path to output file
        data: Dictionary data to write as YAML
        header_lines: List of header comment lines (without # prefix)
    """
    file_path.parent.mkdir(parents=True, exist_ok=True)

    with open(file_path, "w", encoding="utf-8") as f:
        # Write header comments
        for line in header_lines:
            f.write(f"# {line}\n" if line else "#\n")
        f.write("\n")

        # Write YAML data
        yaml.dump(
            data,
            f,
            default_flow_style=False,
            sort_keys=False,
            allow_unicode=True,
        )


def truncate_text(text: str, max_length: int = 100, suffix: str = "...") -> str:
    """Truncate text to max length with suffix if needed.

    Args:
        text: Text to truncate
        max_length: Maximum length before truncation
        suffix: Suffix to append when truncating

    Returns:
        Truncated text with suffix, or original if shorter
    """
    if len(text) <= max_length:
        return text
    return text[:max_length] + suffix


def get_output_path_for_eval_data(config_path: str) -> str:
    """Generate evaluation data output path from config path.

    Args:
        config_path: Path to system config file

    Returns:
        Path for evaluation data file
    """
    path = Path(config_path)
    stem = path.stem
    suffix = path.suffix if path.suffix else ".yaml"
    return str(path.parent / f"{stem}_eval_data{suffix}")


class MetricSuggester:
    """Suggests evaluation metrics based on user's use case using LLM analysis.

    This class analyzes user-provided use case descriptions and suggests
    appropriate metrics from the available evaluation frameworks.

    Example:
        >>> suggester = MetricSuggester(llm_config)
        >>> suggestions = suggester.suggest_metrics(
        ...     "I want to evaluate a RAG chatbot for accuracy"
        ... )
        >>> suggester.generate_sample_config(suggestions, "my_config.yaml")
    """

    def __init__(self, llm_config: LLMConfig):
        """Initialize MetricSuggester with LLM configuration.

        Args:
            llm_config: LLM configuration for making suggestion calls
        """
        self.llm_manager = LLMManager(llm_config)
        self.llm = BaseCustomLLM(
            self.llm_manager.get_model_name(),
            self.llm_manager.get_llm_params(),
        )
        logger.info(
            "MetricSuggester initialized with model: %s",
            self.llm_manager.get_model_name(),
        )

    def suggest_metrics(self, user_prompt: str) -> dict[str, Any]:
        """Analyze user's use case and suggest appropriate metrics.

        Args:
            user_prompt: User's description of their evaluation use case,
                        goals, and expected results

        Returns:
            Dictionary containing:
                - analysis: Use case summary and requirements
                - suggested_metrics: Turn and conversation level metrics
                - custom_metrics: GEval custom metrics if needed
                - data_requirements: Required and optional data fields
                - recommendations: Additional setup recommendations

        Raises:
            LLMError: If LLM call fails or response cannot be parsed
        """
        logger.info("Analyzing use case for metric suggestions...")

        prompt = METRIC_SUGGESTION_PROMPT.format(
            user_prompt=user_prompt,
            available_metrics=AVAILABLE_METRICS,
        )

        try:
            response = self.llm.call(
                prompt,
                temperature=0.1,  # Low temperature for consistent suggestions
                return_single=True,
            )

            # Handle response type (str when return_single=True)
            response_str = response if isinstance(response, str) else response[0]
            suggestions = parse_json_from_llm_response(response_str)

            logger.info("Successfully generated metric suggestions")
            return suggestions

        except json.JSONDecodeError as e:
            logger.error("Failed to parse LLM response as JSON: %s", e)
            raise LLMError(
                f"Failed to parse metric suggestions from LLM response: {e}"
            ) from e
        except LLMError:
            raise
        except Exception as e:
            logger.error("Unexpected error during metric suggestion: %s", e)
            raise LLMError(f"Metric suggestion failed: {e}") from e

    def generate_sample_config(
        self,
        suggestions: dict[str, Any],
        output_path: str = "sample_system.yaml",
        base_config: Optional[dict[str, Any]] = None,
    ) -> str:
        """Generate a sample system.yaml with suggested metrics.

        Args:
            suggestions: Metric suggestions from suggest_metrics()
            output_path: Output file path for the generated configuration
            base_config: Optional base configuration to extend

        Returns:
            Path to the generated configuration file
        """
        logger.info("Generating sample configuration with suggested metrics...")

        # Build metrics from suggestions
        turn_metrics = self._build_turn_metrics(suggestions)
        conv_metrics = self._build_conversation_metrics(suggestions)

        # Build complete config
        config = self._build_config_structure(turn_metrics, conv_metrics, base_config)

        # Build header comments
        header = self._build_config_header(suggestions)

        # Write file
        output_file = Path(output_path)
        write_yaml_with_header(output_file, config, header)

        logger.info("Generated sample configuration: %s", output_file)
        return str(output_file)

    def _build_turn_metrics(self, suggestions: dict[str, Any]) -> dict[str, Any]:
        """Build turn-level metrics configuration from suggestions.

        Args:
            suggestions: Raw suggestions from LLM

        Returns:
            Dictionary of turn-level metric configurations
        """
        metrics = {}

        # Add standard turn-level metrics
        for metric in suggestions.get("suggested_metrics", {}).get("turn_level", []):
            metric_id = metric["metric"]
            metrics[metric_id] = {
                "threshold": metric.get("threshold", DEFAULT_TURN_THRESHOLD),
                "description": metric.get("reason", "Suggested metric"),
                "default": True,
            }
            if metric.get("configuration"):
                metrics[metric_id].update(metric["configuration"])

        # Add custom GEval turn-level metrics
        for custom in suggestions.get("custom_metrics", []):
            if custom.get("level") != "conversation_level":
                metrics[custom["metric"]] = self._build_geval_config(custom)

        return metrics

    def _build_conversation_metrics(
        self, suggestions: dict[str, Any]
    ) -> dict[str, Any]:
        """Build conversation-level metrics configuration from suggestions.

        Args:
            suggestions: Raw suggestions from LLM

        Returns:
            Dictionary of conversation-level metric configurations
        """
        metrics = {}

        # Add standard conversation-level metrics
        for metric in suggestions.get("suggested_metrics", {}).get(
            "conversation_level", []
        ):
            metric_id = metric["metric"]
            metrics[metric_id] = {
                "threshold": metric.get("threshold", DEFAULT_CONVERSATION_THRESHOLD),
                "description": metric.get("reason", "Suggested metric"),
                "default": True,
            }
            if metric.get("configuration"):
                metrics[metric_id].update(metric["configuration"])

        # Add custom GEval conversation-level metrics
        for custom in suggestions.get("custom_metrics", []):
            if custom.get("level") == "conversation_level":
                metrics[custom["metric"]] = self._build_geval_config(custom)

        return metrics

    def _build_geval_config(self, custom_metric: dict[str, Any]) -> dict[str, Any]:
        """Build GEval metric configuration from custom metric suggestion.

        Args:
            custom_metric: Custom metric specification from LLM

        Returns:
            GEval metric configuration dictionary
        """
        config = {
            "criteria": custom_metric["criteria"],
            "evaluation_params": custom_metric.get(
                "evaluation_params", ["query", "response"]
            ),
            "threshold": custom_metric.get("threshold", DEFAULT_CONVERSATION_THRESHOLD),
            "description": custom_metric.get("reason", "Custom evaluation metric"),
        }
        if custom_metric.get("evaluation_steps"):
            config["evaluation_steps"] = custom_metric["evaluation_steps"]
        return config

    def _build_config_header(self, suggestions: dict[str, Any]) -> list[str]:
        """Build header comment lines for the configuration file.

        Args:
            suggestions: Metric suggestions containing analysis and requirements

        Returns:
            List of header comment lines
        """
        analysis = suggestions.get("analysis", {})
        data_req = suggestions.get("data_requirements", {})

        header = [
            "LightSpeed Evaluation Framework - Suggested Configuration",
            "Generated by MetricSuggester based on use case analysis",
            "",
            "Use Case Summary:",
            f"  {analysis.get('use_case_summary', 'N/A')}",
            "",
            "Data Requirements:",
            f"  Required fields: {data_req.get('required_fields', [])}",
            f"  Optional fields: {data_req.get('optional_fields', [])}",
        ]

        if data_req.get("notes"):
            header.append(f"  Notes: {data_req['notes']}")

        header.append("")

        if suggestions.get("recommendations"):
            header.extend(
                [
                    "Recommendations:",
                    f"  {suggestions['recommendations']}",
                    "",
                ]
            )

        return header

    def _build_config_structure(
        self,
        turn_level_metrics: dict[str, Any],
        conversation_level_metrics: dict[str, Any],
        base_config: Optional[dict[str, Any]] = None,
    ) -> dict[str, Any]:
        """Build the complete configuration structure.

        Args:
            turn_level_metrics: Turn-level metrics configuration
            conversation_level_metrics: Conversation-level metrics configuration
            base_config: Optional base configuration to extend

        Returns:
            Complete configuration dictionary
        """
        if base_config:
            config = base_config.copy()
        else:
            config = self._get_default_config()

        config["metrics_metadata"] = {
            "turn_level": turn_level_metrics,
            "conversation_level": conversation_level_metrics,
        }

        return config

    def _get_default_config(self) -> dict[str, Any]:
        """Get default system configuration using framework constants.

        Returns:
            Default configuration dictionary
        """
        return {
            "core": {
                "max_threads": 50,
                "fail_on_invalid_data": True,
                "skip_on_failure": False,
            },
            "llm": {
                "provider": DEFAULT_LLM_PROVIDER,
                "model": DEFAULT_LLM_MODEL,
                "temperature": DEFAULT_LLM_TEMPERATURE,
                "max_tokens": DEFAULT_LLM_MAX_TOKENS,
                "timeout": 300,
                "num_retries": DEFAULT_LLM_RETRIES,
                "cache_dir": DEFAULT_LLM_CACHE_DIR,
                "cache_enabled": True,
            },
            "embedding": {
                "provider": DEFAULT_EMBEDDING_PROVIDER,
                "model": DEFAULT_EMBEDDING_MODEL,
                "cache_dir": DEFAULT_EMBEDDING_CACHE_DIR,
                "cache_enabled": True,
            },
            "api": {
                "enabled": True,
                "api_base": DEFAULT_API_BASE,
                "version": DEFAULT_API_VERSION,
                "endpoint_type": DEFAULT_ENDPOINT_TYPE,
                "timeout": 300,
                "cache_dir": DEFAULT_API_CACHE_DIR,
                "cache_enabled": True,
            },
            "output": {
                "output_dir": DEFAULT_OUTPUT_DIR,
                "base_filename": "evaluation",
                "enabled_outputs": ["csv", "json", "txt"],
            },
            "logging": {
                "source_level": "INFO",
                "package_level": "ERROR",
            },
        }

    def generate_sample_eval_data(
        self,
        suggestions: dict[str, Any],
        output_path: str = "sample_evaluation_data.yaml",
    ) -> str:
        """Generate a sample evaluation data template based on suggestions.

        Args:
            suggestions: Metric suggestions from suggest_metrics()
            output_path: Output file path for the generated template

        Returns:
            Path to the generated evaluation data template
        """
        logger.info("Generating sample evaluation data template...")

        data_req = suggestions.get("data_requirements", {})
        required_fields = data_req.get("required_fields", ["query", "response"])
        optional_fields = data_req.get("optional_fields", [])

        # Build turn template and metrics
        turn_template = self._build_turn_template(required_fields, optional_fields)
        turn_metrics, conv_metrics = self._extract_metric_lists(suggestions)

        # Build evaluation data structure
        eval_data = self._build_eval_data_structure(
            turn_template, turn_metrics, conv_metrics
        )

        # Build header
        header = self._build_eval_data_header(required_fields, optional_fields)

        # Write file
        output_file = Path(output_path)
        write_yaml_with_header(output_file, eval_data, header)

        logger.info("Generated sample evaluation data: %s", output_file)
        return str(output_file)

    def _build_turn_template(
        self,
        required_fields: list[str],
        optional_fields: list[str],
    ) -> dict[str, Any]:
        """Build a sample turn data template.

        Args:
            required_fields: Required data fields
            optional_fields: Optional data fields

        Returns:
            Turn data template dictionary
        """
        all_fields = set(required_fields + optional_fields)
        template: dict[str, Any] = {"query": "<Your query here>"}

        field_templates = {
            "response": None,  # Populated by API
            "expected_response": "<Expected response here>",
            "contexts": ["<Context 1>", "<Context 2>"],
            "expected_keywords": [["keyword1", "keyword2"]],
            "expected_intent": "<Expected intent description>",
            "expected_tool_calls": [
                [[{"tool_name": "example_tool", "arguments": {"arg1": "value1"}}]]
            ],
        }

        for field, value in field_templates.items():
            if field in all_fields:
                template[field] = value

        return template

    def _extract_metric_lists(
        self, suggestions: dict[str, Any]
    ) -> tuple[list[str], list[str]]:
        """Extract metric identifier lists from suggestions.

        Args:
            suggestions: Metric suggestions

        Returns:
            Tuple of (turn_metrics, conversation_metrics) lists
        """
        suggested = suggestions.get("suggested_metrics", {})
        custom = suggestions.get("custom_metrics", [])

        turn_metrics = [m["metric"] for m in suggested.get("turn_level", [])]
        turn_metrics.extend(
            m["metric"] for m in custom if m.get("level") != "conversation_level"
        )

        conv_metrics = [m["metric"] for m in suggested.get("conversation_level", [])]
        conv_metrics.extend(
            m["metric"] for m in custom if m.get("level") == "conversation_level"
        )

        return turn_metrics, conv_metrics

    def _build_eval_data_structure(
        self,
        turn_template: dict[str, Any],
        turn_metrics: list[str],
        conv_metrics: list[str],
    ) -> dict[str, Any]:
        """Build the evaluation data structure.

        Args:
            turn_template: Sample turn data
            turn_metrics: Turn-level metric identifiers
            conv_metrics: Conversation-level metric identifiers

        Returns:
            Complete evaluation data structure
        """
        conversation: dict[str, Any] = {
            "id": "sample_conversation_1",
            "tags": ["sample", "generated"],
            "turns": [turn_template.copy()],
        }

        if turn_metrics:
            conversation["turns"][0]["turn_metrics"] = turn_metrics

        if conv_metrics:
            conversation["conversation_metrics"] = conv_metrics

        return {"conversations": [conversation]}

    def _build_eval_data_header(
        self,
        required_fields: list[str],
        optional_fields: list[str],
    ) -> list[str]:
        """Build header for evaluation data file.

        Args:
            required_fields: Required data fields
            optional_fields: Optional data fields

        Returns:
            List of header comment lines
        """
        header = [
            "LightSpeed Evaluation Framework - Sample Evaluation Data",
            "Generated by MetricSuggester based on use case analysis",
            "",
            "Required fields for your evaluation:",
        ]
        header.extend(f"  - {field}" for field in required_fields)
        header.append("")

        if optional_fields:
            header.append("Optional fields (if applicable):")
            header.extend(f"  - {field}" for field in optional_fields)
            header.append("")

        header.extend(
            [
                "Customize this template with your actual evaluation data.",
                "",
            ]
        )

        return header


def run_metric_suggestion(suggest_config: SuggestConfig, llm_config: LLMConfig) -> None:
    """Run the metric suggestion workflow.

    This function orchestrates the complete metric suggestion process:
    1. Analyzes the user's use case using LLM
    2. Prints suggested metrics with reasoning
    3. Generates configuration and evaluation data files

    Args:
        suggest_config: Suggestion configuration containing user prompt and output settings
        llm_config: LLM configuration for making suggestion calls
    """
    if not suggest_config.enabled:
        logger.info("Metric suggestion is disabled")
        return

    if not suggest_config.prompt:
        logger.warning("No use case prompt provided for metric suggestion")
        print(
            "❌ No use case prompt provided. "
            "Please set 'suggest.prompt' in your configuration."
        )
        return

    print("🔍 Analyzing your use case for metric suggestions...")
    print(f"   Use case: {truncate_text(suggest_config.prompt)}")

    suggester = MetricSuggester(llm_config)

    try:
        suggestions = suggester.suggest_metrics(suggest_config.prompt)
        _print_suggestions(suggestions)
        _generate_output_files(suggester, suggestions, suggest_config)

        print("\n🎉 Metric suggestion complete!")
        print("   Review the generated files and customize as needed.")

    except LLMError as e:
        print(f"\n❌ Metric suggestion failed: {e}")
        logger.error("Metric suggestion failed: %s", e)


def _print_suggestions(suggestions: dict[str, Any]) -> None:
    """Print metric suggestions in a formatted way.

    Args:
        suggestions: Metric suggestions from MetricSuggester
    """
    # Analysis summary
    analysis = suggestions.get("analysis", {})
    print("\n📊 Use Case Analysis:")
    print(f"   Summary: {analysis.get('use_case_summary', 'N/A')}")
    print(f"   Evaluation Type: {analysis.get('evaluation_type', 'N/A')}")
    print(f"   Key Requirements: {', '.join(analysis.get('key_requirements', []))}")

    # Turn-level metrics
    suggested = suggestions.get("suggested_metrics", {})
    print("\n✅ Suggested Turn-Level Metrics:")
    turn_metrics = suggested.get("turn_level", [])
    if turn_metrics:
        for metric in turn_metrics:
            threshold = metric.get("threshold", DEFAULT_TURN_THRESHOLD)
            print(f"   • {metric['metric']} (threshold: {threshold})")
            print(f"     Reason: {metric.get('reason', 'N/A')}")
    else:
        print("   (none)")

    # Conversation-level metrics
    print("\n✅ Suggested Conversation-Level Metrics:")
    conv_metrics = suggested.get("conversation_level", [])
    if conv_metrics:
        for metric in conv_metrics:
            threshold = metric.get("threshold", DEFAULT_CONVERSATION_THRESHOLD)
            print(f"   • {metric['metric']} (threshold: {threshold})")
            print(f"     Reason: {metric.get('reason', 'N/A')}")
    else:
        print("   (none)")

    # Custom metrics
    custom_metrics = suggestions.get("custom_metrics", [])
    if custom_metrics:
        print("\n🛠️ Custom Metrics (GEval):")
        for metric in custom_metrics:
            level = metric.get("level", "turn_level")
            print(f"   • {metric['metric']} ({level})")
            print(f"     Criteria: {truncate_text(metric.get('criteria', 'N/A'))}")
            print(f"     Reason: {metric.get('reason', 'N/A')}")

    # Data requirements
    data_req = suggestions.get("data_requirements", {})
    print("\n📋 Data Requirements:")
    print(f"   Required fields: {', '.join(data_req.get('required_fields', []))}")
    optional = data_req.get("optional_fields", [])
    if optional:
        print(f"   Optional fields: {', '.join(optional)}")
    if data_req.get("notes"):
        print(f"   Notes: {data_req['notes']}")

    # Recommendations
    if suggestions.get("recommendations"):
        print(f"\n💡 Recommendations: {suggestions['recommendations']}")


def _generate_output_files(
    suggester: MetricSuggester,
    suggestions: dict[str, Any],
    suggest_config: SuggestConfig,
) -> None:
    """Generate output configuration files.

    Args:
        suggester: MetricSuggester instance
        suggestions: Metric suggestions
        suggest_config: Suggestion configuration with output path
    """
    config_path = suggester.generate_sample_config(
        suggestions, suggest_config.output_file
    )
    print(f"\n📁 Generated configuration: {config_path}")

    eval_data_path = suggester.generate_sample_eval_data(
        suggestions,
        get_output_path_for_eval_data(suggest_config.output_file),
    )
    print(f"📁 Generated sample evaluation data: {eval_data_path}")
