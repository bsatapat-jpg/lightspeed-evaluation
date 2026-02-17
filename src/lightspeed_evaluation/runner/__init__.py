"""Runner/CLI module for LightSpeed Evaluation Framework."""

from lightspeed_evaluation.runner.evaluation import main, run_evaluation
from lightspeed_evaluation.runner.suggest import main as suggest_main
from lightspeed_evaluation.runner.suggest import run_suggestion

__all__ = ["main", "run_evaluation", "suggest_main", "run_suggestion"]
