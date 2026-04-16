"""CLI behavior when evaluation raises StorageError (e.g. DB schema mismatch)."""

import argparse
from typing import Any

import pytest
from pytest_mock import MockerFixture

from lightspeed_evaluation.core.system.exceptions import StorageError
from lightspeed_evaluation.runner.evaluation import run_evaluation


def _make_eval_args(**kwargs: Any) -> argparse.Namespace:
    """Same defaults as test_evaluation.py for run_evaluation."""
    defaults = {
        "system_config": "config/system.yaml",
        "eval_data": "config/evaluation_data.yaml",
        "output_dir": None,
        "tags": None,
        "conv_ids": None,
        "cache_warmup": False,
    }
    defaults.update(kwargs)
    return argparse.Namespace(**defaults)


def test_run_evaluation_storage_error(
    mocker: MockerFixture, capsys: pytest.CaptureFixture
) -> None:
    """StorageError from the pipeline ends the run with a clear message and exit path."""
    mock_loader = mocker.Mock()
    mock_config = mocker.Mock()
    mock_config.llm.provider = "openai"
    mock_config.llm.model = "gpt-4"
    mock_config.api.enabled = False
    mock_config.storage = []
    mock_loader.system_config = mock_config
    mock_loader.load_system_config.return_value = mock_config

    mock_config_loader_class = mocker.patch(
        "lightspeed_evaluation.runner.evaluation.ConfigLoader"
    )
    mock_config_loader_class.return_value = mock_loader

    mock_eval_data = [mocker.Mock()]
    mock_validator = mocker.patch("lightspeed_evaluation.core.system.DataValidator")
    mock_validator.return_value.load_evaluation_data.return_value = mock_eval_data

    mocker.patch(
        "lightspeed_evaluation.api.evaluate",
        side_effect=StorageError(
            "Database schema mismatch: the existing table 'evaluation_results' "
            "is missing required column(s): score.",
            backend_name="sqlite",
        ),
    )

    result = run_evaluation(_make_eval_args())

    assert result is None
    captured = capsys.readouterr()
    assert "Evaluation failed" in captured.out
    assert "schema mismatch" in captured.out
