"""Storage protocol interface for evaluation results.

This module defines the abstract interface that all storage backends must implement,
plus a small abstract base class with default no-op lifecycle hooks for backends
that only need to satisfy the protocol (e.g. file slot in the pipeline).
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Protocol
from uuid import uuid4

from lightspeed_evaluation.core.models.data import EvaluationResult


@dataclass
class RunInfo:
    """Information about an evaluation run.

    Attributes:
        run_id: Unique identifier for the evaluation run (auto-generated UUID).
        name: Human-readable name for the run.
        started_at: Timestamp when the run started (auto-generated).
    """

    run_id: str = field(default_factory=lambda: str(uuid4()))
    name: str = ""
    started_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))


class StorageProtocol(Protocol):
    """Protocol defining the interface for storage backends.

    Storage backends persist evaluation results. They support both
    incremental saving (one result at a time) and batch saving.
    """

    @property
    def backend_name(self) -> str:
        """Return the name of this storage backend."""
        ...  # pylint: disable=unnecessary-ellipsis  # Required for pyright

    def initialize(self, run_info: RunInfo) -> None:
        """Initialize the backend for a new evaluation run.

        Args:
            run_info: Information about the evaluation run.

        Raises:
            StorageError: If initialization fails.
        """

    def save_result(self, result: EvaluationResult) -> None:
        """Save a single evaluation result incrementally.

        Args:
            result: The evaluation result to save.

        Raises:
            StorageError: If saving fails.
        """

    def save_run(self, results: list[EvaluationResult]) -> None:
        """Save all evaluation results in batch.

        Args:
            results: List of all evaluation results.

        Raises:
            StorageError: If saving fails.
        """

    def finalize(self) -> None:
        """Finalize the backend after evaluation completes.

        Raises:
            StorageError: If finalization fails.
        """

    def close(self) -> None:
        """Close the backend and release resources."""


class BaseStorageBackend(ABC):
    """Abstract storage backend with default no-op lifecycle methods.

    Subclasses must implement :attr:`backend_name`. Override any lifecycle
    method that should perform work (for example :class:`SQLStorageBackend`).
    """

    @property
    @abstractmethod
    def backend_name(self) -> str:
        """Return the name of this storage backend."""

    def initialize(self, run_info: RunInfo) -> None:
        """Prepare the backend for a new run; default does nothing."""

    def save_result(self, result: EvaluationResult) -> None:
        """Persist a single result; default does nothing."""

    def save_run(self, results: list[EvaluationResult]) -> None:
        """Persist a batch of results; default does nothing."""

    def finalize(self) -> None:
        """Complete the run; default does nothing."""

    def close(self) -> None:
        """Release resources; default does nothing."""
