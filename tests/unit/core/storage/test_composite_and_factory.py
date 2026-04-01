"""Tests for composite storage and pipeline storage factory."""

from lightspeed_evaluation.core.models import EvaluationResult
from lightspeed_evaluation.core.storage import (
    BaseStorageBackend,
    CompositeStorageBackend,
    NoOpStorageBackend,
    RunInfo,
    SQLStorageBackend,
    create_pipeline_storage_backend,
)
from lightspeed_evaluation.core.storage.config import (
    DatabaseBackendConfig,
    FileBackendConfig,
)


class TestCreatePipelineStorageBackend:
    """Tests for create_pipeline_storage_backend."""

    def test_empty_config_returns_noop(self) -> None:
        """No backends configured -> single no-op backend."""
        backend = create_pipeline_storage_backend([])
        assert isinstance(backend, NoOpStorageBackend)
        assert backend.backend_name == "noop"

    def test_file_only_returns_file_noop(self) -> None:
        """File backend is a no-op in the pipeline (outputs go via output layer)."""
        backend = create_pipeline_storage_backend([FileBackendConfig()])
        assert isinstance(backend, NoOpStorageBackend)
        assert backend.backend_name == "file"

    def test_sqlite_returns_sql_backend(self) -> None:
        """Database config yields SQL storage implementation."""
        backend = create_pipeline_storage_backend(
            [DatabaseBackendConfig(type="sqlite", database=":memory:")]
        )
        assert isinstance(backend, SQLStorageBackend)
        backend.close()

    def test_file_and_sqlite_returns_composite(self) -> None:
        """Multiple backends are composed."""
        backend = create_pipeline_storage_backend(
            [
                FileBackendConfig(),
                DatabaseBackendConfig(type="sqlite", database=":memory:"),
            ]
        )
        assert isinstance(backend, CompositeStorageBackend)
        assert "file" in backend.backend_name
        assert "sqlite" in backend.backend_name
        backend.close()


class TestCompositeStorageBackend:  # pylint: disable=too-few-public-methods
    """Tests for CompositeStorageBackend."""

    def test_delegates_lifecycle(self) -> None:
        """Initialize / finalize / close run on all children."""
        calls: list[str] = []

        class TrackingBackend(BaseStorageBackend):
            """Records selected lifecycle calls; other hooks use base no-ops."""

            @property
            def backend_name(self) -> str:
                """Return a fixed name."""
                return "track"

            def initialize(self, run_info: RunInfo) -> None:
                """Record initialize."""
                _ = run_info
                calls.append("init")

            def save_run(self, results: list[EvaluationResult]) -> None:
                """Record save_run."""
                _ = results
                calls.append("save_run")

            def finalize(self) -> None:
                """Record finalize."""
                calls.append("finalize")

            def close(self) -> None:
                """Record close."""
                calls.append("close")

        t1, t2 = TrackingBackend(), TrackingBackend()
        composite = CompositeStorageBackend([t1, t2])
        run = RunInfo(name="t")
        composite.initialize(run)
        composite.save_run([])
        composite.finalize()
        composite.close()

        assert calls == [
            "init",
            "init",
            "save_run",
            "save_run",
            "finalize",
            "finalize",
            "close",
            "close",
        ]
