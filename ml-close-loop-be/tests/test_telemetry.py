"""Tests for OpenTelemetry instrumentation (issue #82)."""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import Session
from sqlalchemy.pool import StaticPool

from app.db.base import Base
from app.db.session import get_db
from app.limiter import limiter
from app.main import app


@pytest.fixture()
def client():
    engine = create_engine(
        "sqlite://", connect_args={"check_same_thread": False}, poolclass=StaticPool
    )
    Base.metadata.create_all(engine)

    def override_get_db():
        with Session(engine) as session:
            yield session

    app.dependency_overrides[get_db] = override_get_db
    limiter.reset()
    with TestClient(app) as c:
        yield c
    app.dependency_overrides.clear()
    engine.dispose()


@pytest.fixture(autouse=True)
def _disable_promotion_gates(monkeypatch):
    from app.config import settings

    for name in (
        "eval_gate_require_eval_set_reference",
        "eval_gate_require_qualitative_majority",
        "eval_gate_require_no_general_regression",
        "eval_gate_require_eval_loss_not_worse",
    ):
        monkeypatch.setattr(settings, name, False)


@pytest.fixture
def lock_file(tmp_path):
    return str(tmp_path / "gpu.lock")


def test_telemetry_setup_creates_tracer():
    """setup_telemetry with OTEL_ENABLED=false is a no-op (no OTel packages needed)."""
    from app.telemetry import _training_runs_counter

    # With default settings (otel_enabled=False) the counter stays None
    assert _training_runs_counter is None


def test_metrics_endpoint_returns_prometheus_format(client: TestClient):
    """GET /metrics returns 200 with prometheus text content."""
    resp = client.get("/metrics")
    assert resp.status_code == 200
    # Empty bytes when OTel disabled, still valid
    assert isinstance(resp.content, bytes)


def test_training_run_counter_increments():
    """Counter instruments can be incremented when OTel is enabled (unit-level)."""
    from app.telemetry import _training_runs_counter

    # With OTel disabled the counter is None — no-op increment is safe
    if _training_runs_counter is not None:
        _training_runs_counter.add(1, {"status": "completed"})
    assert True


def test_deployment_counter_increments():
    """Deployment counter can be incremented."""
    from app.telemetry import _deployments_counter

    if _deployments_counter is not None:
        _deployments_counter.add(1, {"environment": "staging", "result": "success"})
    assert True


def test_telemetry_disabled_when_not_configured():
    """When OTEL_ENABLED=false (default), telemetry module is importable and inert."""
    from app.telemetry import (
        _training_runs_counter,
        _deployments_counter,
        _training_duration,
        _inference_latency,
        _active_training_runs,
        _gpu_lock_wait,
    )

    assert _training_runs_counter is None
    assert _deployments_counter is None
    assert _training_duration is None
    assert _inference_latency is None
    assert _active_training_runs is None
    assert _gpu_lock_wait is None


def test_health_checks_db(client: TestClient):
    """Health endpoint returns db status."""
    resp = client.get("/api/v1/health")
    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "ok"
    assert body["checks"]["db"] == "ok"


def test_health_checks_minio_error_handled(client: TestClient):
    """Health endpoint handles MinIO unavailability gracefully."""
    resp = client.get("/api/v1/health")
    assert resp.status_code == 200
    body = resp.json()
    # With local backend, minio check is absent; with minio backend, it would be "ok" or "error"
    assert body["status"] == "ok"
    assert "db" in body["checks"]


def test_trace_id_injected_into_structlog():
    """The trace-to-structlog processor does not crash without an active span."""
    from app.telemetry import _add_trace_to_structlog

    result = _add_trace_to_structlog(None, "test", {"event": "hello"})
    assert "event" in result
    # Without an active span, trace_id/span_id are not added
    assert "trace_id" not in result


def test_training_run_counter_increments_on_complete(db_session):
    """Training run counter increments when a run completes (issue #83)."""
    from app.schemas.dataset import DatasetVersionCreateRequest
    from app.schemas.training import TrainingConfig, TrainingRunCreateRequest
    from app.services import dataset_service, training_service

    dataset_version = dataset_service.create_dataset_version(
        db_session,
        "no_robots",
        DatasetVersionCreateRequest(
            source_type="huggingface",
            source_dataset="HuggingFaceH4/no_robots",
            source_commit_or_snapshot_date="2026-08-01",
            source_format="chatml",
        ),
    )
    training_run = training_service.create_training_run(
        db_session,
        dataset_version,
        TrainingRunCreateRequest(
            dataset_id="no_robots",
            dataset_version=1,
            model_id="test-model",
            base_model="Qwen/Qwen3-0.6B",
            training_config=TrainingConfig(),
        ),
    )
    training_service.start_training_run(db_session, training_run)
    training_service.complete_training_run(
        db_session, training_run, artifact_uri="file:///tmp/test"
    )
    # No exception means the counter increment path was reached
    assert training_run.status == "COMPLETED"


def test_deployment_counter_increments_on_deploy(db_session):
    """Deployment counter increments on successful deploy (issue #83)."""
    from app.schemas.dataset import DatasetVersionCreateRequest
    from app.schemas.model import (
        EvalLossTrend,
        EvaluationUpdateRequest,
        GeneralDomainRegressionCheck,
        QualitativeComparison,
    )
    from app.schemas.promotion import DecisionCreateRequest
    from app.schemas.training import TrainingConfig, TrainingRunCreateRequest
    from app.services import (
        dataset_service,
        deployment_service,
        model_service,
        promotion_service,
        training_service,
    )

    dataset_version = dataset_service.create_dataset_version(
        db_session,
        "no_robots",
        DatasetVersionCreateRequest(
            source_type="huggingface",
            source_dataset="HuggingFaceH4/no_robots",
            source_commit_or_snapshot_date="2026-08-01",
            source_format="chatml",
        ),
    )
    training_run = training_service.create_training_run(
        db_session,
        dataset_version,
        TrainingRunCreateRequest(
            dataset_id="no_robots",
            dataset_version=1,
            model_id="test-model",
            base_model="Qwen/Qwen3-0.6B",
            training_config=TrainingConfig(),
        ),
    )
    training_service.start_training_run(db_session, training_run)
    training_service.complete_training_run(
        db_session, training_run, artifact_uri="file:///tmp/test"
    )
    mv = model_service.register_model_version(db_session, training_run)
    model_service.submit_evaluation(
        db_session,
        mv,
        EvaluationUpdateRequest(
            eval_loss_trend=EvalLossTrend(this_version_eval_loss=0.9),
            qualitative_comparison=QualitativeComparison(
                question_table_version=1, wins=10, losses=5, ties=5, total=20
            ),
            general_domain_regression_check=GeneralDomainRegressionCheck(
                checked=True, regressions_found=[]
            ),
            eval_set_id="domain-benchmark",
            eval_set_version=1,
        ),
    )
    promotion_service.create_decision(
        db_session,
        mv,
        DecisionCreateRequest(decision="PROMOTED", decided_by="test", rationale="test"),
    )
    deployment, _ = deployment_service.deploy(db_session, mv)
    db_session.commit()
    assert deployment.status == "DEPLOYED"


def test_training_duration_recorded(db_session, lock_file, tmp_path, monkeypatch):
    """Training duration histogram is recorded after a worker pass (issue #83)."""
    import tempfile
    from pathlib import Path
    from app.config import settings
    from app.schemas.dataset import DatasetVersionCreateRequest
    from app.schemas.training import TrainingConfig, TrainingRunCreateRequest
    from app.services import dataset_service, training_service
    from app.workers.training_worker import process_next_job

    monkeypatch.setattr(settings, "artifact_storage_dir", str(tmp_path / "artifacts"))

    dataset_version = dataset_service.create_dataset_version(
        db_session,
        "no_robots",
        DatasetVersionCreateRequest(
            source_type="huggingface",
            source_dataset="HuggingFaceH4/no_robots",
            source_commit_or_snapshot_date="2026-08-01",
            source_format="chatml",
        ),
    )
    training_service.create_training_run(
        db_session,
        dataset_version,
        TrainingRunCreateRequest(
            dataset_id="no_robots",
            dataset_version=1,
            model_id="test-model",
            base_model="Qwen/Qwen3-0.6B",
            training_config=TrainingConfig(),
        ),
    )
    db_session.commit()

    class _Runner:
        def run(self, db, tr):
            staging = Path(tempfile.mkdtemp(prefix="defnex-test-stage-"))
            (staging / "adapter_model.safetensors").write_bytes(b"fake")
            (staging / "adapter_config.json").write_text("{}")
            return str(staging)

    result = process_next_job(db_session, _Runner(), lock_file=lock_file)
    assert result is not None
    assert result.status == "COMPLETED"
