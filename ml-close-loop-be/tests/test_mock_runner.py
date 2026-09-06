from pathlib import Path

import pytest

from app.config import settings
from app.schemas.dataset import DatasetVersionCreateRequest
from app.schemas.training import TrainingConfig, TrainingRunCreateRequest
from app.services import dataset_service, training_service
from app.workers.mock_runner import MockTrainingRunner
from app.workers.training_worker import process_next_job


@pytest.fixture(autouse=True)
def _artifact_sandbox(tmp_path, monkeypatch):
    """Keep immutable per-version artifacts (register_model_version finalizes the runner's
    staging output) out of the repo's data/ dir — one fresh per-test temp dir."""
    monkeypatch.setattr(settings, "artifact_storage_dir", str(tmp_path))


def _queued_training_run(db_session):
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
    return training_service.create_training_run(
        db_session,
        dataset_version,
        TrainingRunCreateRequest(
            dataset_id="no_robots",
            dataset_version=1,
            model_id="qwen-sft-domain-x",
            base_model="Qwen/Qwen3.8-27B",
            training_config=TrainingConfig(),
        ),
    )


def test_run_returns_staging_directory_with_fake_adapter(db_session):
    """Mock runner matches the real runner's contract (issue #38): it drives the run forward
    and returns a staging DIRECTORY containing a fake adapter, not a flat artifact URI."""
    training_run = type("_Run", (), {"training_run_id": "run-abc123"})()
    training_run.status = "RUNNING"

    staging = MockTrainingRunner().run(db_session, training_run)

    staging_dir = Path(staging)
    assert staging_dir.is_dir()
    assert (staging_dir / "adapter_model.safetensors").is_file()
    assert (staging_dir / "adapter_config.json").is_file()


def test_process_next_job_completes_end_to_end_via_mock_runner(db_session):
    training_run = _queued_training_run(db_session)

    processed = process_next_job(db_session, MockTrainingRunner())

    assert processed.training_run_id == training_run.training_run_id
    assert processed.status == "COMPLETED"

    from app.services import model_service

    model_version = model_service.get_model_version(db_session, "qwen-sft-domain-x", 1)
    assert model_version is not None
    assert model_version.status == "REGISTERED"
    # The staged output was finalized into an immutable per-version directory, and the run's
    # artifact pointer now points at it (issue #38): `{base}/{model_id}/{name}/...`.
    artifact_path = Path(model_version.artifacts[0]["uri"].removeprefix("file://"))
    assert artifact_path.is_dir()
    assert (artifact_path / "adapter_model.safetensors").is_file()
    assert processed.artifact_uri == model_version.artifacts[0]["uri"]
