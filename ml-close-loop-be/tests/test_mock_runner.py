from pathlib import Path

from app.schemas.dataset import DatasetVersionCreateRequest
from app.schemas.training import TrainingConfig, TrainingRunCreateRequest
from app.services import dataset_service, training_service
from app.workers.mock_runner import MockTrainingRunner
from app.workers.training_worker import process_next_job


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


def test_run_writes_a_fake_artifact_file_and_returns_its_uri():
    training_run = type("_Run", (), {"training_run_id": "run-abc123"})()

    artifact_uri = MockTrainingRunner().run(training_run)

    assert artifact_uri.startswith("file://")
    artifact_path = Path(artifact_uri.removeprefix("file://"))
    assert artifact_path.is_file()
    assert artifact_path.read_text() == "mock artifact for run-abc123"


def test_process_next_job_completes_end_to_end_via_mock_runner(db_session):
    training_run = _queued_training_run(db_session)

    processed = process_next_job(db_session, MockTrainingRunner())

    assert processed.training_run_id == training_run.training_run_id
    assert processed.status == "COMPLETED"
    assert processed.artifact_uri.startswith("file://")
    assert Path(processed.artifact_uri.removeprefix("file://")).is_file()
