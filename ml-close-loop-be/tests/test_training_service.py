import pytest

from app.schemas.dataset import DatasetVersionCreateRequest
from app.schemas.training import TrainingConfig, TrainingRunCreateRequest
from app.services import dataset_service, training_service


def _dataset_version(db_session):
    return dataset_service.create_dataset_version(
        db_session,
        "no_robots",
        DatasetVersionCreateRequest(
            source_type="huggingface",
            source_dataset="HuggingFaceH4/no_robots",
            source_commit_or_snapshot_date="2026-08-01",
            source_format="chatml",
        ),
    )


def _create_request(**overrides):
    defaults = {
        "dataset_id": "no_robots",
        "dataset_version": 1,
        "model_id": "qwen-sft-domain-x",
        "base_model": "Qwen/Qwen3.8-27B",
        "training_config": TrainingConfig(),
    }
    defaults.update(overrides)
    return TrainingRunCreateRequest(**defaults)


def test_create_training_run_starts_pending(db_session):
    dataset_version = _dataset_version(db_session)

    training_run = training_service.create_training_run(db_session, dataset_version, _create_request())

    assert training_run.training_run_id.startswith("run-")
    assert training_run.status == "PENDING"
    assert training_run.dataset_version_id == dataset_version.id
    assert training_run.model_id == "qwen-sft-domain-x"
    assert training_run.base_model == "Qwen/Qwen3.8-27B"
    assert training_run.training_config["peft_method"] == "dora"


def test_completed_lifecycle(db_session):
    dataset_version = _dataset_version(db_session)
    training_run = training_service.create_training_run(db_session, dataset_version, _create_request())

    training_service.start_training_run(db_session, training_run)
    assert training_run.status == "RUNNING"

    training_service.complete_training_run(db_session, training_run, artifact_uri="file:///tmp/adapter")
    assert training_run.status == "COMPLETED"
    assert training_run.artifact_uri == "file:///tmp/adapter"


def test_failed_lifecycle(db_session):
    dataset_version = _dataset_version(db_session)
    training_run = training_service.create_training_run(db_session, dataset_version, _create_request())

    training_service.start_training_run(db_session, training_run)
    training_service.fail_training_run(db_session, training_run, error_message="OOM")
    assert training_run.status == "FAILED"
    assert training_run.error_message == "OOM"


@pytest.mark.parametrize(
    ("from_status", "transition"),
    [
        ("PENDING", "complete"),
        ("PENDING", "fail"),
        ("COMPLETED", "start"),
        ("FAILED", "start"),
    ],
)
def test_invalid_transitions_are_rejected(db_session, from_status, transition):
    dataset_version = _dataset_version(db_session)
    training_run = training_service.create_training_run(db_session, dataset_version, _create_request())
    training_run.status = from_status

    action = {
        "start": lambda: training_service.start_training_run(db_session, training_run),
        "complete": lambda: training_service.complete_training_run(db_session, training_run, "uri"),
        "fail": lambda: training_service.fail_training_run(db_session, training_run, "err"),
    }[transition]

    with pytest.raises(ValueError):
        action()
