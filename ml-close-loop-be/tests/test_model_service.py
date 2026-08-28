import pytest

from app.schemas.dataset import DatasetVersionCreateRequest
from app.schemas.training import TrainingConfig, TrainingRunCreateRequest
from app.services import dataset_service, model_service, training_service


def _completed_training_run(db_session, **overrides):
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
    defaults = {
        "dataset_id": "no_robots",
        "dataset_version": 1,
        "model_id": "qwen-sft-domain-x",
        "base_model": "Qwen/Qwen3.8-27B",
        "training_config": TrainingConfig(),
    }
    defaults.update(overrides)
    training_run = training_service.create_training_run(
        db_session, dataset_version, TrainingRunCreateRequest(**defaults)
    )
    training_service.start_training_run(db_session, training_run)
    training_service.complete_training_run(db_session, training_run, artifact_uri="file:///tmp/adapter")
    return training_run


def test_register_model_version_from_completed_run(db_session):
    training_run = _completed_training_run(db_session)

    model_version = model_service.register_model_version(db_session, training_run)

    assert model_version.model_id == "qwen-sft-domain-x"
    assert model_version.version == 1
    assert model_version.status == "REGISTERED"
    assert model_version.training_run_id == training_run.training_run_id
    assert model_version.base_model == "Qwen/Qwen3.8-27B"
    assert model_version.training_config["peft_method"] == "dora"
    assert model_version.artifacts == [{"type": "adapter", "uri": "file:///tmp/adapter"}]


def test_register_model_version_increments_version_per_model_id(db_session):
    first_run = _completed_training_run(db_session)
    model_service.register_model_version(db_session, first_run)

    second_run = _completed_training_run(db_session)
    second_version = model_service.register_model_version(db_session, second_run)

    assert second_version.version == 2


def test_register_model_version_requires_completed_status(db_session):
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
            model_id="qwen-sft-domain-x",
            base_model="Qwen/Qwen3.8-27B",
            training_config=TrainingConfig(),
        ),
    )

    with pytest.raises(ValueError):
        model_service.register_model_version(db_session, training_run)
