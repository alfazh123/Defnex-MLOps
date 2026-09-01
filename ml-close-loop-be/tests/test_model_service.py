import pytest

from app.schemas.dataset import DatasetVersionCreateRequest
from app.schemas.model import (
    EvalLossTrend,
    EvaluationUpdateRequest,
    GeneralDomainRegressionCheck,
    QualitativeComparison,
)
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
    training_service.complete_training_run(
        db_session, training_run, artifact_uri="file:///tmp/adapter"
    )
    return training_run


def test_register_model_version_from_completed_run(db_session):
    training_run = _completed_training_run(db_session)

    model_version = model_service.register_model_version(db_session, training_run)

    assert model_version.model_id == "qwen-sft-domain-x"
    assert model_version.version == 1
    assert model_version.status == "REGISTERED"
    assert model_version.training_run_id == training_run.training_run_id
    assert model_version.base_model == "Qwen/Qwen3.8-27B"
    assert model_version.training_config["peft_method"] == "lora"
    assert model_version.artifacts == [
        {"type": "adapter", "uri": "file:///tmp/adapter"}
    ]


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


def test_submit_evaluation_partial_stays_registered(db_session):
    training_run = _completed_training_run(db_session)
    model_version = model_service.register_model_version(db_session, training_run)

    model_service.submit_evaluation(
        db_session,
        model_version,
        EvaluationUpdateRequest(
            eval_loss_trend=EvalLossTrend(this_version_eval_loss=0.84)
        ),
    )

    assert model_version.status == "REGISTERED"
    evaluation = model_service.get_evaluation(model_version)
    assert evaluation.eval_loss_trend.this_version_eval_loss == 0.84
    assert evaluation.qualitative_comparison is None
    assert evaluation.general_domain_regression_check is None


def test_submit_evaluation_all_three_transitions_to_evaluated(db_session):
    training_run = _completed_training_run(db_session)
    model_version = model_service.register_model_version(db_session, training_run)

    model_service.submit_evaluation(
        db_session,
        model_version,
        EvaluationUpdateRequest(
            eval_loss_trend=EvalLossTrend(this_version_eval_loss=0.84)
        ),
    )
    model_service.submit_evaluation(
        db_session,
        model_version,
        EvaluationUpdateRequest(
            qualitative_comparison=QualitativeComparison(
                question_table_version=1, wins=13, losses=5, ties=2, total=20
            )
        ),
    )
    assert model_version.status == "REGISTERED"

    model_service.submit_evaluation(
        db_session,
        model_version,
        EvaluationUpdateRequest(
            general_domain_regression_check=GeneralDomainRegressionCheck(
                checked=True, regressions_found=[]
            )
        ),
    )

    assert model_version.status == "EVALUATED"
    evaluation = model_service.get_evaluation(model_version)
    assert evaluation.eval_loss_trend.this_version_eval_loss == 0.84
    assert evaluation.qualitative_comparison.wins == 13
    assert evaluation.general_domain_regression_check.checked is True
