import pytest

from app.schemas.dataset import DatasetVersionCreateRequest
from app.schemas.model import (
    EvalLossTrend,
    EvaluationUpdateRequest,
    GeneralDomainRegressionCheck,
    QualitativeComparison,
)
from app.schemas.promotion import DecisionCreateRequest, RollbackRequest
from app.schemas.training import TrainingConfig, TrainingRunCreateRequest
from app.services import (
    dataset_service,
    model_service,
    promotion_service,
    training_service,
)


def _evaluated_model_version(db_session):
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
    training_service.start_training_run(db_session, training_run)
    training_service.complete_training_run(
        db_session, training_run, artifact_uri="file:///tmp/adapter"
    )
    model_version = model_service.register_model_version(db_session, training_run)

    model_service.submit_evaluation(
        db_session,
        model_version,
        EvaluationUpdateRequest(
            eval_loss_trend=EvalLossTrend(this_version_eval_loss=0.84),
            qualitative_comparison=QualitativeComparison(
                question_table_version=1, wins=13, losses=5, ties=2, total=20
            ),
            general_domain_regression_check=GeneralDomainRegressionCheck(
                checked=True, regressions_found=[]
            ),
        ),
    )
    assert model_version.status == "EVALUATED"
    return model_version


def test_create_decision_promotes_evaluated_version(db_session):
    model_version = _evaluated_model_version(db_session)

    decision = promotion_service.create_decision(
        db_session,
        model_version,
        DecisionCreateRequest(
            decision="PROMOTED",
            decided_by="reviewer-1",
            rationale="All three signals aligned.",
        ),
    )

    assert decision.decision == "PROMOTED"
    assert decision.rationale == "All three signals aligned."
    assert decision.rollback_of_version is None
    assert decision.evidence_snapshot["qualitative_comparison"]["wins"] == 13
    assert model_version.status == "PROMOTED"
    assert model_version.promotion_decision_ref == decision.decision_id


def test_create_decision_rejects_evaluated_version(db_session):
    model_version = _evaluated_model_version(db_session)

    decision = promotion_service.create_decision(
        db_session,
        model_version,
        DecisionCreateRequest(
            decision="REJECTED",
            decided_by="reviewer-1",
            rationale="General-domain regression.",
        ),
    )

    assert decision.decision == "REJECTED"
    assert model_version.status == "REJECTED"


@pytest.mark.parametrize(
    "status", ["REGISTERED", "PROMOTED", "REJECTED", "DEPLOYED", "RETIRED"]
)
def test_create_decision_rejects_invalid_source_status(db_session, status):
    model_version = _evaluated_model_version(db_session)
    model_version.status = status

    with pytest.raises(ValueError):
        promotion_service.create_decision(
            db_session,
            model_version,
            DecisionCreateRequest(
                decision="PROMOTED", decided_by="reviewer-1", rationale="n/a"
            ),
        )


def _promoted_model_version(db_session):
    model_version = _evaluated_model_version(db_session)
    promotion_service.create_decision(
        db_session,
        model_version,
        DecisionCreateRequest(
            decision="PROMOTED",
            decided_by="reviewer-1",
            rationale="All three signals aligned.",
        ),
    )
    return model_version


def test_rollback_deploys_target_and_records_decision(db_session):
    target = _promoted_model_version(db_session)

    decision = promotion_service.rollback(
        db_session,
        target,
        RollbackRequest(
            rollback_of_version=target.version,
            decided_by="reviewer-1",
            rationale="Prod regression.",
        ),
    )

    assert decision.decision == "ROLLBACK"
    assert decision.rollback_of_version == target.version
    assert decision.evidence_snapshot is None
    assert decision.rationale == "Prod regression."
    assert target.status == "DEPLOYED"
    assert target.promotion_decision_ref == decision.decision_id


def test_rollback_retires_previously_deployed_version(db_session):
    target = _promoted_model_version(db_session)
    other = _promoted_model_version(db_session)
    other.status = "DEPLOYED"
    db_session.flush()

    decision = promotion_service.rollback(
        db_session,
        target,
        RollbackRequest(
            rollback_of_version=target.version, rationale="Prod regression."
        ),
    )

    assert other.status == "RETIRED"
    assert target.status == "DEPLOYED"
    assert decision.model_version_id == other.id


@pytest.mark.parametrize("status", ["REGISTERED", "EVALUATED", "REJECTED", "DEPLOYED"])
def test_rollback_rejects_non_promoted_non_retired_target(db_session, status):
    target = _evaluated_model_version(db_session)
    target.status = status

    with pytest.raises(ValueError):
        promotion_service.rollback(
            db_session,
            target,
            RollbackRequest(rollback_of_version=target.version, rationale="n/a"),
        )


def test_to_schema_derives_model_id_and_version(db_session):
    model_version = _evaluated_model_version(db_session)
    decision = promotion_service.create_decision(
        db_session,
        model_version,
        DecisionCreateRequest(
            decision="PROMOTED", decided_by="reviewer-1", rationale="ok"
        ),
    )

    schema = promotion_service.to_schema(decision)

    assert schema.model_id == "qwen-sft-domain-x"
    assert schema.version == model_version.version
    assert schema.decision == "PROMOTED"
