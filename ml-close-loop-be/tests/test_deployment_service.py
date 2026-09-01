from app.models.deployment import Deployment
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
    deployment_service,
    model_service,
    promotion_service,
    training_service,
)
from app.services.serving import MockServingBackend


def _promoted_model_version(db_session, dataset_version=None):
    """Drive one more version of `qwen-sft-domain-x` all the way to PROMOTED."""
    if dataset_version is None:
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
    promotion_service.create_decision(
        db_session,
        model_version,
        DecisionCreateRequest(
            decision="PROMOTED", decided_by="reviewer-1", rationale="Signals aligned."
        ),
    )
    assert model_version.status == "PROMOTED"
    return model_version, dataset_version


def test_deploy_moves_pointer_and_calls_serving_backend(db_session):
    model_version, _ = _promoted_model_version(db_session)
    backend = MockServingBackend()

    deployment, previous = deployment_service.deploy(
        db_session, model_version, backend=backend
    )

    assert previous is None
    assert model_version.status == "DEPLOYED"
    assert deployment.model_id == "qwen-sft-domain-x"
    assert deployment.model_version == model_version.version
    assert deployment.status == "DEPLOYED"
    assert deployment.environment == "default"
    assert backend.deployed == [("qwen-sft-domain-x", model_version.version)]


def test_deploy_retires_the_previously_deployed_version(db_session):
    v1, dataset_version = _promoted_model_version(db_session)
    deployment_service.deploy(db_session, v1)
    v2, _ = _promoted_model_version(db_session, dataset_version)

    deployment, previous = deployment_service.deploy(db_session, v2)

    assert previous is v1
    assert v1.status == "RETIRED"
    assert v2.status == "DEPLOYED"
    assert deployment.model_version == v2.version
    # Deployment rows are pointer history; the current pointer is the newest row, not a status flip.
    rows = {d.model_version: d.status for d in db_session.query(Deployment).all()}
    assert rows == {v1.version: "DEPLOYED", v2.version: "DEPLOYED"}


def test_get_deployment_status_is_all_null_before_any_deploy(db_session):
    model_version, _ = _promoted_model_version(db_session)

    status = deployment_service.get_deployment_status(
        db_session, model_version.model_id
    )

    assert status.model_id == "qwen-sft-domain-x"
    assert status.current_deployed_version is None
    assert status.deployed_at is None
    assert status.status is None


def test_get_deployment_status_follows_the_latest_deploy(db_session):
    v1, dataset_version = _promoted_model_version(db_session)
    deployment_service.deploy(db_session, v1)
    v2, _ = _promoted_model_version(db_session, dataset_version)
    deployment, _ = deployment_service.deploy(db_session, v2)

    status = deployment_service.get_deployment_status(db_session, "qwen-sft-domain-x")

    assert status.current_deployed_version == v2.version
    assert status.status == "DEPLOYED"
    assert status.deployed_at == deployment.deployed_at


def test_rollback_moves_the_deployment_pointer_back(db_session):
    """US-017's rollback delegates to `deploy`, so the pointer resource must follow it."""
    v1, dataset_version = _promoted_model_version(db_session)
    deployment_service.deploy(db_session, v1)
    v2, _ = _promoted_model_version(db_session, dataset_version)
    deployment_service.deploy(db_session, v2)

    decision = promotion_service.rollback(
        db_session,
        v1,
        RollbackRequest(
            rollback_of_version=v1.version,
            decided_by="reviewer-1",
            rationale="Regression.",
        ),
    )

    assert v1.status == "DEPLOYED"
    assert v2.status == "RETIRED"
    # DecisionRecord.version is "the version being rolled back from" - now a real one, not the fallback.
    assert decision.model_version_id == v2.id
    assert decision.rollback_of_version == v1.version
    pointer = deployment_service.get_deployment_status(db_session, "qwen-sft-domain-x")
    assert pointer.current_deployed_version == v1.version


def test_to_deploy_result_reports_the_superseded_version(db_session):
    v1, dataset_version = _promoted_model_version(db_session)
    deployment_service.deploy(db_session, v1)
    v2, _ = _promoted_model_version(db_session, dataset_version)
    deployment, previous = deployment_service.deploy(db_session, v2)

    result = deployment_service.to_deploy_result(deployment, previous)

    assert result.model_id == "qwen-sft-domain-x"
    assert result.current_deployed_version == v2.version
    assert result.previous_deployed_version == v1.version
