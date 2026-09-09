import pytest

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
from app.services.artifact_storage import (
    ArtifactChecksumError,
    LocalFilesystemArtifactStorage,
)


@pytest.fixture(autouse=True)
def _disable_promotion_gates(monkeypatch):
    """These tests exercise deployment mechanics, not the #43 eval gate
    (gate criteria are covered by tests/test_promotion_gate.py)."""
    from app.config import settings

    for name in (
        "eval_gate_require_eval_set_reference",
        "eval_gate_require_qualitative_majority",
        "eval_gate_require_no_general_regression",
        "eval_gate_require_eval_loss_not_worse",
    ):
        monkeypatch.setattr(settings, name, False)


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


def test_deploy_records_target_environment(db_session):
    """Issue #67: `deploy` records the explicitly targeted environment on the deployment row
    instead of hard-coding the single configured one."""
    model_version, _ = _promoted_model_version(db_session)

    deployment, _ = deployment_service.deploy(
        db_session, model_version, environment="staging"
    )

    assert deployment.environment == "staging"
    assert _latest_deployment_row(db_session).environment == "staging"


def test_deploy_defaults_to_configured_environment(db_session):
    """Issue #67: an omitted environment keeps the pre-#67 behavior (settings default)."""
    model_version, _ = _promoted_model_version(db_session)

    deployment, _ = deployment_service.deploy(db_session, model_version)

    assert deployment.environment == "default"


def test_deploy_environment_resolves_to_environment_row(db_session):
    """Issue #67: the deployment's environment navigates to the matching Environment row when one
    exists (no FK hard-join; unmatched environment resolves to None)."""
    from app.models.environment import Environment

    db_session.add(Environment(name="staging", description="Integration validation."))
    model_version, _ = _promoted_model_version(db_session)

    deployment, _ = deployment_service.deploy(
        db_session, model_version, environment="staging"
    )

    assert deployment.environment_obj.name == "staging"
    assert deployment.environment_obj.description == "Integration validation."
    # A never-seeded environment value resolves to None, never a lazy-load error.
    deployment_unknown, _ = deployment_service.deploy(
        db_session, model_version, environment="canary"
    )
    assert deployment_unknown.environment_obj is None


def _latest_deployment_row(db_session):
    from sqlalchemy import select

    return db_session.scalars(
        select(Deployment).order_by(Deployment.deployed_at.desc())
    ).first()


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


def test_deploy_same_version_twice(db_session):
    v1, _ = _promoted_model_version(db_session)
    deployment_service.deploy(db_session, v1)

    deployment, previous = deployment_service.deploy(db_session, v1)

    assert previous is None
    assert v1.status == "DEPLOYED"
    assert deployment.model_version == v1.version
    assert deployment.status == "DEPLOYED"
    rows = {d.model_version: d.status for d in db_session.query(Deployment).all()}
    assert rows == {v1.version: "DEPLOYED"}


def test_deploy_when_no_previous_deployment(db_session):
    model_version, _ = _promoted_model_version(db_session)

    deployment, previous = deployment_service.deploy(db_session, model_version)

    assert previous is None
    assert deployment.status == "DEPLOYED"
    assert model_version.status == "DEPLOYED"


def test_rollback_deploys_target_version(db_session):
    v1, dataset_version = _promoted_model_version(db_session)
    deployment_service.deploy(db_session, v1)
    v2, _ = _promoted_model_version(db_session, dataset_version)
    deployment_service.deploy(db_session, v2)

    promotion_service.rollback(
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
    status = deployment_service.get_deployment_status(db_session, "qwen-sft-domain-x")
    assert status.current_deployed_version == v1.version
    assert status.status == "DEPLOYED"


def test_get_deployment_status_after_deploy(db_session):
    model_version, _ = _promoted_model_version(db_session)
    deployment, _ = deployment_service.deploy(db_session, model_version)

    status = deployment_service.get_deployment_status(
        db_session, model_version.model_id
    )

    assert status.model_id == "qwen-sft-domain-x"
    assert status.current_deployed_version == model_version.version
    assert status.status == "DEPLOYED"
    assert status.deployed_at == deployment.deployed_at


def _actively_finalized_model_version(db_session, tmp_path):
    """Register a version whose artifact is finalized into the immutable store (so it carries a
    recorded SHA-256 checksum), promoted to PROMOTED, ready to deploy."""
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
    staging = tmp_path / "staging"
    staging.mkdir()
    (staging / "adapter_model.safetensors").write_bytes(b"trained-weights")
    model_version = model_service.register_model_version(
        db_session,
        training_run,
        staging_dir=str(staging),
        storage=LocalFilesystemArtifactStorage(base_dir=tmp_path / "store"),
    )
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
    return model_version


def test_deploy_verifies_checksum_for_intact_artifact(db_session, tmp_path):
    """Issue #62: a finalized artifact with a recorded checksum that matches on disk deploys
    normally; the pointer moves and the version becomes DEPLOYED."""
    model_version = _actively_finalized_model_version(db_session, tmp_path)
    backend = MockServingBackend()

    deployment, previous = deployment_service.deploy(
        db_session, model_version, backend=backend
    )

    assert previous is None
    assert model_version.status == "DEPLOYED"
    assert deployment.model_version == model_version.version
    assert model_version.artifacts[0]["checksum"]
    assert backend.deployed == [("qwen-sft-domain-x", model_version.version)]


def test_deploy_refuses_corrupted_artifact_before_pointer_moves(db_session, tmp_path):
    """Issue #62: tampering with a finalized artifact payload after registration means its
    SHA-256 no longer matches metadata.json. deploy must refuse (no pointer move, no status
    flip, backend never loads) and raise ArtifactChecksumError."""
    model_version = _actively_finalized_model_version(db_session, tmp_path)
    uri = model_version.artifacts[0]["uri"]
    from app.services.artifact_storage import _uri_to_path

    (  # corrupt the on-disk payload
        _uri_to_path(uri) / "adapter_model.safetensors"
    ).write_bytes(b"tampered")
    backend = MockServingBackend()

    try:
        deployment_service.deploy(db_session, model_version, backend=backend)
    except ArtifactChecksumError as exc:
        assert "checksum mismatch" in str(exc)
    else:
        raise AssertionError("expected ArtifactChecksumError for corrupted artifact")

    assert model_version.status == "PROMOTED"  # pointer never moved
    assert backend.deployed == []  # adapter never loaded
    assert backend.unloaded == []
