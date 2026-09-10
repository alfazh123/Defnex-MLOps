import pytest

from app.config import settings
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


def _evaluated_model_version(db_session, *, dataset_license=None):
    """`dataset_license` (issue #134) lets license-warning tests set the training dataset's
    license without duplicating this whole fixture; default (None) keeps every existing call
    site's behavior unchanged."""
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
    if dataset_license is not None:
        dataset_version.license = dataset_license
        db_session.flush()
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
            eval_set_id="domain-benchmark",
            eval_set_version=1,
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
    assert schema.eval_set_id == "domain-benchmark"
    assert schema.eval_set_version == 1


def test_create_decision_snapshots_eval_set_reference(db_session):
    model_version = _evaluated_model_version(db_session)
    model_version.eval_set_version = 99

    decision = promotion_service.create_decision(
        db_session,
        model_version,
        DecisionCreateRequest(
            decision="PROMOTED", decided_by="reviewer-1", rationale="ok"
        ),
    )

    assert decision.eval_set_id == "domain-benchmark"
    assert decision.eval_set_version == 99

    schema = promotion_service.to_schema(decision)

    assert schema.model_id == "qwen-sft-domain-x"
    assert schema.version == model_version.version
    assert schema.decision == "PROMOTED"


def test_rollback_creates_decision_with_null_evidence(db_session):
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
    assert decision.evidence_snapshot is None


def test_rollback_sets_rollback_of_version(db_session):
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

    assert decision.rollback_of_version == target.version


def test_create_decision_rejects_already_promoted(db_session):
    model_version = _promoted_model_version(db_session)

    with pytest.raises(ValueError):
        promotion_service.create_decision(
            db_session,
            model_version,
            DecisionCreateRequest(
                decision="PROMOTED", decided_by="reviewer-1", rationale="n/a"
            ),
        )


def test_promotion_blocked_when_gate_fails(db_session):
    model_version = _evaluated_model_version(db_session)
    model_version.qualitative_comparison = {"wins": 5, "losses": 13, "ties": 2}

    with pytest.raises(promotion_service.EvalGateBlocked):
        promotion_service.create_decision(
            db_session,
            model_version,
            DecisionCreateRequest(
                decision="PROMOTED", decided_by="reviewer-1", rationale="n/a"
            ),
        )


def test_rejection_is_not_blocked_by_gate(db_session):
    model_version = _evaluated_model_version(db_session)
    model_version.eval_set_id = None
    model_version.eval_set_version = None

    decision = promotion_service.create_decision(
        db_session,
        model_version,
        DecisionCreateRequest(
            decision="REJECTED", decided_by="reviewer-1", rationale="n/a"
        ),
    )

    assert decision.decision == "REJECTED"
    assert model_version.status == "REJECTED"


# ── License tracking + promotion warning (issue #134) ──────────────────────────


def test_base_model_license_known_and_unknown():
    """Config lookup keyed by base_model (app/config.py BASE_MODEL_LICENSES default includes
    the ticket's example: Qwen/Qwen3.8-27B -> apache-2.0). An unconfigured base model resolves
    to None (unknown), never a guessed value."""
    assert promotion_service._base_model_license("Qwen/Qwen3.8-27B") == "apache-2.0"
    assert promotion_service._base_model_license("totally/unknown-model") is None


@pytest.mark.parametrize(
    "license_str,expected",
    [
        ("cc-by-nc-4.0", True),
        ("CC-BY-NC-4.0", True),  # case-insensitive
        ("apache-2.0", False),
        ("mit", False),
        (None, False),
        ("", False),
    ],
)
def test_is_non_commercial_license_placeholder_heuristic(license_str, expected):
    """PLACEHOLDER heuristic (Open Decision, issue #134): only feeds the warning, never a
    block. Confirms the configured pattern list matches the ticket's actual non-commercial
    license (cc-by-nc-4.0) case-insensitively and does not flag a permissive/absent license."""
    assert promotion_service._is_non_commercial_license(license_str) is expected


def test_to_schema_surfaces_dataset_and_base_model_license_no_warning(db_session):
    """Happy path: base model license is always surfaced; a dataset with no license recorded
    and a permissive base model produces no warning."""
    model_version = _evaluated_model_version(db_session)
    decision = promotion_service.create_decision(
        db_session,
        model_version,
        DecisionCreateRequest(
            decision="PROMOTED", decided_by="reviewer-1", rationale="ok"
        ),
    )

    schema = promotion_service.to_schema(decision)

    assert schema.dataset_license is None
    assert schema.base_model_license == "apache-2.0"
    assert schema.license_warning is None


def test_promotion_request_warns_on_non_commercial_dataset_license(db_session):
    """Issue #134 AC: a dataset with a non-commercial license (the ticket's actual Phase A
    example, cc-by-nc-4.0) triggers an explicit warning on a promotion request - it must not
    silently pass through. The decision itself still succeeds (warning, not a hard block)."""
    model_version = _evaluated_model_version(db_session, dataset_license="cc-by-nc-4.0")

    decision = promotion_service.create_decision(
        db_session,
        model_version,
        DecisionCreateRequest(
            decision="PROMOTED", decided_by="reviewer-1", rationale="ok"
        ),
    )

    assert decision.decision == "PROMOTED"  # not blocked
    schema = promotion_service.to_schema(decision)
    assert schema.dataset_license == "cc-by-nc-4.0"
    assert schema.license_warning is not None
    assert "non-commercial" in schema.license_warning
    assert "production" in schema.license_warning


def test_rejection_does_not_warn_even_with_non_commercial_license(db_session):
    """A REJECTED decision is not heading toward production, so it never carries the warning
    even when the dataset license would otherwise trigger one."""
    model_version = _evaluated_model_version(db_session, dataset_license="cc-by-nc-4.0")

    decision = promotion_service.create_decision(
        db_session,
        model_version,
        DecisionCreateRequest(
            decision="REJECTED", decided_by="reviewer-1", rationale="n/a"
        ),
    )

    schema = promotion_service.to_schema(decision)
    assert schema.dataset_license == "cc-by-nc-4.0"
    assert schema.license_warning is None


def test_promotion_no_warning_for_permissive_dataset_license(db_session):
    model_version = _evaluated_model_version(db_session, dataset_license="mit")

    decision = promotion_service.create_decision(
        db_session,
        model_version,
        DecisionCreateRequest(
            decision="PROMOTED", decided_by="reviewer-1", rationale="ok"
        ),
    )

    schema = promotion_service.to_schema(decision)
    assert schema.license_warning is None


def test_non_commercial_pattern_list_is_configurable(db_session, monkeypatch):
    """The non-commercial pattern list is a config, not a hardcoded rule (issue #134 Open
    Decision) - widening or narrowing it changes what warns, without a code change."""
    monkeypatch.setattr(settings, "non_commercial_license_patterns", "some-custom-tag")
    model_version = _evaluated_model_version(db_session, dataset_license="cc-by-nc-4.0")

    decision = promotion_service.create_decision(
        db_session,
        model_version,
        DecisionCreateRequest(
            decision="PROMOTED", decided_by="reviewer-1", rationale="ok"
        ),
    )

    # cc-by-nc-4.0 no longer matches the (replaced) pattern list -> no warning.
    assert promotion_service.to_schema(decision).license_warning is None
