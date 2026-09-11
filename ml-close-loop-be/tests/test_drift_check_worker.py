"""Periodic post-promotion drift-check tests (issue #136): re-running the golden-set
evaluation for DEPLOYED model versions, storing results separately from the promotion-time
evaluation, and notifying admins on a new regression. No real GPU (mock ServingBackend only,
per CLAUDE.md's GPU constraint)."""

from datetime import datetime, timezone

from app.models.drift_check import ModelDriftCheck
from app.models.notification import Notification
from app.models.user import User
from app.schemas.dataset import DatasetVersionCreateRequest
from app.schemas.eval_set import EvalSetVersionCreateRequest
from app.schemas.training import TrainingConfig, TrainingRunCreateRequest
from app.services import (
    dataset_service,
    eval_set_service,
    model_service,
    training_service,
)
from app.services.drift_check_service import (
    TRIGGER_SCHEDULED_DRIFT_CHECK,
    has_new_regression,
    run_drift_check_for_version,
)
from app.services.evaluation_engine import compute_evaluation_update
from app.services.serving import InferenceError, MockServingBackend
from app.workers.drift_check_worker import run_once


def _deployed_model_version(
    db_session, *, model_id="qwen-sft-domain-x", eval_loss=0.84
):
    """A model version taken all the way to DEPLOYED, with a promotion-time evaluation
    baseline already recorded on it (mirrors the real REGISTERED -> EVALUATED -> PROMOTED ->
    DEPLOYED lifecycle, but sets `status` directly for the PROMOTED -> DEPLOYED leg - the same
    shortcut test_promotion_service.py / test_inference.py / test_deployment_single_source.py
    already use, since deployment_service.deploy's GPU/lock machinery is out of scope here)."""
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
            model_id=model_id,
            base_model="Qwen/Qwen3.8-27B",
            training_config=TrainingConfig(),
        ),
    )
    training_service.claim_training_run(db_session, training_run)
    training_run.eval_loss = eval_loss
    training_service.complete_training_run(
        db_session, training_run, artifact_uri="file:///tmp/adapter"
    )
    model_version = model_service.register_model_version(db_session, training_run)

    eval_version = eval_set_service.create_eval_set_version(
        db_session,
        "domain-benchmark",
        EvalSetVersionCreateRequest(
            records=[
                {"messages": [{"role": "user", "content": "golden-question-0"}]},
                {"messages": [{"role": "user", "content": "golden-question-1"}]},
            ]
        ),
    ).version
    model_service.trigger_evaluation(
        db_session,
        model_version,
        eval_set_id="domain-benchmark",
        eval_set_version=eval_version,
    )
    model_service.submit_evaluation(
        db_session,
        model_version,
        compute_evaluation_update(db_session, model_version, MockServingBackend()),
    )
    assert model_version.status == "EVALUATED"
    # Baseline: both golden-set records passed at promotion time.
    assert model_version.general_domain_regression_check["regressions_found"] == []

    model_version.status = "DEPLOYED"
    db_session.flush()
    return model_version


def _admin(db_session, username="admin1"):
    admin = User(
        username=username,
        hashed_password="x",
        role="admin",
        created_at=datetime.now(timezone.utc),
    )
    db_session.add(admin)
    db_session.flush()
    return admin


class _SecondRecordFailsBackend(MockServingBackend):
    """Passes every golden-set record except index 1 - simulates a regression appearing after
    promotion that was not present in the baseline."""

    def generate(self, prompt: str, model_id: str, version: int) -> str:
        if "golden-question-1" in prompt:
            raise InferenceError("simulated post-deploy regression")
        return super().generate(prompt, model_id, version)


# --- has_new_regression -------------------------------------------------------------------


def test_has_new_regression_true_for_previously_passing_record():
    baseline = {"checked": True, "regressions_found": []}
    new = {"checked": True, "regressions_found": [{"record_index": 1, "prompt": "x"}]}
    assert has_new_regression(baseline, new) is True


def test_has_new_regression_false_when_same_record_already_failed():
    baseline = {
        "checked": True,
        "regressions_found": [{"record_index": 1, "prompt": "x"}],
    }
    new = {"checked": True, "regressions_found": [{"record_index": 1, "prompt": "x"}]}
    assert has_new_regression(baseline, new) is False


def test_has_new_regression_false_when_no_regressions_in_new_run():
    baseline = {"checked": True, "regressions_found": []}
    new = {"checked": True, "regressions_found": []}
    assert has_new_regression(baseline, new) is False


def test_has_new_regression_fail_safe_when_no_baseline():
    new = {"checked": True, "regressions_found": [{"record_index": 0, "prompt": "x"}]}
    assert has_new_regression(None, new) is True


# --- run_drift_check_for_version -----------------------------------------------------------


def test_drift_check_records_separately_from_promotion_evaluation(db_session):
    model_version = _deployed_model_version(db_session)
    promotion_snapshot = dict(model_version.qualitative_comparison)

    record = run_drift_check_for_version(
        db_session, model_version, MockServingBackend()
    )

    assert record is not None
    assert record.trigger == TRIGGER_SCHEDULED_DRIFT_CHECK
    assert record.new_regression_detected is False
    # ModelVersion's own snapshot (the promotion baseline) is untouched.
    assert model_version.status == "DEPLOYED"
    assert model_version.qualitative_comparison == promotion_snapshot
    # The periodic result lives in its own table, queryable independently.
    rows = db_session.query(ModelDriftCheck).all()
    assert len(rows) == 1
    assert rows[0].id == record.id


def test_drift_check_score_drop_triggers_admin_notification(db_session):
    model_version = _deployed_model_version(db_session)
    admin = _admin(db_session)

    record = run_drift_check_for_version(
        db_session, model_version, _SecondRecordFailsBackend()
    )

    assert record.new_regression_detected is True
    assert (
        record.general_domain_regression_check["regressions_found"][0]["record_index"]
        == 1
    )

    notifications = db_session.query(Notification).filter_by(user_id=admin.id).all()
    assert len(notifications) == 1
    assert notifications[0].type == "MODEL_QUALITY_DRIFT_DETECTED"
    assert model_version.model_id in notifications[0].message


def test_drift_check_no_new_regression_does_not_notify(db_session):
    model_version = _deployed_model_version(db_session)
    admin = _admin(db_session)

    run_drift_check_for_version(db_session, model_version, MockServingBackend())

    notifications = db_session.query(Notification).filter_by(user_id=admin.id).all()
    assert notifications == []


def test_drift_check_skips_model_version_without_eval_set(db_session):
    """Same posture as evaluation_worker.process_next_evaluation: a ValueError from
    compute_evaluation_update (no eval_set_id/version) is caught, logged, and skipped - never
    crashes the cycle, and writes no row."""
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
            model_id="qwen-no-eval-set",
            base_model="Qwen/Qwen3.8-27B",
            training_config=TrainingConfig(),
        ),
    )
    training_service.claim_training_run(db_session, training_run)
    training_service.complete_training_run(
        db_session, training_run, artifact_uri="file:///tmp/adapter"
    )
    model_version = model_service.register_model_version(db_session, training_run)
    model_version.status = "DEPLOYED"  # no eval_set_id/eval_set_version ever set
    db_session.flush()

    result = run_drift_check_for_version(
        db_session, model_version, MockServingBackend()
    )

    assert result is None
    assert db_session.query(ModelDriftCheck).count() == 0


# --- run_once (the scheduled job) -----------------------------------------------------------


def test_run_once_checks_every_deployed_version_and_skips_others(db_session):
    deployed = _deployed_model_version(db_session, model_id="deployed-model")
    dataset_version = dataset_service.create_dataset_version(
        db_session,
        "another-ds",
        DatasetVersionCreateRequest(
            source_type="huggingface",
            source_dataset="HuggingFaceH4/no_robots",
            source_commit_or_snapshot_date="2026-08-01",
            source_format="chatml",
        ),
    )
    other_run = training_service.create_training_run(
        db_session,
        dataset_version,
        TrainingRunCreateRequest(
            dataset_id="another-ds",
            dataset_version=1,
            model_id="registered-only",
            base_model="Qwen/Qwen3.8-27B",
            training_config=TrainingConfig(),
        ),
    )
    training_service.claim_training_run(db_session, other_run)
    training_service.complete_training_run(
        db_session, other_run, artifact_uri="file:///tmp/adapter2"
    )
    not_deployed = model_service.register_model_version(db_session, other_run)
    db_session.flush()

    checked = run_once(db_session, backend=MockServingBackend())

    assert [c.model_version_id for c in checked] == [deployed.id]
    assert not_deployed.status == "REGISTERED"


def test_run_once_returns_empty_when_nothing_deployed(db_session):
    assert run_once(db_session, backend=MockServingBackend()) == []
