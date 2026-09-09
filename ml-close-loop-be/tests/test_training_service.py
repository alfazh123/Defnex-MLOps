import pytest
from datetime import datetime, timedelta, timezone

from app.schemas.dataset import DatasetVersionCreateRequest
from app.schemas.training import TrainingConfig, TrainingRunCreateRequest
from app.services import dataset_service, model_service, training_service


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

    training_run = training_service.create_training_run(
        db_session, dataset_version, _create_request()
    )

    assert training_run.training_run_id.startswith("run-")
    assert training_run.status == "PENDING"
    assert training_run.dataset_version_id == dataset_version.id
    assert training_run.model_id == "qwen-sft-domain-x"
    assert training_run.base_model == "Qwen/Qwen3.8-27B"
    assert training_run.training_config["peft_method"] == "lora"


def test_completed_lifecycle(db_session):
    dataset_version = _dataset_version(db_session)
    training_run = training_service.create_training_run(
        db_session, dataset_version, _create_request()
    )

    training_service.start_training_run(db_session, training_run)
    assert training_run.status == "RUNNING"

    training_service.complete_training_run(
        db_session, training_run, artifact_uri="file:///tmp/adapter"
    )
    assert training_run.status == "COMPLETED"
    assert training_run.artifact_uri == "file:///tmp/adapter"


def test_failed_lifecycle(db_session):
    dataset_version = _dataset_version(db_session)
    training_run = training_service.create_training_run(
        db_session, dataset_version, _create_request()
    )

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
    training_run = training_service.create_training_run(
        db_session, dataset_version, _create_request()
    )
    training_run.status = from_status

    action = {
        "start": lambda: training_service.start_training_run(db_session, training_run),
        "complete": lambda: training_service.complete_training_run(
            db_session, training_run, "uri"
        ),
        "fail": lambda: training_service.fail_training_run(
            db_session, training_run, "err"
        ),
    }[transition]

    with pytest.raises(ValueError):
        action()


def test_claim_training_run_claims_only_once(db_session):
    """CAS claim (issue #33): the first worker flips PENDING -> RUNNING and wins; the same
    call on an already claimed (or non-PENDING) run loses."""
    dataset_version = _dataset_version(db_session)
    training_run = training_service.create_training_run(
        db_session, dataset_version, _create_request()
    )
    assert training_run.status == "PENDING"

    assert training_service.claim_training_run(db_session, training_run) is True
    assert training_run.status == "RUNNING"

    assert training_service.claim_training_run(db_session, training_run) is False


def test_claim_training_run_looses_on_pending_mismatch(db_session):
    """CAS must not claim a run that is no longer PENDING at the SQL level."""
    dataset_version = _dataset_version(db_session)
    training_run = training_service.create_training_run(
        db_session, dataset_version, _create_request()
    )
    training_run.status = "RUNNING"
    db_session.flush()

    assert training_service.claim_training_run(db_session, training_run) is False


def test_claim_sets_started_at_and_completion_sets_finished_at(db_session):
    """Issue #38: the run's start/end are recorded alongside the claim and completion, and
    copied onto the registered model version."""
    dataset_version = _dataset_version(db_session)
    training_run = training_service.create_training_run(
        db_session, dataset_version, _create_request()
    )
    assert training_run.started_at is None

    training_service.claim_training_run(db_session, training_run)
    assert training_run.started_at is not None
    assert training_run.finished_at is None

    training_service.complete_training_run(db_session, training_run, artifact_uri="uri")
    assert training_run.finished_at is not None
    assert training_run.finished_at >= training_run.started_at


def test_update_training_progress_rejects_non_running_run(db_session):
    dataset_version = _dataset_version(db_session)
    training_run = training_service.create_training_run(
        db_session, dataset_version, _create_request()
    )

    with pytest.raises(ValueError):
        training_service.update_training_progress(
            db_session, training_run, current_step=1
        )


def test_update_training_progress_overwrites_fields_on_running_run(db_session):
    dataset_version = _dataset_version(db_session)
    training_run = training_service.create_training_run(
        db_session, dataset_version, _create_request()
    )
    training_service.claim_training_run(db_session, training_run)

    training_service.update_training_progress(
        db_session,
        training_run,
        epoch=1,
        current_step=100,
        train_loss=0.5,
        eval_loss=0.45,
    )
    training_service.update_training_progress(
        db_session, training_run, current_step=200, train_loss=0.4
    )

    assert training_run.current_epoch == 1
    assert training_run.current_step == 200
    assert training_run.train_loss == 0.4
    assert training_run.eval_loss == 0.45


def _seed_training_runs(session, n):
    for idx in range(n):
        dv = dataset_service.create_dataset_version(
            session,
            f"ds-{idx}",
            DatasetVersionCreateRequest(
                source_type="huggingface",
                source_dataset=f"HuggingFaceH4/ds-{idx}",
                source_commit_or_snapshot_date="2026-08-01",
                source_format="chatml",
            ),
        )
        run = training_service.create_training_run(
            session, dv, _create_request(model_id=f"model-{idx}")
        )
        training_service.start_training_run(session, run)
        training_service.complete_training_run(session, run, artifact_uri=f"uri-{idx}")
        model_service.register_model_version(session, run)
    session.commit()


def test_list_training_runs_has_no_n_plus_1(count_queries):
    with count_queries() as (session, counters):
        _seed_training_runs(session, 10)
        counters["n"] = 0
        runs, _ = training_service.list_training_runs(session, limit=10, offset=0)
        [training_service.to_schema(r) for r in runs]
        count_10 = counters["n"]

    with count_queries() as (session, counters):
        _seed_training_runs(session, 1)
        counters["n"] = 0
        runs, _ = training_service.list_training_runs(session, limit=1, offset=0)
        [training_service.to_schema(r) for r in runs]
        count_1 = counters["n"]

    # 9 extra rows must not add 2 lazy loads each (dataset_version + model_versions) -
    # selectinload keeps the growth to a couple of extra statements.
    assert count_10 - count_1 < 9


def _running_run(db_session, model_id="qwen-sft-domain-x"):
    dataset_version = _dataset_version(db_session)
    training_run = training_service.create_training_run(
        db_session, dataset_version, _create_request(model_id=model_id)
    )
    training_service.claim_training_run(db_session, training_run)
    return training_run


def test_claim_seeds_heartbeat_at(db_session):
    """The claim seeds `heartbeat_at` so a worker that crashes immediately after claiming
    still has a timestamp to go stale from (issue #60 AC #5)."""
    run = _running_run(db_session)
    assert run.status == "RUNNING"
    assert run.heartbeat_at is not None


def test_touch_heartbeat_updates_only_running_run(db_session):
    run = _running_run(db_session)
    assert run.heartbeat_at is not None

    assert training_service.touch_heartbeat(db_session, run.training_run_id) == 1
    db_session.flush()
    db_session.refresh(run)
    # column is stored naive-UTC (SQLite DateTime drops tzinfo); assert it advanced to "now"
    assert run.heartbeat_at is not None
    assert (
        datetime.now(timezone.utc).replace(tzinfo=None) - run.heartbeat_at
    ).total_seconds() < 5


def test_touch_heartbeat_noop_when_run_not_running(db_session):
    dataset_version = _dataset_version(db_session)
    run = training_service.create_training_run(
        db_session, dataset_version, _create_request()
    )
    assert run.status == "PENDING"
    assert training_service.touch_heartbeat(db_session, run.training_run_id) == 0
    assert run.heartbeat_at is None


def test_running_to_stale_transition_is_valid(db_session):
    run = _running_run(db_session)
    training_service._transition(run, "STALE")
    assert run.status == "STALE"


def test_stale_to_running_retry_is_valid(db_session):
    run = _running_run(db_session)
    training_service._transition(run, "STALE")
    training_service._transition(run, "RUNNING")
    assert run.status == "RUNNING"


def test_mark_stale_runs_flags_lapsed_heartbeat(db_session):
    run = _running_run(db_session)
    run.heartbeat_at = datetime.now(timezone.utc) - timedelta(seconds=3600)
    db_session.flush()

    count = training_service.mark_stale_runs(db_session, threshold_seconds=60)
    db_session.commit()
    db_session.refresh(run)

    assert count == 1
    assert run.status == "STALE"


def test_mark_stale_runs_keeps_fresh_run_running(db_session):
    run = _running_run(db_session)
    run.heartbeat_at = datetime.now(timezone.utc)
    db_session.flush()

    count = training_service.mark_stale_runs(db_session, threshold_seconds=3600)

    assert count == 0
    assert run.status == "RUNNING"


def test_mark_stale_runs_uses_started_at_when_no_heartbeat(db_session):
    """A RUNNING run with no heartbeat at all (crash before any heartbeat persisted) is
    staled from its start time, not left RUNNING forever."""
    dataset_version = _dataset_version(db_session)
    run = training_service.create_training_run(
        db_session, dataset_version, _create_request()
    )
    training_service.claim_training_run(db_session, run)
    run.heartbeat_at = None
    run.started_at = datetime.now(timezone.utc) - timedelta(seconds=3600)
    db_session.flush()

    count = training_service.mark_stale_runs(db_session, threshold_seconds=60)
    db_session.commit()
    db_session.refresh(run)

    assert count == 1
    assert run.status == "STALE"


def test_stale_run_can_be_reclaimed(db_session):
    """AC #4: a STALE run (crash recovery) can be claimed again and re-run."""
    run = _running_run(db_session)
    training_service._transition(run, "STALE")
    db_session.flush()

    assert training_service.claim_training_run(db_session, run) is True
    assert run.status == "RUNNING"


def test_completed_and_failed_cannot_return_to_running(db_session):
    run = _running_run(db_session)
    training_service._transition(run, "COMPLETED")
    with pytest.raises(ValueError):
        training_service._transition(run, "RUNNING")

    run2 = _running_run(db_session, model_id="qwen-sft-domain-y")
    training_service._transition(run2, "FAILED")
    with pytest.raises(ValueError):
        training_service._transition(run2, "RUNNING")


def _failed_run(db_session, model_id="qwen-sft-domain-x"):
    dataset_version = _dataset_version(db_session)
    training_run = training_service.create_training_run(
        db_session, dataset_version, _create_request(model_id=model_id)
    )
    training_service.claim_training_run(db_session, training_run)
    training_service.fail_training_run(db_session, training_run, error_message="OOM")
    return training_run


def test_retry_failed_run_creates_new_pending_run_with_retry_of(db_session):
    """Issue #61: a FAILED run can be retried into a fresh PENDING run whose `retry_of`
    points back at the original; the original record is not mutated (immutable history)."""
    failed = _failed_run(db_session)
    failed_id = failed.training_run_id
    failed_config = failed.training_config

    new_run = training_service.retry_training_run(db_session, failed)

    assert new_run is not failed
    assert new_run.status == "PENDING"
    assert new_run.retry_of == failed_id
    assert new_run.dataset_version_id == failed.dataset_version_id
    assert new_run.model_id == failed.model_id
    assert new_run.base_model == failed.base_model
    assert new_run.training_config == failed_config
    assert new_run.triggered_by == failed.triggered_by
    # Original stays FAILED and untouched.
    assert failed.status == "FAILED"
    assert failed.error_message == "OOM"
    assert failed.retry_of is None


@pytest.mark.parametrize("status", ["PENDING", "RUNNING", "COMPLETED", "STALE"])
def test_retry_rejects_non_failed_run(db_session, status):
    """Retry is only legal from FAILED; any other status raises (endpoint maps to 409)."""
    dataset_version = _dataset_version(db_session)
    run = training_service.create_training_run(
        db_session, dataset_version, _create_request()
    )
    run.status = status
    db_session.flush()

    with pytest.raises(ValueError, match="must be FAILED"):
        training_service.retry_training_run(db_session, run)


def test_to_schema_exposes_stale_status(db_session):
    """STALE must round-trip through the API schema (it is a valid status), not fail
    Pydantic validation in `to_schema`."""
    run = _running_run(db_session)
    training_service._transition(run, "STALE")
    db_session.flush()

    schema = training_service.to_schema(run)
    assert schema.status == "STALE"
