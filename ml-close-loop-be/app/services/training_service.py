import uuid
from datetime import datetime, timedelta, timezone

from sqlalchemy import case, select, update
from sqlalchemy.orm import Session

from app.config import settings
from app.models.dataset import DatasetVersion as DatasetVersionModel
from app.models.training import TrainingRun
from app.schemas.training import (
    GpuHoursReportRow,
    PRIORITY_LEVELS,
    TrainingRun as TrainingRunSchema,
    TrainingRunCreateRequest,
)
from app.services import audit_service

# Issue #135: rank used to order the claim query `priority DESC, created_at ASC`.
# Any value outside PRIORITY_LEVELS (should not happen -- validated at the request
# schema) ranks as "normal" rather than erroring the claim query.
_PRIORITY_RANK = {level: rank for rank, level in enumerate(PRIORITY_LEVELS)}

# PRD §9's lifecycle prose says QUEUED; the frozen TrainingRunStatus enum (openapi.yaml,
# mlops-api-contract.md §3.4) uses PENDING for the same "not started yet" state (see US-007's note).
# STALE (issue #60, PRD §10.2/§10.3): a worker stopped reporting heartbeat beyond the threshold;
# distinct from FAILED and reclaimable (STALE -> RUNNING retry).
_VALID_TRANSITIONS: dict[str, set[str]] = {
    "PENDING": {"RUNNING"},
    "RUNNING": {"COMPLETED", "FAILED", "STALE"},
    "STALE": {"RUNNING"},
    "COMPLETED": set(),
    "FAILED": set(),
}


def create_training_run(
    db: Session, dataset_version: DatasetVersionModel, request: TrainingRunCreateRequest
) -> TrainingRun:
    """Create a TrainingRun in PENDING status without waiting for training to finish (PRD §9)."""

    training_run = TrainingRun(
        training_run_id=f"run-{uuid.uuid4().hex[:6]}",
        dataset_version_id=dataset_version.id,
        model_id=request.model_id,
        base_model=request.base_model,
        training_config=request.training_config.model_dump(),
        status="PENDING",
        triggered_by=request.triggered_by,
        compute_resource_id=request.compute_resource_id,
        priority=request.priority,
        created_at=datetime.now(timezone.utc),
    )
    db.add(training_run)
    db.flush()
    return training_run


def next_claimable_run(db: Session) -> TrainingRun | None:
    """Select the next PENDING/STALE run to claim (issue #135 fair-use queue).

    Ordered `priority DESC, created_at ASC` instead of pure FIFO: within the same
    priority tier the oldest run still wins, but a "high" run jumps ahead of an
    older "low"/"normal" one. Does not mutate anything -- caller still runs it
    through `claim_training_run` for the atomic compare-and-set.
    """

    rank = case(
        _PRIORITY_RANK, value=TrainingRun.priority, else_=_PRIORITY_RANK["normal"]
    )
    return db.scalar(
        select(TrainingRun)
        .where(TrainingRun.status.in_(["PENDING", "STALE"]))
        .order_by(rank.desc(), TrainingRun.created_at.asc())
    )


def gpu_hours_report(db: Session) -> list[GpuHoursReportRow]:
    """Aggregate GPU-hours per (triggered_by, model_id) from TrainingRun timestamps
    that already exist (issue #135) -- no new table, no invented metric beyond a
    plain wall-clock sum. Only runs that actually started (`started_at` set)
    contribute; a run still RUNNING is measured up to now so in-flight GPU time is
    visible. Aggregated in Python rather than dialect-specific SQL (e.g. Postgres
    EXTRACT(EPOCH ...) vs SQLite) since TrainingRun volume is small and this keeps
    the query portable across the SQLite-dev / Postgres-prod split (PRD §23.1).
    """

    # started_at/finished_at columns are plain (timezone-naive) TIMESTAMP, always
    # written as UTC wall-clock (datetime.now(timezone.utc)) elsewhere in this
    # service; a raw column select (bypassing the ORM identity map) reads them back
    # naive, so `now` is stripped of tzinfo too to keep the subtraction consistent.
    now = datetime.now(timezone.utc).replace(tzinfo=None)
    rows = db.execute(
        select(
            TrainingRun.triggered_by,
            TrainingRun.model_id,
            TrainingRun.started_at,
            TrainingRun.finished_at,
        ).where(TrainingRun.started_at.is_not(None))
    ).all()

    totals: dict[tuple[str | None, str], dict[str, float]] = {}
    for triggered_by, model_id, started_at, finished_at in rows:
        key = (triggered_by, model_id)
        bucket = totals.setdefault(key, {"seconds": 0.0, "count": 0})
        end = finished_at or now
        bucket["seconds"] += (end - started_at).total_seconds()
        bucket["count"] += 1

    return [
        GpuHoursReportRow(
            triggered_by=triggered_by,
            model_id=model_id,
            run_count=int(bucket["count"]),
            gpu_hours=round(bucket["seconds"] / 3600, 4),
        )
        for (triggered_by, model_id), bucket in sorted(
            totals.items(), key=lambda kv: (kv[0][0] or "", kv[0][1])
        )
    ]


def _transition(training_run: TrainingRun, new_status: str) -> None:
    allowed = _VALID_TRANSITIONS[training_run.status]
    if new_status not in allowed:
        raise ValueError(
            f"Cannot transition training run {training_run.training_run_id} "
            f"from {training_run.status} to {new_status}"
        )
    training_run.status = new_status


def start_training_run(db: Session, training_run: TrainingRun) -> TrainingRun:
    """PENDING -> RUNNING (PRD §10 worker step 2)."""

    _transition(training_run, "RUNNING")
    db.flush()
    return training_run


def claim_training_run(db: Session, training_run: TrainingRun) -> bool:
    """Atomically claim a claimable (PENDING or STALE) run for execution (issue #33/#60).

    Compare-and-set on the status column: only the caller that flips
    PENDING|STALE -> RUNNING at the SQL level wins. Two worker processes that have
    both selected the same PENDING row can then not both win; the loser gets
    `False` and must not execute the runner. Works on SQLite (where
    `SELECT ... FOR UPDATE` is a no-op) and on PostgreSQL. STALE is re-claimable so
    a timed-out run can be retried (issue #60), up to `settings.max_stale_retries`
    times before the run is forced to FAILED (P2-6).
    """

    was_stale = training_run.status == "STALE"
    if was_stale:
        new_count = (training_run.retry_count or 0) + 1
        if new_count > settings.max_stale_retries:
            training_run.status = "FAILED"
            training_run.error_message = (
                f"Exceeded max stale retries ({settings.max_stale_retries})"
            )
            training_run.finished_at = datetime.now(timezone.utc)
            db.flush()
            return False
        training_run.retry_count = new_count

    result = db.execute(
        update(TrainingRun)
        .where(
            TrainingRun.training_run_id == training_run.training_run_id,
            TrainingRun.status.in_(["PENDING", "STALE"]),
        )
        .values(status="RUNNING")
        .execution_options(synchronize_session=False)
    )
    if result.rowcount != 1:
        return False
    now = datetime.now(timezone.utc)
    training_run.status = "RUNNING"
    training_run.started_at = now
    # Seed the heartbeat at claim time so a worker that crashes right after claiming
    # still has a timestamp to go stale from (issue #60).
    training_run.heartbeat_at = now
    return True


def touch_heartbeat(db: Session, training_run_id: str) -> int:
    """Atomically refresh `heartbeat_at` for a RUNNING run by id (issue #60).

    Standalone SQL update with `synchronize_session=False` so it can be driven from a
    worker heartbeat thread that does not hold the run's ORM object. Returns the number
    of rows updated (0 when the run is no longer RUNNING). Caller commits.
    """

    result = db.execute(
        update(TrainingRun)
        .where(
            TrainingRun.training_run_id == training_run_id,
            TrainingRun.status == "RUNNING",
        )
        .values(heartbeat_at=datetime.now(timezone.utc))
        .execution_options(synchronize_session=False)
    )
    return result.rowcount


def stale_runs(
    db: Session, *, threshold_seconds: int | None = None
) -> list[TrainingRun]:
    """Detector (issue #60, PRD §10.2/§10.3): RUNNING runs whose liveness has lapsed.

    A run is stale when its last heartbeat (or, before any heartbeat, its start time)
    is older than `stale_threshold_seconds`. Returns the stale runs without mutating
    them; `mark_stale_runs` applies the transition. Caller commits.
    """

    threshold = (
        threshold_seconds
        if threshold_seconds is not None
        else settings.stale_threshold_seconds
    )
    from sqlalchemy import func, or_, select

    cutoff = datetime.now(timezone.utc) - timedelta(seconds=threshold)
    return list(
        db.scalars(
            select(TrainingRun).where(
                TrainingRun.status == "RUNNING",
                or_(
                    TrainingRun.heartbeat_at.is_(None),
                    func.coalesce(TrainingRun.heartbeat_at, TrainingRun.started_at)
                    < cutoff,
                ),
            )
        ).all()
    )


def mark_stale_runs(db: Session, *, threshold_seconds: int | None = None) -> int:
    """Transition lapsed RUNNING runs to STALE (issue #60). Returns the count marked.

    The bulk UPDATE runs on the SQL level without touching the ORM objects, so the
    caller should consider the session state; returns how many rows flipped so the
    worker can log recovery. Caller commits.
    """

    threshold = (
        threshold_seconds
        if threshold_seconds is not None
        else settings.stale_threshold_seconds
    )
    from sqlalchemy import func, or_, select

    cutoff = datetime.now(timezone.utc) - timedelta(seconds=threshold)
    result = db.execute(
        update(TrainingRun)
        .where(
            TrainingRun.status == "RUNNING",
            or_(
                TrainingRun.heartbeat_at.is_(None),
                func.coalesce(TrainingRun.heartbeat_at, TrainingRun.started_at)
                < cutoff,
            ),
        )
        .values(status="STALE")
        .execution_options(synchronize_session=False)
    )
    count = result.rowcount
    if count > 0:
        stale_ids = db.scalars(
            select(TrainingRun.training_run_id).where(
                TrainingRun.status == "STALE",
                func.coalesce(TrainingRun.heartbeat_at, TrainingRun.started_at)
                < cutoff,
            )
        ).all()
        from app.services.alerting import alert_training_stale

        for run_id in stale_ids:
            alert_training_stale(job_id=run_id)
    return count


def update_training_progress(
    db: Session,
    training_run: TrainingRun,
    *,
    epoch: int | None = None,
    current_step: int | None = None,
    train_loss: float | None = None,
    eval_loss: float | None = None,
) -> TrainingRun:
    """Overwrite the live progress fields on a RUNNING training run (issue #38).

    Called by the training runner on each progress event; the caller is responsible for
    committing so `GET /training-runs/{id}` observes non-NULL values while still RUNNING.
    """

    if training_run.status != "RUNNING":
        raise ValueError(
            f"Cannot record training progress for run {training_run.training_run_id} "
            f"in status {training_run.status} (must be RUNNING)"
        )
    if epoch is not None:
        training_run.current_epoch = epoch
    if current_step is not None:
        training_run.current_step = current_step
    if train_loss is not None:
        training_run.train_loss = train_loss
    if eval_loss is not None:
        training_run.eval_loss = eval_loss
    db.flush()
    return training_run


def complete_training_run(
    db: Session, training_run: TrainingRun, artifact_uri: str
) -> TrainingRun:
    """RUNNING -> COMPLETED, recording the artifact location (PRD §10 worker steps 6-8)."""

    _transition(training_run, "COMPLETED")
    training_run.artifact_uri = artifact_uri
    training_run.finished_at = datetime.now(timezone.utc)
    db.flush()
    from app.telemetry import _training_runs_counter

    if _training_runs_counter is not None:
        _training_runs_counter.add(1, {"status": "COMPLETED"})
    return training_run


def fail_training_run(
    db: Session, training_run: TrainingRun, error_message: str
) -> TrainingRun:
    """RUNNING -> FAILED, recording the error (PRD §9's "error information ketika gagal")."""
    _transition(training_run, "FAILED")
    training_run.error_message = error_message
    training_run.finished_at = datetime.now(timezone.utc)
    db.flush()
    # ponytail: ORM object sometimes reverts dirty attrs in-memory after flush
    # under concurrent session use (subprocess watchdog thread). A forced refresh
    # ensures the returned object's in-memory state matches the committed DB row.
    db.refresh(training_run)
    from app.telemetry import _training_runs_counter

    if _training_runs_counter is not None:
        _training_runs_counter.add(1, {"status": "FAILED"})
    from app.services.alerting import alert_training_failed

    alert_training_failed(job_id=training_run.training_run_id, error=error_message)
    return training_run


def set_external_job_id(
    db: Session, training_run: TrainingRun, external_job_id: str
) -> TrainingRun:
    """Persist the provider-assigned external_job_id on the run (issue #74).

    Called by the worker after TrainingProvider.submit() returns so the job can be
    polled/cancelled via the provider contract. Caller commits.
    """
    training_run.external_job_id = external_job_id
    db.flush()
    return training_run


def retry_training_run(
    db: Session, training_run: TrainingRun, actor_id: int | None = None
) -> TrainingRun:
    """Create a new PENDING run as a retry of a FAILED run (issue #61, PRD §10.5).

    The original run is never mutated; only FAILED runs are retryable.
    """

    if training_run.status != "FAILED":
        raise ValueError(
            f"Cannot retry training run {training_run.training_run_id} "
            f"with status {training_run.status}: must be FAILED"
        )

    new_run = TrainingRun(
        training_run_id=f"run-{uuid.uuid4().hex[:6]}",
        dataset_version_id=training_run.dataset_version_id,
        model_id=training_run.model_id,
        base_model=training_run.base_model,
        training_config=training_run.training_config,
        status="PENDING",
        triggered_by=training_run.triggered_by,
        compute_resource_id=training_run.compute_resource_id,
        retry_of=training_run.training_run_id,
        created_at=datetime.now(timezone.utc),
    )
    db.add(new_run)
    db.flush()

    # issue #129 (audit AC "cancel/retry training"): no "cancel" endpoint exists in this codebase
    # today (grepped app/api + app/services - only retry does), so only retry is audited here;
    # see PR description for that gap noted separately, per CLAUDE.md's no-silent-scope-change rule.
    audit_service.record_audit(
        db,
        actor_id=actor_id,
        action=audit_service.RETRY_TRAINING,
        resource_type="training_run",
        resource_id=training_run.training_run_id,
        before={"status": "FAILED"},
        after={"status": "PENDING", "new_training_run_id": new_run.training_run_id},
    )
    return new_run


def get_training_run(db: Session, training_run_id: str) -> TrainingRun | None:
    return db.get(TrainingRun, training_run_id)


def list_training_runs(
    db: Session,
    limit: int = 20,
    offset: int = 0,
    status: str | None = None,
    model: str | None = None,
) -> tuple[list[TrainingRun], int]:
    from sqlalchemy import func, select
    from sqlalchemy.orm import selectinload

    base_filter = select(TrainingRun).options(
        selectinload(TrainingRun.dataset_version),
        selectinload(TrainingRun.model_versions),
    )

    if status is not None:
        base_filter = base_filter.where(TrainingRun.status == status)
    if model is not None:
        base_filter = base_filter.where(TrainingRun.model_id.ilike(f"%{model}%"))

    total = db.scalar(select(func.count()).select_from(base_filter.subquery()))
    runs = list(
        db.scalars(
            base_filter.order_by(TrainingRun.created_at.desc())
            .limit(limit)
            .offset(offset)
        ).all()
    )
    return runs, total


def to_schema(training_run: TrainingRun) -> TrainingRunSchema:
    """Compose the flat ORM row into the nested TrainingRun response schema.

    `model_version` is the forward link to the model version this run produced — populated once
    the worker's internal Register call has succeeded on COMPLETED (openapi.yaml TrainingRun),
    None before that. This is the only place the contract links a training_run_id forward to its
    model version.
    """

    return TrainingRunSchema(
        training_run_id=training_run.training_run_id,
        status=training_run.status,
        dataset_id=training_run.dataset_version.dataset_id,
        dataset_version=training_run.dataset_version.version,
        model_id=training_run.model_id,
        base_model=training_run.base_model,
        training_config=training_run.training_config,
        triggered_by=training_run.triggered_by,
        created_at=training_run.created_at,
        current_epoch=training_run.current_epoch,
        current_step=training_run.current_step,
        train_loss=training_run.train_loss,
        eval_loss=training_run.eval_loss,
        model_version=training_run.model_versions[-1].version
        if training_run.model_versions
        else None,
        retry_of=training_run.retry_of,
    )
