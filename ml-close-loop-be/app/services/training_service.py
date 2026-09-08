import uuid
from datetime import datetime, timedelta, timezone

from sqlalchemy import update
from sqlalchemy.orm import Session

from app.config import settings
from app.models.dataset import DatasetVersion as DatasetVersionModel
from app.models.training import TrainingRun
from app.schemas.training import (
    TrainingRun as TrainingRunSchema,
    TrainingRunCreateRequest,
)

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
        created_at=datetime.now(timezone.utc),
    )
    db.add(training_run)
    db.flush()
    return training_run


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
    a timed-out run can be retried (issue #60).
    """

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


def stale_runs(db: Session, *, threshold_seconds: int | None = None) -> list[TrainingRun]:
    """Detector (issue #60, PRD §10.2/§10.3): RUNNING runs whose liveness has lapsed.

    A run is stale when its last heartbeat (or, before any heartbeat, its start time)
    is older than `stale_threshold_seconds`. Returns the stale runs without mutating
    them; `mark_stale_runs` applies the transition. Caller commits.
    """

    threshold = threshold_seconds if threshold_seconds is not None else settings.stale_threshold_seconds
    from sqlalchemy import func, or_, select

    cutoff = datetime.now(timezone.utc) - timedelta(seconds=threshold)
    return list(
        db.scalars(
            select(TrainingRun).where(
                TrainingRun.status == "RUNNING",
                or_(
                    TrainingRun.heartbeat_at.is_(None),
                    func.coalesce(
                        TrainingRun.heartbeat_at, TrainingRun.started_at
                    ) < cutoff,
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

    threshold = threshold_seconds if threshold_seconds is not None else settings.stale_threshold_seconds
    from sqlalchemy import func, or_

    cutoff = datetime.now(timezone.utc) - timedelta(seconds=threshold)
    result = db.execute(
        update(TrainingRun)
        .where(
            TrainingRun.status == "RUNNING",
            or_(
                TrainingRun.heartbeat_at.is_(None),
                func.coalesce(TrainingRun.heartbeat_at, TrainingRun.started_at) < cutoff,
            ),
        )
        .values(status="STALE")
        .execution_options(synchronize_session=False)
    )
    return result.rowcount


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
    return training_run


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
    )
