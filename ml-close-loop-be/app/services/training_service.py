import uuid
from datetime import datetime, timezone

from sqlalchemy import update
from sqlalchemy.orm import Session

from app.models.dataset import DatasetVersion as DatasetVersionModel
from app.models.training import TrainingRun
from app.schemas.training import (
    TrainingRun as TrainingRunSchema,
    TrainingRunCreateRequest,
)

# PRD §9's lifecycle prose says QUEUED; the frozen TrainingRunStatus enum (openapi.yaml,
# mlops-api-contract.md §3.4) uses PENDING for the same "not started yet" state (see US-007's note).
_VALID_TRANSITIONS: dict[str, set[str]] = {
    "PENDING": {"RUNNING"},
    "RUNNING": {"COMPLETED", "FAILED"},
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
    """Atomically claim a PENDING run for execution (issue #33).

    Compare-and-set on the status column: only the caller that flips
    PENDING -> RUNNING at the SQL level wins. Two worker processes that have
    both selected the same PENDING row can then not both win; the loser gets
    `False` and must not execute the runner. Works on SQLite (where
    `SELECT ... FOR UPDATE` is a no-op) and on PostgreSQL.
    """

    result = db.execute(
        update(TrainingRun)
        .where(
            TrainingRun.training_run_id == training_run.training_run_id,
            TrainingRun.status == "PENDING",
        )
        .values(status="RUNNING")
        .execution_options(synchronize_session=False)
    )
    if result.rowcount != 1:
        return False
    training_run.status = "RUNNING"
    training_run.started_at = datetime.now(timezone.utc)
    return True


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
