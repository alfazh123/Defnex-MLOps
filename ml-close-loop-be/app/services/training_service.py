import uuid
from datetime import datetime, timezone

from sqlalchemy.orm import Session

from app.models.dataset import DatasetVersion as DatasetVersionModel
from app.models.training import TrainingRun
from app.schemas.training import (
    TrainingRun as TrainingRunSchema,
    TrainingRunCreateRequest,
)
from app.services import unsloth_client

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


def complete_training_run(
    db: Session, training_run: TrainingRun, artifact_uri: str
) -> TrainingRun:
    """RUNNING -> COMPLETED, recording the artifact location (PRD §10 worker steps 6-8)."""

    _transition(training_run, "COMPLETED")
    training_run.artifact_uri = artifact_uri
    db.flush()
    return training_run


def fail_training_run(
    db: Session, training_run: TrainingRun, error_message: str
) -> TrainingRun:
    """RUNNING -> FAILED, recording the error (PRD §9's "error information ketika gagal")."""

    _transition(training_run, "FAILED")
    training_run.error_message = error_message
    db.flush()
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

    base_filter = select(TrainingRun)

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


async def sync_status_from_unsloth(
    db: Session, training_run: TrainingRun
) -> TrainingRun:
    """Fetch status from Unsloth Studio and update the DB record accordingly."""

    status_data = await unsloth_client.get_training_status(training_run.training_run_id)
    unsloth_status = status_data.get("status", "")

    if unsloth_status == "running":
        if training_run.status == "PENDING":
            _transition(training_run, "RUNNING")
        training_run.current_epoch = status_data.get(
            "current_epoch", training_run.current_epoch
        )
        training_run.current_step = status_data.get(
            "current_step", training_run.current_step
        )
        training_run.train_loss = status_data.get("train_loss", training_run.train_loss)
        training_run.eval_loss = status_data.get("eval_loss", training_run.eval_loss)
    elif unsloth_status == "completed":
        if training_run.status == "PENDING":
            _transition(training_run, "RUNNING")
        if training_run.status == "RUNNING":
            _transition(training_run, "COMPLETED")
        training_run.artifact_uri = status_data.get(
            "artifact_uri", training_run.artifact_uri
        )
    elif unsloth_status == "failed":
        if training_run.status == "PENDING":
            _transition(training_run, "RUNNING")
        if training_run.status == "RUNNING":
            _transition(training_run, "FAILED")
        training_run.error_message = status_data.get(
            "error_message", "Unknown error from Unsloth Studio"
        )

    db.flush()
    return training_run


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
