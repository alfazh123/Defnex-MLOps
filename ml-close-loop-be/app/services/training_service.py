import uuid
from datetime import datetime, timezone

from sqlalchemy.orm import Session

from app.models.dataset import DatasetVersion as DatasetVersionModel
from app.models.training import TrainingRun
from app.schemas.training import TrainingRun as TrainingRunSchema, TrainingRunCreateRequest

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


def complete_training_run(db: Session, training_run: TrainingRun, artifact_uri: str) -> TrainingRun:
    """RUNNING -> COMPLETED, recording the artifact location (PRD §10 worker steps 6-8)."""

    _transition(training_run, "COMPLETED")
    training_run.artifact_uri = artifact_uri
    db.flush()
    return training_run


def fail_training_run(db: Session, training_run: TrainingRun, error_message: str) -> TrainingRun:
    """RUNNING -> FAILED, recording the error (PRD §9's "error information ketika gagal")."""

    _transition(training_run, "FAILED")
    training_run.error_message = error_message
    db.flush()
    return training_run


def get_training_run(db: Session, training_run_id: str) -> TrainingRun | None:
    return db.get(TrainingRun, training_run_id)


def to_schema(training_run: TrainingRun) -> TrainingRunSchema:
    """Compose the flat ORM row into the nested TrainingRun response schema.

    `model_version` stays None — it's only populated once a completed run is registered
    into the model registry (US-011/US-012, not yet implemented; see openapi.yaml's own
    note that it's set by the backend's internal Register call on COMPLETED).
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
        model_version=None,
    )
