import uuid
from datetime import datetime, timezone

from sqlalchemy.orm import Session

from app.models.dataset import DatasetVersion as DatasetVersionModel
from app.models.training import TrainingRun
from app.schemas.training import TrainingRunCreateRequest

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
