from datetime import datetime, timezone

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.models.model import Model, ModelVersion
from app.models.training import TrainingRun


def register_model_version(db: Session, training_run: TrainingRun) -> ModelVersion:
    """Register a ModelVersion from a COMPLETED TrainingRun with full lineage
    (model-artifact-versioning-lineage.md §6/§7 Register operation).

    `dataset_id`/`dataset_version`/`dataset_validation_report_ref` are not stored as columns
    (same flat-ORM/derived-nested pattern as US-011's note) - they're reachable via
    `training_run.dataset_version` and left for a future `to_schema()` (US-013) to derive.
    """

    if training_run.status != "COMPLETED":
        raise ValueError(
            f"Cannot register a model version from training run "
            f"{training_run.training_run_id} in status {training_run.status} (must be COMPLETED)"
        )

    model_id = training_run.model_id
    model = db.get(Model, model_id)
    if model is None:
        model = Model(model_id=model_id)
        db.add(model)
        db.flush()

    latest_version = db.scalar(
        select(ModelVersion.version)
        .where(ModelVersion.model_id == model_id)
        .order_by(ModelVersion.version.desc())
    )
    next_version = (latest_version or 0) + 1

    model_version = ModelVersion(
        model_id=model_id,
        version=next_version,
        status="REGISTERED",
        training_run_id=training_run.training_run_id,
        base_model=training_run.base_model,
        training_config=training_run.training_config,
        artifacts=[{"type": "adapter", "uri": training_run.artifact_uri}],
        created_at=datetime.now(timezone.utc),
    )
    db.add(model_version)
    db.flush()
    return model_version
