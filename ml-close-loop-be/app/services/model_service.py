from datetime import datetime, timezone

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.models.model import Model, ModelVersion
from app.models.training import TrainingRun
from app.schemas.model import (
    EvaluationObject,
    EvaluationUpdateRequest,
    ModelRegistryRecord,
    ModelSummary,
)


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


def list_models(
    db: Session, status: str | None = None, search: str | None = None
) -> list[ModelSummary]:
    """List every model_id with its latest version and status (openapi.yaml GET /models),
    optionally filtered to models whose latest version is currently in `status`
    and/or whose model_id matches a search substring."""

    models = db.scalars(select(Model)).all()
    summaries = []
    for model in models:
        if not model.versions:
            continue
        latest = model.versions[-1]
        if status is not None and latest.status != status:
            continue
        if search is not None and search.lower() not in model.model_id.lower():
            continue
        summaries.append(
            ModelSummary(
                model_id=model.model_id,
                latest_version=latest.version,
                status=latest.status,
            )
        )
    return summaries


def get_model_version(db: Session, model_id: str, version: int) -> ModelVersion | None:
    return db.scalar(
        select(ModelVersion).where(
            ModelVersion.model_id == model_id, ModelVersion.version == version
        )
    )


def get_evaluation(model_version: ModelVersion) -> EvaluationObject:
    """Build the current evaluation object from a ModelVersion's stored signal fields
    (openapi.yaml GET .../evaluation) - any signal not yet submitted stays null."""

    return EvaluationObject(
        eval_loss_trend=model_version.eval_loss_trend,
        qualitative_comparison=model_version.qualitative_comparison,
        general_domain_regression_check=model_version.general_domain_regression_check,
    )


def submit_evaluation(
    db: Session, model_version: ModelVersion, update: EvaluationUpdateRequest
) -> ModelVersion:
    """Merge a partial evaluation payload onto a ModelVersion (model-artifact-versioning-lineage.md
    §5, openapi.yaml POST .../evaluation) and auto-transition REGISTERED -> EVALUATED once all
    three signal fields are present (model-promotion-approval-workflow.md §2) - partial data does
    not qualify. Caller (API layer) is responsible for the 409 "not editable" guard."""

    if update.eval_loss_trend is not None:
        model_version.eval_loss_trend = update.eval_loss_trend.model_dump()
    if update.qualitative_comparison is not None:
        model_version.qualitative_comparison = (
            update.qualitative_comparison.model_dump()
        )
    if update.general_domain_regression_check is not None:
        model_version.general_domain_regression_check = (
            update.general_domain_regression_check.model_dump()
        )

    all_signals_present = (
        model_version.eval_loss_trend is not None
        and model_version.qualitative_comparison is not None
        and model_version.general_domain_regression_check is not None
    )
    if all_signals_present and model_version.status == "REGISTERED":
        model_version.status = "EVALUATED"

    db.flush()
    return model_version


def to_schema(model_version: ModelVersion) -> ModelRegistryRecord:
    """Build the full lineage response (openapi.yaml ModelRegistryRecord) from a ModelVersion row.

    `dataset_id`/`dataset_version` are derived via the training_run -> dataset_version
    relationship chain (not stored as columns - see US-011/US-012's notes)."""

    dataset_version = model_version.training_run.dataset_version
    return ModelRegistryRecord(
        model_id=model_version.model_id,
        version=model_version.version,
        status=model_version.status,
        training_run_id=model_version.training_run_id,
        base_model=model_version.base_model,
        dataset_id=dataset_version.dataset_id,
        dataset_version=dataset_version.version,
        dataset_validation_report_ref=model_version.dataset_validation_report_ref,
        training_config=model_version.training_config,
        created_at=model_version.created_at,
        created_by=model_version.created_by,
        evaluation=get_evaluation(model_version),
        artifacts=model_version.artifacts,
        promotion_decision_ref=model_version.promotion_decision_ref,
        previous_model_id=model_version.previous_model_id,
    )
