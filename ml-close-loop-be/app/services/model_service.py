import os
import re
import subprocess
from datetime import datetime, timezone
from pathlib import Path

from sqlalchemy import select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session, selectinload

from app.models.model import Model, ModelVersion
from app.models.training import TrainingRun
from app.schemas.model import (
    EvaluationObject,
    EvaluationUpdateRequest,
    ModelRegistryRecord,
    ModelSummary,
)
from app.services.artifact_storage import LocalFilesystemArtifactStorage

_MAX_VERSION_RETRIES = 10


def _slug(value: str) -> str:
    """Turn a free identifier (e.g. a HuggingFace `org/name`) into a path/URL-safe slug.
    Alphanumerics and dots are kept; runs of any other character collapse to a single dash
    (so `Qwen/Qwen3.8-27B` -> `Qwen-Qwen3.8-27B` has no doubled separators)."""
    slugged = re.sub(r"[^A-Za-z0-9.]+", "-", value)
    slugged = re.sub(r"-+", "-", slugged).strip("-")
    return slugged or "model"


def build_version_name(model_id: str, base_model: str, version: int) -> str:
    """Version name following `{project}-{base_model}-v{N}` (issue #38), with `project`
    being the `model_id`. `model_id` is request-validated to `[A-Za-z0-9._-]`; `base_model`
    is slugged so the combined name is safe to use as a filesystem directory name."""
    return f"{model_id}-{_slug(base_model)}-v{version}"


def _current_git_commit() -> str | None:
    """Best-effort backend git commit the run started from (issue #38). Respects the
    GIT_COMMIT env var (set in builds that bake a commit), else reads `git rev-parse HEAD`."""
    override = os.environ.get("GIT_COMMIT")
    if override:
        return override
    try:
        out = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
            timeout=5,
        ).stdout.strip()
        return out or None
    except (subprocess.SubprocessError, OSError):
        return None


def _ensure_model_row(db: Session, model_id: str) -> None:
    """Guarantee a `models` row exists, tolerating a concurrent first-registration race."""
    if db.get(Model, model_id) is not None:
        return
    try:
        with db.begin_nested():
            db.add(Model(model_id=model_id))
            db.flush()
    except IntegrityError:
        # lost the race creating the row; a concurrent writer won the unique PK — either way
        # the row exists for our FK, so transparently continue.
        pass


def _allocate_version(
    db: Session, model_id: str, training_run: TrainingRun
) -> tuple[ModelVersion, int]:
    """Atomically allocate the next `(model_id, version)` pair.

    Instead of `SELECT MAX(version)+1` then a blind INSERT (which races and surfaces as an
    IntegrityError/500 under two simultaneous registrations — issue #38), the version is
    computed and inserted inside a SAVEPOINT. If a concurrent writer claims the same version
    first, the unique constraint fires, the savepoint rolls back, and the caller retries with
    the now-updated MAX. Works on SQLite and PostgreSQL.
    """
    for _ in range(_MAX_VERSION_RETRIES):
        try:
            with db.begin_nested():
                latest = db.scalar(
                    select(ModelVersion.version)
                    .where(ModelVersion.model_id == model_id)
                    .order_by(ModelVersion.version.desc())
                )
                next_version = (latest or 0) + 1
                model_version = ModelVersion(
                    model_id=model_id,
                    version=next_version,
                    name=build_version_name(
                        model_id, training_run.base_model, next_version
                    ),
                    status="REGISTERED",
                    training_run_id=training_run.training_run_id,
                    base_model=training_run.base_model,
                    training_config=training_run.training_config,
                    git_commit=_current_git_commit(),
                    training_started_at=training_run.started_at,
                    training_completed_at=training_run.finished_at,
                    artifacts=[{"type": "adapter", "uri": training_run.artifact_uri}],
                    created_at=datetime.now(timezone.utc),
                )
                db.add(model_version)
                db.flush()
            return model_version, next_version
        except IntegrityError:
            # Unique (model_id, version) violated — another writer took this version. The
            # SAVEPOINT rolled back this attempt; the outer transaction is still usable for
            # the next iteration with a freshly computed version.
            continue
    raise RuntimeError(
        f"could not allocate a model version for {model_id!r} "
        f"after {_MAX_VERSION_RETRIES} attempts"
    )


def register_model_version(
    db: Session,
    training_run: TrainingRun,
    *,
    staging_dir: str | None = None,
    storage: LocalFilesystemArtifactStorage | None = None,
) -> ModelVersion:
    """Register a ModelVersion from a COMPLETED TrainingRun with full lineage
    (model-artifact-versioning-lineage.md §6/§7 Register operation).

    Version allocation is atomic (see `_allocate_version`). When `staging_dir` is provided
    (the runner-driven path, issue #38), the staged training output is moved into an immutable
    per-version directory named `{model_id}-{base_model_slug}-v{N}` and accompanied by a
    `metadata.json`; writing to an already-populated version path is rejected. When
    `staging_dir` is None (unit-test / flat artifact_uri path), the raw `artifact_uri` is kept.

    `dataset_id`/`dataset_version` are reachable via `training_run.dataset_version` (not stored
    as columns — same flat-ORM/derived-nested pattern, see US-011/US-012 notes).
    """

    if training_run.status != "COMPLETED":
        raise ValueError(
            f"Cannot register a model version from training run "
            f"{training_run.training_run_id} in status {training_run.status} (must be COMPLETED)"
        )

    model_id = training_run.model_id
    _ensure_model_row(db, model_id)

    model_version, next_version = _allocate_version(db, model_id, training_run)

    if staging_dir is not None:
        metadata = {
            "model_id": model_id,
            "name": model_version.name,
            "version": next_version,
            "training_run_id": training_run.training_run_id,
            "dataset_id": training_run.dataset_version.dataset_id,
            "dataset_version": training_run.dataset_version.version,
            "base_model": training_run.base_model,
            "training_config": training_run.training_config,
            "git_commit": model_version.git_commit,
            "started_at": training_run.started_at,
            "finished_at": training_run.finished_at,
        }
        final_uri = (storage or LocalFilesystemArtifactStorage()).finalize_version(
            model_id, model_version.name, Path(staging_dir), metadata
        )
        model_version.artifacts = [{"type": "adapter", "uri": final_uri}]
        training_run.artifact_uri = final_uri
        db.flush()

    return model_version


def list_models(
    db: Session, status: str | None = None, search: str | None = None
) -> list[ModelSummary]:
    """List every model_id with its latest version and status (openapi.yaml GET /models),
    optionally filtered to models whose latest version is currently in `status`
    and/or whose model_id matches a search substring."""

    models = db.scalars(select(Model).options(selectinload(Model.versions))).all()
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

    if update.eval_set_id is not None:
        model_version.eval_set_id = update.eval_set_id
    if update.eval_set_version is not None:
        model_version.eval_set_version = update.eval_set_version

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
        name=model_version.name,
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
        eval_set_id=model_version.eval_set_id,
        eval_set_version=model_version.eval_set_version,
        artifacts=model_version.artifacts,
        promotion_decision_ref=model_version.promotion_decision_ref,
        previous_model_id=model_version.previous_model_id,
    )
