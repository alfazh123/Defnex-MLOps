from datetime import datetime, timezone

from sqlalchemy import func, select
from sqlalchemy.orm import Session

from app.models.dataset import Dataset, DatasetVersion as DatasetVersionModel
from app.models.feedback import Feedback
from app.schemas.dataset import (
    DatasetManifest,
    DatasetSummary,
    DatasetVersion as DatasetVersionSchema,
    DatasetVersionCreateRequest,
)


def register_dataset(db: Session, dataset_id: str) -> Dataset:
    """Get-or-create the Dataset row for `dataset_id` (PRD §7 "mendaftarkan dataset")."""

    dataset = db.get(Dataset, dataset_id)
    if dataset is None:
        dataset = Dataset(dataset_id=dataset_id)
        db.add(dataset)
        db.flush()
    return dataset


def _next_version(db: Session, dataset_id: str) -> int:
    latest_version = db.scalar(
        select(DatasetVersionModel.version)
        .where(DatasetVersionModel.dataset_id == dataset_id)
        .order_by(DatasetVersionModel.version.desc())
    )
    return (latest_version or 0) + 1


def create_dataset_version(
    db: Session, dataset_id: str, request: DatasetVersionCreateRequest
) -> DatasetVersionModel:
    """Create the next version for `dataset_id`, registering the dataset if needed.

    No intake/normalization pipeline exists yet, so the version is created directly in
    `PROCESSED` status to unblock the validation→training→deploy closed loop.
    """

    register_dataset(db, dataset_id)

    version = DatasetVersionModel(
        dataset_id=dataset_id,
        version=_next_version(db, dataset_id),
        status="PROCESSED",
        source_type=request.source_type,
        source_url_or_hf_id=request.source_dataset,
        source_commit_or_snapshot_date=request.source_commit_or_snapshot_date,
        source_format=request.source_format,
        seed=None,
        row_count=None,
        cleaning_steps_applied=[],
        created_at=datetime.now(timezone.utc),
        created_by=None,
    )
    db.add(version)
    db.commit()
    db.refresh(version)
    return version


def create_dataset_version_from_feedback(
    db: Session, dataset_id: str, feedback_ids: list[str], source_format: str
) -> DatasetVersionModel:
    """Create the next version for `dataset_id` from curated feedback (issue #42).

    Every id in `feedback_ids` must resolve to an existing, `APPROVED` `Feedback` row - a
    missing id raises `ValueError("... not found")` (API 404), a not-yet-`APPROVED` one raises
    `ValueError("... not APPROVED")` (API 409); no version is created either way. Selection is
    explicit and manual (the request lists exact ids) - there is no automatic
    threshold-based trigger, matching issue #42's stated scope.
    """

    rows = list(
        db.scalars(select(Feedback).where(Feedback.feedback_id.in_(feedback_ids)))
    )
    found_ids = {row.feedback_id for row in rows}
    missing = [fid for fid in feedback_ids if fid not in found_ids]
    if missing:
        raise ValueError(f"feedback_id(s) not found: {', '.join(missing)}")

    not_approved = [
        row.feedback_id for row in rows if row.curation_status != "APPROVED"
    ]
    if not_approved:
        raise ValueError(f"feedback_id(s) not APPROVED: {', '.join(not_approved)}")

    register_dataset(db, dataset_id)

    version = DatasetVersionModel(
        dataset_id=dataset_id,
        version=_next_version(db, dataset_id),
        status="PROCESSED",
        source_type="feedback",
        source_feedback_ids=feedback_ids,
        source_url_or_hf_id=None,
        source_commit_or_snapshot_date=None,
        source_format=source_format,
        seed=None,
        row_count=len(feedback_ids),
        cleaning_steps_applied=[],
        created_at=datetime.now(timezone.utc),
        created_by=None,
    )
    db.add(version)
    db.commit()
    db.refresh(version)
    return version


def get_dataset_version(
    db: Session, dataset_id: str, version: int
) -> DatasetVersionModel | None:
    return db.scalar(
        select(DatasetVersionModel).where(
            DatasetVersionModel.dataset_id == dataset_id,
            DatasetVersionModel.version == version,
        )
    )


def list_datasets(
    db: Session,
    limit: int = 20,
    offset: int = 0,
    status: str | None = None,
    search: str | None = None,
) -> tuple[list[DatasetSummary], int]:
    """Every dataset with its latest version + that version's status, for GET /datasets.

    No single ORM entity maps to this composed view, so it returns the response
    schema directly instead of an ORM instance (unlike the other functions here).
    """

    # Subquery: datasets that have at least one version
    ds_with_versions = (
        select(DatasetVersionModel.dataset_id)
        .group_by(DatasetVersionModel.dataset_id)
        .subquery()
    )

    base_filter = Dataset.dataset_id.in_(select(ds_with_versions.c.dataset_id))

    # Filter by latest version status
    if status is not None:
        latest_status = (
            select(DatasetVersionModel.dataset_id)
            .where(DatasetVersionModel.status == status)
            .group_by(DatasetVersionModel.dataset_id)
            .subquery()
        )
        base_filter = base_filter & Dataset.dataset_id.in_(
            select(latest_status.c.dataset_id)
        )

    # Search by dataset_id substring
    if search is not None:
        base_filter = base_filter & Dataset.dataset_id.ilike(f"%{search}%")

    # Count
    total = db.scalar(select(func.count()).select_from(Dataset).where(base_filter))

    datasets = db.scalars(
        select(Dataset)
        .where(base_filter)
        .order_by(Dataset.dataset_id)
        .limit(limit)
        .offset(offset)
    )
    return [
        DatasetSummary(
            dataset_id=dataset.dataset_id,
            latest_version=dataset.versions[-1].version,
            status=dataset.versions[-1].status,
        )
        for dataset in datasets
    ], total


def list_dataset_versions(db: Session, dataset_id: str) -> list[DatasetVersionModel]:
    return list(
        db.scalars(
            select(DatasetVersionModel)
            .where(DatasetVersionModel.dataset_id == dataset_id)
            .order_by(DatasetVersionModel.version.desc())
        )
    )


def to_schema(version: DatasetVersionModel) -> DatasetVersionSchema:
    """Compose the flat ORM row into the nested DatasetVersion response schema."""

    return DatasetVersionSchema(
        dataset_id=version.dataset_id,
        version=version.version,
        status=version.status,
        manifest=DatasetManifest(
            source_type=version.source_type,
            source_url_or_hf_id=version.source_url_or_hf_id,
            source_commit_or_snapshot_date=version.source_commit_or_snapshot_date,
            source_format=version.source_format,
            seed=version.seed,
            row_count=version.row_count,
            cleaning_steps_applied=version.cleaning_steps_applied,
            source_feedback_ids=version.source_feedback_ids,
            created_at=version.created_at,
            created_by=version.created_by,
        ),
    )
