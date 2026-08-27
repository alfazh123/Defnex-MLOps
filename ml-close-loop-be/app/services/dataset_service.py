from datetime import datetime, timezone

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.models.dataset import Dataset, DatasetVersion as DatasetVersionModel
from app.schemas.dataset import (
    DatasetManifest,
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


def create_dataset_version(
    db: Session, dataset_id: str, request: DatasetVersionCreateRequest
) -> DatasetVersionModel:
    """Create the next version for `dataset_id`, registering the dataset if needed.

    No intake/normalization pipeline exists yet (not in scope for this story), so the
    manifest is populated directly from the request and the version is left in
    `PROCESSING`, matching openapi.yaml's documented POST response.
    """

    register_dataset(db, dataset_id)

    latest_version = db.scalar(
        select(DatasetVersionModel.version)
        .where(DatasetVersionModel.dataset_id == dataset_id)
        .order_by(DatasetVersionModel.version.desc())
    )
    next_version = (latest_version or 0) + 1

    version = DatasetVersionModel(
        dataset_id=dataset_id,
        version=next_version,
        status="PROCESSING",
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


def get_dataset_version(db: Session, dataset_id: str, version: int) -> DatasetVersionModel | None:
    return db.scalar(
        select(DatasetVersionModel).where(
            DatasetVersionModel.dataset_id == dataset_id,
            DatasetVersionModel.version == version,
        )
    )


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
            source_url_or_hf_id=version.source_url_or_hf_id,
            source_commit_or_snapshot_date=version.source_commit_or_snapshot_date,
            source_format=version.source_format,
            seed=version.seed,
            row_count=version.row_count,
            cleaning_steps_applied=version.cleaning_steps_applied,
            created_at=version.created_at,
            created_by=version.created_by,
        ),
    )
