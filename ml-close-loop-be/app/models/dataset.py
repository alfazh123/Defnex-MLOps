from datetime import datetime

from sqlalchemy import ForeignKey, JSON, String, UniqueConstraint
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.db.base import Base


class Dataset(Base):
    """A registered dataset (identified by a slug, e.g. `no_robots`)."""

    __tablename__ = "datasets"

    dataset_id: Mapped[str] = mapped_column(String, primary_key=True)

    versions: Mapped[list["DatasetVersion"]] = relationship(
        back_populates="dataset", order_by="DatasetVersion.version"
    )


class DatasetVersion(Base):
    """A single version of a dataset's manifest (dataset-lifecycle-and-schema.md §2)."""

    __tablename__ = "dataset_versions"
    __table_args__ = (UniqueConstraint("dataset_id", "version", name="uq_dataset_version"),)

    id: Mapped[int] = mapped_column(primary_key=True)
    dataset_id: Mapped[str] = mapped_column(ForeignKey("datasets.dataset_id"))
    version: Mapped[int] = mapped_column()
    status: Mapped[str] = mapped_column(String, default="PENDING")

    # DatasetManifest fields (openapi.yaml DatasetManifest / dataset-lifecycle-and-schema.md §2)
    source_url_or_hf_id: Mapped[str | None] = mapped_column(String, nullable=True)
    source_commit_or_snapshot_date: Mapped[str | None] = mapped_column(String, nullable=True)
    source_format: Mapped[str] = mapped_column(String)
    seed: Mapped[int | None] = mapped_column(nullable=True)
    row_count: Mapped[int | None] = mapped_column(nullable=True)
    cleaning_steps_applied: Mapped[list[str]] = mapped_column(JSON, default=list)
    created_at: Mapped[datetime] = mapped_column()
    created_by: Mapped[str | None] = mapped_column(String, nullable=True)

    dataset: Mapped["Dataset"] = relationship(back_populates="versions")
    validation_reports: Mapped[list["ValidationReport"]] = relationship(
        back_populates="dataset_version"
    )
