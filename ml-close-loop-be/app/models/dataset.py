from datetime import datetime

from sqlalchemy import ForeignKey, JSON, String, UniqueConstraint
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.db.base import Base


class Dataset(Base):
    """A registered dataset (identified by a slug, e.g. `no_robots`)."""

    __tablename__ = "datasets"

    dataset_id: Mapped[str] = mapped_column(String, primary_key=True)
    display_name: Mapped[str | None] = mapped_column(String, nullable=True)
    description: Mapped[str | None] = mapped_column(String, nullable=True)

    versions: Mapped[list["DatasetVersion"]] = relationship(
        back_populates="dataset", order_by="DatasetVersion.version"
    )


class DatasetVersion(Base):
    """A single version of a dataset's manifest (dataset-lifecycle-and-schema.md §2)."""

    __tablename__ = "dataset_versions"
    __table_args__ = (
        UniqueConstraint("dataset_id", "version", name="uq_dataset_version"),
    )

    id: Mapped[int] = mapped_column(primary_key=True)
    dataset_id: Mapped[str] = mapped_column(ForeignKey("datasets.dataset_id"))
    version: Mapped[int] = mapped_column()
    status: Mapped[str] = mapped_column(String, default="PENDING")

    # DatasetManifest fields (openapi.yaml DatasetManifest / dataset-lifecycle-and-schema.md §2)
    # `source_type` (was request-only, never persisted, before issue #42) records where a
    # version actually came from - "huggingface"/"file_upload" (external) or "feedback"
    # (issue #42, curated from approved Feedback rows). Nullable for versions created before
    # this column existed. `source_feedback_ids` is the traceable origin for the "feedback"
    # case - the exact Feedback rows that became this version's records.
    source_type: Mapped[str | None] = mapped_column(String, nullable=True)
    source_feedback_ids: Mapped[list[str] | None] = mapped_column(JSON, nullable=True)
    source_url_or_hf_id: Mapped[str | None] = mapped_column(String, nullable=True)
    source_commit_or_snapshot_date: Mapped[str | None] = mapped_column(
        String, nullable=True
    )
    source_format: Mapped[str] = mapped_column(String)
    # Dataset license for this version (issue #134): e.g. "cc-by-nc-4.0", "apache-2.0". Nullable
    # for versions created before this column existed. Not enforced automatically - surfaced by
    # the promotion flow (promotion_service.py) as a human-facing warning when a non-commercial
    # license is heading toward a production promotion; the exact "commercial use" definition is
    # an open governance decision, so nothing here silently blocks a promotion on this value.
    license: Mapped[str | None] = mapped_column(String, nullable=True)
    seed: Mapped[int | None] = mapped_column(nullable=True)
    row_count: Mapped[int | None] = mapped_column(nullable=True)
    cleaning_steps_applied: Mapped[list[str]] = mapped_column(JSON, default=list)
    created_at: Mapped[datetime] = mapped_column()
    created_by: Mapped[str | None] = mapped_column(String, nullable=True)
    raw_file_uri: Mapped[str | None] = mapped_column(String, nullable=True)
    canonical_file_uri: Mapped[str | None] = mapped_column(String, nullable=True)

    dataset: Mapped["Dataset"] = relationship(back_populates="versions")
    validation_reports: Mapped[list["ValidationReport"]] = relationship(
        back_populates="dataset_version"
    )
    training_runs: Mapped[list["TrainingRun"]] = relationship(
        back_populates="dataset_version"
    )
