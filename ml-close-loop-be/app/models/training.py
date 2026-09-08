from datetime import datetime

from sqlalchemy import ForeignKey, JSON, String
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.db.base import Base


class TrainingRun(Base):
    """An asynchronous training job with its config snapshot and outcome (PRD §9)."""

    __tablename__ = "training_runs"

    training_run_id: Mapped[str] = mapped_column(String, primary_key=True)
    dataset_version_id: Mapped[int] = mapped_column(
        ForeignKey("dataset_versions.id"), index=True
    )
    model_id: Mapped[str] = mapped_column(String)
    base_model: Mapped[str] = mapped_column(String)
    training_config: Mapped[dict] = mapped_column(JSON)
    status: Mapped[str] = mapped_column(String, default="PENDING")
    triggered_by: Mapped[str | None] = mapped_column(String, nullable=True)
    created_at: Mapped[datetime] = mapped_column()
    # Issue #38: wall-clock bounds of the actual training execution, single source of truth
    # for `ModelVersion.training_started_at`/`training_completed_at` at registration time.
    started_at: Mapped[datetime | None] = mapped_column(nullable=True)
    finished_at: Mapped[datetime | None] = mapped_column(nullable=True)
    # Issue #60: worker liveness. Persisted periodically while a run is RUNNING;
    # a run whose heartbeat has not advanced past the stale threshold is reclaimed
    # as STALE (PRD §10.4), so a crashed/frozen worker never leaves it RUNNING forever.
    heartbeat_at: Mapped[datetime | None] = mapped_column(nullable=True)

    current_epoch: Mapped[int | None] = mapped_column(nullable=True)
    current_step: Mapped[int | None] = mapped_column(nullable=True)
    train_loss: Mapped[float | None] = mapped_column(nullable=True)
    eval_loss: Mapped[float | None] = mapped_column(nullable=True)

    artifact_uri: Mapped[str | None] = mapped_column(String, nullable=True)
    external_job_id: Mapped[str | None] = mapped_column(String, nullable=True)
    error_message: Mapped[str | None] = mapped_column(String, nullable=True)

    dataset_version: Mapped["DatasetVersion"] = relationship(
        back_populates="training_runs"
    )
    model_versions: Mapped[list["ModelVersion"]] = relationship(
        back_populates="training_run"
    )
