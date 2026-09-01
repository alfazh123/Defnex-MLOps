from datetime import datetime

from sqlalchemy import ForeignKey, JSON, String
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.db.base import Base


class TrainingRun(Base):
    """An asynchronous training job with its config snapshot and outcome (PRD §9)."""

    __tablename__ = "training_runs"

    training_run_id: Mapped[str] = mapped_column(String, primary_key=True)
    dataset_version_id: Mapped[int] = mapped_column(ForeignKey("dataset_versions.id"))
    model_id: Mapped[str] = mapped_column(String)
    base_model: Mapped[str] = mapped_column(String)
    training_config: Mapped[dict] = mapped_column(JSON)
    status: Mapped[str] = mapped_column(String, default="PENDING")
    triggered_by: Mapped[str | None] = mapped_column(String, nullable=True)
    created_at: Mapped[datetime] = mapped_column()

    current_epoch: Mapped[int | None] = mapped_column(nullable=True)
    current_step: Mapped[int | None] = mapped_column(nullable=True)
    train_loss: Mapped[float | None] = mapped_column(nullable=True)
    eval_loss: Mapped[float | None] = mapped_column(nullable=True)

    artifact_uri: Mapped[str | None] = mapped_column(String, nullable=True)
    error_message: Mapped[str | None] = mapped_column(String, nullable=True)

    dataset_version: Mapped["DatasetVersion"] = relationship(
        back_populates="training_runs"
    )
    model_versions: Mapped[list["ModelVersion"]] = relationship(
        back_populates="training_run"
    )
