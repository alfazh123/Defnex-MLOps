from datetime import datetime

from sqlalchemy import ForeignKey, JSON, Index, String, UniqueConstraint, text
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.db.base import Base


class Model(Base):
    """A continuing fine-tuning line (model-artifact-versioning-lineage.md §2)."""

    __tablename__ = "models"

    model_id: Mapped[str] = mapped_column(String, primary_key=True)

    versions: Mapped[list["ModelVersion"]] = relationship(
        back_populates="model", order_by="ModelVersion.version"
    )


class ModelVersion(Base):
    """A registered model artifact version (model-artifact-versioning-lineage.md §6/§8).

    Evaluation fields (§5) are persisted directly on this record - no separate
    `evaluation_id` foreign key / table, per US-014's acceptance criteria.
    """

    __tablename__ = "model_versions"
    __table_args__ = (
        UniqueConstraint("model_id", "version", name="uq_model_version"),
        Index(
            "uq_model_versions_one_deployed",
            "model_id",
            unique=True,
            sqlite_where=text("status = 'DEPLOYED'"),
            postgresql_where=text("status = 'DEPLOYED'"),
        ),
    )

    id: Mapped[int] = mapped_column(primary_key=True)
    model_id: Mapped[str] = mapped_column(ForeignKey("models.model_id"), index=True)
    version: Mapped[int] = mapped_column()
    # Stable version name following `{model_id}-{base_model_slug}-v{N}` (issue #38). Stored
    # on the row AND exposed via ModelRegistryRecord; also names the immutable artifact dir.
    name: Mapped[str | None] = mapped_column(String, nullable=True)
    status: Mapped[str] = mapped_column(String, default="REGISTERED")

    training_run_id: Mapped[str] = mapped_column(
        ForeignKey("training_runs.training_run_id"), index=True
    )
    base_model: Mapped[str] = mapped_column(String)
    training_config: Mapped[dict] = mapped_column(JSON)
    # Issue #64: deterministic id of `training_config` (sha256 truncated to 16 hex chars),
    # so the §37 lineage chain can point at one config hash per version.
    training_config_hash: Mapped[str | None] = mapped_column(String, nullable=True)
    dataset_validation_report_ref: Mapped[str | None] = mapped_column(
        String, nullable=True
    )
    # Issue #38 lineage: backend git commit the run started from, plus the wall-clock bounds
    # of the training execution. Sourced from the producing training_run (single source of truth).
    git_commit: Mapped[str | None] = mapped_column(String, nullable=True)
    training_started_at: Mapped[datetime | None] = mapped_column(nullable=True)
    training_completed_at: Mapped[datetime | None] = mapped_column(nullable=True)
    artifacts: Mapped[list[dict]] = mapped_column(JSON, default=list)

    eval_loss_trend: Mapped[dict | None] = mapped_column(JSON, nullable=True)
    qualitative_comparison: Mapped[dict | None] = mapped_column(JSON, nullable=True)
    general_domain_regression_check: Mapped[dict | None] = mapped_column(
        JSON, nullable=True
    )

    # The golden/eval set this version's evaluation was measured against
    # (model-promotion-approval-workflow.md §13, issue #43).
    eval_set_id: Mapped[str | None] = mapped_column(String, nullable=True)
    eval_set_version: Mapped[int | None] = mapped_column(nullable=True)

    created_at: Mapped[datetime] = mapped_column()
    created_by: Mapped[str | None] = mapped_column(String, nullable=True)

    promotion_decision_ref: Mapped[str | None] = mapped_column(String, nullable=True)
    previous_model_id: Mapped[str | None] = mapped_column(String, nullable=True)

    model: Mapped["Model"] = relationship(back_populates="versions")
    training_run: Mapped["TrainingRun"] = relationship(back_populates="model_versions")
    promotion_decisions: Mapped[list["PromotionDecision"]] = relationship(
        back_populates="model_version"
    )
