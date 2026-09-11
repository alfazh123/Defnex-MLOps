from datetime import datetime

from sqlalchemy import Boolean, ForeignKey, JSON, String
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.db.base import Base


class ModelDriftCheck(Base):
    """One periodic post-promotion evaluation run against the golden/eval set (issue #136,
    PRD §41 Principle 4 / §28.1) for a `ModelVersion` that is already `DEPLOYED`.

    Deliberately a separate append-only table rather than overwriting
    `ModelVersion.eval_loss_trend` / `qualitative_comparison` / `general_domain_regression_check`
    (app/models/model.py) - those three columns hold only the single latest snapshot (no
    history) and are the promotion-time evaluation that a DEPLOYED model must never lose or
    have silently replaced. They also double as the "baseline" this table's rows are diffed
    against (`app/services/drift_check_service.py`) - there is no need to duplicate that
    promotion evaluation into a `trigger="promotion"` row here, only new periodic runs are
    written, always tagged `trigger=TRIGGER_SCHEDULED_DRIFT_CHECK`. The `trigger` column exists
    so a future write path (if one is ever added) can share this table without the two kinds of
    evaluation record becoming ambiguous, per the issue's acceptance criteria.
    """

    __tablename__ = "model_drift_checks"

    id: Mapped[int] = mapped_column(primary_key=True)
    model_version_id: Mapped[int] = mapped_column(
        ForeignKey("model_versions.id"), index=True
    )
    trigger: Mapped[str] = mapped_column(String, index=True)

    eval_set_id: Mapped[str | None] = mapped_column(String, nullable=True)
    eval_set_version: Mapped[int | None] = mapped_column(nullable=True)
    eval_loss_trend: Mapped[dict | None] = mapped_column(JSON, nullable=True)
    qualitative_comparison: Mapped[dict | None] = mapped_column(JSON, nullable=True)
    general_domain_regression_check: Mapped[dict | None] = mapped_column(
        JSON, nullable=True
    )

    # Whether this run surfaced a golden-set record failure that the promotion-time baseline
    # did not have (see drift_check_service.has_new_regression) - the "significant" drop this
    # issue asks to notify on, expressed as a binary "new regression" flag rather than an
    # invented numeric threshold (CLAUDE.md: don't invent evaluation metrics/thresholds).
    new_regression_detected: Mapped[bool] = mapped_column(Boolean, default=False)

    created_at: Mapped[datetime] = mapped_column()

    model_version: Mapped["ModelVersion"] = relationship()  # noqa: F821
