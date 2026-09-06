from datetime import datetime

from sqlalchemy import ForeignKey, JSON, String
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.db.base import Base


class PromotionDecision(Base):
    """A human promotion, rejection, or rollback decision (model-promotion-approval-workflow.md §8).

    One schema covers all three decision types (§8 Decision 2), differing only in
    `decision`. `evidence_snapshot` is a frozen copy of the model version's evaluation
    object at decision time, not a live reference (§8 Decision 4) - null for ROLLBACK.
    """

    __tablename__ = "promotion_decisions"

    decision_id: Mapped[str] = mapped_column(String, primary_key=True)
    model_version_id: Mapped[int] = mapped_column(
        ForeignKey("model_versions.id"), index=True
    )
    decision: Mapped[str] = mapped_column(String)
    decided_by: Mapped[str | None] = mapped_column(String, nullable=True)
    decided_at: Mapped[datetime] = mapped_column()
    evidence_snapshot: Mapped[dict | None] = mapped_column(JSON, nullable=True)
    # Frozen eval-set reference the decision's evidence was measured against (issue #43) -
    # null for ROLLBACK, like `evidence_snapshot`.
    eval_set_id: Mapped[str | None] = mapped_column(String, nullable=True)
    eval_set_version: Mapped[int | None] = mapped_column(nullable=True)
    rationale: Mapped[str] = mapped_column(String)
    rollback_of_version: Mapped[int | None] = mapped_column(nullable=True)

    model_version: Mapped["ModelVersion"] = relationship(
        back_populates="promotion_decisions"
    )
