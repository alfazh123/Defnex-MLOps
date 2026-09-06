from datetime import datetime

from sqlalchemy import ForeignKey, JSON, String, UniqueConstraint
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.db.base import Base


class EvalSet(Base):
    """A curated golden/eval set (identified by a slug, e.g. `domain-benchmark`).

    The eval set is the comparison surface for the H8 leakage check during dataset
    validation (validation-rules.md H8) and the fixed question set evaluations are
    measured against (model-promotion-approval-workflow.md §13). It is stored
    separately from training data, with its own version counter.
    """

    __tablename__ = "eval_sets"

    eval_set_id: Mapped[str] = mapped_column(String, primary_key=True)

    versions: Mapped[list["EvalSetVersion"]] = relationship(
        back_populates="eval_set", order_by="EvalSetVersion.version"
    )


class EvalSetVersion(Base):
    """A single version of an eval set's record content."""

    __tablename__ = "eval_set_versions"
    __table_args__ = (
        UniqueConstraint("eval_set_id", "version", name="uq_eval_set_version"),
    )

    id: Mapped[int] = mapped_column(primary_key=True)
    eval_set_id: Mapped[str] = mapped_column(ForeignKey("eval_sets.eval_set_id"))
    version: Mapped[int] = mapped_column()
    records: Mapped[list] = mapped_column(JSON, default=list)
    created_at: Mapped[datetime] = mapped_column()
    created_by: Mapped[str | None] = mapped_column(String, nullable=True)

    eval_set: Mapped["EvalSet"] = relationship(back_populates="versions")
