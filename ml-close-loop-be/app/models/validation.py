from datetime import datetime

from sqlalchemy import ForeignKey, JSON, String
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.db.base import Base


class ValidationReport(Base):
    """A dataset-level validation run report (validation-rules.md §5/§7).

    Identity is `{dataset_id}/{version}` + `run_at` — re-running validation
    produces additional rows, there is no separate report version counter.
    """

    __tablename__ = "validation_reports"

    id: Mapped[int] = mapped_column(primary_key=True)
    dataset_version_id: Mapped[int] = mapped_column(ForeignKey("dataset_versions.id"))
    rule_set_version: Mapped[str] = mapped_column(String)
    run_at: Mapped[datetime] = mapped_column()
    record_count: Mapped[int] = mapped_column()
    status_counts: Mapped[dict] = mapped_column(JSON)
    warnings_summary: Mapped[dict] = mapped_column(JSON, default=dict)
    dataset_statistics: Mapped[dict] = mapped_column(JSON, default=dict)
    gate_decision: Mapped[str] = mapped_column(String)
    gate_reason: Mapped[str] = mapped_column(String)

    dataset_version: Mapped["DatasetVersion"] = relationship(back_populates="validation_reports")
