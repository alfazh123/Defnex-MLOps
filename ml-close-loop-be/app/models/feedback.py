from datetime import datetime

from sqlalchemy import ForeignKey, String
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.db.base import Base


class Feedback(Base):
    """User feedback on an inference response, curated into candidate rows for the next
    dataset version (issue #42). References the concrete `ModelVersion` that produced the
    response (`InferenceResponse.model_id`/`version`, issue #41) so it can be traced back.

    `curation_status` is a small state machine (PENDING -> APPROVED|REJECTED), guarded in
    `feedback_service` the same way `TrainingRun`/`ModelVersion` guard their own transitions -
    never edited directly.
    """

    __tablename__ = "feedback"

    feedback_id: Mapped[str] = mapped_column(String, primary_key=True)
    model_version_id: Mapped[int] = mapped_column(
        ForeignKey("model_versions.id"), index=True
    )
    prompt: Mapped[str] = mapped_column(String)
    response: Mapped[str] = mapped_column(String)
    rating: Mapped[int] = mapped_column()
    correction: Mapped[str | None] = mapped_column(String, nullable=True)
    curation_status: Mapped[str] = mapped_column(String, default="PENDING")
    submitted_by: Mapped[str | None] = mapped_column(String, nullable=True)
    submitted_at: Mapped[datetime] = mapped_column()
    curated_by: Mapped[str | None] = mapped_column(String, nullable=True)
    curated_at: Mapped[datetime | None] = mapped_column(nullable=True)

    model_version: Mapped["ModelVersion"] = relationship()
