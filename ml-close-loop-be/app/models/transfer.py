from datetime import datetime

from sqlalchemy import String
from sqlalchemy.orm import Mapped, mapped_column

from app.db.base import Base


class ArtifactTransfer(Base):
    """Artifact transfer record (PRD §20.3, issue #72).

    Tracks cross-host or cross-storage artifact transfers with checksum
    verification.  Deployment is blocked when verification fails (PRD §29).
    """

    __tablename__ = "artifact_transfers"

    transfer_id: Mapped[str] = mapped_column(String, primary_key=True)
    artifact_uri: Mapped[str] = mapped_column(String)
    source_host: Mapped[str] = mapped_column(String)
    target_host: Mapped[str] = mapped_column(String)
    checksum_before: Mapped[str] = mapped_column(String)
    checksum_after: Mapped[str | None] = mapped_column(String, nullable=True)
    status: Mapped[str] = mapped_column(String, default="PENDING")
    created_at: Mapped[datetime] = mapped_column()
    completed_at: Mapped[datetime | None] = mapped_column(nullable=True)
    error_message: Mapped[str | None] = mapped_column(String, nullable=True)
