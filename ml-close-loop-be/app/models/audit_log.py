from datetime import datetime

from sqlalchemy import JSON, Integer, String
from sqlalchemy.orm import Mapped, mapped_column

from app.db.base import Base


class AuditLog(Base):
    """Append-only audit trail for critical actions (issue #129, follow-up to closed #86 whose
    "audit log" acceptance criterion was never actually implemented).

    One row per critical transition: who did it (`actor_id`), what happened (`action`), on which
    resource, the before/after state, whether it succeeded, and why. Nothing here is ever
    updated or deleted by application code - callers only ever go through
    `app.services.audit_service.record_audit`, which only ever INSERTs.
    """

    __tablename__ = "audit_logs"

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    # Nullable: some actions (e.g. a failed login with an unknown username) have no resolvable
    # actor. Deliberately a plain int, not a ForeignKey - the actor's user row can be deleted
    # later (issue #142 retention) without breaking the historical audit record.
    actor_id: Mapped[int | None] = mapped_column(Integer, nullable=True, index=True)
    action: Mapped[str] = mapped_column(String, index=True)
    resource_type: Mapped[str] = mapped_column(String, index=True)
    resource_id: Mapped[str] = mapped_column(String, index=True)
    before_json: Mapped[dict | None] = mapped_column(JSON, nullable=True)
    after_json: Mapped[dict | None] = mapped_column(JSON, nullable=True)
    result: Mapped[str] = mapped_column(String)
    reason: Mapped[str | None] = mapped_column(String, nullable=True)
    created_at: Mapped[datetime] = mapped_column(index=True)
