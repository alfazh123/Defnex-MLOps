from datetime import datetime

from sqlalchemy import Integer, String, Text
from sqlalchemy.orm import Mapped, mapped_column

from app.db.base import Base


class IdempotencyKey(Base):
    """Durable `X-Idempotency-Key` cache (issue #124, follow-up #81).

    Replaces the in-process Python dicts that used to live in `deployment_service.py` and
    `intake_validate.py`: a dict is per-process state, so with >1 API worker or a process
    restart the "same request -> same job" guarantee broke silently. This table is the swap-in
    Postgres backing (PRD v2 target is Postgres+Redis+Celery, but Redis isn't wired yet - don't
    jump ahead of the stack that's actually in) - callers only ever go through
    `app.services.idempotency_service`, so swapping the storage backend later doesn't touch them.

    `key` is the primary key, which is itself a unique constraint (a table can have at most one
    row per PK value) - satisfies the "unique constraint on key" acceptance criterion without a
    redundant second constraint.
    """

    __tablename__ = "idempotency_keys"

    key: Mapped[str] = mapped_column(String, primary_key=True)
    endpoint: Mapped[str] = mapped_column(String)
    response_status: Mapped[int] = mapped_column(Integer)
    response_body_json: Mapped[str] = mapped_column(Text)
    created_at: Mapped[datetime] = mapped_column()
    expires_at: Mapped[datetime] = mapped_column(index=True)
