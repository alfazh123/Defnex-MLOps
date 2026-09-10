from datetime import datetime

from sqlalchemy import String, UniqueConstraint
from sqlalchemy.orm import Mapped, mapped_column

from app.db.base import Base


class RevokedToken(Base):
    """Durable JWT revocation store (P2-1)."""

    __tablename__ = "revoked_tokens"
    __table_args__ = (UniqueConstraint("jti", name="uq_revoked_tokens_jti"),)

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    jti: Mapped[str] = mapped_column(String, index=True)
    revoked_at: Mapped[datetime] = mapped_column()
    expires_at: Mapped[datetime] = mapped_column()
