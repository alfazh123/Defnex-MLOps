from datetime import datetime

from sqlalchemy import String, UniqueConstraint
from sqlalchemy.orm import Mapped, mapped_column

from app.db.base import Base


class RevokedRefreshToken(Base):
    """Durable refresh-token revocation store (issue #125).

    Access tokens are short-lived (settings.jwt_expire_minutes) and are never
    checked against this table - natural expiry is fast enough. Only refresh
    tokens are persisted here, and only two kinds of rows exist:

    - one row per rotated/used refresh token, keyed by its own `jti` - used to
      detect reuse of an already-rotated token (theft indicator);
    - one row per compromised token *family*, keyed by the synthetic jti
      ``f"family-revoked:{family_id}"`` - written when reuse is detected (or
      on logout) to invalidate every token that shares that family, including
      ones not individually recorded here.
    """

    __tablename__ = "revoked_refresh_tokens"
    __table_args__ = (UniqueConstraint("jti", name="uq_revoked_refresh_tokens_jti"),)

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    jti: Mapped[str] = mapped_column(String, index=True)
    family_id: Mapped[str] = mapped_column(String, index=True)
    revoked_at: Mapped[datetime] = mapped_column()
    expires_at: Mapped[datetime] = mapped_column()
