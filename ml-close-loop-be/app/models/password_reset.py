from datetime import datetime

from sqlalchemy import Boolean, ForeignKey, String
from sqlalchemy.orm import Mapped, mapped_column

from app.db.base import Base


class PasswordResetToken(Base):
    """Single-use, short-lived password-reset token (issue #177).

    `token` is the primary key (a `secrets.token_urlsafe(32)` value, unguessable), matching
    `IdempotencyKey`'s pattern of keying by the token itself rather than an autoincrement id.
    """

    __tablename__ = "password_reset_tokens"

    token: Mapped[str] = mapped_column(String, primary_key=True)
    user_id: Mapped[int] = mapped_column(ForeignKey("users.id"), index=True)
    used: Mapped[bool] = mapped_column(Boolean, default=False)
    created_at: Mapped[datetime] = mapped_column()
    expires_at: Mapped[datetime] = mapped_column(index=True)
