from datetime import datetime

from sqlalchemy import ForeignKey, Integer, String
from sqlalchemy.orm import Mapped, mapped_column

from app.db.base import Base


class Notification(Base):
    """In-app notification for a job/deployment lifecycle event (issue #130).

    One row per (user, event) - e.g. the job owner gets one row when their training run
    completes/fails, admins get one row each when a production deploy fails or a candidate needs
    an approval decision. Read via `GET /api/v1/notifications`; `read_at` is set by the
    mark-as-read action and stays `None` until then.
    """

    __tablename__ = "notifications"

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    user_id: Mapped[int] = mapped_column(Integer, ForeignKey("users.id"), index=True)
    type: Mapped[str] = mapped_column(String, index=True)
    message: Mapped[str] = mapped_column(String)
    resource_ref: Mapped[str | None] = mapped_column(String, nullable=True)
    read_at: Mapped[datetime | None] = mapped_column(nullable=True)
    created_at: Mapped[datetime] = mapped_column(index=True)
