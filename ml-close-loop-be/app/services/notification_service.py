"""In-app notifications (issue #130): job/deployment lifecycle events surfaced to the relevant
user - the job owner (resolved from the free-text `triggered_by`/`created_by` fields already on
TrainingRun/DatasetVersion) or, for events with no single owner (a production deploy failure, an
approval decision waiting on a human), every admin.

`record_notification` is the single write path; callers only ever INSERT. Never commits
(services flush, routers commit) - the caller's existing commit persists it, matching how
`app.services.alerting` is already wired into the same call sites for the out-of-band webhook.
"""

from datetime import datetime, timezone

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.models.notification import Notification
from app.models.user import User

TRAINING_COMPLETED = "TRAINING_COMPLETED"
TRAINING_FAILED = "TRAINING_FAILED"
VALIDATION_FAILED = "VALIDATION_FAILED"
PRODUCTION_DEPLOYMENT_FAILED = "PRODUCTION_DEPLOYMENT_FAILED"
APPROVAL_REQUIRED = "APPROVAL_REQUIRED"


def record_notification(
    db: Session,
    *,
    user_id: int,
    type: str,
    message: str,
    resource_ref: str | None = None,
) -> Notification:
    """Append one notification row for `user_id`. Caller commits."""
    notification = Notification(
        user_id=user_id,
        type=type,
        message=message,
        resource_ref=resource_ref,
        created_at=datetime.now(timezone.utc),
    )
    db.add(notification)
    db.flush()
    return notification


def notify_user_by_username(
    db: Session,
    *,
    username: str | None,
    type: str,
    message: str,
    resource_ref: str | None = None,
) -> Notification | None:
    """Resolve `username` (a free-text field like TrainingRun.triggered_by or
    DatasetVersion.created_by - not a guaranteed FK) to a real user and notify them.

    Returns None without writing anything when there's no such user (unknown/blank
    triggered_by is common in tests and for runs triggered outside the API) - a missing
    notification target is not itself an error worth raising over.
    """
    if not username:
        return None
    user = db.scalar(select(User).where(User.username == username))
    if user is None:
        return None
    return record_notification(
        db, user_id=user.id, type=type, message=message, resource_ref=resource_ref
    )


def notify_admins(
    db: Session,
    *,
    type: str,
    message: str,
    resource_ref: str | None = None,
) -> list[Notification]:
    """Notify every admin user - used for events with no single job owner (production deploy
    failures, a candidate reaching EVALUATED and needing a promote/reject decision)."""
    admin_ids = db.scalars(select(User.id).where(User.role == "admin")).all()
    return [
        record_notification(
            db, user_id=admin_id, type=type, message=message, resource_ref=resource_ref
        )
        for admin_id in admin_ids
    ]


def list_notifications(
    db: Session,
    *,
    user_id: int,
    limit: int = 20,
    offset: int = 0,
    unread_only: bool = False,
) -> tuple[list[Notification], int]:
    """A user's own notifications, newest first (issue #130 AC: list + pagination)."""
    from sqlalchemy import func

    query = select(Notification).where(Notification.user_id == user_id)
    if unread_only:
        query = query.where(Notification.read_at.is_(None))

    total = db.scalar(select(func.count()).select_from(query.subquery()))
    rows = list(
        db.scalars(
            query.order_by(Notification.created_at.desc()).limit(limit).offset(offset)
        ).all()
    )
    return rows, total or 0


def mark_read(db: Session, notification: Notification) -> Notification:
    """Mark one notification read (idempotent - re-marking an already-read row just keeps its
    original `read_at`). Caller commits."""
    if notification.read_at is None:
        notification.read_at = datetime.now(timezone.utc)
        db.flush()
    return notification
