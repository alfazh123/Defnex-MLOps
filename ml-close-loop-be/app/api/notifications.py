from fastapi import APIRouter, Depends, Query
from sqlalchemy.orm import Session

from app.api.deps import PaginationParams, get_current_user, get_pagination
from app.api.errors import APIError
from app.db.session import get_db
from app.models.notification import Notification
from app.models.user import User
from app.schemas.common import ErrorResponse, PaginatedResponse
from app.schemas.notification import NotificationEntry, NotificationMarkReadResponse
from app.services import notification_service

router = APIRouter(tags=["Notifications"])


@router.get("/notifications", response_model=PaginatedResponse[NotificationEntry])
def list_notifications(
    db: Session = Depends(get_db),
    user: User = Depends(get_current_user),
    pg: PaginationParams = Depends(get_pagination),
    unread_only: bool = Query(False),
) -> PaginatedResponse[NotificationEntry]:
    """List the current user's own notifications, newest first (issue #130). Reuses
    `get_pagination` for page/size."""
    rows, total = notification_service.list_notifications(
        db,
        user_id=user.id,
        limit=pg.limit,
        offset=pg.offset,
        unread_only=unread_only,
    )
    return PaginatedResponse(
        items=[NotificationEntry.model_validate(r) for r in rows],
        total=total,
        page=pg.page,
        size=pg.size,
        pages=PaginationParams.pages_from(total, pg.size),
    )


@router.post(
    "/notifications/{notification_id}/read",
    response_model=NotificationMarkReadResponse,
    responses={404: {"model": ErrorResponse}},
)
def mark_notification_read(
    notification_id: int,
    db: Session = Depends(get_db),
    user: User = Depends(get_current_user),
) -> NotificationMarkReadResponse:
    """Mark one of the current user's own notifications read (issue #130)."""
    notification = db.get(Notification, notification_id)
    if notification is None or notification.user_id != user.id:
        # Same 404 whether the row doesn't exist or belongs to someone else - never confirm
        # another user's notification id exists.
        raise APIError(
            404, "NOTIFICATION_NOT_FOUND", f"notification {notification_id} not found"
        )
    notification_service.mark_read(db, notification)
    db.commit()
    return NotificationMarkReadResponse(
        id=notification.id, read_at=notification.read_at
    )
