from datetime import datetime

from pydantic import BaseModel, ConfigDict


class NotificationEntry(BaseModel):
    """A single in-app notification (issue #130): a job/deployment lifecycle event surfaced to
    the relevant user (job owner or admin)."""

    model_config = ConfigDict(from_attributes=True)

    id: int
    user_id: int
    type: str
    message: str
    resource_ref: str | None = None
    read_at: datetime | None = None
    created_at: datetime


class NotificationMarkReadResponse(BaseModel):
    """Response for POST .../notifications/{id}/read."""

    id: int
    read_at: datetime
