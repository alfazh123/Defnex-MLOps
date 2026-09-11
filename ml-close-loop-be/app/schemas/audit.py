from datetime import datetime

from pydantic import BaseModel, ConfigDict


class AuditLogEntry(BaseModel):
    """A single audit_logs row (issue #129): who did what, to which resource, when, with what
    before/after state, whether it succeeded, and why."""

    model_config = ConfigDict(from_attributes=True)

    id: int
    actor_id: int | None = None
    action: str
    resource_type: str
    resource_id: str
    before_json: dict | None = None
    after_json: dict | None = None
    result: str
    reason: str | None = None
    created_at: datetime
