from datetime import datetime

from fastapi import APIRouter, Depends, Query
from sqlalchemy.orm import Session

from app.api.deps import PaginationParams, get_pagination, require_admin
from app.db.session import get_db
from app.models.user import User
from app.schemas.audit import AuditLogEntry
from app.schemas.common import PaginatedResponse
from app.services import audit_service

router = APIRouter(tags=["Audit"])


@router.get("/audit-logs", response_model=PaginatedResponse[AuditLogEntry])
def list_audit_logs(
    db: Session = Depends(get_db),
    _admin: User = Depends(require_admin),
    pg: PaginationParams = Depends(get_pagination),
    actor_id: int | None = Query(None),
    action: str | None = Query(None),
    resource_type: str | None = Query(None),
    resource_id: str | None = Query(None),
    date_from: datetime | None = Query(None),
    date_to: datetime | None = Query(None),
) -> PaginatedResponse[AuditLogEntry]:
    """Read-only audit trail (issue #129), admin only. Filterable by actor, action, resource
    (type and/or id), and a created_at date range; reuses `get_pagination` for page/size."""
    rows, total = audit_service.list_audit_logs(
        db,
        limit=pg.limit,
        offset=pg.offset,
        actor_id=actor_id,
        action=action,
        resource_type=resource_type,
        resource_id=resource_id,
        date_from=date_from,
        date_to=date_to,
    )
    return PaginatedResponse(
        items=[AuditLogEntry.model_validate(r) for r in rows],
        total=total,
        page=pg.page,
        size=pg.size,
        pages=PaginationParams.pages_from(total, pg.size),
    )
