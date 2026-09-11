"""Audit logging (issue #129, follow-up to closed #86 whose audit AC was never implemented).

`record_audit` is the single write path onto the append-only `audit_logs` table; callers only
ever INSERT, never UPDATE/DELETE. It is called from the existing service-layer transition points
(login, promote/reject, deploy/rollback, retry training, infra config / credential_ref change) -
see the PR description for the exact file:line call sites. Never commits (services flush,
routers commit).
"""

from datetime import datetime, timezone

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.models.audit_log import AuditLog

# Free-text but not arbitrary: one word per critical action named in issue #129 / closed #86 so
# `action=` values stay grep-able and don't silently drift as new call sites are added.
LOGIN = "LOGIN"
PROMOTE = "PROMOTE"
REJECT = "REJECT"
DEPLOY = "DEPLOY"
ROLLBACK = "ROLLBACK"
RETRY_TRAINING = "RETRY_TRAINING"
INFRA_CONFIG_CREATE = "INFRA_CONFIG_CREATE"
INFRA_CONFIG_UPDATE = "INFRA_CONFIG_UPDATE"
INFRA_CONFIG_DELETE = "INFRA_CONFIG_DELETE"

RESULT_SUCCESS = "SUCCESS"
RESULT_FAILURE = "FAILURE"


def record_audit(
    db: Session,
    *,
    actor_id: int | None,
    action: str,
    resource_type: str,
    resource_id: str,
    before: dict | None = None,
    after: dict | None = None,
    result: str = RESULT_SUCCESS,
    reason: str | None = None,
) -> AuditLog:
    """Append one audit_logs row. Caller commits (services never commit, see CLAUDE.md)."""
    entry = AuditLog(
        actor_id=actor_id,
        action=action,
        resource_type=resource_type,
        resource_id=resource_id,
        before_json=before,
        after_json=after,
        result=result,
        reason=reason,
        created_at=datetime.now(timezone.utc),
    )
    db.add(entry)
    db.flush()
    return entry


def list_audit_logs(
    db: Session,
    *,
    limit: int = 20,
    offset: int = 0,
    actor_id: int | None = None,
    action: str | None = None,
    resource_type: str | None = None,
    resource_id: str | None = None,
    date_from: datetime | None = None,
    date_to: datetime | None = None,
) -> tuple[list[AuditLog], int]:
    """Filtered, paginated read of the audit trail (issue #129 AC: filter by actor/action/
    resource/date range). Read-only - the endpoint above this is GET-only."""
    from sqlalchemy import func

    query = select(AuditLog)
    if actor_id is not None:
        query = query.where(AuditLog.actor_id == actor_id)
    if action is not None:
        query = query.where(AuditLog.action == action)
    if resource_type is not None:
        query = query.where(AuditLog.resource_type == resource_type)
    if resource_id is not None:
        query = query.where(AuditLog.resource_id == resource_id)
    if date_from is not None:
        query = query.where(AuditLog.created_at >= date_from)
    if date_to is not None:
        query = query.where(AuditLog.created_at <= date_to)

    total = db.scalar(select(func.count()).select_from(query.subquery()))
    rows = list(
        db.scalars(
            query.order_by(AuditLog.created_at.desc()).limit(limit).offset(offset)
        ).all()
    )
    return rows, total or 0
