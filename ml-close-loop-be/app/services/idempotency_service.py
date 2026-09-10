"""Durable `X-Idempotency-Key` cache (issue #124, follow-up #81).

`deployment_service.py` and (the commit endpoint of) `intake_validate.py` each used to keep an
in-process `dict[str, tuple[dict, float]]` keyed by the caller-supplied idempotency key. That
cache lived in one Python process's memory: with >1 API worker process, or across a restart, a
retried request with the same `X-Idempotency-Key` would no longer find the earlier response and
would silently re-run the operation (e.g. double-deploy, double dataset-version commit).

This module replaces the dict with reads/writes against the `idempotency_keys` Postgres table
(`app.models.idempotency.IdempotencyKey`) so the guarantee is backed by durable storage shared by
every worker process. The PRD v2 target stack adds Redis, but Redis isn't wired yet (still
broker-free) - a Postgres table is the dependency that's actually in place today. Every caller
goes through `get_cached_response` / `store_response` / `cleanup_expired`, so swapping the
backing store later (e.g. to Redis) only means changing this module, not its callers.

Per the repo's router/service split, this module never calls `db.commit()` - it only
`db.flush()`es, so the caller (a router) keeps owning the transaction boundary.
"""

from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone
from typing import Any, NamedTuple

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.models.idempotency import IdempotencyKey

DEFAULT_TTL_SECONDS = 3600  # 1 hour - matches the TTL the old in-process caches used.


class CachedResponse(NamedTuple):
    status: int
    body: Any


def _utc_now() -> datetime:
    """Naive UTC "now", used for both storage and comparison.

    SQLite (the test/dev engine, see tests/conftest.py) has no native timezone-aware DATETIME
    type; a tz-aware value written through the plain `DateTime` column type used here round-trips
    unreliably on SQLite. Staying naive-but-always-UTC end to end sidesteps that entirely, and is
    still correct on Postgres (the authoritative target per CLAUDE.md), which is fine storing
    naive UTC timestamps as long as nothing else writes local time into the same column.
    """
    return datetime.now(timezone.utc).replace(tzinfo=None)


def get_cached_response(db: Session, key: str | None) -> CachedResponse | None:
    """Look up the durable record for `key`.

    Returns `None` when `key` is `None` (idempotency is opt-in, per the existing header
    contract), when no row exists, or when the row's TTL has passed. An expired row is deleted
    (flushed, not committed - the router still owns the commit) as a side effect of the lookup,
    so the same key can be reused for a brand-new response right away instead of permanently
    colliding with the PK of a dead row.
    """
    if key is None:
        return None
    row = db.get(IdempotencyKey, key)
    if row is None:
        return None
    if row.expires_at < _utc_now():
        db.delete(row)
        db.flush()
        return None
    return CachedResponse(
        status=row.response_status, body=json.loads(row.response_body_json)
    )


def store_response(
    db: Session,
    key: str | None,
    *,
    endpoint: str,
    status: int,
    body: Any,
    ttl_seconds: int = DEFAULT_TTL_SECONDS,
) -> None:
    """Persist the response for `key` so a retried request carrying the same
    `X-Idempotency-Key` replays it instead of re-running the operation. No-op when `key` is
    `None`."""
    if key is None:
        return
    now = _utc_now()
    db.add(
        IdempotencyKey(
            key=key,
            endpoint=endpoint,
            response_status=status,
            response_body_json=json.dumps(body),
            created_at=now,
            expires_at=now + timedelta(seconds=ttl_seconds),
        )
    )
    db.flush()


def cleanup_expired(db: Session) -> int:
    """Delete every row whose TTL has passed and return how many were removed.

    Callable cleanup (issue #124 AC) rather than a running scheduled job - no scheduler is wired
    in this codebase yet, and wiring one is out of scope for this issue. A future Celery beat
    task (once Celery is actually in, per CLAUDE.md) can call this directly.
    """
    rows = list(
        db.scalars(select(IdempotencyKey).where(IdempotencyKey.expires_at < _utc_now()))
    )
    for row in rows:
        db.delete(row)
    db.flush()
    return len(rows)
