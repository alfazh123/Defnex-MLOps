"""Scheduled drift-detection job (issue #136): re-runs the golden-set evaluation for every
`DEPLOYED` model version and records the result via `drift_check_service`.

Unlike `training_worker.py`/`evaluation_worker.py` this is not a `run_forever` poll loop - a
weekly-ish quality re-check does not need sub-second latency, and CLAUDE.md explicitly says not
to wire this project to Celery beat before a `JobManager` abstraction exists. `run_once` does one
full pass over all `DEPLOYED` versions and returns; an external scheduler (cron, a CI scheduled
workflow, etc.) is expected to invoke `python -m app.workers.drift_check_worker` on whatever
cadence is chosen - the job itself is idempotent-per-call (each call just appends new
`ModelDriftCheck` rows), so how often it's invoked is an ops decision, not a code one.
"""

from __future__ import annotations

import structlog
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.db.session import SessionLocal
from app.models.drift_check import ModelDriftCheck
from app.models.model import ModelVersion
from app.services.drift_check_service import run_drift_check_for_version
from app.services.serving import ServingBackend, get_serving_backend

logger = structlog.get_logger(__name__)


def run_once(
    db: Session, backend: ServingBackend | None = None
) -> list[ModelDriftCheck]:
    """Re-check every currently `DEPLOYED` model version once. Returns the `ModelDriftCheck`
    rows created (versions skipped for lack of an eval set are omitted, not raised)."""
    backend = backend or get_serving_backend()
    deployed_versions = db.scalars(
        select(ModelVersion)
        .where(ModelVersion.status == "DEPLOYED")
        .order_by(ModelVersion.id)
    ).all()

    results: list[ModelDriftCheck] = []
    for model_version in deployed_versions:
        result = run_drift_check_for_version(db, model_version, backend)
        if result is not None:
            results.append(result)
    return results


def main() -> None:
    db = SessionLocal()
    try:
        checked = run_once(db)
        db.commit()
        logger.info("drift_check_cycle_complete", checked=len(checked))
    finally:
        db.close()


if __name__ == "__main__":
    main()
