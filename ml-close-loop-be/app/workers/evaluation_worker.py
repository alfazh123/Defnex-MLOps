"""Async evaluation worker (issue #128): polls for model versions with `evaluation_requested`
set, the same claim-then-run pattern `training_worker.py` uses for `TrainingRun` (PENDING/STALE
-> RUNNING via a compare-and-set UPDATE). Computes evaluation server-side through
`evaluation_engine.compute_evaluation_update` (real inference via the existing `ServingBackend`
against the stored golden/eval set, issue #43) instead of trusting caller-supplied numbers, then
applies the result through the existing `model_service.submit_evaluation` merge + auto-transition.

Run standalone via `python -m app.workers.evaluation_worker`, same shape as
`app/workers/training_worker.py`.
"""

from __future__ import annotations

import time

import structlog
from sqlalchemy import select, update
from sqlalchemy.orm import Session

from app.db.session import SessionLocal
from app.models.model import ModelVersion
from app.services import model_service
from app.services.evaluation_engine import compute_evaluation_update
from app.services.serving import ServingBackend, get_serving_backend

logger = structlog.get_logger(__name__)


def claim_pending_evaluation(db: Session, model_version: ModelVersion) -> bool:
    """Atomically claim `model_version` for evaluation (compare-and-set `evaluation_requested`
    True -> False), mirroring `training_service.claim_training_run`'s CAS so two worker
    processes racing on the same row cannot both run the (non-cheap, ServingBackend-calling)
    evaluation. Works on SQLite and PostgreSQL."""

    result = db.execute(
        update(ModelVersion)
        .where(
            ModelVersion.id == model_version.id,
            ModelVersion.evaluation_requested.is_(True),
        )
        .values(evaluation_requested=False)
        .execution_options(synchronize_session=False)
    )
    if result.rowcount != 1:
        return False
    model_version.evaluation_requested = False
    return True


def process_next_evaluation(
    db: Session, backend: ServingBackend | None = None
) -> ModelVersion | None:
    """One worker iteration: pick the oldest model version with `evaluation_requested`, claim
    it, compute the evaluation, and apply it. Returns the processed version, or None if the
    queue is empty or the claim was lost to a concurrent worker.

    A version whose evaluation cannot be computed (no eval set reference, eval set not found,
    adapter failed to load - all raise `ValueError` from `compute_evaluation_update`) is logged
    and left as-is: `evaluation_requested` was already consumed by the claim, so it stays
    unevaluated until the caller re-triggers it (same "skip this cycle, never crash the worker
    loop" posture as `training_worker.process_next_job`'s lock-timeout handling)."""

    model_version = db.scalar(
        select(ModelVersion)
        .where(ModelVersion.evaluation_requested.is_(True))
        .order_by(ModelVersion.id)
    )
    if model_version is None:
        return None
    if not claim_pending_evaluation(db, model_version):
        return None

    backend = backend or get_serving_backend()
    try:
        update_request = compute_evaluation_update(db, model_version, backend)
    except ValueError as exc:
        logger.warning(
            "evaluation_skipped",
            model_id=model_version.model_id,
            version=model_version.version,
            reason=str(exc),
        )
        return model_version

    model_service.submit_evaluation(db, model_version, update_request)
    return model_version


def run_forever(poll_interval: float = 5.0) -> None:
    """Poll for queued evaluations, independent of the FastAPI process - same shape as
    `training_worker.run_forever`."""

    while True:
        db = SessionLocal()
        try:
            job = process_next_evaluation(db)
            db.commit()
        finally:
            db.close()
        if job is None:
            time.sleep(poll_interval)


if __name__ == "__main__":
    run_forever()
