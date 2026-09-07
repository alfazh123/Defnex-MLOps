import time
from typing import Protocol

import structlog
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.config import settings
from app.db.session import SessionLocal
from app.models.training import TrainingRun
from app.services import model_service, training_service
from app.workers.gpu_lock import gpu_lock
from app.workers.gpu_orchestrator import (
    ServingCoordinator,
    ServingInterrupted,
    ServingStopFailed,
    VRAMNotFree,
    make_coordinator,
)

logger = structlog.get_logger(__name__)


class TrainingRunner(Protocol):
    """Executes a training run's actual work. Swappable (mock now, real Unsloth runner later)
    without changing the worker or the API contract (PRD §9/§10, US-009's acceptance criteria).
    """

    def run(self, training_run: TrainingRun) -> str:
        """Run training for `training_run` and return the resulting artifact_uri, or raise."""
        ...


def process_next_job(
    db: Session,
    runner: TrainingRunner,
    *,
    lock_file: str | None = None,
    lock_timeout: float | None = None,
    coordinator: ServingCoordinator | None = None,
) -> TrainingRun | None:
    """One worker iteration (PRD §10 steps 1-8): pick the oldest PENDING run, claim it
    atomically, run it under the exclusive GPU lock, persist the outcome.

    Returns the processed run, or None if the queue is empty or the claim was lost
    to a concurrent worker. A lock-queue timeout does not fail or lose the run: the
    worker simply skips this poll and the run stays PENDING for the next iteration.

    Serving orchestration (issue #39): the `coordinator` (default `make_coordinator()`
    from settings — a no-op when `SERVING_CONTROL=mock`) wraps the training block so
    serving is stopped and VRAM verified free *before* training starts, and restarted
    after — all inside the same `gpu_lock` from #33, never a second coordination
    sequence. If serving cannot be stopped or VRAM never frees up, the run is not
    started; it stays PENDING (skipped this poll) with the reason logged — a busy GPU
    never fails or loses a run, matching the lock-timeout behavior.
    """

    training_run = db.scalar(
        select(TrainingRun)
        .where(TrainingRun.status == "PENDING")
        .order_by(TrainingRun.created_at)
    )
    if training_run is None:
        return None

    coordinator = coordinator or make_coordinator()
    try:
        with gpu_lock(
            lock_file or settings.gpu_lock_file,
            lock_timeout if lock_timeout is not None else settings.gpu_lock_timeout,
        ):
            try:
                with coordinator.cycle():
                    if not training_service.claim_training_run(db, training_run):
                        return None
                    try:
                        artifact_uri = runner.run(training_run)
                    except Exception as exc:
                        training_service.fail_training_run(
                            db, training_run, error_message=str(exc)
                        )
                    else:
                        training_service.complete_training_run(
                            db, training_run, artifact_uri=artifact_uri
                        )
                        # The internal Register call openapi.yaml documents as running on
                        # COMPLETED — without it nothing in a running system ever creates a
                        # ModelVersion, so the loop never closes.
                        model_service.register_model_version(db, training_run)
            except (VRAMNotFree, ServingStopFailed) as exc:
                # Serving could not be made safe for training: the run is not started,
                # stays PENDING, and the poll is skipped so the next iteration retries it.
                logger.warning("serving_preflight_failed", reason=str(exc))
                return None
    except TimeoutError:
        return None
    return training_run


def run_forever(
    runner: TrainingRunner,
    poll_interval: float = 5.0,
    coordinator: ServingCoordinator | None = None,
) -> None:
    """Poll for queued training runs, independent of the FastAPI process (PRD §10).
    Run standalone via `python -m app.workers.training_worker`.

    A SIGTERM that lands mid-cycle (`ServingInterrupted`) has already restarted serving
    in the cycle's `finally`; it now stops the loop so the worker exits cleanly instead
    of spinning after being asked to shut down.
    """

    while True:
        db = SessionLocal()
        try:
            job = process_next_job(db, runner, coordinator=coordinator)
            db.commit()
        except ServingInterrupted:
            # Serving was already restarted in the cycle's `finally` (see gpu_orchestrator);
            # stop the loop so the worker exits cleanly after receiving SIGTERM. The
            # interrupted run's uncommitted transaction is rolled back by db.close().
            return
        finally:
            db.close()
        if job is None:
            time.sleep(poll_interval)


if __name__ == "__main__":
    from app.workers.mock_runner import MockTrainingRunner

    # Mock runner until the VM/Unsloth environment exists (PRD §20); swap the runner here only.
    run_forever(MockTrainingRunner())
