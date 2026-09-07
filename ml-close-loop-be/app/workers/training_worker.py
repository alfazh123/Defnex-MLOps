import time
from typing import Protocol

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.config import settings
from app.db.session import SessionLocal
from app.models.training import TrainingRun
from app.services import model_service, training_service
from app.workers.gpu_lock import gpu_lock


class TrainingRunner(Protocol):
    """Executes a training run's actual work. Swappable (mock in tests, real Unsloth runner in
    production — issue #38) without changing the worker or the API contract (PRD §9/§10,
    US-009's acceptance criteria).

    `run` returns a staging directory containing the trained output; the worker registers it
    into an immutable per-version location.
    """

    def run(self, db: Session, training_run: TrainingRun) -> str:
        """Run training for `training_run`, streaming progress into `db`, and return a staging
        directory with the trained output, or raise."""
        ...


def process_next_job(
    db: Session,
    runner: TrainingRunner,
    *,
    lock_file: str | None = None,
    lock_timeout: float | None = None,
) -> TrainingRun | None:
    """One worker iteration (PRD §10 steps 1-8): pick the oldest PENDING run, claim it
    atomically, run it under the exclusive GPU lock, persist the outcome.

    Returns the processed run, or None if the queue is empty or the claim was lost
    to a concurrent worker. A lock-queue timeout does not fail or lose the run: the
    worker simply skips this poll and the run stays PENDING for the next iteration.
    """

    training_run = db.scalar(
        select(TrainingRun)
        .where(TrainingRun.status == "PENDING")
        .order_by(TrainingRun.created_at)
    )
    if training_run is None:
        return None

    try:
        with gpu_lock(
            lock_file or settings.gpu_lock_file,
            lock_timeout if lock_timeout is not None else settings.gpu_lock_timeout,
        ):
            if not training_service.claim_training_run(db, training_run):
                return None
            try:
                staging_dir = runner.run(db, training_run)
            except Exception as exc:
                training_service.fail_training_run(
                    db, training_run, error_message=str(exc)
                )
            else:
                training_service.complete_training_run(
                    db, training_run, artifact_uri=staging_dir
                )
                # The internal Register call openapi.yaml documents as running on
                # COMPLETED — without it nothing in a running system ever creates a
                # ModelVersion, so the loop never closes. It also finalizes the staged
                # training output into its immutable per-version artifact (issue #38).
                model_service.register_model_version(
                    db, training_run, staging_dir=staging_dir
                )
    except TimeoutError:
        return None
    return training_run


def run_forever(runner: TrainingRunner, poll_interval: float = 5.0) -> None:
    """Poll for queued training runs, independent of the FastAPI process (PRD §10).
    Run standalone via `python -m app.workers.training_worker`.
    """

    while True:
        db = SessionLocal()
        try:
            job = process_next_job(db, runner)
            db.commit()
        finally:
            db.close()
        if job is None:
            time.sleep(poll_interval)


if __name__ == "__main__":
    from app.workers.unsloth_runner import UnslothTrainingRunner

    # Real runner: spawns Unsloth training in a separate venv (issue #38). Mock is test-only.
    run_forever(UnslothTrainingRunner())
