import threading
import time
from typing import Protocol

import structlog
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


class ProviderRunnerAdapter:
    """Adapt a TrainingProvider (issue #74, PRD §9.4) to the TrainingRunner interface.

    The provider's submit/get_status/collect_result cycle is wrapped into a single
    blocking `run()` call so the existing worker loop works unchanged. For
    LocalSubprocessProvider this is effectively a direct subprocess call; for remote
    providers (GPU VPS, Colab) the adapter would poll get_status in a loop.
    """

    def __init__(self, provider):
        self._provider = provider

    def run(self, db: Session, training_run: TrainingRun) -> str:
        external_job_id = self._provider.submit(db, training_run)
        # For LocalSubprocessProvider, submit() spawns a daemon thread and returns
        # immediately. Poll until the job completes or fails.
        import time as _time

        while True:
            status = self._provider.get_status(external_job_id)
            if status.status == "COMPLETED":
                return self._provider.collect_result(external_job_id)
            if status.status in ("FAILED", "CANCELLED"):
                raise RuntimeError(
                    status.error_message or f"Job {external_job_id} {status.status}"
                )
            _time.sleep(0.1)


def _heartbeat_loop(
    stop: threading.Event,
    db: Session,
    run_id: str,
    interval: float,
) -> None:
    """Persist a heartbeat for `run_id` every `interval` seconds until stopped (issue #60).

    Runs on a daemon thread SHARING the worker's `db` session (the claim is not committed
    until the runner/loop commits, so a separate connection could not observe RUNNING).
    It only issues an atomic `touch_heartbeat` UPDATE with `synchronize_session=False` and
    never begins/commits a transaction, so the runner's own commits flush the heartbeat
    along with progress. Caller calls `stop.set()` after the runner returns to exit.
    """

    next_tick = time.monotonic() + interval
    while not stop.is_set():
        now = time.monotonic()
        if now < next_tick:
            stop.wait(next_tick - now)
            continue
        next_tick = time.monotonic() + interval
        try:
            training_service.touch_heartbeat(db, run_id)
        except Exception:
            # Heartbeats are best-effort liveness; a failed write must not fail the job.
            logger.warning("heartbeat_failed", training_run_id=run_id, exc_info=True)


def process_next_job(
    db: Session,
    runner: TrainingRunner,
    *,
    lock_file: str | None = None,
    lock_timeout: float | None = None,
    coordinator: ServingCoordinator | None = None,
    heartbeat_interval: float | None = None,
) -> TrainingRun | None:
    """One worker iteration (PRD §10 steps 1-8): pick the next claimable run (PENDING or
    STALE, `priority DESC, created_at ASC` -- issue #135 fair-use queue, oldest-first
    within a priority tier), claim it atomically, run it under the exclusive GPU lock,
    persist the outcome.

    Returns the processed run, or None if the queue is empty or the claim was lost
    to a concurrent worker. A lock-queue timeout does not fail or lose the run: the
    worker simply skips this poll and the run stays PENDING for the next iteration.

    Heartbeat / stale detection (issue #60): before picking a run, the detector
    `mark_stale_runs` reclaims any RUNNING run whose heartbeat has lapsed (a crashed
    worker's orphaned run), so it is never stuck RUNNING forever and gets re-claimable.
    While the runner executes, a daemon heartbeat thread persists `heartbeat_at` every
    `heartbeat_interval` seconds so the run stays alive (PRD §10.4).

    Serving orchestration (issue #39): the `coordinator` (default `make_coordinator()`
    from settings — a no-op when `SERVING_CONTROL=mock`) wraps the training block so
    serving is stopped and VRAM verified free *before* training starts, and restarted
    after — all inside the same `gpu_lock` from #33, never a second coordination
    sequence. If serving cannot be stopped or VRAM never frees up, the run is not
    started; it stays PENDING (skipped this poll) with the reason logged — a busy GPU
    never fails or loses a run, matching the lock-timeout behavior.
    """

    training_service.mark_stale_runs(db)
    training_run = training_service.next_claimable_run(db)
    if training_run is None:
        return None

    coordinator = coordinator or make_coordinator()
    from app.telemetry import _active_training_runs

    try:
        with gpu_lock(
            lock_file or settings.gpu_lock_file,
            lock_timeout if lock_timeout is not None else settings.gpu_lock_timeout,
        ):
            try:
                with coordinator.cycle():
                    if not training_service.claim_training_run(db, training_run):
                        return None
                    if _active_training_runs is not None:
                        _active_training_runs.add(1)
                    interval = (
                        heartbeat_interval
                        if heartbeat_interval is not None
                        else settings.heartbeat_interval_seconds
                    )
                    stop = threading.Event()
                    # ponytail: heartbeat daemon thread shares the worker's Session (single
                    # atomic UPDATE, no transaction begin/commit); switch to a separate
                    # session only once the claim is committed independently or Celery owns
                    # the job lifecycle.
                    heartbeat = threading.Thread(
                        target=_heartbeat_loop,
                        args=(stop, db, training_run.training_run_id, interval),
                        daemon=True,
                    )
                    heartbeat.start()
                    run_start = time.monotonic()
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
                        # ModelVersion, so the loop never closes. It also finalizes the
                        # staged training output into its immutable per-version artifact
                        # (issue #38).
                        model_service.register_model_version(
                            db, training_run, staging_dir=staging_dir
                        )
                    finally:
                        if _active_training_runs is not None:
                            _active_training_runs.add(-1)
                        from app.telemetry import _training_duration

                        if _training_duration is not None:
                            _training_duration.record(time.monotonic() - run_start)
                        stop.set()
                        heartbeat.join(timeout=2)
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

    Accepts either a TrainingRunner (legacy) or a TrainingProvider (issue #74).
    If a TrainingProvider is passed, it is automatically wrapped with ProviderRunnerAdapter.

    A SIGTERM that lands mid-cycle (`ServingInterrupted`) has already restarted serving
    in the cycle's `finally`; it now stops the loop so the worker exits cleanly instead
    of spinning after being asked to shut down.
    """

    # Auto-adapt TrainingProvider to TrainingRunner if needed (issue #74).
    if hasattr(runner, "submit") and not hasattr(runner, "run"):
        runner = ProviderRunnerAdapter(runner)

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
    from app.providers.training_provider import LocalSubprocessProvider

    # Prefer the provider abstraction (issue #74) over the raw runner.
    # LocalSubprocessProvider wraps UnslothTrainingRunner with the submit/get_status/
    # collect_result contract so future GPU VPS / Colab providers can be swapped in.
    run_forever(LocalSubprocessProvider())


def get_provider_for_resource(resource):
    """Return the appropriate TrainingProvider based on ComputeResource.provider_type.

    provider_type "local" → LocalSubprocessProvider (local subprocess)
    provider_type "gpu_vps" → GPUVPSProvider (SSH remote execution, issue #76)
    provider_type "colab" → ColabProvider (ephemeral on-demand, issue #77)
    """
    from app.providers.training_provider import (
        ColabProvider,
        GPUVPSProvider,
        LocalSubprocessProvider,
    )

    if resource.provider_type == "gpu_vps":
        return GPUVPSProvider(resource)
    if resource.provider_type == "colab":
        return ColabProvider()
    return LocalSubprocessProvider()
