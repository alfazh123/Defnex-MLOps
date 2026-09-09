"""TrainingProvider abstraction (issue #74, PRD §9.1/§9.4).

Defines the domain contract for training execution backends. The domain must not
depend directly on any specific provider implementation — swap the provider, not
the caller.

The provider contract is:

  submit(db, training_run) -> external_job_id
  get_status(external_job_id) -> JobStatus
  cancel(external_job_id) -> None
  collect_result(external_job_id) -> staging_dir_path

`LocalSubprocessProvider` wraps the existing `UnslothTrainingRunner` (issue #38)
as the first concrete implementation — subprocess-based, local execution.
"""

from __future__ import annotations

import json
import subprocess
import tempfile
import threading
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

import structlog
from sqlalchemy.orm import Session

from app.config import settings
from app.models.training import TrainingRun

logger = structlog.get_logger(__name__)

_MAX_STDERR = 1000


@dataclass
class JobStatus:
    """Outcome of a get_status call on a submitted job."""

    status: str  # "RUNNING", "COMPLETED", "FAILED", "CANCELLED"
    error_message: str | None = None


class TrainingProvider(Protocol):
    """PRD §9.4 provider contract — the domain boundary for training execution.

    Implementations may be local (subprocess), remote (GPU VPS), or on-demand
    (Colab). The domain calls these methods; it never touches the runner directly.
    """

    def submit(self, db: Session, training_run: TrainingRun) -> str:
        """Submit a training job. Sets external_job_id on the run via db.flush().

        Returns the external_job_id assigned to this job.
        """
        ...

    def get_status(self, external_job_id: str) -> JobStatus:
        """Poll the status of a previously submitted job."""
        ...

    def cancel(self, external_job_id: str) -> None:
        """Request cancellation of a running job."""
        ...

    def collect_result(self, external_job_id: str) -> str:
        """Collect the result of a completed job. Returns staging directory path.

        Raises if the job is not completed or has failed.
        """
        ...


@dataclass
class _RunningJob:
    """Internal bookkeeping for a subprocess-based job."""

    training_run_id: str
    proc: subprocess.Popen | None = None
    staging_dir: str = ""
    completed: bool = False
    failed: bool = False
    error_message: str = ""
    result_dir: str = ""


class LocalSubprocessProvider:
    """Concrete TrainingProvider that spawns training in a local subprocess (PRD §9.1).

    Wraps the existing UnslothTrainingRunner's subprocess logic but exposed through
    the submit/get_status/cancel/collect_result contract so the domain can swap in
    GPU VPS or Colab providers later without changing the worker or API layer.
    """

    def __init__(
        self,
        *,
        python_executable: str | None = None,
        script_path: Path | str | None = None,
        timeout_seconds: int | None = None,
    ):
        self._python = python_executable or settings.training_python
        self._script = str(
            script_path if script_path is not None else settings.training_script_path
        )
        self._timeout = (
            timeout_seconds
            if timeout_seconds is not None
            else settings.training_timeout_seconds
        )
        self._jobs: dict[str, _RunningJob] = {}

    def submit(self, db: Session, training_run: TrainingRun) -> str:
        """Spawn the training subprocess and return its external_job_id (PRD §9.4).

        Populates `training_run.external_job_id` via flush so the caller can persist
        it with their commit.
        """
        external_job_id = f"job-{uuid.uuid4().hex[:8]}"
        staging = Path(tempfile.mkdtemp(prefix="defnex-training-"))

        cmd = [
            self._python,
            "-u",
            self._script,
            "--config",
            json.dumps(training_run.training_config),
            "--staging",
            str(staging),
        ]

        proc = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1,
        )

        job = _RunningJob(
            training_run_id=training_run.training_run_id,
            proc=proc,
            staging_dir=str(staging),
        )
        self._jobs[external_job_id] = job

        training_run.external_job_id = external_job_id
        db.flush()

        logger.info(
            "training_submitted",
            training_run_id=training_run.training_run_id,
            external_job_id=external_job_id,
            pid=proc.pid,
        )

        # Spawn a daemon thread that reads stdout/stderr and waits for completion.
        # This avoids blocking the caller (PRD §10.1 — no sync HTTP handler).
        thread = threading.Thread(
            target=self._wait_for_process, args=(external_job_id,), daemon=True
        )
        thread.start()

        return external_job_id

    def get_status(self, external_job_id: str) -> JobStatus:
        """Poll whether the subprocess has finished (PRD §9.4)."""
        job = self._jobs.get(external_job_id)
        if job is None:
            raise ValueError(f"Unknown job: {external_job_id}")
        if job.failed:
            return JobStatus(status="FAILED", error_message=job.error_message)
        if job.completed:
            return JobStatus(status="COMPLETED")
        return JobStatus(status="RUNNING")

    def cancel(self, external_job_id: str) -> None:
        """Kill the subprocess (PRD §9.4)."""
        job = self._jobs.get(external_job_id)
        if job is None:
            raise ValueError(f"Unknown job: {external_job_id}")
        if job.proc and job.proc.poll() is None:
            job.proc.kill()
            job.failed = True
            job.error_message = "Cancelled by operator"

    def collect_result(self, external_job_id: str) -> str:
        """Return the staging directory path for a completed job (PRD §9.4).

        Raises if the job is not completed or has failed.
        """
        job = self._jobs.get(external_job_id)
        if job is None:
            raise ValueError(f"Unknown job: {external_job_id}")
        if job.failed:
            raise RuntimeError(f"Job {external_job_id} failed: {job.error_message}")
        if not job.completed:
            raise RuntimeError(f"Job {external_job_id} is not yet completed")
        return job.result_dir or job.staging_dir

    def _wait_for_process(self, external_job_id: str) -> None:
        """Background thread: drain stderr, wait for exit, set job outcome."""
        job = self._jobs.get(external_job_id)
        if job is None or job.proc is None:
            return

        stderr_lines: list[str] = []

        def _drain_stderr():
            if job.proc and job.proc.stderr:
                for line in job.proc.stderr:
                    stderr_lines.append(line)

        stderr_thread = threading.Thread(target=_drain_stderr, daemon=True)
        stderr_thread.start()

        timed_out = [False]

        def _enforce_timeout():
            timed_out[0] = True
            if job.proc and job.proc.poll() is None:
                try:
                    job.proc.kill()
                except OSError:
                    pass
            logger.warning(
                "training_timeout_kill",
                training_run_id=job.training_run_id,
                timeout_seconds=self._timeout,
            )

        watchdog = None
        if self._timeout and self._timeout > 0:
            watchdog = threading.Timer(self._timeout, _enforce_timeout)
            watchdog.daemon = True
            watchdog.start()

        try:
            if job.proc:
                job.proc.wait()
        finally:
            if watchdog:
                watchdog.cancel()

        stderr_thread.join(timeout=5)

        if timed_out[0]:
            job.failed = True
            job.error_message = f"Training exceeded {self._timeout}s and was killed"
        elif job.proc and job.proc.returncode != 0:
            stderr = "".join(stderr_lines)[-_MAX_STDERR:]
            job.failed = True
            job.error_message = (
                f"Training exited with code {job.proc.returncode}: {stderr.strip()[-_MAX_STDERR:]}"
                if stderr
                else f"Training exited with code {job.proc.returncode}"
            )
        else:
            job.completed = True
            job.result_dir = job.staging_dir

        logger.info(
            "training_process_finished",
            training_run_id=job.training_run_id,
            external_job_id=external_job_id,
            success=job.completed,
        )
