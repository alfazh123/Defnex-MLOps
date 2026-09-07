"""Real Unsloth training runner (issue #38).

The worker's default runner. Executes training by spawning a standalone Unsloth training
script (`app/training/run_training.py`) in a SEPARATE venv/subprocess — the app/serving venv
never imports Unsloth. The script streams newline-delimited JSON progress to stdout; this
runner parses it and persists the four progress columns to the TrainingRun (committing per
event so `GET /training-runs/{id}` observes live non-NULL values while still RUNNING), then
returns the staging output directory for the worker to register immutably.
"""

import json
import subprocess
import tempfile
import threading
from pathlib import Path

import structlog

from app.config import settings
from app.models.training import TrainingRun
from app.services import training_service

logger = structlog.get_logger(__name__)

_MAX_STDERR = 1000


class UnslothTrainingRunner:
    """Run a training subprocess, stream its progress into the DB, and return the staging
    directory containing the trained adapter."""

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

    def run(self, db, training_run: TrainingRun) -> str:
        """Spawn the training subprocess and drive the run to a staging directory.

        Raises on failure (non-zero exit, subprocess error, or timeout); the worker translates
        that into a FAILED run so no run stays stuck RUNNING (issue #38).
        """
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

        timed_out = [False]

        def _enforce_timeout() -> None:
            timed_out[0] = True
            try:
                proc.kill()
            except OSError:
                pass
            try:
                logger.warning(
                    "training_timeout_kill",
                    training_run_id=training_run.training_run_id,
                    timeout_seconds=self._timeout,
                )
            except Exception:
                pass

        watchdog = None
        if self._timeout and self._timeout > 0:
            watchdog = threading.Timer(self._timeout, _enforce_timeout)
            watchdog.daemon = True
            watchdog.start()

        stderr_lines: list[str] = []

        def _drain_stderr() -> None:
            for line in proc.stderr:
                stderr_lines.append(line)

        stderr_thread = threading.Thread(target=_drain_stderr, daemon=True)
        stderr_thread.start()

        try:
            for line in proc.stdout:
                self._handle_line(db, training_run, line, staging)
        except (BrokenPipeError, OSError, UnicodeDecodeError):
            pass

        proc.wait()
        if watchdog:
            watchdog.cancel()

        if proc.returncode != 0:
            raise self._failure(
                training_run, proc.returncode, stderr_lines, timed_out[0]
            )

        return str(staging)

    def _handle_line(
        self, db, training_run: TrainingRun, line: str, staging: Path
    ) -> None:
        line = line.strip()
        if not line:
            return
        try:
            payload = json.loads(line)
        except json.JSONDecodeError:
            return
        event = payload.get("event")
        if event == "progress":
            training_service.update_training_progress(
                db,
                training_run,
                epoch=payload.get("epoch"),
                current_step=payload.get("step"),
                train_loss=payload.get("train_loss"),
                eval_loss=payload.get("eval_loss"),
            )
            db.commit()
        elif event == "done":
            logger.info(
                "training_done",
                training_run_id=training_run.training_run_id,
                staging=str(staging),
            )

    def _failure(
        self,
        training_run: TrainingRun,
        returncode: int,
        stderr_lines: list[str],
        timed_out: bool,
    ) -> Exception:
        stderr = "".join(stderr_lines)[-_MAX_STDERR:]
        if timed_out:
            return TimeoutError(
                f"training for {training_run.training_run_id} exceeded "
                f"{self._timeout}s and was killed"
            )
        if stderr:
            return RuntimeError(
                f"training for {training_run.training_run_id} exited with code "
                f"{returncode}: {stderr.strip()[-_MAX_STDERR:]}"
            )
        return RuntimeError(
            f"training for {training_run.training_run_id} exited with code {returncode}"
        )
