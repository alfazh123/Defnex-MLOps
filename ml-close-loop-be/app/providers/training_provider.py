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
from app.models.compute_resource import ComputeResource
from app.models.training import TrainingRun
from app.services.ssh import SSHRemoteHost, SecretRef, PasswordCredential

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
                f"Training exited with code {job.proc.returncode}: {stderr.strip()}"
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


# ---------------------------------------------------------------------------
# GPUVPSProvider — remote SSH-based training provider (PRD §9.5)
# ---------------------------------------------------------------------------


@dataclass
class _RemoteJob:
    """Bookkeeping for an SSH-submitted remote job."""

    training_run_id: str
    external_job_id: str
    remote_staging_dir: str
    failed: bool = False
    error_message: str = ""


class GPUVPSProvider:
    """GPU VPS training provider — executes training on remote GPU via SSH (PRD §9.5).

    Uses SSHRemoteHost from app/services/ssh.py for remote execution. The provider
    uploads the training script, spawns a remote process, and polls its status via SSH.

    Cross-host GPU locking is deferred to #86 (infrastructure). This provider relies
    on single-resource assignment: no two workers claim the same ComputeResource
    simultaneously.
    """

    def __init__(self, resource: ComputeResource) -> None:
        self._resource = resource
        self._host = resource.ssh_host or resource.host or ""
        self._port = resource.ssh_port or 22
        self._username = resource.ssh_username or "root"
        self._credential = self._resolve_credential(resource.credential_ref)
        self._jobs: dict[str, _RemoteJob] = {}

    def _resolve_credential(self, credential_ref: str | None) -> object:
        """Resolve credential from a secret:// reference using the SSH module."""
        if credential_ref and credential_ref.startswith("secret://"):
            return SecretRef(ref=credential_ref)
        # Fallback: use environment-based password for root.
        import os

        return PasswordCredential(
            username=self._username,
            password=os.environ.get("GPU_VPS_PASSWORD", ""),
        )

    def _ssh(self):
        """Create a new SSHRemoteHost connection for this resource."""
        return SSHRemoteHost(
            self._host,
            port=self._port,
            credential=self._credential,
            connect_timeout=settings.ssh_connect_timeout,
        )

    def submit(self, db: Session, training_run: TrainingRun) -> str:
        """Upload training script and spawn remote process (PRD §9.5).

        Returns the remote PID as external_job_id.
        """
        external_job_id = f"gpu-vps-{uuid.uuid4().hex[:8]}"
        remote_staging = f"/tmp/defnex-training-{uuid.uuid4().hex[:8]}"

        with self._ssh() as host:
            # Create staging directory on remote
            host.execute(["mkdir", "-p", remote_staging])

            # Upload training script via SFTP
            script_path = Path(settings.training_script_path)
            if script_path.exists():
                remote_script = f"{remote_staging}/run_training.py"
                host.upload(script_path, remote_script)
            else:
                remote_script = settings.training_script_path

            # Build the command
            config_json = json.dumps(training_run.training_config)
            cmd = [
                "python3",
                remote_script,
                "--config",
                config_json,
                "--staging",
                remote_staging,
            ]

            # Start remote process in background: nohup ... &
            # Use nohup + & so the process survives SSH disconnect.
            bg_cmd = (
                "nohup "
                + " ".join(cmd)
                + f" > {remote_staging}/stdout.log 2> {remote_staging}/stderr.log & echo $!"
            )
            result = host.execute(["sh", "-c", bg_cmd])
            remote_pid = result.stdout.strip()

        job = _RemoteJob(
            training_run_id=training_run.training_run_id,
            external_job_id=external_job_id,
            remote_staging_dir=remote_staging,
        )
        self._jobs[external_job_id] = job

        training_run.external_job_id = external_job_id
        db.flush()

        logger.info(
            "gpu_vps_submitted",
            training_run_id=training_run.training_run_id,
            external_job_id=external_job_id,
            remote_pid=remote_pid,
            host=self._host,
        )

        return external_job_id

    def get_status(self, external_job_id: str) -> JobStatus:
        """Check if the remote process is alive via SSH (PRD §9.5)."""
        job = self._jobs.get(external_job_id)
        if job is None:
            raise ValueError(f"Unknown job: {external_job_id}")
        if job.failed:
            return JobStatus(status="FAILED", error_message=job.error_message)

        try:
            # Check remote PID via ps
            with self._ssh() as host:
                pid = external_job_id.removeprefix("gpu-vps-")
                result = host.execute(
                    [
                        "sh",
                        "-c",
                        f"ps -p $(cat {job.remote_staging_dir}/.pid 2>/dev/null || echo {pid}) > /dev/null 2>&1 && echo alive || echo dead",
                    ]
                )
                output = result.stdout.strip()
                if "alive" in output:
                    return JobStatus(status="RUNNING")
                # Process not running — check for success marker
                check = host.execute(["test", "-f", f"{job.remote_staging_dir}/.done"])
                if check.returncode == 0:
                    return JobStatus(status="COMPLETED")
                # Fallback: check remote process exit code from the PID log
                exit_check = host.execute(
                    [
                        "sh",
                        "-c",
                        f"PID=$(grep -o '[0-9]*' {job.remote_staging_dir}/stdout.log | head -1); "
                        f"wait $PID 2>/dev/null; echo $? 2>/dev/null || echo 1",
                    ]
                )
                try:
                    exit_code = int(exit_check.stdout.strip())
                except (ValueError, TypeError):
                    exit_code = 1
                # Check stderr from log
                err_check = host.execute(
                    ["cat", f"{job.remote_staging_dir}/stderr.log"]
                )
                if exit_code != 0 or err_check.stdout.strip():
                    return JobStatus(
                        status="FAILED",
                        error_message=(
                            err_check.stdout.strip()[-500:]
                            if err_check.stdout.strip()
                            else f"Remote process exited with code {exit_code}"
                        ),
                    )
                return JobStatus(status="COMPLETED")
        except Exception as exc:
            job.failed = True
            job.error_message = f"SSH error: {exc}"
            return JobStatus(status="FAILED", error_message=job.error_message)

    def cancel(self, external_job_id: str) -> None:
        """Kill the remote process via SSH (PRD §9.5)."""
        job = self._jobs.get(external_job_id)
        if job is None:
            raise ValueError(f"Unknown job: {external_job_id}")
        try:
            with self._ssh() as host:
                pid = external_job_id.removeprefix("gpu-vps-")
                host.execute(["sh", "-c", f"kill {pid} 2>/dev/null; true"])
        except Exception:
            pass
        job.failed = True
        job.error_message = "Cancelled by operator"

    def collect_result(self, external_job_id: str) -> str:
        """Download artifacts from remote via SFTP to a local staging dir."""
        job = self._jobs.get(external_job_id)
        if job is None:
            raise ValueError(f"Unknown job: {external_job_id}")
        if job.failed:
            raise RuntimeError(f"Job {external_job_id} failed: {job.error_message}")

        local_staging = Path(tempfile.mkdtemp(prefix="defnex-gpu-vps-result-"))
        with self._ssh() as host:
            # Download the remote staging directory contents
            result = host.execute(["ls", job.remote_staging_dir])
            for filename in result.stdout.strip().splitlines():
                if filename and not filename.startswith("."):
                    remote_file = f"{job.remote_staging_dir}/{filename}"
                    local_file = local_staging / filename
                    host.download(remote_file, local_file)

        return str(local_staging)


# ---------------------------------------------------------------------------
# ColabProvider — ephemeral on-demand Colab runner (PRD §9.6/§9.7/§9.8)
# ---------------------------------------------------------------------------


@dataclass
class _ColabClaim:
    """Bookkeeping for a Colab claim token."""

    training_run_id: str
    status: str = "QUEUED"
    error_message: str = ""
    artifact_uri: str = ""


class ColabProvider:
    """Colab runner provider — ephemeral, on-demand (PRD §9.6).

    Unlike GPUVPSProvider which runs training on a persistent remote host,
    ColabProvider is designed for on-demand execution via Google Colab notebooks.
    The notebook polls for pending jobs, claims one, executes training, and
    uploads artifacts directly to MinIO.

    Job state is tracked in-memory like the other providers. The notebook
    reports back to the control plane via API endpoints (heartbeat, complete,
    fail) which update the TrainingRun in the database directly.
    """

    def __init__(self) -> None:
        self._claims: dict[str, _ColabClaim] = {}

    def submit(self, db: Session, training_run: TrainingRun) -> str:
        """Generate a claim token for a Colab notebook to pick up (PRD §9.6).

        The notebook polls for PENDING jobs assigned to this compute_resource_id,
        claims the job, executes training, and uploads artifacts to MinIO.
        Returns a claim token (UUID) as external_job_id.
        """
        claim_token = str(uuid.uuid4())
        training_run.external_job_id = claim_token
        self._claims[claim_token] = _ColabClaim(
            training_run_id=training_run.training_run_id,
        )
        db.flush()

        logger.info(
            "colab_submitted",
            training_run_id=training_run.training_run_id,
            external_job_id=claim_token,
            compute_resource_id=training_run.compute_resource_id,
        )
        return claim_token

    def get_status(self, external_job_id: str) -> JobStatus:
        """Check if a Colab job has been claimed and is running.

        QUEUED → waiting for notebook to pick up (reported as RUNNING to
        the worker adapter which keeps polling). COMPLETED/FAILED are set
        when the notebook reports back.
        """
        claim = self._claims.get(external_job_id)
        if claim is None:
            raise ValueError(f"Unknown job: {external_job_id}")
        if claim.status == "COMPLETED":
            return JobStatus(status="COMPLETED")
        if claim.status == "FAILED":
            return JobStatus(status="FAILED", error_message=claim.error_message)
        # QUEUED or RUNNING — still waiting for notebook
        return JobStatus(status="RUNNING")

    def cancel(self, external_job_id: str) -> None:
        """Invalidate a Colab claim token (PRD §9.6).

        Sets the claim to FAILED with a cancellation message.
        The notebook should detect this via the API and stop execution.
        """
        claim = self._claims.get(external_job_id)
        if claim is None:
            raise ValueError(f"Unknown job: {external_job_id}")
        claim.status = "FAILED"
        claim.error_message = "Cancelled by operator"

    def collect_result(self, external_job_id: str) -> str:
        """Return the artifact URI where Colab uploaded artifacts.

        The Colab notebook uploads artifacts to MinIO via presigned URL.
        This returns the MinIO URI for the artifacts.
        """
        claim = self._claims.get(external_job_id)
        if claim is None:
            raise ValueError(f"Unknown job: {external_job_id}")
        if claim.status == "FAILED":
            raise RuntimeError(f"Job {external_job_id} failed: {claim.error_message}")
        if claim.status != "COMPLETED":
            raise RuntimeError(f"Job {external_job_id} is not yet completed")
        return claim.artifact_uri

    def mark_completed(self, external_job_id: str, artifact_uri: str = "") -> None:
        """Mark a claim as completed (called by notebook via API)."""
        claim = self._claims.get(external_job_id)
        if claim is not None:
            claim.status = "COMPLETED"
            claim.artifact_uri = artifact_uri

    def mark_failed(self, external_job_id: str, error_message: str = "") -> None:
        """Mark a claim as failed (called by notebook via API)."""
        claim = self._claims.get(external_job_id)
        if claim is not None:
            claim.status = "FAILED"
            claim.error_message = error_message
