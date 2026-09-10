"""Standalone worker for GPU VPS — polls control plane for jobs (PRD §9.5).

Runs on the remote GPU VPS server. Claims PENDING training jobs assigned to this
resource, executes training locally, uploads artifacts to MinIO via presigned URL,
and reports heartbeat/completion back to the control plane.

Usage::

    REMOTE_WORKER_API_URL=http://control-plane:8000 \\
    REMOTE_WORKER_POLL_INTERVAL=5 \\
    python -m app.workers.remote_worker
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

import httpx
import structlog

from app.config import settings
from app.services.artifact_storage import MinioArtifactStorage

logger = structlog.get_logger(__name__)


def _get_pending_jobs(client: httpx.Client, resource_id: int) -> list[dict[str, Any]]:
    """Fetch PENDING training runs assigned to this compute resource."""
    resp = client.get(
        "/api/v1/training-runs",
        params={"status": "PENDING"},
    )
    resp.raise_for_status()
    runs = resp.json().get("runs", [])
    return [r for r in runs if r.get("compute_resource_id") == resource_id]


def _claim_job(client: httpx.Client, run_id: str) -> bool:
    """Atomically claim a PENDING training run (PENDING → RUNNING)."""
    resp = client.post(f"/api/v1/training-runs/{run_id}/claim")
    if resp.status_code == 200:
        return True
    logger.warning("claim_failed", run_id=run_id, status=resp.status_code)
    return False


def _report_heartbeat(client: httpx.Client, run_id: str) -> None:
    """POST heartbeat to keep the run alive (PRD §10.4)."""
    try:
        client.post(f"/api/v1/training-runs/{run_id}/heartbeat")
    except Exception:
        logger.warning("heartbeat_post_failed", run_id=run_id, exc_info=True)


def _complete_job(client: httpx.Client, run_id: str, artifact_uri: str) -> None:
    """Report training completion with artifact location."""
    client.post(
        f"/api/v1/training-runs/{run_id}/complete",
        json={"artifact_uri": artifact_uri},
    )


def _fail_job(client: httpx.Client, run_id: str, error: str) -> None:
    """Report training failure."""
    client.post(
        f"/api/v1/training-runs/{run_id}/fail",
        json={"error_message": error},
    )


def _execute_training_locally(run: dict[str, Any], staging_dir: Path) -> None:
    """Execute training on the local GPU VPS (PRD §9.5).

    Spawns the training script as a subprocess and waits for completion.
    Writes a .done marker on success.
    """
    import subprocess

    config_json = json.dumps(run.get("training_config", {}))
    cmd = [
        settings.training_python,
        "-u",
        settings.training_script_path,
        "--config",
        config_json,
        "--staging",
        str(staging_dir),
    ]

    proc = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    stdout, stderr = proc.communicate(timeout=settings.training_timeout_seconds or None)

    if proc.returncode != 0:
        raise RuntimeError(
            f"Training failed with code {proc.returncode}: {stderr[-500:]}"
        )
    (staging_dir / ".done").touch()


def _upload_artifact_to_minio(staging_dir: Path, key_prefix: str) -> str:
    """Upload training artifacts to MinIO via presigned URL (PRD §13.3).

    Returns the s3:// URI of the uploaded artifact.
    """
    storage = MinioArtifactStorage()
    uploaded: list[str] = []

    for path in sorted(staging_dir.rglob("*")):
        if path.is_file() and not path.name.startswith("."):
            rel = path.relative_to(staging_dir).as_posix()
            object_key = f"{key_prefix}/{rel}"
            presigned_url = storage.generate_presigned_upload_url(
                object_key, expires_seconds=3600
            )
            with open(path, "rb") as f:
                resp = httpx.put(
                    presigned_url,
                    content=f.read(),
                    headers={"Content-Type": "application/octet-stream"},
                    timeout=300,
                )
                resp.raise_for_status()
            uploaded.append(object_key)

    bucket = storage._bucket
    return f"s3://{bucket}/{key_prefix}/"


def run_forever() -> None:
    """Main polling loop for the remote GPU VPS worker (PRD §9.5)."""
    if not settings.remote_worker_api_url:
        logger.error("remote_worker_api_url_not_set")
        raise SystemExit("REMOTE_WORKER_API_URL must be set")

    resource_id = int(__import__("os").environ.get("REMOTE_WORKER_RESOURCE_ID", "0"))
    if not resource_id:
        logger.error("remote_worker_resource_id_not_set")
        raise SystemExit("REMOTE_WORKER_RESOURCE_ID must be set")

    logger.info(
        "remote_worker_starting",
        api_url=settings.remote_worker_api_url,
        resource_id=resource_id,
        poll_interval=settings.remote_worker_poll_interval,
    )

    client = httpx.Client(
        base_url=settings.remote_worker_api_url,
        timeout=30.0,
    )

    while True:
        try:
            pending = _get_pending_jobs(client, resource_id)
            for run in pending:
                run_id = run["training_run_id"]
                logger.info("job_found", run_id=run_id)

                if not _claim_job(client, run_id):
                    continue

                staging_dir = Path(
                    __import__("tempfile").mkdtemp(prefix="defnex-remote-")
                )
                try:
                    _execute_training_locally(run, staging_dir)

                    key_prefix = f"{run.get('model_id', 'unknown')}/{run_id}"
                    artifact_uri = _upload_artifact_to_minio(staging_dir, key_prefix)

                    _complete_job(client, run_id, artifact_uri)
                    logger.info(
                        "job_completed",
                        run_id=run_id,
                        artifact_uri=artifact_uri,
                    )
                except Exception as exc:
                    error_msg = str(exc)[-500:]
                    _fail_job(client, run_id, error_msg)
                    logger.error(
                        "job_failed",
                        run_id=run_id,
                        error=error_msg,
                        exc_info=True,
                    )
                finally:
                    _report_heartbeat(client, run_id)

        except httpx.HTTPError as exc:
            logger.warning("poll_http_error", error=str(exc))
        except Exception:
            logger.error("remote_worker_unhandled", exc_info=True)

        time.sleep(settings.remote_worker_poll_interval)


if __name__ == "__main__":
    run_forever()
