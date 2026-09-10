"""GPUVPSProvider and remote_worker tests (issue #76, PRD §9.5).

Tests the GPUVPSProvider (SSH-based remote training) and the remote_worker
polling/claim/upload logic using mock SSH and httpx.
"""

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import Session

from app.config import settings
from app.db.base import Base
from app.models.compute_resource import ComputeResource
from app.providers.training_provider import GPUVPSProvider
from app.schemas.dataset import DatasetVersionCreateRequest
from app.schemas.training import TrainingConfig, TrainingRunCreateRequest
from app.services import dataset_service, training_service
from app.services.ssh import ExecResult


@pytest.fixture(autouse=True)
def _artifact_sandbox(tmp_path, monkeypatch):
    monkeypatch.setattr(settings, "artifact_storage_dir", str(tmp_path))


@pytest.fixture
def db(tmp_path):
    engine = create_engine(
        f"sqlite:///{tmp_path / 'gpu_vps.db'}", connect_args={"timeout": 30}
    )
    Base.metadata.create_all(engine)
    with Session(engine) as session:
        yield session
    engine.dispose()


@pytest.fixture
def resource(db):
    res = ComputeResource(
        name="gpu-vps-test",
        provider_type="gpu_vps",
        host="10.0.0.5",
        ssh_host="10.0.0.5",
        ssh_port=22,
        ssh_username="root",
        credential_ref="secret://GPU_VPS_PASSWORD",
    )
    db.add(res)
    db.flush()
    return res


def _create_run(db):
    dataset_version = dataset_service.create_dataset_version(
        db,
        "test_ds",
        DatasetVersionCreateRequest(
            source_type="huggingface",
            source_dataset="test/dataset",
            source_commit_or_snapshot_date="2026-08-01",
            source_format="chatml",
        ),
    )
    return training_service.create_training_run(
        db,
        dataset_version,
        TrainingRunCreateRequest(
            dataset_id="test_ds",
            dataset_version=1,
            model_id="qwen-test",
            base_model="Qwen/Qwen3.8-27B",
            training_config=TrainingConfig(),
            compute_resource_id=1,
        ),
    )


# --- GPUVPSProvider submit tests ---


@patch("app.providers.training_provider.SSHRemoteHost")
def test_gpu_vps_submit_creates_remote_staging_dir(mock_ssh_cls, db, resource):
    """submit() creates a staging directory on the remote host."""
    mock_host = MagicMock()
    mock_host.execute.return_value = ExecResult(
        stdout="12345\n", stderr="", returncode=0
    )
    mock_host.__enter__ = MagicMock(return_value=mock_host)
    mock_host.__exit__ = MagicMock(return_value=False)
    mock_ssh_cls.return_value = mock_host

    run = _create_run(db)
    provider = GPUVPSProvider(resource)

    external_job_id = provider.submit(db, run)

    assert external_job_id.startswith("gpu-vps-")
    # Verify mkdir -p was called for staging
    mkdir_calls = [c for c in mock_host.execute.call_args_list if c[0][0][0] == "mkdir"]
    assert len(mkdir_calls) > 0


@patch("app.providers.training_provider.SSHRemoteHost")
def test_gpu_vps_submit_uploads_script(mock_ssh_cls, db, resource, tmp_path):
    """submit() uploads the training script via SFTP."""
    mock_host = MagicMock()
    mock_host.execute.return_value = ExecResult(
        stdout="9999\n", stderr="", returncode=0
    )
    mock_host.__enter__ = MagicMock(return_value=mock_host)
    mock_host.__exit__ = MagicMock(return_value=False)
    mock_ssh_cls.return_value = mock_host

    # Create a fake training script
    script = tmp_path / "run_training.py"
    script.write_text("print('hello')")
    with patch.object(settings, "training_script_path", str(script)):
        run = _create_run(db)
        provider = GPUVPSProvider(resource)
        provider.submit(db, run)

    assert mock_host.upload.called


@patch("app.providers.training_provider.SSHRemoteHost")
def test_gpu_vps_submit_spawns_remote_process(mock_ssh_cls, db, resource):
    """submit() spawns a background process and returns PID as external_job_id."""
    mock_host = MagicMock()
    mock_host.execute.return_value = ExecResult(stdout="42\n", stderr="", returncode=0)
    mock_host.__enter__ = MagicMock(return_value=mock_host)
    mock_host.__exit__ = MagicMock(return_value=False)
    mock_ssh_cls.return_value = mock_host

    run = _create_run(db)
    provider = GPUVPSProvider(resource)
    external_job_id = provider.submit(db, run)

    # The last execute call should be the background spawn
    last_call = mock_host.execute.call_args_list[-1]
    cmd = last_call[0][0]
    assert "nohup" in cmd[0] or any("nohup" in c for c in cmd)
    assert run.external_job_id == external_job_id


# --- GPUVPSProvider get_status tests ---


@patch("app.providers.training_provider.SSHRemoteHost")
def test_gpu_vps_get_status_running(mock_ssh_cls, db, resource):
    """get_status returns RUNNING when remote process is alive."""
    mock_host = MagicMock()
    mock_host.execute.return_value = ExecResult(
        stdout="alive\n", stderr="", returncode=0
    )
    mock_host.__enter__ = MagicMock(return_value=mock_host)
    mock_host.__exit__ = MagicMock(return_value=False)
    mock_ssh_cls.return_value = mock_host

    run = _create_run(db)
    provider = GPUVPSProvider(resource)
    eid = provider.submit(db, run)

    status = provider.get_status(eid)
    assert status.status == "RUNNING"


@patch("app.providers.training_provider.SSHRemoteHost")
def test_gpu_vps_get_status_completed(mock_ssh_cls, db, resource):
    """get_status returns COMPLETED when remote process finished successfully."""
    mock_host = MagicMock()
    # submit() uses 2 execute calls (mkdir + spawn); get_status uses 3 (ps, test -f, cat)
    mock_host.execute.side_effect = [
        ExecResult(stdout="", stderr="", returncode=0),  # submit: mkdir
        ExecResult(stdout="42\n", stderr="", returncode=0),  # submit: spawn
        ExecResult(
            stdout="dead\n", stderr="", returncode=1
        ),  # get_status: ps -p → dead
        ExecResult(
            stdout="", stderr="", returncode=0
        ),  # get_status: test -f .done → exists
    ]
    mock_host.__enter__ = MagicMock(return_value=mock_host)
    mock_host.__exit__ = MagicMock(return_value=False)
    mock_ssh_cls.return_value = mock_host

    run = _create_run(db)
    provider = GPUVPSProvider(resource)
    eid = provider.submit(db, run)

    status = provider.get_status(eid)
    assert status.status == "COMPLETED"


# --- GPUVPSProvider cancel tests ---


@patch("app.providers.training_provider.SSHRemoteHost")
def test_gpu_vps_cancel_kills_process(mock_ssh_cls, db, resource):
    """cancel() sends kill to remote process."""
    mock_host = MagicMock()
    mock_host.execute.return_value = ExecResult(stdout="42\n", stderr="", returncode=0)
    mock_host.__enter__ = MagicMock(return_value=mock_host)
    mock_host.__exit__ = MagicMock(return_value=False)
    mock_ssh_cls.return_value = mock_host

    run = _create_run(db)
    provider = GPUVPSProvider(resource)
    eid = provider.submit(db, run)
    provider.cancel(eid)

    status = provider.get_status(eid)
    assert status.status == "FAILED"
    assert "Cancelled" in status.error_message


# --- GPUVPSProvider collect_result tests ---


@patch("app.providers.training_provider.SSHRemoteHost")
def test_gpu_vps_collect_result_downloads_artifact(mock_ssh_cls, db, resource):
    """collect_result downloads artifacts from remote via SFTP."""
    mock_host = MagicMock()
    mock_host.execute.side_effect = [
        # submit: mkdir
        ExecResult(stdout="", stderr="", returncode=0),
        # submit: spawn
        ExecResult(stdout="42\n", stderr="", returncode=0),
        # collect_result: ls remote staging
        ExecResult(
            stdout="adapter_model.safetensors\nadapter_config.json\n",
            stderr="",
            returncode=0,
        ),
    ]
    mock_host.__enter__ = MagicMock(return_value=mock_host)
    mock_host.__exit__ = MagicMock(return_value=False)
    mock_ssh_cls.return_value = mock_host

    run = _create_run(db)
    provider = GPUVPSProvider(resource)
    eid = provider.submit(db, run)

    # Mark as completed manually for the test
    provider._jobs[eid].failed = False

    local_dir = provider.collect_result(eid)
    assert Path(local_dir).is_dir()


# --- Provider factory test ---


def test_provider_factory_selects_gpu_vps_by_type():
    """get_provider_for_resource returns GPUVPSProvider for gpu_vps type."""
    from app.workers.training_worker import get_provider_for_resource

    res = ComputeResource(
        name="test",
        provider_type="gpu_vps",
        ssh_host="10.0.0.5",
    )
    provider = get_provider_for_resource(res)
    assert isinstance(provider, GPUVPSProvider)


def test_provider_factory_selects_local_by_default():
    """get_provider_for_resource returns LocalSubprocessProvider for local type."""
    from app.providers.training_provider import LocalSubprocessProvider
    from app.workers.training_worker import get_provider_for_resource

    res = ComputeResource(name="test", provider_type="local")
    provider = get_provider_for_resource(res)
    assert isinstance(provider, LocalSubprocessProvider)


# --- Remote worker tests ---


def test_remote_worker_claims_pending_job(tmp_path):
    """remote_worker._claim_job returns True on 200 response."""
    from app.workers.remote_worker import _claim_job

    mock_client = MagicMock()
    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_client.post.return_value = mock_resp

    result = _claim_job(mock_client, "run-abc")
    assert result is True
    mock_client.post.assert_called_once_with("/api/v1/training-runs/run-abc/claim")


def test_remote_worker_uploads_to_minio_via_presigned(tmp_path):
    """_upload_artifact_to_minio uses presigned PUT URLs for each file."""
    from app.workers.remote_worker import _upload_artifact_to_minio

    # Create staging dir with test files
    staging = tmp_path / "staging"
    staging.mkdir()
    (staging / "adapter.safetensors").write_bytes(b"model-data")
    (staging / "config.json").write_text('{"lora": true}')
    (staging / ".hidden").touch()  # Should be skipped

    with patch("app.workers.remote_worker.MinioArtifactStorage") as mock_storage_cls:
        mock_storage = MagicMock()
        mock_storage.generate_presigned_upload_url.return_value = (
            "http://minio:9000/bucket/key?presigned"
        )
        mock_storage._bucket = "artifacts"
        mock_storage_cls.return_value = mock_storage

        with patch("app.workers.remote_worker.httpx.put") as mock_put:
            mock_put.return_value = MagicMock(status_code=200)
            uri = _upload_artifact_to_minio(staging, "test/run-123")

    assert uri.startswith("s3://artifacts/")
    # Should upload 2 files (adapter.safetensors + config.json), skip .hidden
    assert mock_put.call_count == 2


def test_remote_worker_reports_heartbeat(tmp_path):
    """_report_heartbeat sends POST to /heartbeat endpoint."""
    from app.workers.remote_worker import _report_heartbeat

    mock_client = MagicMock()
    _report_heartbeat(mock_client, "run-xyz")
    mock_client.post.assert_called_once_with("/api/v1/training-runs/run-xyz/heartbeat")


# --- SSH connection error test ---


@patch("app.providers.training_provider.SSHRemoteHost")
def test_ssh_connection_error_returns_failed_status(mock_ssh_cls, db, resource):
    """get_status returns FAILED if SSH connection fails."""
    # First SSH session: submit succeeds
    submit_host = MagicMock()
    submit_host.execute.side_effect = [
        ExecResult(stdout="", stderr="", returncode=0),  # mkdir
        ExecResult(stdout="42\n", stderr="", returncode=0),  # spawn
    ]
    submit_host.__enter__ = MagicMock(return_value=submit_host)
    submit_host.__exit__ = MagicMock(return_value=False)

    # Second SSH session: get_status fails
    fail_host = MagicMock()
    fail_host.execute.side_effect = ConnectionError("SSH unreachable")
    fail_host.__enter__ = MagicMock(return_value=fail_host)
    fail_host.__exit__ = MagicMock(return_value=False)

    mock_ssh_cls.side_effect = [submit_host, fail_host]

    run = _create_run(db)
    provider = GPUVPSProvider(resource)
    eid = provider.submit(db, run)

    status = provider.get_status(eid)
    assert status.status == "FAILED"
    assert "SSH error" in status.error_message
