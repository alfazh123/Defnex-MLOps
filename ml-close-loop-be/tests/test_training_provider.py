"""TrainingProvider abstraction tests (issue #74, PRD §9.1/§9.4).

Tests the TrainingProvider Protocol, LocalSubprocessProvider, and
ProviderRunnerAdapter. The provider wraps subprocess execution behind the
submit/get_status/cancel/collect_result contract so the domain can later swap
in GPU VPS or Colab providers.
"""

import sys
import time
from pathlib import Path

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import Session

from app.config import settings
from app.db.base import Base
from app.providers.training_provider import (
    LocalSubprocessProvider,
    _RunningJob,
)
from app.schemas.dataset import DatasetVersionCreateRequest
from app.schemas.training import TrainingConfig, TrainingRunCreateRequest
from app.services import dataset_service, training_service
from app.workers.training_worker import ProviderRunnerAdapter, process_next_job


@pytest.fixture(autouse=True)
def _artifact_sandbox(tmp_path, monkeypatch):
    monkeypatch.setattr(settings, "artifact_storage_dir", str(tmp_path))


@pytest.fixture
def db(tmp_path):
    engine = create_engine(
        f"sqlite:///{tmp_path / 'provider.db'}", connect_args={"timeout": 30}
    )
    Base.metadata.create_all(engine)
    with Session(engine) as session:
        yield session
    engine.dispose()


def _create_run(db, *, model_id="qwen-sft-domain-x"):
    dataset_version = dataset_service.create_dataset_version(
        db,
        "no_robots",
        DatasetVersionCreateRequest(
            source_type="huggingface",
            source_dataset="HuggingFaceH4/no_robots",
            source_commit_or_snapshot_date="2026-08-01",
            source_format="chatml",
        ),
    )
    return training_service.create_training_run(
        db,
        dataset_version,
        TrainingRunCreateRequest(
            dataset_id="no_robots",
            dataset_version=1,
            model_id=model_id,
            base_model="Qwen/Qwen3.8-27B",
            training_config=TrainingConfig(),
        ),
    )


def _write_script(tmp_path: Path, body: str) -> Path:
    script = tmp_path / "fake_training.py"
    script.write_text(body)
    return script


SUCCESS_SCRIPT = """
import json, sys
from pathlib import Path
staging = Path(sys.argv[sys.argv.index('--staging') + 1])
staging.mkdir(parents=True, exist_ok=True)
print(json.dumps({"event": "progress", "epoch": 1, "step": 5, "train_loss": 0.5}), flush=True)
print(json.dumps({"event": "progress", "epoch": 1, "step": 6, "train_loss": 0.4}), flush=True)
(staging / "adapter_model.safetensors").write_bytes(b"trained")
(staging / "adapter_config.json").write_text('{"lora": true}')
print(json.dumps({"event": "done"}), flush=True)
"""

FAIL_SCRIPT = """
import sys
print("CUDA OOM", file=sys.stderr, flush=True)
sys.exit(1)
"""


def _provider(tmp_path, *, timeout=None, python=None) -> LocalSubprocessProvider:
    return LocalSubprocessProvider(
        python_executable=python or sys.executable,
        script_path=_write_script(tmp_path, SUCCESS_SCRIPT),
        timeout_seconds=timeout,
    )


# --- submit / get_status / collect_result contract ---


def test_submit_sets_external_job_id(db, tmp_path):
    """AC: Kolom external_job_id diisi sebagai bagian dari kontrak provider."""
    run = _create_run(db)
    training_service.claim_training_run(db, run)
    provider = _provider(tmp_path)

    external_job_id = provider.submit(db, run)

    assert external_job_id is not None
    assert external_job_id.startswith("job-")
    assert run.external_job_id == external_job_id


def test_submit_flushes_to_db(db, tmp_path):
    """external_job_id is visible from the same session after submit (flushed)."""
    run = _create_run(db)
    training_service.claim_training_run(db, run)
    provider = _provider(tmp_path)

    provider.submit(db, run)
    db.flush()

    refreshed = training_service.get_training_run(db, run.training_run_id)
    assert refreshed.external_job_id is not None


def test_get_status_running_while_subprocess_alive(tmp_path):
    """get_status returns RUNNING while the subprocess has not exited."""
    provider = LocalSubprocessProvider(
        python_executable=sys.executable,
        script_path=_write_script(
            tmp_path,
            "import time\nwhile True: time.sleep(0.1)\n",
        ),
        timeout_seconds=30,
    )

    job_id = "job-test-running"
    proc = __import__("subprocess").Popen(
        [sys.executable, "-c", "import time; time.sleep(30)"],
        stdout=__import__("subprocess").PIPE,
        stderr=__import__("subprocess").PIPE,
    )
    provider._jobs[job_id] = _RunningJob(
        training_run_id="run-test", proc=proc, staging_dir="/tmp/fake"
    )

    status = provider.get_status(job_id)
    assert status.status == "RUNNING"

    proc.kill()
    proc.wait()


def test_get_status_completed_after_success(db, tmp_path):
    """get_status returns COMPLETED after a successful subprocess finishes."""
    run = _create_run(db)
    training_service.claim_training_run(db, run)
    provider = _provider(tmp_path)

    external_job_id = provider.submit(db, run)
    # Wait for the background thread to finish
    deadline = time.monotonic() + 10
    while time.monotonic() < deadline:
        if provider.get_status(external_job_id).status != "RUNNING":
            break
        time.sleep(0.1)

    status = provider.get_status(external_job_id)
    assert status.status == "COMPLETED"


def test_collect_result_returns_staging_dir(db, tmp_path):
    """AC: collect_result returns the staging directory with trained artifacts."""
    run = _create_run(db)
    training_service.claim_training_run(db, run)
    provider = _provider(tmp_path)

    external_job_id = provider.submit(db, run)
    deadline = time.monotonic() + 10
    while time.monotonic() < deadline:
        if provider.get_status(external_job_id).status != "RUNNING":
            break
        time.sleep(0.1)

    staging = provider.collect_result(external_job_id)
    assert Path(staging).is_dir()
    assert (Path(staging) / "adapter_model.safetensors").exists()


def test_collect_result_raises_on_failure(db, tmp_path):
    """collect_result raises if the job failed."""
    run = _create_run(db)
    training_service.claim_training_run(db, run)
    provider = LocalSubprocessProvider(
        python_executable=sys.executable,
        script_path=_write_script(tmp_path, FAIL_SCRIPT),
    )

    external_job_id = provider.submit(db, run)
    deadline = time.monotonic() + 10
    while time.monotonic() < deadline:
        if provider.get_status(external_job_id).status != "RUNNING":
            break
        time.sleep(0.1)

    with pytest.raises(RuntimeError, match="failed"):
        provider.collect_result(external_job_id)


def test_collect_result_raises_on_pending():
    """collect_result raises if the job has not completed yet."""
    provider = LocalSubprocessProvider()
    provider._jobs["job-pending"] = _RunningJob(
        training_run_id="run-x", staging_dir="/tmp/fake"
    )
    with pytest.raises(RuntimeError, match="not yet completed"):
        provider.collect_result("job-pending")


def test_cancel_kills_subprocess(db, tmp_path):
    """cancel kills the running subprocess and marks the job as failed."""
    run = _create_run(db)
    training_service.claim_training_run(db, run)
    provider = LocalSubprocessProvider(
        python_executable=sys.executable,
        script_path=_write_script(tmp_path, "import time; time.sleep(30)\n"),
    )

    external_job_id = provider.submit(db, run)
    time.sleep(0.2)  # let subprocess start

    provider.cancel(external_job_id)
    status = provider.get_status(external_job_id)
    assert status.status == "FAILED"
    assert "Cancelled" in status.error_message


def test_get_status_unknown_job_raises():
    provider = LocalSubprocessProvider()
    with pytest.raises(ValueError, match="Unknown job"):
        provider.get_status("job-nonexistent")


def test_cancel_unknown_job_raises():
    provider = LocalSubprocessProvider()
    with pytest.raises(ValueError, match="Unknown job"):
        provider.cancel("job-nonexistent")


def test_collect_result_unknown_job_raises():
    provider = LocalSubprocessProvider()
    with pytest.raises(ValueError, match="Unknown job"):
        provider.collect_result("job-nonexistent")


# --- ProviderRunnerAdapter ---


def test_adapter_runs_to_completion(db, tmp_path):
    """ProviderRunnerAdapter adapts TrainingProvider.submit → TrainingRunner.run."""
    run = _create_run(db)
    training_service.claim_training_run(db, run)
    provider = _provider(tmp_path)
    adapter = ProviderRunnerAdapter(provider)

    staging_dir = adapter.run(db, run)

    assert Path(staging_dir).is_dir()
    assert (Path(staging_dir) / "adapter_model.safetensors").exists()
    assert run.external_job_id is not None


def test_adapter_raises_on_failure(db, tmp_path):
    """ProviderRunnerAdapter propagates provider failures as exceptions."""
    run = _create_run(db)
    training_service.claim_training_run(db, run)
    provider = LocalSubprocessProvider(
        python_executable=sys.executable,
        script_path=_write_script(tmp_path, FAIL_SCRIPT),
    )
    adapter = ProviderRunnerAdapter(provider)

    with pytest.raises(RuntimeError, match="exited with code"):
        adapter.run(db, run)


# --- Worker integration with provider ---


def test_worker_process_next_job_with_provider(db, tmp_path):
    """process_next_job works with a TrainingProvider wrapped by ProviderRunnerAdapter."""
    run = _create_run(db)
    provider = _provider(tmp_path)
    adapter = ProviderRunnerAdapter(provider)
    lock = str(tmp_path / "gpu.lock")

    processed = process_next_job(db, adapter, lock_file=lock)

    assert processed.training_run_id == run.training_run_id
    assert processed.status == "COMPLETED"
    assert processed.external_job_id is not None


def test_worker_registers_model_version_via_provider(db, tmp_path):
    """The worker closes the loop: provider → COMPLETED → ModelVersion registered."""
    _create_run(db)
    provider = _provider(tmp_path)
    adapter = ProviderRunnerAdapter(provider)
    lock = str(tmp_path / "gpu.lock")

    processed = process_next_job(db, adapter, lock_file=lock)

    assert processed.status == "COMPLETED"
    from app.services import model_service

    model_version = model_service.get_model_version(db, "qwen-sft-domain-x", 1)
    assert model_version is not None
    assert model_version.status == "REGISTERED"
