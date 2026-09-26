"""Real Unsloth training runner tests (issue #38).

The runner is a subprocess orchestrator: it spawns a standalone training script and streams its
JSON progress into the DB. Unsloth itself is never imported here (no GPU, separate venv) — the
runner is exercised against tiny fake scripts that emit the documented stdout contract, which is
exactly how the production script behaves.
"""

import sys
import time
from pathlib import Path

import pytest
import sqlalchemy as sa
from sqlalchemy import create_engine
from sqlalchemy.orm import Session

from app.config import settings
from app.db.base import Base
from app.schemas.dataset import DatasetVersionCreateRequest
from app.schemas.training import TrainingConfig, TrainingRunCreateRequest
from app.services import dataset_service, model_service, training_service
from app.workers.training_worker import process_next_job
from app.workers.unsloth_runner import UnslothTrainingRunner


@pytest.fixture(autouse=True)
def _artifact_sandbox(tmp_path, monkeypatch):
    """Successful worker passes finalize staged artifacts into the immutable store via
    register_model_version; keep that out of the repo's data/ dir with a fresh per-test dir."""
    monkeypatch.setattr(settings, "artifact_storage_dir", str(tmp_path))


@pytest.fixture
def db(tmp_path):
    """File-backed SQLite session (all connections share one database, like production).

    The in-memory `db` fixture keeps ONE private database per thread; the runner's
    per-event progress commits race the watchdog timer thread and the spawned training child
    under pytest, which intermittently reloads into a thread-private empty database. A real
    file has no such split-brain, so these subprocess-driven tests pin the file engine here.
    """
    engine = create_engine(
        f"sqlite:///{tmp_path / 'runner.db'}", connect_args={"timeout": 30}
    )

    @sa.event.listens_for(engine, "connect")
    def _enable_wal(conn, _record):
        conn.execute("PRAGMA journal_mode=WAL")

    Base.metadata.create_all(engine)
    with Session(engine) as session:
        yield session
    engine.dispose()


def _run_in_dir(db, *, model_id="qwen-sft-domain-x") -> "object":
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
    # Issue #208: the runner now resolves the run's dataset pin to real bytes before
    # spawning the trainer, and refuses to run without them. This fixture used to create a
    # dataset version with no `canonical_file_uri` at all, which under the new rule is
    # (correctly) untrainable -- so these runner tests would have been exercising the pin
    # failure path instead of the runner plumbing they are named for. Committing bytes here
    # keeps them testing the runner *and* exercises pinning for real. `artifact_storage_dir`
    # is already redirected to tmp_path by the `_artifact_sandbox` fixture.
    from app.services.artifact_storage import LocalFilesystemArtifactStorage

    store = LocalFilesystemArtifactStorage(settings.artifact_storage_dir)
    payload = b'{"id": "r1", "messages": [{"role": "user", "content": "hi"}]}\n'
    dataset_version.canonical_file_uri = store.put_immutable(
        "datasets/no_robots/v1/train.jsonl", payload
    )
    db.flush()
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
print(json.dumps({"event": "progress", "epoch": 1, "step": 5, "train_loss": 0.5, "eval_loss": 0.4}), flush=True)
print(json.dumps({"event": "progress", "epoch": 1, "step": 6, "train_loss": 0.4}), flush=True)
(staging / "adapter_model.safetensors").write_bytes(b"trained")
(staging / "adapter_config.json").write_text('{"lora": true}')
print(json.dumps({"event": "done", "files": ["adapter_model.safetensors"]}), flush=True)
"""

OOM_SCRIPT = """
import sys
print("Traceback (most recent call last):", file=sys.stderr, flush=True)
print("torch.cuda.OutOfMemoryError: CUDA out of memory.", file=sys.stderr, flush=True)
sys.exit(1)
"""

SLEEP_FOREVER_SCRIPT = """
import json, sys
print(json.dumps({"event": "progress", "epoch": 1, "step": 1}), flush=True)
sys.stdout.flush()
import time
while True:
    time.sleep(1)
"""


def _runner(script: Path, timeout=None, python=None) -> UnslothTrainingRunner:
    return UnslothTrainingRunner(
        python_executable=python or sys.executable,
        script_path=script,
        timeout_seconds=timeout,
    )


def test_runner_streams_progress_into_db_and_returns_staging(db, tmp_path):
    """Issue #38 core acceptance: as the (fake) training emits progress, the four progress
    columns are persisted to the still-RUNNING run, and a staging directory is returned."""
    run = _run_in_dir(db)
    assert training_service.claim_training_run(db, run)

    staging = _runner(_write_script(tmp_path, SUCCESS_SCRIPT)).run(db, run)

    db.expire_all()
    refreshed = training_service.get_training_run(db, run.training_run_id)
    assert refreshed.status == "RUNNING"
    assert refreshed.current_epoch == 1
    assert refreshed.current_step == 6
    assert refreshed.train_loss == 0.4
    assert refreshed.eval_loss == 0.4

    staged = Path(staging)
    assert staged.is_dir()
    assert (staged / "adapter_model.safetensors").read_bytes() == b"trained"


def test_runner_nonzero_exit_raises_with_stderr(db, tmp_path):
    """Issue #38: a failing training process (non-zero exit / CUDA OOM) raises; the worker
    turns that into a FAILED run."""
    with pytest.raises(RuntimeError, match="out of memory"):
        _runner(_write_script(tmp_path, OOM_SCRIPT)).run(db, _run_in_dir(db))


def test_runner_timeout_kills_and_raises_timeout_error(db, tmp_path):
    """Issue #38: a submission thread or a hung process must not leave the run stuck RUNNING —
    the timeout watchdog kills it and raises TimeoutError."""
    run = _run_in_dir(db)
    assert training_service.claim_training_run(db, run)

    started = time.monotonic()
    with pytest.raises(TimeoutError, match="exceeded"):
        _runner(_write_script(tmp_path, SLEEP_FOREVER_SCRIPT), timeout=1).run(db, run)
    assert time.monotonic() - started < 10


def test_worker_failure_sets_failed_and_registers_nothing(db, tmp_path):
    """Issue #38 required test (failure path through the real worker): a script that dies
    with a CUDA OOM marks the run FAILED with the error, never leaves it RUNNING, and no
    ModelVersion is registered."""
    run = _run_in_dir(db)
    lock = str(tmp_path / "gpu.lock")

    processed = process_next_job(
        db,
        _runner(_write_script(tmp_path, OOM_SCRIPT)),
        lock_file=lock,
    )

    assert processed.training_run_id == run.training_run_id
    assert processed.status == "FAILED"
    assert "out of memory" in processed.error_message
    assert processed.finished_at is not None
    assert model_service.get_model_version(db, "qwen-sft-domain-x", 1) is None


def test_worker_timeout_sets_failed_and_registers_nothing(db, tmp_path):
    """Issue #38 required test (timeout through the real worker): a hung script becomes
    FAILED, not stuck RUNNING, and nothing is registered."""
    _run_in_dir(db)
    lock = str(tmp_path / "gpu.lock")

    processed = process_next_job(
        db,
        _runner(_write_script(tmp_path, SLEEP_FOREVER_SCRIPT), timeout=1),
        lock_file=lock,
    )

    assert processed.status == "FAILED"
    assert "exceeded" in processed.error_message
    assert model_service.get_model_version(db, "qwen-sft-domain-x", 1) is None


def test_worker_success_registers_version_with_immutable_artifact(db, tmp_path):
    """Issue #38: success finalizes the staged output into the immutable per-version dir,
    names it `{project}-{base_model}-v{N}`, and rewires the run's artifact_uri to it."""
    _run_in_dir(db)
    lock = str(tmp_path / "gpu.lock")

    processed = process_next_job(
        db, _runner(_write_script(tmp_path, SUCCESS_SCRIPT)), lock_file=lock
    )

    assert processed.status == "COMPLETED"
    model_version = model_service.get_model_version(db, "qwen-sft-domain-x", 1)
    assert model_version is not None
    assert model_version.name == "qwen-sft-domain-x-Qwen-Qwen3.8-27B-v1"

    artifact_uri = model_version.artifacts[0]["uri"]
    assert artifact_uri.startswith("file://")
    artifact_dir = Path(artifact_uri.removeprefix("file://"))
    assert artifact_dir.is_dir()
    assert artifact_dir.name == model_version.name
    assert (artifact_dir / "adapter_model.safetensors").read_bytes() == b"trained"
    assert (artifact_dir / "metadata.json").is_file()
    assert processed.artifact_uri == artifact_uri


def test_runner_ignores_garbage_and_non_contract_json(db, tmp_path):
    """Robustness: non-JSON or unknown-event lines on stdout must not crash the runner; a
    clean exit still returns the staging directory."""
    script_body = (
        "import json, sys\n"
        'print("not json at all", flush=True)\n'
        'print(json.dumps({"event": "unrelated", "foo": 1}), flush=True)\n'
        "sys.exit(0)\n"
    )
    staging = _runner(_write_script(tmp_path, script_body)).run(db, _run_in_dir(db))

    assert Path(staging).is_dir()


def test_progress_visible_from_separate_session_while_training_runs(tmp_path):
    """Issue #38 core acceptance (cross-session visibility): progress committed by the runner
    must be visible from a DIFFERENT database session while training is still in progress.
    The runner's per-event `db.commit()` is a real DB-level commit (not just a flush), so a
    second session reading the same file sees live non-NULL progress columns."""
    import threading

    from sqlalchemy import create_engine as ce
    from sqlalchemy.orm import Session as S

    db_path = str(tmp_path / "progress.db")
    engine = ce(f"sqlite:///{db_path}", connect_args={"timeout": 30})
    Base.metadata.create_all(engine)

    with S(engine) as setup:
        run = _run_in_dir(setup, model_id="live-progress")
        setup.commit()
        run_id = run.training_run_id
    engine.dispose()

    # Training script: emit one progress event, wait for observer, then exit.
    script_body = (
        "import json, sys, time\n"
        'print(json.dumps({"event": "progress", "epoch": 1, "step": 1, "train_loss": 0.9}), flush=True)\n'
        "time.sleep(3)\n"
        'print(json.dumps({"event": "progress", "epoch": 1, "step": 2, "train_loss": 0.8}), flush=True)\n'
        "sys.exit(0)\n"
    )
    script = tmp_path / "live_training.py"
    script.write_text(script_body)

    eng = ce(f"sqlite:///{db_path}", connect_args={"timeout": 30})
    observed = {}

    def run_training():
        with S(eng) as session:
            run_obj = training_service.get_training_run(session, run_id)
            training_service.claim_training_run(session, run_obj)
            _runner(script).run(session, run_obj)
            session.commit()

    t = threading.Thread(target=run_training)
    t.start()

    # Poll from a SEPARATE session until the first progress event lands.
    reader_eng = ce(f"sqlite:///{db_path}", connect_args={"timeout": 30})
    deadline = time.monotonic() + 10
    while time.monotonic() < deadline:
        with S(reader_eng) as reader:
            row = training_service.get_training_run(reader, run_id)
            if row and row.current_step is not None:
                observed["status"] = row.status
                observed["step"] = row.current_step
                observed["loss"] = row.train_loss
                break
        time.sleep(0.2)
    reader_eng.dispose()

    t.join(timeout=15)
    eng.dispose()

    assert observed["status"] == "RUNNING"
    assert observed["step"] >= 1
    assert observed["loss"] == 0.9
