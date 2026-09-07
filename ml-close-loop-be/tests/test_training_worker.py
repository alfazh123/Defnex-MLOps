import tempfile
import threading
import time
from pathlib import Path

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import Session

from app.config import settings
from app.db.base import Base
from app.schemas.dataset import DatasetVersionCreateRequest
from app.schemas.training import TrainingConfig, TrainingRunCreateRequest
from app.services import dataset_service, training_service
from app.workers.gpu_lock import gpu_lock
from app.workers.training_worker import process_next_job


@pytest.fixture(autouse=True)
def _artifact_sandbox(tmp_path, monkeypatch):
    """Keep immutable per-version artifacts (register_model_version finalizes the runner's
    staging output) out of the repo's data/ dir — one fresh per-test temp dir, since the
    same model/version names collide across tests (immutability is the point)."""
    monkeypatch.setattr(settings, "artifact_storage_dir", str(tmp_path))


def _staging_dir(training_run_id: str) -> str:
    """A real staging directory, matching the runner contract (issue #38) — run() returns a
    PATH to a dir holding the adapter, never a flat file uri."""
    staging = Path(tempfile.mkdtemp(prefix="defnex-test-stage-"))
    (staging / "adapter_model.safetensors").write_bytes(b"fake")
    (staging / "adapter_config.json").write_text(f'{{"run": "{training_run_id}"}}')
    return str(staging)


class _StubRunner:
    def __init__(self, staging=None, error=None):
        self.staging = staging
        self.error = error
        self.calls = []
        self._guard = threading.Lock()

    def run(self, db, training_run):
        with self._guard:
            self.calls.append(training_run.training_run_id)
        if self.error:
            raise self.error
        return self.staging or _staging_dir(training_run.training_run_id)


class _StubRunnerBlocking(_StubRunner):
    """Holds onto the lock while running, to widen the concurrent-claim window."""

    def __init__(self, hold=0.3, **kwargs):
        super().__init__(**kwargs)
        self.hold = hold

    def run(self, db, training_run):
        with self._guard:
            self.calls.append(training_run.training_run_id)
        time.sleep(self.hold)
        if self.error:
            raise self.error
        return self.staging or _staging_dir(training_run.training_run_id)


@pytest.fixture
def lock_file(tmp_path):
    return str(tmp_path / "gpu.lock")


def _queued_training_run(db_session):
    dataset_version = dataset_service.create_dataset_version(
        db_session,
        "no_robots",
        DatasetVersionCreateRequest(
            source_type="huggingface",
            source_dataset="HuggingFaceH4/no_robots",
            source_commit_or_snapshot_date="2026-08-01",
            source_format="chatml",
        ),
    )
    return training_service.create_training_run(
        db_session,
        dataset_version,
        TrainingRunCreateRequest(
            dataset_id="no_robots",
            dataset_version=1,
            model_id="qwen-sft-domain-x",
            base_model="Qwen/Qwen3.8-27B",
            training_config=TrainingConfig(),
        ),
    )


def _assert_finalized_immutable_artifact(run_or_uri):
    """After a successful worker pass the artifact must be the finalized per-version path
    `{base}/{model_id}/{name}/` (issue #38), not the raw staging dir."""
    if isinstance(run_or_uri, str):
        uri = run_or_uri
    else:
        uri = run_or_uri.artifact_uri
    assert uri.startswith("file://"), uri
    path = Path(uri.removeprefix("file://"))
    assert path.is_dir()
    assert (path / "adapter_model.safetensors").is_file()
    return path


def test_process_next_job_returns_none_when_queue_empty(db_session, lock_file):
    assert process_next_job(db_session, _StubRunner(), lock_file=lock_file) is None


def test_process_next_job_completes_on_success(db_session, lock_file):
    training_run = _queued_training_run(db_session)
    runner = _StubRunner()

    processed = process_next_job(db_session, runner, lock_file=lock_file)

    assert processed.training_run_id == training_run.training_run_id
    assert processed.status == "COMPLETED"
    assert runner.calls == [training_run.training_run_id]
    # the staged adapter was moved into the immutable per-version path
    _assert_finalized_immutable_artifact(processed)


def test_process_next_job_fails_on_runner_exception(db_session, lock_file):
    _queued_training_run(db_session)
    runner = _StubRunner(error=RuntimeError("out of memory"))

    processed = process_next_job(db_session, runner, lock_file=lock_file)

    assert processed.status == "FAILED"
    assert processed.error_message == "out of memory"

    from app.services import model_service

    assert model_service.get_model_version(db_session, "qwen-sft-domain-x", 1) is None


def test_process_next_job_picks_oldest_pending_first(db_session, lock_file):
    first = _queued_training_run(db_session)
    second = _queued_training_run(db_session)
    second.created_at = first.created_at.replace(year=first.created_at.year + 1)
    db_session.flush()

    processed = process_next_job(db_session, _StubRunner(), lock_file=lock_file)

    assert processed.training_run_id == first.training_run_id


def test_process_next_job_with_runner_error_sets_failed_status(db_session, lock_file):
    _queued_training_run(db_session)

    processed = process_next_job(
        db_session, _StubRunner(error=RuntimeError("cuda oom")), lock_file=lock_file
    )

    assert processed.status == "FAILED"


def test_process_next_job_with_runner_error_records_message(db_session, lock_file):
    _queued_training_run(db_session)

    processed = process_next_job(
        db_session, _StubRunner(error=RuntimeError("cuda oom")), lock_file=lock_file
    )

    assert processed.status == "FAILED"
    assert processed.error_message == "cuda oom"


def test_process_next_job_registers_model_version_on_success(db_session, lock_file):
    training_run = _queued_training_run(db_session)

    process_next_job(db_session, _StubRunner(), lock_file=lock_file)

    from app.services import model_service

    model_version = model_service.get_model_version(db_session, "qwen-sft-domain-x", 1)
    assert model_version is not None
    assert model_version.status == "REGISTERED"
    assert model_version.training_run_id == training_run.training_run_id
    # issue #38 naming + stored per-version name
    assert model_version.name == "qwen-sft-domain-x-Qwen-Qwen3.8-27B-v1"


def test_process_next_job_sequential_second_call_finds_no_pending_run(
    db_session, lock_file
):
    """Two process_next_job calls over one PENDING run: the first claims and runs it, the
    second finds an empty PENDING queue, so the runner executes once."""
    training_run = _queued_training_run(db_session)
    runner = _StubRunner()

    first = process_next_job(db_session, runner, lock_file=lock_file)
    second = process_next_job(db_session, runner, lock_file=lock_file)

    assert first.training_run_id == training_run.training_run_id
    assert second is None
    assert runner.calls == [training_run.training_run_id]
    assert training_run.status == "COMPLETED"


def test_runner_failure_never_leaves_run_in_running(db_session, lock_file):
    """If starting/executing fails (external service unreachable), the run must not hang
    forever in RUNNING: process_next_job resolves it to FAILED with an explicit error."""
    _queued_training_run(db_session)
    runner = _StubRunner(error=ConnectionError("unreachable"))

    processed = process_next_job(db_session, runner, lock_file=lock_file)

    assert processed.status == "FAILED"
    assert processed.error_message == "unreachable"


def test_runner_exception_releases_lock_and_next_job_still_runs(db_session, lock_file):
    """Issue #33 edge case: a runner exception must release the GPU lock so a later job
    can still execute."""
    _queued_training_run(db_session)
    first = process_next_job(
        db_session, _StubRunner(error=RuntimeError("boom")), lock_file=lock_file
    )
    assert first.status == "FAILED"

    _queued_training_run(db_session)
    second = process_next_job(db_session, _StubRunner(), lock_file=lock_file)
    assert second.status == "COMPLETED"


def test_lock_timeout_leaves_run_pending_and_retries(db_session, lock_file):
    """Issue #33 edge case: when the GPU lock is held beyond the wait timeout, the run is
    not silently lost — it stays PENDING and the next poll executes it."""
    _queued_training_run(db_session)
    runner = _StubRunner()

    with gpu_lock(lock_file, timeout=5.0):
        skipped = process_next_job(
            db_session, runner, lock_file=lock_file, lock_timeout=0.1
        )

    assert skipped is None
    assert runner.calls == []

    processed = process_next_job(db_session, runner, lock_file=lock_file)
    assert processed.status == "COMPLETED"
    assert runner.calls == [processed.training_run_id]


def test_lock_timeout_does_not_fail_queued_run(db_session, lock_file):
    """The run behind a contended lock must stay PENDING (not FAILED) — only a crashed worker
    that has already claimed it can end a run in FAILED."""
    queued = _queued_training_run(db_session)

    with gpu_lock(lock_file, timeout=5.0):
        skipped = process_next_job(
            db_session,
            _StubRunner(),
            lock_file=lock_file,
            lock_timeout=0.1,
        )

    assert skipped is None
    assert (
        training_service.get_training_run(db_session, queued.training_run_id).status
        == "PENDING"
    )


def test_concurrent_workers_run_single_pending_job_once(tmp_path):
    """Issue #33 required test (concurrency): two worker threads over ONE PENDING run
    execute the runner exactly once. Real concurrency needs a file-backed SQLite shared by
    two connections; the claim is fenced by the compare-and-set AND the shared GPU lock."""
    db_path = str(tmp_path / "concurrent.db")
    lock = str(tmp_path / "gpu.lock")
    runner = _StubRunnerBlocking()

    engine = create_engine(f"sqlite:///{db_path}", connect_args={"timeout": 30})
    Base.metadata.create_all(engine)
    with Session(engine) as setup:
        _queued_training_run(setup)
        setup.commit()
    engine.dispose()

    def worker():
        eng = create_engine(f"sqlite:///{db_path}", connect_args={"timeout": 30})
        try:
            with Session(eng) as session:
                process_next_job(session, runner, lock_file=lock, lock_timeout=10.0)
                session.commit()
        finally:
            eng.dispose()

    threads = [threading.Thread(target=worker), threading.Thread(target=worker)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=30)

    assert not threads[0].is_alive() and not threads[1].is_alive()
    assert len(runner.calls) == 1


def test_no_api_handler_calls_claim_or_start_training_run():
    """Regression: the only executor of a training run is the worker. No request handler
    may transition a run out of PENDING."""
    import pathlib

    api_dir = pathlib.Path(__file__).resolve().parents[1] / "app" / "api"
    offenders = []
    for py in api_dir.rglob("*.py"):
        text = py.read_text()
        if "start_training_run" in text or "claim_training_run" in text:
            offenders.append(str(py))
    assert offenders == [], (
        f"api handlers must not call training run claims: {offenders}"
    )
