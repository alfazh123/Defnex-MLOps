import threading
import time

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import Session

from app.db.base import Base
from app.schemas.dataset import DatasetVersionCreateRequest
from app.schemas.training import TrainingConfig, TrainingRunCreateRequest
from app.services import dataset_service, training_service
from app.workers.gpu_lock import gpu_lock
from app.workers.training_worker import process_next_job


class _StubRunner:
    def __init__(self, artifact_uri=None, error=None):
        self.artifact_uri = artifact_uri
        self.error = error
        self.calls = []
        self._guard = threading.Lock()

    def run(self, training_run):
        with self._guard:
            self.calls.append(training_run.training_run_id)
        if self.error:
            raise self.error
        return self.artifact_uri


class _StubRunnerBlocking(_StubRunner):
    """Holds onto the lock while running, to widen the concurrent-claim window."""

    def __init__(self, hold=0.3, **kwargs):
        super().__init__(**kwargs)
        self.hold = hold

    def run(self, training_run):
        with self._guard:
            self.calls.append(training_run.training_run_id)
        time.sleep(self.hold)
        if self.error:
            raise self.error
        return self.artifact_uri


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


def test_process_next_job_returns_none_when_queue_empty(db_session, lock_file):
    assert process_next_job(db_session, _StubRunner(), lock_file=lock_file) is None


def test_process_next_job_completes_on_success(db_session, lock_file):
    training_run = _queued_training_run(db_session)
    runner = _StubRunner(artifact_uri="file:///tmp/adapter")

    processed = process_next_job(db_session, runner, lock_file=lock_file)

    assert processed.training_run_id == training_run.training_run_id
    assert processed.status == "COMPLETED"
    assert processed.artifact_uri == "file:///tmp/adapter"
    assert runner.calls == [training_run.training_run_id]


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

    processed = process_next_job(
        db_session, _StubRunner(artifact_uri="uri"), lock_file=lock_file
    )

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

    process_next_job(
        db_session, _StubRunner(artifact_uri="file:///tmp/adapter"), lock_file=lock_file
    )

    from app.services import model_service

    model_version = model_service.get_model_version(db_session, "qwen-sft-domain-x", 1)
    assert model_version is not None
    assert model_version.status == "REGISTERED"
    assert model_version.training_run_id == training_run.training_run_id


def test_process_next_job_sequential_second_call_finds_no_pending_run(
    db_session, lock_file
):
    """Two process_next_job calls over one PENDING run: the first claims and runs it, the
    second finds an empty PENDING queue, so the runner executes once."""
    training_run = _queued_training_run(db_session)
    runner = _StubRunner(artifact_uri="file:///tmp/adapter")

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
    second = process_next_job(
        db_session, _StubRunner(artifact_uri="file:///tmp/adapter"), lock_file=lock_file
    )
    assert second.status == "COMPLETED"


def test_lock_timeout_leaves_run_pending_and_retries(db_session, lock_file):
    """Issue #33 edge case: when the GPU lock is held beyond the wait timeout, the run is
    not silently lost — it stays PENDING and the next poll executes it."""
    _queued_training_run(db_session)
    runner = _StubRunner(artifact_uri="file:///tmp/adapter")

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
            _StubRunner(artifact_uri="uri"),
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
    runner = _StubRunnerBlocking(artifact_uri="file:///tmp/adapter")

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
