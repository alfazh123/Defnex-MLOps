import contextlib
import pathlib
import signal
import subprocess
import sys
import tempfile
import threading
import time
import types

import pytest
from pydantic import ValidationError
from sqlalchemy import create_engine
from sqlalchemy.orm import Session

from app.config import Settings, settings
from app.db.base import Base
from app.services import training_service
from app.workers.gpu_orchestrator import (
    MockServingControl,
    NvidiaSmiVRAMReader,
    NoopServingCoordinator,
    RealServingCoordinator,
    ServingStartFailed,
    ServingStopFailed,
    ShellServingControl,
    StubVRAMReader,
    VRAMNotFree,
    make_coordinator,
    serving_cycle,
)
from app.workers.training_worker import process_next_job


@pytest.fixture(autouse=True)
def _artifact_sandbox(tmp_path, monkeypatch):
    """Keep immutable per-version artifacts (register_model_version finalizes the runner's
    staging output) out of the repo's data/ dir — same convention as test_training_worker.py."""
    monkeypatch.setattr(settings, "artifact_storage_dir", str(tmp_path))


def _staging_dir(training_run_id: str) -> str:
    """A real staging directory, matching the runner contract (issue #38) — run() returns a
    PATH to a dir holding the adapter, never a flat file uri."""
    staging = pathlib.Path(tempfile.mkdtemp(prefix="defnex-test-stage-"))
    (staging / "adapter_model.safetensors").write_bytes(b"fake")
    (staging / "adapter_config.json").write_text(f'{{"run": "{training_run_id}"}}')
    return str(staging)


class _StubRunner:
    def __init__(self, artifact_uri=None, error=None, hold=0.0):
        self.artifact_uri = artifact_uri
        self.error = error
        self.hold = hold
        self.calls = []
        self._guard = threading.Lock()

    def run(self, db, training_run):
        with self._guard:
            self.calls.append(training_run.training_run_id)
        if self.hold:
            time.sleep(self.hold)
        if self.error:
            raise self.error
        return _staging_dir(training_run.training_run_id)


class _StreamRunner(_StubRunner):
    """Runner that also writes training start/finish markers into a shared, locked
    event stream so tests can prove serving events never interleave mid-training."""

    def __init__(self, stream, stream_lock, **kwargs):
        super().__init__(**kwargs)
        self._stream = stream
        self._stream_lock = stream_lock

    def run(self, db, training_run):
        with self._stream_lock:
            self._stream.append(("train_start", training_run.training_run_id))
        try:
            return super().run(db, training_run)
        finally:
            with self._stream_lock:
                self._stream.append(("train_finish", training_run.training_run_id))


class _StubCoordinator:
    """Wraps a `serving_cycle` with an injected mock control + stub VRAM."""

    def __init__(self, control, vram, threshold_mb=8192, timeout=5.0, poll=0.01):
        self.control = control
        self.vram = vram
        self.threshold_mb = threshold_mb
        self.timeout = timeout
        self.poll = poll

    @contextlib.contextmanager
    def cycle(self):
        with serving_cycle(
            control=self.control,
            vram=self.vram,
            threshold_mb=self.threshold_mb,
            timeout=self.timeout,
            poll=self.poll,
        ):
            yield


def _queued_training_run(db_session):
    from app.schemas.dataset import DatasetVersionCreateRequest
    from app.schemas.training import TrainingConfig, TrainingRunCreateRequest
    from app.services import dataset_service

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


@pytest.fixture
def lock_file(tmp_path):
    return str(tmp_path / "gpu.lock")


# ── serving_cycle unit tests ─────────────────────────────────────────────


def test_cycle_stop_wait_vram_train_restart_order():
    """Required: normal cycle — stop serving, verify VRAM free, train, restart serving.
    Asserts the order of calls, not just the final result."""
    control = MockServingControl()
    vram = StubVRAMReader(free_mb=16_000)

    with serving_cycle(
        control=control, vram=vram, threshold_mb=8192, timeout=5.0, poll=0.01
    ):
        assert control.events == ["stop"]

    assert control.events == ["stop", "start", "health_check"]
    assert vram.reads >= 1


def test_cycle_restarts_serving_when_training_raises():
    """Required edge case: training throws -> serving is still restarted."""
    control = MockServingControl()
    vram = StubVRAMReader(free_mb=16_000)

    with pytest.raises(RuntimeError, match="oom"):
        with serving_cycle(
            control=control, vram=vram, threshold_mb=8192, timeout=5.0, poll=0.01
        ):
            raise RuntimeError("oom")

    assert control.events == ["stop", "start", "health_check"]


def test_cycle_vram_timeout_raises_without_training_and_restarts():
    """Required edge case: VRAM never free until timeout -> training not started,
    serving restarted, explicit final state (VRAMNotFree)."""
    control = MockServingControl()
    vram = StubVRAMReader(free_mb=1000)  # forever below threshold

    with pytest.raises(VRAMNotFree, match="did not reach"):
        with serving_cycle(
            control=control,
            vram=vram,
            threshold_mb=8192,
            timeout=0.05,
            poll=0.01,
        ):
            pytest.fail("training body must not run when VRAM is never free")

    assert control.events == ["stop", "start", "health_check"]
    assert vram.reads >= 1


def test_cycle_stop_failure_raises_before_training_and_restarts():
    """Required edge case: stop serving fails -> training never starts; system brought
    back to a consistent (serving-restarted) state, not left half-way."""
    control = MockServingControl()
    control.stop_error = RuntimeError("cannot stop vllm")
    vram = StubVRAMReader(free_mb=16_000)

    with pytest.raises(ServingStopFailed, match="cannot stop vllm"):
        with serving_cycle(
            control=control, vram=vram, threshold_mb=8192, timeout=5.0, poll=0.01
        ):
            pytest.fail("training body must not run when serving stop fails")

    assert control.events == ["stop", "start", "health_check"]
    assert vram.reads == 0  # VRAM is never consulted after a failed stop


# ── process_next_job integration (serving orchestration inside the GPU lock) ──


def test_process_next_job_full_service_cycle_order(db_session, lock_file):
    """Required: full worker cycle with stubbed serving control + VRAM. Asserts the
    stop -> verify-VRAM -> train -> start sequence happened in order."""
    queued = _queued_training_run(db_session)
    runner = _StubRunner(artifact_uri=_staging_dir(queued.training_run_id))
    control = MockServingControl()
    vram = StubVRAMReader(free_mb=16_000)
    coordinator = _StubCoordinator(control, vram)

    processed = process_next_job(
        db_session, runner, lock_file=lock_file, coordinator=coordinator
    )

    assert processed.status == "COMPLETED"
    assert runner.calls == [processed.training_run_id]
    assert control.events == ["stop", "start", "health_check"]
    assert vram.reads >= 1


def test_process_next_job_vram_never_free_leaves_run_pending_and_restarts(
    db_session, lock_file
):
    """Required edge case: VRAM not free by deadline -> training NOT started, serving
    restarted, run stays PENDING (requeued) with an explicit, recorded reason."""
    queued = _queued_training_run(db_session)
    runner = _StubRunner(artifact_uri=_staging_dir("vram-never-free"))
    control = MockServingControl()
    vram = StubVRAMReader(free_mb=500)  # never free
    coordinator = _StubCoordinator(control, vram, timeout=0.05)

    skipped = process_next_job(
        db_session, runner, lock_file=lock_file, coordinator=coordinator
    )

    assert skipped is None
    assert runner.calls == []  # training did not start
    assert control.events == ["stop", "start", "health_check"]  # serving restored
    assert (
        training_service.get_training_run(db_session, queued.training_run_id).status
        == "PENDING"
    )


def test_process_next_job_stop_failure_leaves_run_pending(db_session, lock_file):
    """Required edge case: serving stop fails -> training NOT started, serving restored,
    run stays PENDING (system not left half-way)."""
    queued = _queued_training_run(db_session)
    runner = _StubRunner(artifact_uri=_staging_dir("stop-failure"))
    control = MockServingControl()
    control.stop_error = RuntimeError("vllm busy")
    vram = StubVRAMReader(free_mb=16_000)
    coordinator = _StubCoordinator(control, vram)

    skipped = process_next_job(
        db_session, runner, lock_file=lock_file, coordinator=coordinator
    )

    assert skipped is None
    assert runner.calls == []
    assert control.events == ["stop", "start", "health_check"]
    assert (
        training_service.get_training_run(db_session, queued.training_run_id).status
        == "PENDING"
    )


def test_process_next_job_restart_on_runner_exception(db_session, lock_file):
    """Required edge case: training raises -> serving still restarted (stub start called),
    run ends FAILED (the #33 semantics preserved)."""
    _queued_training_run(db_session)
    runner = _StubRunner(error=RuntimeError("cuda oom"))
    control = MockServingControl()
    vram = StubVRAMReader(free_mb=16_000)
    coordinator = _StubCoordinator(control, vram)

    processed = process_next_job(
        db_session, runner, lock_file=lock_file, coordinator=coordinator
    )

    assert processed.status == "FAILED"
    assert processed.error_message == "cuda oom"
    assert control.events == ["stop", "start", "health_check"]


def test_concurrent_second_job_cannot_interleave_serving_into_first_cycle(tmp_path):
    """Required concurrency: a second training request arriving mid-cycle must not
    inject serving stop/start into the middle of the first cycle. Both workers
    serialize on the same GPU lock, so the second cycle only starts once the first
    has finished training AND restarted serving — the event stream proves no serving
    event ever lands between a train_start and its train_finish."""
    stream: list[tuple] = []
    stream_lock = threading.Lock()
    db_path = str(tmp_path / "concurrent.db")
    lock = str(tmp_path / "gpu.lock")

    runner = _StreamRunner(
        stream, stream_lock, artifact_uri=_staging_dir("concurrent-test")
    )
    vram = StubVRAMReader(free_mb=16_000)

    class _ThreadSafeControl(MockServingControl):
        """Records into the shared, locked stream alongside the runner markers."""

        def stop(self, *a, **kw):
            with stream_lock:
                stream.append(("stop",))
            return super().stop(*a, **kw)

        def start(self, *a, **kw):
            with stream_lock:
                stream.append(("start",))
            return super().start(*a, **kw)

        def health_check(self, *a, **kw):
            with stream_lock:
                stream.append(("health_check",))
            return super().health_check(*a, **kw)

    shared_control = _ThreadSafeControl()

    engine = create_engine(f"sqlite:///{db_path}", connect_args={"timeout": 30})
    Base.metadata.create_all(engine)
    with Session(engine) as setup:
        _queued_training_run(setup)
        _queued_training_run(setup)
        setup.commit()
    engine.dispose()

    def _shared_coordinator():
        return _StubCoordinator(shared_control, vram, timeout=0.05)

    def worker():
        eng = create_engine(f"sqlite:///{db_path}", connect_args={"timeout": 30})
        try:
            with Session(eng) as session:
                # Loop like the real run_forever: a claim loss returns None (another worker
                # won that run) but the queue may still hold runs, so re-poll until empty.
                # Bound the loop so a stuck PENDING run can't spin forever in the test.
                for _ in range(10):
                    if (
                        process_next_job(
                            session,
                            runner,
                            lock_file=lock,
                            lock_timeout=10.0,
                            coordinator=_shared_coordinator(),
                        )
                        is None
                    ):
                        break
                    session.commit()
        finally:
            eng.dispose()

    threads = [threading.Thread(target=worker), threading.Thread(target=worker)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=30)

    assert not threads[0].is_alive() and not threads[1].is_alive()
    assert len(runner.calls) == 2

    serving_events = {"stop", "start", "health_check"}
    in_training = False
    for event, *rest in stream:
        if event == "train_start":
            assert not in_training
            in_training = True
        elif event == "train_finish":
            assert in_training
            in_training = False
        else:
            assert event in serving_events, f"unexpected stream event {event!r}"
            assert not in_training, (
                f"serving event {event!r} interleaved mid-training: {stream}"
            )


def test_sigterm_mid_cycle_restarts_serving_and_exits(tmp_path):
    """Required: recovery also happens on the SIGTERM path — a worker asked to stop
    mid-cycle still restarts serving (verified via a marker the child wrote with the
    recorded serving events)."""
    marker = str(tmp_path / "serving_restarted")
    script = (
        "import pathlib, sys, time\n"
        "from app.workers.gpu_orchestrator import MockServingControl, StubVRAMReader,"
        " serving_cycle\n"
        "control = MockServingControl()\n"
        "vram = StubVRAMReader(16384)\n"
        "try:\n"
        "    with serving_cycle(control=control, vram=vram, threshold_mb=8192,"
        " timeout=60):\n"
        "        print('entered', flush=True)\n"
        "        time.sleep(30)\n"
        "except BaseException:\n"
        "    pass\n"
        "pathlib.Path(sys.argv[1]).write_text(','.join(control.events))\n"
    )
    child = subprocess.Popen(
        [sys.executable, "-c", script, marker],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        cwd=str(pathlib.Path(__file__).resolve().parents[1]),
    )
    assert child.stdout.readline().strip() == b"entered"
    child.send_signal(signal.SIGTERM)
    child.wait(timeout=15)

    assert child.returncode == 0
    # Serving was restarted on the way out: stop (entry) then start + health_check (finally).
    assert pathlib.Path(marker).read_text() == "stop,start,health_check"


def test_sigterm_mid_training_keeps_run_pending_and_restarts_serving(tmp_path):
    """Regression for the SIGTERM-through-the-full-worker path: a SIGTERM arriving
    while training is mid-flight must NOT be mistaken for a training failure (which
    would mark the run FAILED and commit it) and must NOT leave the worker spinning.
    The run rolls back to PENDING, serving is restarted, and the worker exits cleanly."""
    db_path = str(tmp_path / "sigterm_worker.db")
    marker = str(tmp_path / "outcome")
    script = """\
import contextlib, pathlib, sys, time
from unittest import mock
from sqlalchemy import create_engine, select
from sqlalchemy.orm import Session
from app.db.base import Base
from app.models.training import TrainingRun
from app.schemas.dataset import DatasetVersionCreateRequest
from app.schemas.training import TrainingConfig, TrainingRunCreateRequest
from app.services import dataset_service, training_service
from app.workers import training_worker
from app.workers.gpu_orchestrator import (
    MockServingControl, StubVRAMReader, serving_cycle,
)

db_path, marker = sys.argv[1], sys.argv[2]
engine = create_engine(f"sqlite:///{db_path}", connect_args={"timeout": 30})
Base.metadata.create_all(engine)
with Session(engine) as setup:
    dv = dataset_service.create_dataset_version(
        setup, "no_robots",
        DatasetVersionCreateRequest(
            source_type="huggingface",
            source_dataset="HuggingFaceH4/no_robots",
            source_commit_or_snapshot_date="2026-08-01",
            source_format="chatml",
        ),
    )
    training_service.create_training_run(
        setup, dv,
        TrainingRunCreateRequest(
            dataset_id="no_robots", dataset_version=1,
            model_id="qwen-sft-domain-x",
            base_model="Qwen/Qwen3.8-27B",
            training_config=TrainingConfig(),
        ),
    )
    setup.commit()

control = MockServingControl()
vram = StubVRAMReader(16384)

@contextlib.contextmanager
def coordinator_factory():
    with serving_cycle(
        control=control, vram=vram, threshold_mb=8192, timeout=60, poll=0.01,
    ):
        yield

class Coordinator:
    def cycle(self):
        return coordinator_factory()

class SlowRunner:
    def run(self, db, training_run):
        print("started", flush=True)
        time.sleep(30)
        return "file:///tmp/adapter"

def session_factory():
    return Session(engine)

with mock.patch.object(training_worker, "SessionLocal", session_factory):
    training_worker.run_forever(
        SlowRunner(), poll_interval=0.2, coordinator=Coordinator()
    )

with Session(engine) as check:
    run = check.scalar(select(TrainingRun))
    pathlib.Path(marker).write_text(f"{','.join(control.events)}|{run.status}")
"""
    child = subprocess.Popen(
        [sys.executable, "-c", script, db_path, marker],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        cwd=str(pathlib.Path(__file__).resolve().parents[1]),
    )
    assert child.stdout.readline().strip() == b"started"
    started_at = time.monotonic()
    child.send_signal(signal.SIGTERM)
    child.wait(timeout=15)

    assert child.returncode == 0
    assert time.monotonic() - started_at < 15  # exited promptly, did not keep polling
    assert pathlib.Path(marker).read_text() == "stop,start,health_check|PENDING"


# ── real-impl wiring (no GPU touched): subprocess mocked, settings driven explicitly ──


def test_shell_control_runs_configured_command_and_noops_empty(monkeypatch):
    """`ShellServingControl` runs exactly the configured command and treats an
    unconfigured (empty) command as a no-op success."""
    calls = []

    def fake_run(*args, **kwargs):
        calls.append(args)
        return types.SimpleNamespace(returncode=0)

    monkeypatch.setattr("app.workers.gpu_orchestrator.subprocess.run", fake_run)
    control = ShellServingControl(
        stop_cmd="docker stop vllm", start_cmd="", health_cmd=""
    )
    control.stop()
    control.start()  # empty command -> no subprocess call
    assert control.health_check() is True
    assert len(calls) == 1
    assert calls[0][0] == "docker stop vllm"


def test_shell_control_nonzero_returncode_raises(monkeypatch):
    """The Protocol contract — stop/start raise on failure, health_check returns False —
    is honored by the real implementation behind a failing (`returncode=1`) command."""
    monkeypatch.setattr(
        "app.workers.gpu_orchestrator.subprocess.run",
        lambda *a, **k: types.SimpleNamespace(returncode=1),
    )
    with pytest.raises(ServingStopFailed):
        ShellServingControl(stop_cmd="stop").stop()
    with pytest.raises(ServingStartFailed):
        ShellServingControl(start_cmd="start").start()
    assert ShellServingControl(health_cmd="curl /health").health_check() is False


def test_shell_control_timeout_propagates(monkeypatch):
    """A hung command must not stall the worker forever — the subprocess timeout is
    honored and the wrapped cycle turns it into a ServingStopFailed pre-flight failure."""

    def fake_run(cmd, **kwargs):
        raise subprocess.TimeoutExpired(cmd=cmd, timeout=kwargs["timeout"])

    monkeypatch.setattr("app.workers.gpu_orchestrator.subprocess.run", fake_run)
    with pytest.raises(subprocess.TimeoutExpired):
        ShellServingControl(stop_cmd="hung", timeout=1.0).stop()


def test_make_coordinator_default_is_noop():
    """`SERVING_CONTROL=mock` (default) must yield a no-op coordinator — the worker
    never touches serving, preserving pre-#39 behavior."""
    assert isinstance(make_coordinator(), NoopServingCoordinator)


def test_make_coordinator_shell_wires_real_cycle(monkeypatch):
    """`SERVING_CONTROL=shell` maps each setting to its real implementation:
    shell command control + real nvidia-smi reader into a RealServingCoordinator.
    Constructing the reader is safe (only `free_mb()` would touch the GPU)."""
    monkeypatch.setattr(settings, "serving_control", "shell")
    monkeypatch.setattr(settings, "serving_stop_cmd", "stop")
    monkeypatch.setattr(settings, "serving_start_cmd", "start")
    monkeypatch.setattr(settings, "serving_health_cmd", "health")
    monkeypatch.setattr(settings, "vram_reader", "nvidia_smi")
    monkeypatch.setattr(settings, "vram_free_threshold_mb", 10000)

    coordinator = make_coordinator()
    assert isinstance(coordinator, RealServingCoordinator)
    assert isinstance(coordinator._control, ShellServingControl)
    assert isinstance(coordinator._vram, NvidiaSmiVRAMReader)


def test_make_coordinator_shell_rejects_mock_vram(monkeypatch):
    """Shell mode with a mock VRAM reader must fail loudly, never silently fake the
    free-VRAM verification (the config validator is the first guard; this is the
    second, for programmatically-constructed settings)."""
    monkeypatch.setattr(settings, "serving_control", "shell")
    monkeypatch.setattr(settings, "vram_reader", "mock")
    with pytest.raises(ValueError, match="nvidia_smi"):
        make_coordinator()


# ── Settings validator: no half-configured safety pipeline ──


def _shell_settings(**overrides):
    base = {
        "serving_control": "shell",
        "serving_stop_cmd": "stop",
        "serving_start_cmd": "start",
        "serving_health_cmd": "health",
        "vram_reader": "nvidia_smi",
        "vram_free_threshold_mb": 4096,
    }
    base.update(overrides)
    return Settings(_env_file=None, **base)


def test_settings_shell_requires_explicit_threshold():
    """No built-in VRAM threshold default until a governance decision exists —
    shell mode refuses to start unless the operator explicitly picks a value."""
    kwargs = {
        "serving_control": "shell",
        "serving_stop_cmd": "stop",
        "serving_start_cmd": "start",
        "serving_health_cmd": "health",
        "vram_reader": "nvidia_smi",
    }
    with pytest.raises(ValidationError, match="VRAM_FREE_THRESHOLD_MB"):
        Settings(_env_file=None, **kwargs)  # threshold left at its default -> not set


def test_settings_shell_rejects_mock_vram_reader():
    with pytest.raises(ValidationError, match="VRAM_READER"):
        _shell_settings(vram_reader="mock")


def test_settings_shell_rejects_empty_command():
    with pytest.raises(ValidationError, match="SERVING_STOP_CMD"):
        _shell_settings(serving_stop_cmd="")


def test_settings_shell_accepts_fully_configured():
    configured = _shell_settings(vram_free_threshold_mb=16384)
    assert configured.serving_control == "shell"
    assert configured.vram_reader == "nvidia_smi"
    assert configured.vram_free_threshold_mb == 16384
