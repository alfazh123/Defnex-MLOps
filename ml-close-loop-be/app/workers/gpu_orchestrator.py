"""Cross-service GPU orchestration for training (issue #39).

Issue #33's flock only serializes *training* workers against each other; it knows
nothing about the VRAM held by the running serving process (vLLM, #40). Before
training starts, the serving service must be stopped and VRAM verified free so a
training run cannot OOM and take the serving process serving live traffic down
with it. After training (success or failure) serving must be restarted and its
health confirmed — on every exit path, including exceptions and SIGTERM.

All of this happens inside the *same* GPU lock that #33 introduced (a single
point of execution in `training_worker.process_next_job`), not a second
coordination sequence, so a second training request cannot interleave serving
stop/start commands into the middle of an in-flight cycle.

Like `TrainingRunner`, `ServingBackend` and `ArtifactStorage`, the two external
boundaries here are `Protocol`s with mock/no-op implementations so tests and
no-GPU local dev never touch a GPU. The real implementations
(`ShellServingControl`, `NvidiaSmiVRAMReader`) are only constructed when the
corresponding settings opt in (`SERVING_CONTROL=shell`, `VRAM_READER=nvidia_smi`)
and are never invoked by the test suite (project rule: don't run GPU-touching
commands unasked).
"""

from __future__ import annotations

import contextlib
import signal
import subprocess
import threading
import time
from typing import Protocol

import structlog

from app.config import settings

logger = structlog.get_logger(__name__)


class ServingControl(Protocol):
    """Stops/starts the serving service and checks whether it is serving again.

    Real implementation stops/runs/health-checks the concrete serving stack
    (`ShellServingControl`); tests and no-GPU dev use a mock that records calls.
    """

    def stop(self) -> None:
        """Stop serving so training can use the GPU. Raise on failure."""
        ...

    def start(self) -> None:
        """Restart serving after training. Raise on failure."""
        ...

    def health_check(self) -> bool:
        """True when serving is actually serving again (not merely 'process exists')."""
        ...


class VRAMReader(Protocol):
    """Reads how much GPU memory is currently free (MB).

    Real implementation shells out to `nvidia-smi`; tests supply a stub value.
    """

    def free_mb(self) -> int:
        """Free GPU memory in MB. Raise if it cannot be determined."""
        ...


class ServingCoordinator(Protocol):
    """Wraps a single training run inside the stop-serving / train / restart-serving cycle."""

    @contextlib.contextmanager
    def cycle(self):
        """Enter: serving stopped, VRAM verified free. Exit (always): serving restarted."""
        ...


class ServingStopFailed(Exception):
    """Serving could not be stopped, so the GPU cannot be made safe for training."""


class VRAMNotFree(Exception):
    """VRAM was still above the free threshold when the wait deadline passed."""


class ServingStartFailed(Exception):
    """Serving did not come back up after training ended."""


class ServingInterrupted(BaseException):
    """Internal: a SIGTERM arrived mid-cycle; unwinds to restart serving.

    `BaseException`, not `Exception`: like `KeyboardInterrupt`/`SystemExit`, it is a
    control-flow unwind, so `except Exception` blocks inside `process_next_job` (the
    runner-error handler) must NOT catch it — otherwise a SIGTERM during training would
    be mistaken for a training failure, mark the run FAILED and commit it, and the
    worker would keep polling instead of shutting down.
    """


class NoopServingCoordinator:
    """Serving orchestration disabled (`SERVING_CONTROL=mock`, the default).

    `cycle()` does nothing: the worker's pre-#39 behavior is preserved exactly —
    training starts immediately under the GPU lock and serving is never touched.
    """

    @contextlib.contextmanager
    def cycle(self):
        yield


class MockServingControl:
    """Records serving lifecycle calls so tests can assert the order of operations."""

    def __init__(self) -> None:
        self.events: list[str] = []
        self.stop_error: Exception | None = None
        self.start_error: Exception | None = None
        self.healthy: bool = True

    def stop(self) -> None:
        self.events.append("stop")
        if self.stop_error is not None:
            raise self.stop_error

    def start(self) -> None:
        self.events.append("start")
        if self.start_error is not None:
            raise self.start_error

    def health_check(self) -> bool:
        self.events.append("health_check")
        return self.healthy


class StubVRAMReader:
    """A stub VRAM reader whose reported free memory and read attempts are test-visible."""

    def __init__(self, free_mb: int) -> None:
        self._free_mb = free_mb
        self.reads = 0

    def set_free_mb(self, free_mb: int) -> None:
        self._free_mb = free_mb

    def free_mb(self) -> int:
        self.reads += 1
        return self._free_mb


@contextlib.contextmanager
def serving_cycle(
    *,
    control: ServingControl,
    vram: VRAMReader,
    threshold_mb: int,
    timeout: float,
    poll: float = 0.5,
    sleep=time.sleep,
):
    """Stop serving, wait until VRAM is free, run `yield`, then always restart serving.

    On entry: `control.stop()` (a failure raises `ServingStopFailed` before any
    training can start), then polls `vram.free_mb()` until it is `>= threshold_mb`
    (a deadline of `timeout` seconds raises `VRAMNotFree`).

    On exit — the success path, an exception from the training block, a failed
    stop, a VRAM timeout, or a SIGTERM: `control.start()` then
    `control.health_check()` run from the `finally`, so serving always comes back
    no matter how the cycle ends. A SIGTERM handler raises `ServingInterrupted` to
    unwind through that same `finally` instead of letting the process die with the
    GPU still occupied by a half-finished cycle.
    """

    # signal.signal() is only legal from the main thread; the worker also invokes this
    # cycle from test threads, where the SIGTERM unwind is unavailable from Python anyway
    # (the thread only ends when the process does). Skip registration there, not an error.
    registered = False
    if threading.current_thread() is threading.main_thread():
        previous_handler = signal.getsignal(signal.SIGTERM)

        def _on_sigterm(signum, _frame):
            # Unwind to the finally block so serving is restarted rather than leaving the
            # GPU occupied when the worker is asked to stop mid-training.
            raise ServingInterrupted("SIGTERM received during serving cycle")

        signal.signal(signal.SIGTERM, _on_sigterm)
        registered = True

    try:
        try:
            control.stop()
        except Exception as exc:  # noqa: BLE001 - surface as a domain failure, never a raw backend error
            raise ServingStopFailed(str(exc)) from exc

        deadline = time.monotonic() + timeout
        while True:
            try:
                free = vram.free_mb()
            except Exception:  # noqa: BLE001 - a VRAM read failure must not crash the worker
                free = 0
            if free >= threshold_mb:
                break
            if time.monotonic() >= deadline:
                raise VRAMNotFree(
                    f"VRAM did not reach {threshold_mb} MB free within {timeout}s "
                    f"(last read {free} MB); training not started"
                )
            sleep(poll)

        yield
    finally:
        if registered:
            signal.signal(signal.SIGTERM, previous_handler)
        # Best-effort restart: a serving control failure on the way out is logged, never
        # allowed to mask an in-flight training outcome.
        try:
            control.start()
        except Exception:  # noqa: BLE001 - restart must not mask the cycle's outcome
            logger.warning("serving_restart_failed", exc_info=True)
        try:
            if not control.health_check():
                logger.warning("serving_health_check_failed", healthy=False)
        except Exception:  # noqa: BLE001 - health-check failure is logged, never raised
            logger.warning("serving_health_check_failed", exc_info=True)


class ShellServingControl:
    """Real serving control: stop/start/health-check via shell commands from settings.

    Active only when `SERVING_CONTROL=shell`. If a command is not configured the
    matching operation is a no-op (so an unconfigured stack fails open rather than
    blocking training on a command that was never set). Commands are run with a
    short timeout so a hung serving stop/start cannot stall the worker forever.
    """

    def __init__(
        self,
        stop_cmd: str = "",
        start_cmd: str = "",
        health_cmd: str = "",
        timeout: float = 60.0,
    ) -> None:
        self._stop_cmd = stop_cmd
        self._start_cmd = start_cmd
        self._health_cmd = health_cmd
        self._timeout = timeout

    def _run(self, cmd: str) -> bool:
        if not cmd:
            return True
        result = subprocess.run(
            cmd, shell=True, timeout=self._timeout, capture_output=True
        )
        return result.returncode == 0

    def stop(self) -> None:
        if not self._run(self._stop_cmd):
            raise ServingStopFailed(f"serving stop command failed: {self._stop_cmd!r}")

    def start(self) -> None:
        if not self._run(self._start_cmd):
            raise ServingStartFailed(
                f"serving start command failed: {self._start_cmd!r}"
            )

    def health_check(self) -> bool:
        return self._run(self._health_cmd)


class NvidiaSmiVRAMReader:
    """Reads free GPU memory by parsing `nvidia-smi --query-gpu=memory.free`.

    Takes the first GPU's line — the project targets a single shared H100, so one
    line per GPU is never interleaved in a way that matters here.
    """

    def free_mb(self) -> int:
        output = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.free", "--format=csv,noheader,nounits"],
            capture_output=True,
            text=True,
            timeout=30,
            check=True,
        )
        line = output.stdout.strip().splitlines()[0].strip()
        return int(line)


class RealServingCoordinator:
    """The production serving cycle: stop serving, verify VRAM, train, restart serving."""

    def __init__(
        self,
        control: ServingControl,
        vram: VRAMReader,
        threshold_mb: int,
        timeout: float,
        poll: float = 0.5,
    ) -> None:
        self._control = control
        self._vram = vram
        self._threshold_mb = threshold_mb
        self._timeout = timeout
        self._poll = poll

    @contextlib.contextmanager
    def cycle(self):
        with serving_cycle(
            control=self._control,
            vram=self._vram,
            threshold_mb=self._threshold_mb,
            timeout=self._timeout,
            poll=self._poll,
        ):
            yield


def make_coordinator() -> ServingCoordinator:
    """Build the worker's serving coordinator from settings.

    `SERVING_CONTROL=mock` (default) returns a `NoopServingCoordinator` that never
    touches serving — preserving the pre-#39 worker behavior for tests and no-GPU
    dev. `SERVING_CONTROL=shell` returns the real cycle wired from env vars.
    """
    if settings.serving_control == "shell":
        control: ServingControl = ShellServingControl(
            stop_cmd=settings.serving_stop_cmd,
            start_cmd=settings.serving_start_cmd,
            health_cmd=settings.serving_health_cmd,
            timeout=settings.serving_command_timeout,
        )
        if settings.vram_reader == "nvidia_smi":
            vram: VRAMReader = NvidiaSmiVRAMReader()
        else:
            vram = StubVRAMReader(free_mb=settings.vram_free_threshold_mb)
        return RealServingCoordinator(
            control=control,
            vram=vram,
            threshold_mb=settings.vram_free_threshold_mb,
            timeout=settings.vram_check_timeout,
            poll=settings.vram_check_poll,
        )
    return NoopServingCoordinator()
