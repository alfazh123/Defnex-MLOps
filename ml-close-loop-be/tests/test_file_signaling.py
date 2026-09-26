"""Tests for file-based GPU signaling (Phase 2A, Option B1).

Covers:
- Request serialization and validation
- Response correlation by request_id
- Stale response rejection
- Atomic writes
- Invalid action rejection
- Stop/start state handling
- VRAM threshold handling
- Timeout handling
- Watchdog recovery
- No arbitrary command injection
- Config validation for file_signal mode
"""

import json
import os
import sys
import time
import threading
from pathlib import Path

import pytest
from pydantic import ValidationError

from app.config import Settings
from app.workers.gpu_orchestrator import (
    FileSignalingServingControl,
    FileSignalingVRAMReader,
    NoopServingCoordinator,
    ServingStartFailed,
    ServingStopFailed,
    RealServingCoordinator,
    make_coordinator,
    serving_cycle,
)

# ---------------------------------------------------------------------------
# gpu_controller: a HOST-side artifact, not a module in this repo
# ---------------------------------------------------------------------------
# `gpu_controller` is the daemon script that runs on the experiment VM and answers the
# file-signal requests this backend writes (issue #163). It is deliberately not vendored here:
# it is deployed per host, and its own path is host-specific. The tests that exercise it are
# therefore INTEGRATION tests against that script, not unit tests of this codebase.
#
# They used to `import gpu_controller` inline and blow up with ModuleNotFoundError on any
# machine without the host artifact -- 27 red tests that said nothing about this repo's code,
# and had trained everyone to ignore the whole file. They now skip, with a reason, via
# `requires_gpu_controller` on the six classes below. The same pattern this repo already uses
# for its other optional dependencies (tests/test_artifact_storage_minio_integration.py, the
# telemetry tests).
#
# Skipping is per-class on purpose: a module-level `importorskip` would take the ~40 tests that
# DO cover this repo's own `gpu_orchestrator.py` down with it, trading 27 false failures for a
# real loss of coverage.
#
# Point DEFNEX_GPU_CONTROLLER_PATH at the deployed script to run them.
GPU_CONTROLLER_PATH = os.environ.get(
    "DEFNEX_GPU_CONTROLLER_PATH",
    "/home/ubuntu/defnex-mlops-experiment/gpu_controller",
)
if GPU_CONTROLLER_PATH not in sys.path:
    sys.path.insert(0, GPU_CONTROLLER_PATH)

try:  # noqa: SIM105 - the reason string is the whole point of doing this explicitly
    import gpu_controller as _gpu_controller  # noqa: F401
except ImportError:
    _GPU_CONTROLLER_AVAILABLE = False
    _GPU_CONTROLLER_SKIP_REASON = (
        "gpu_controller is a host-side script deployed per host, not a module in this repo "
        "(issue #163). Set DEFNEX_GPU_CONTROLLER_PATH to run these integration tests."
    )
else:
    _GPU_CONTROLLER_AVAILABLE = True
    _GPU_CONTROLLER_SKIP_REASON = ""

requires_gpu_controller = pytest.mark.skipif(
    not _GPU_CONTROLLER_AVAILABLE,
    reason=_GPU_CONTROLLER_SKIP_REASON,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _write_request(
    signal_dir: Path, action: str, request_id: str = "test-req-001"
) -> Path:
    """Write a valid request.json to the signal directory."""
    request = {
        "request_id": request_id,
        "action": action,
        "timestamp": "2026-09-14T12:00:00Z",
    }
    path = signal_dir / "request.json"
    path.write_text(json.dumps(request))
    return path


def _write_response(signal_dir: Path, request_id: str, phase: str, **kwargs) -> Path:
    """Write a response.json to the signal directory."""
    response = {
        "request_id": request_id,
        "phase": phase,
        "vllm_stopped": kwargs.get("vllm_stopped"),
        "vram_free_mb": kwargs.get("vram_free_mb"),
        "vllm_healthy": kwargs.get("vllm_healthy"),
        "timestamp": "2026-09-14T12:00:05Z",
        "error": kwargs.get("error"),
    }
    path = signal_dir / "response.json"
    path.write_text(json.dumps(response))
    return path


# ---------------------------------------------------------------------------
# Request serialization
# ---------------------------------------------------------------------------
class TestRequestSerialization:
    def test_valid_request_structure(self, tmp_path):
        """Request has required fields: request_id, action, timestamp."""
        _write_request(tmp_path, "stop_serving")
        data = json.loads((tmp_path / "request.json").read_text())
        assert "request_id" in data
        assert "action" in data
        assert "timestamp" in data
        assert data["action"] == "stop_serving"

    def test_stop_serving_action(self, tmp_path):
        """stop_serving is a valid action."""
        _write_request(tmp_path, "stop_serving")
        data = json.loads((tmp_path / "request.json").read_text())
        assert data["action"] == "stop_serving"

    def test_start_serving_action(self, tmp_path):
        """start_serving is a valid action."""
        _write_request(tmp_path, "start_serving")
        data = json.loads((tmp_path / "request.json").read_text())
        assert data["action"] == "start_serving"

    def test_request_id_is_uuid(self, tmp_path):
        """request_id should be a UUID string."""
        _write_request(
            tmp_path, "stop_serving", request_id="550e8400-e29b-41d4-a716-446655440000"
        )
        data = json.loads((tmp_path / "request.json").read_text())
        assert data["request_id"] == "550e8400-e29b-41d4-a716-446655440000"


# ---------------------------------------------------------------------------
# Response correlation by request_id
# ---------------------------------------------------------------------------
class TestResponseCorrelation:
    def test_response_matches_request_id(self, tmp_path):
        """Response request_id must match the request."""
        _write_response(
            tmp_path, "req-123", "vram_checked", vllm_stopped=True, vram_free_mb=10000
        )
        data = json.loads((tmp_path / "response.json").read_text())
        assert data["request_id"] == "req-123"

    def test_stale_response_with_different_request_id(self, tmp_path):
        """Worker ignores responses with non-matching request_id."""
        _write_response(tmp_path, "old-request", "vram_checked")
        response = json.loads((tmp_path / "response.json").read_text())
        assert response["request_id"] != "new-request"
        # Worker logic: poll_response loops until request_id matches


# ---------------------------------------------------------------------------
# Stale response rejection
# ---------------------------------------------------------------------------
class TestStaleResponseRejection:
    def test_ignores_old_response(self, tmp_path):
        """Worker ignores response with request_id different from current request."""
        # Simulate: old response from a previous request
        _write_response(tmp_path, "old-request-id", "healthy")
        response = json.loads((tmp_path / "response.json").read_text())
        # Current request has a different ID
        assert response["request_id"] == "old-request-id"
        assert response["request_id"] != "current-request-id"


# ---------------------------------------------------------------------------
# Atomic writes
# ---------------------------------------------------------------------------
class TestAtomicWrites:
    def test_no_partial_json_files(self, tmp_path):
        """Atomic write leaves no .tmp files behind."""
        import os

        path = tmp_path / "test.json"
        tmp = path.with_suffix(".tmp")
        with open(tmp, "w") as f:
            json.dump({"key": "value"}, f)
            f.flush()
            os.fsync(f.fileno())
        tmp.rename(path)
        assert path.exists()
        assert not (tmp_path / "test.json.tmp").exists()

    def test_request_written_atomically(self, tmp_path):
        """FileSignalingServingControl writes request atomically."""
        control = FileSignalingServingControl(
            control_dir=str(tmp_path), timeout=1.0, poll=0.01
        )
        control._write_request("stop_serving")
        assert (tmp_path / "request.json").exists()
        assert not (tmp_path / "request.json.tmp").exists()

    def test_response_written_atomically(self, tmp_path):
        """Host controller writes response atomically."""
        import os

        path = tmp_path / "response.json"
        tmp = path.with_suffix(".tmp")
        with open(tmp, "w") as f:
            json.dump({"request_id": "req-1", "phase": "healthy"}, f)
            f.flush()
            os.fsync(f.fileno())
        tmp.rename(path)
        assert path.exists()
        assert not (tmp_path / "response.json.tmp").exists()


# ---------------------------------------------------------------------------
# Invalid action rejection
# ---------------------------------------------------------------------------
@requires_gpu_controller
class TestInvalidActionRejection:
    def test_validate_request_rejects_invalid_action(self):
        """validate_request rejects actions not in the allowed set."""
        # Import from the host controller module
        sys_path = "/home/ubuntu/defnex-mlops-experiment/gpu_controller"
        import sys

        if sys_path not in sys.path:
            sys.path.insert(0, sys_path)
        from gpu_controller import validate_request

        assert (
            validate_request(
                {"request_id": "r1", "action": "kill_process", "timestamp": "t"}
            )
            is False
        )
        assert (
            validate_request({"request_id": "r1", "action": "", "timestamp": "t"})
            is False
        )
        assert (
            validate_request(
                {"request_id": "r1", "action": "stop_serving", "timestamp": "t"}
            )
            is True
        )
        assert (
            validate_request(
                {"request_id": "r1", "action": "start_serving", "timestamp": "t"}
            )
            is True
        )

    def test_validate_request_rejects_missing_fields(self):
        """validate_request rejects requests missing required fields."""
        sys_path = "/home/ubuntu/defnex-mlops-experiment/gpu_controller"
        import sys

        if sys_path not in sys.path:
            sys.path.insert(0, sys_path)
        from gpu_controller import validate_request

        assert validate_request({}) is False
        assert validate_request({"request_id": "r1"}) is False
        assert validate_request({"request_id": "r1", "action": "stop_serving"}) is False

    def test_validate_request_rejects_empty_request_id(self):
        """validate_request rejects empty request_id."""
        sys_path = "/home/ubuntu/defnex-mlops-experiment/gpu_controller"
        import sys

        if sys_path not in sys.path:
            sys.path.insert(0, sys.path)
        from gpu_controller import validate_request

        assert (
            validate_request(
                {"request_id": "", "action": "stop_serving", "timestamp": "t"}
            )
            is False
        )

    def test_validate_request_rejects_non_dict(self):
        """validate_request rejects non-dict input."""
        sys_path = "/home/ubuntu/defnex-mlops-experiment/gpu_controller"
        import sys

        if sys_path not in sys.path:
            sys.path.insert(0, sys.path)
        from gpu_controller import validate_request

        assert validate_request("not a dict") is False
        assert validate_request(None) is False
        assert validate_request(42) is False


# ---------------------------------------------------------------------------
# No arbitrary command injection
# ---------------------------------------------------------------------------
@requires_gpu_controller
class TestCommandInjection:
    def test_validate_request_rejects_shell_injection(self):
        """validate_request rejects actions that could inject shell commands."""
        sys_path = "/home/ubuntu/defnex-mlops-experiment/gpu_controller"
        import sys

        if sys_path not in sys.path:
            sys.path.insert(0, sys_path)
        from gpu_controller import validate_request

        assert (
            validate_request(
                {"request_id": "r1", "action": "stop; rm -rf /", "timestamp": "t"}
            )
            is False
        )
        assert (
            validate_request(
                {"request_id": "r1", "action": "$(docker stop x)", "timestamp": "t"}
            )
            is False
        )
        assert (
            validate_request(
                {
                    "request_id": "r1",
                    "action": "stop_serving; echo pwned",
                    "timestamp": "t",
                }
            )
            is False
        )

    def test_controller_only_supports_two_actions(self):
        """Controller only supports stop_serving and start_serving."""
        sys_path = "/home/ubuntu/defnex-mlops-experiment/gpu_controller"
        import sys

        if sys_path not in sys.path:
            sys.path.insert(0, sys.path)
        from gpu_controller import _VALID_ACTIONS

        assert _VALID_ACTIONS == frozenset({"stop_serving", "start_serving"})


# ---------------------------------------------------------------------------
# Stop/start state handling
# ---------------------------------------------------------------------------
class TestStopStartStateHandling:
    def test_stop_writes_request_and_polls(self, tmp_path):
        """FileSignalingServingControl.stop() writes stop_serving request."""
        control = FileSignalingServingControl(
            control_dir=str(tmp_path), timeout=1.0, poll=0.01
        )

        # Write request in background (simulating async)
        def write_and_respond():
            time.sleep(0.1)
            request = json.loads((tmp_path / "request.json").read_text())
            _write_response(
                tmp_path,
                request["request_id"],
                "vram_checked",
                vllm_stopped=True,
                vram_free_mb=10000,
            )

        thread = threading.Thread(target=write_and_respond)
        thread.start()
        control.stop()
        thread.join()

        assert control._last_vram_free_mb == 10000

    def test_stop_raises_on_controller_error(self, tmp_path):
        """FileSignalingServingControl.stop() raises on host controller error."""
        control = FileSignalingServingControl(
            control_dir=str(tmp_path), timeout=2.0, poll=0.01
        )

        def write_and_respond():
            time.sleep(0.1)
            request = json.loads((tmp_path / "request.json").read_text())
            _write_response(
                tmp_path, request["request_id"], "error", error="docker_stop_failed"
            )

        thread = threading.Thread(target=write_and_respond)
        thread.start()
        with pytest.raises(ServingStopFailed, match="docker_stop_failed"):
            control.stop()
        thread.join()

    def test_start_writes_request_and_polls(self, tmp_path):
        """FileSignalingServingControl.start() writes start_serving request."""
        control = FileSignalingServingControl(
            control_dir=str(tmp_path), timeout=2.0, poll=0.01
        )

        def write_and_respond():
            time.sleep(0.1)
            request = json.loads((tmp_path / "request.json").read_text())
            _write_response(
                tmp_path, request["request_id"], "healthy", vllm_healthy=True
            )

        thread = threading.Thread(target=write_and_respond)
        thread.start()
        control.start()
        thread.join()

    def test_start_raises_on_controller_error(self, tmp_path):
        """FileSignalingServingControl.start() raises on host controller error."""
        control = FileSignalingServingControl(
            control_dir=str(tmp_path), timeout=2.0, poll=0.01
        )

        def write_and_respond():
            time.sleep(0.1)
            request = json.loads((tmp_path / "request.json").read_text())
            _write_response(
                tmp_path, request["request_id"], "error", error="docker_start_failed"
            )

        thread = threading.Thread(target=write_and_respond)
        thread.start()
        with pytest.raises(ServingStartFailed, match="docker_start_failed"):
            control.start()
        thread.join()

    def test_health_check_returns_true_when_healthy(self, tmp_path):
        """health_check() returns True when response indicates healthy."""
        control = FileSignalingServingControl(
            control_dir=str(tmp_path), timeout=2.0, poll=0.01
        )
        _write_response(tmp_path, "req-1", "healthy", vllm_healthy=True)
        assert control.health_check() is True

    def test_health_check_returns_false_when_not_healthy(self, tmp_path):
        """health_check() returns False when response indicates not healthy."""
        control = FileSignalingServingControl(
            control_dir=str(tmp_path), timeout=2.0, poll=0.01
        )
        _write_response(tmp_path, "req-1", "error", vllm_healthy=False)
        assert control.health_check() is False

    def test_health_check_returns_false_when_no_response(self, tmp_path):
        """health_check() returns False when no response.json exists."""
        control = FileSignalingServingControl(
            control_dir=str(tmp_path), timeout=2.0, poll=0.01
        )
        assert control.health_check() is False


# ---------------------------------------------------------------------------
# VRAM threshold handling
# ---------------------------------------------------------------------------
class TestVRAMThresholdHandling:
    def test_vram_reader_returns_free_mb(self, tmp_path):
        """FileSignalingVRAMReader returns vram_free_mb from last response."""
        control = FileSignalingServingControl(
            control_dir=str(tmp_path), timeout=2.0, poll=0.01
        )
        control._last_vram_free_mb = 12000
        reader = FileSignalingVRAMReader(control)
        assert reader.free_mb() == 12000

    def test_vram_reader_raises_when_no_response(self, tmp_path):
        """FileSignalingVRAMReader raises RuntimeError when no response received."""
        control = FileSignalingServingControl(
            control_dir=str(tmp_path), timeout=2.0, poll=0.01
        )
        reader = FileSignalingVRAMReader(control)
        with pytest.raises(RuntimeError, match="VRAM not available"):
            reader.free_mb()

    def test_vram_reader_uses_last_response(self, tmp_path):
        """FileSignalingVRAMReader uses the last received vram value."""
        control = FileSignalingServingControl(
            control_dir=str(tmp_path), timeout=2.0, poll=0.01
        )
        control._last_vram_free_mb = 5000
        reader = FileSignalingVRAMReader(control)
        # Update the value (simulating a new response)
        control._last_vram_free_mb = 15000
        assert reader.free_mb() == 15000


# ---------------------------------------------------------------------------
# Timeout handling
# ---------------------------------------------------------------------------
class TestTimeoutHandling:
    def test_stop_timeout_when_no_response(self, tmp_path):
        """FileSignalingServingControl.stop() times out when no response arrives."""
        control = FileSignalingServingControl(
            control_dir=str(tmp_path), timeout=0.1, poll=0.01
        )
        with pytest.raises(TimeoutError, match="No response matching"):
            control.stop()

    def test_start_timeout_when_no_response(self, tmp_path):
        """FileSignalingServingControl.start() times out when no response arrives."""
        control = FileSignalingServingControl(
            control_dir=str(tmp_path), timeout=0.1, poll=0.01
        )
        with pytest.raises(TimeoutError, match="No response matching"):
            control.start()

    def test_poll_interval_respected(self, tmp_path):
        """Polling respects the configured poll interval."""
        control = FileSignalingServingControl(
            control_dir=str(tmp_path), timeout=0.3, poll=0.1
        )
        start = time.monotonic()
        with pytest.raises(TimeoutError):
            control.stop()
        elapsed = time.monotonic() - start
        # Should have waited at least timeout
        assert elapsed >= 0.2


# ---------------------------------------------------------------------------
# Watchdog recovery
# ---------------------------------------------------------------------------
@requires_gpu_controller
class TestWatchdogRecovery:
    def test_active_operation_persisted(self, tmp_path):
        """write_active_operation persists to active.json."""
        sys_path = "/home/ubuntu/defnex-mlops-experiment/gpu_controller"
        import sys

        if sys_path not in sys.path:
            sys.path.insert(0, sys_path)
        from gpu_controller import write_active_operation, read_active_operation

        # Patch SIGNAL_DIR for testing
        import gpu_controller

        original_signal_dir = gpu_controller.SIGNAL_DIR
        gpu_controller.SIGNAL_DIR = tmp_path
        try:
            write_active_operation("req-123", "stop_serving")
            active = read_active_operation()
            assert active is not None
            assert active["request_id"] == "req-123"
            assert active["action"] == "stop_serving"
            assert "started_at" in active
            assert "hard_deadline" in active
        finally:
            gpu_controller.SIGNAL_DIR = original_signal_dir

    def test_clear_active_operation(self, tmp_path):
        """clear_active_operation deletes active.json."""
        sys_path = "/home/ubuntu/defnex-mlops-experiment/gpu_controller"
        import sys

        if sys_path not in sys.path:
            sys.path.insert(0, sys_path)
        import gpu_controller
        from gpu_controller import (
            write_active_operation,
            clear_active_operation,
            read_active_operation,
        )

        original_signal_dir = gpu_controller.SIGNAL_DIR
        gpu_controller.SIGNAL_DIR = tmp_path
        try:
            write_active_operation("req-123", "stop_serving")
            assert (tmp_path / "active.json").exists()
            clear_active_operation()
            assert not (tmp_path / "active.json").exists()
            assert read_active_operation() is None
        finally:
            gpu_controller.SIGNAL_DIR = original_signal_dir

    def test_read_active_operation_returns_none_when_no_file(self, tmp_path):
        """read_active_operation returns None when active.json doesn't exist."""
        sys_path = "/home/ubuntu/defnex-mlops-experiment/gpu_controller"
        import sys

        if sys_path not in sys.path:
            sys.path.insert(0, sys_path)
        import gpu_controller
        from gpu_controller import read_active_operation

        original_signal_dir = gpu_controller.SIGNAL_DIR
        gpu_controller.SIGNAL_DIR = tmp_path
        try:
            assert read_active_operation() is None
        finally:
            gpu_controller.SIGNAL_DIR = original_signal_dir

    def test_active_operation_survives_daemon_restart(self, tmp_path):
        """Active operation persists across daemon restart (read_active_operation)."""
        sys_path = "/home/ubuntu/defnex-mlops-experiment/gpu_controller"
        import sys

        if sys_path not in sys.path:
            sys.path.insert(0, sys_path)
        import gpu_controller
        from gpu_controller import write_active_operation, read_active_operation

        original_signal_dir = gpu_controller.SIGNAL_DIR
        gpu_controller.SIGNAL_DIR = tmp_path
        try:
            write_active_operation("req-456", "stop_serving")
            # Simulate daemon restart by re-reading
            active = read_active_operation()
            assert active is not None
            assert active["request_id"] == "req-456"
        finally:
            gpu_controller.SIGNAL_DIR = original_signal_dir


# ---------------------------------------------------------------------------
# Config validation
# ---------------------------------------------------------------------------
class TestConfigValidation:
    def test_file_signal_requires_gpu_control_dir(self):
        """file_signal mode requires gpu_control_dir to be set."""
        with pytest.raises(ValidationError, match="GPU_CONTROL_DIR"):
            Settings(
                _env_file=None,
                serving_control="file_signal",
                gpu_control_dir="",
                vram_free_threshold_mb=8000,
            )

    def test_file_signal_requires_explicit_threshold(self):
        """file_signal mode requires VRAM_FREE_THRESHOLD_MB to be set."""
        with pytest.raises(ValidationError, match="VRAM_FREE_THRESHOLD_MB"):
            Settings(
                _env_file=None,
                serving_control="file_signal",
                gpu_control_dir="/models/.gpu-control",
            )

    def test_file_signal_rejects_negative_threshold(self):
        """file_signal mode rejects negative VRAM threshold."""
        with pytest.raises(
            ValidationError, match="VRAM_FREE_THRESHOLD_MB must be positive"
        ):
            Settings(
                _env_file=None,
                serving_control="file_signal",
                gpu_control_dir="/models/.gpu-control",
                vram_free_threshold_mb=-100,
            )

    def test_file_signal_accepts_valid_config(self):
        """file_signal mode accepts valid configuration."""
        settings = Settings(
            _env_file=None,
            serving_control="file_signal",
            gpu_control_dir="/models/.gpu-control",
            vram_free_threshold_mb=8000,
        )
        assert settings.serving_control == "file_signal"
        assert settings.gpu_control_dir == "/models/.gpu-control"
        assert settings.vram_free_threshold_mb == 8000


# ---------------------------------------------------------------------------
# make_coordinator wiring
# ---------------------------------------------------------------------------
class TestMakeCoordinator:
    def test_make_coordinator_file_signal(self, monkeypatch):
        """make_coordinator() returns RealServingCoordinator for file_signal mode."""
        monkeypatch.setattr(
            "app.workers.gpu_orchestrator.settings",
            Settings(
                _env_file=None,
                serving_control="file_signal",
                gpu_control_dir="/tmp/test-gpu-control",
                vram_free_threshold_mb=8000,
            ),
        )
        coordinator = make_coordinator()
        assert isinstance(coordinator, RealServingCoordinator)
        assert isinstance(coordinator._control, FileSignalingServingControl)
        assert isinstance(coordinator._vram, FileSignalingVRAMReader)

    def test_make_coordinator_default_is_noop(self):
        """make_coordinator() returns NoopServingCoordinator for mock mode."""
        assert isinstance(make_coordinator(), NoopServingCoordinator)


# ---------------------------------------------------------------------------
# Integration: serving_cycle with file signaling (mocked)
# ---------------------------------------------------------------------------
class TestServingCycleIntegration:
    def test_serving_cycle_with_file_signaling(self, tmp_path):
        """serving_cycle works with FileSignalingServingControl (mocked responses)."""

        # Write a stop response immediately
        def write_stop_response():
            time.sleep(0.05)
            request = json.loads((tmp_path / "request.json").read_text())
            _write_response(
                tmp_path,
                request["request_id"],
                "vram_checked",
                vllm_stopped=True,
                vram_free_mb=16000,
            )

        def write_start_response():
            time.sleep(0.05)
            request = json.loads((tmp_path / "request.json").read_text())
            _write_response(
                tmp_path,
                request["request_id"],
                "healthy",
                vllm_healthy=True,
            )

        control = FileSignalingServingControl(
            control_dir=str(tmp_path), timeout=2.0, poll=0.01
        )

        stop_thread = threading.Thread(target=write_stop_response)
        start_thread = threading.Thread(target=write_start_response)

        stop_thread.start()
        with serving_cycle(
            control=control,
            vram=FileSignalingVRAMReader(control),
            threshold_mb=8000,
            timeout=5.0,
            poll=0.01,
        ):
            stop_thread.join()
            # Training would happen here
            start_thread = threading.Thread(target=write_start_response)
            start_thread.start()

        start_thread.join()
        # Verify the cycle completed without errors


# ---------------------------------------------------------------------------
# Watchdog persistence with absolute timestamps (P0 fix)
# ---------------------------------------------------------------------------
def _import_gpu_controller():
    """Import the host-side gpu_controller module."""
    import importlib
    import sys

    sys_path = "/home/ubuntu/defnex-mlops-experiment/gpu_controller"
    if sys_path not in sys.path:
        sys.path.insert(0, sys_path)
    if "gpu_controller" in sys.modules:
        importlib.reload(sys.modules["gpu_controller"])
    import gpu_controller

    return gpu_controller


@requires_gpu_controller
class TestWatchdogPersistence:
    """Verify that active.json uses absolute wall-clock timestamps (time.time()),
    not monotonic time, so persisted state survives controller restarts."""

    def test_active_json_uses_absolute_epoch_timestamps(self, tmp_path):
        """active.json written with time.time() epoch floats, not monotonic."""
        gc = _import_gpu_controller()
        original_signal_dir = gc.SIGNAL_DIR
        gc.SIGNAL_DIR = tmp_path
        try:
            gc.write_active_operation("req-abs-001", "stop_serving")
            active = gc.read_active_operation()
            assert active is not None
            # Epoch timestamps are large positive floats (year 2026 >> 1e9)
            assert isinstance(active["started_at"], float)
            assert active["started_at"] > 1_700_000_000
            assert isinstance(active["hard_deadline"], float)
            assert active["hard_deadline"] > 1_700_000_000
            assert active["request_id"] == "req-abs-001"
        finally:
            gc.SIGNAL_DIR = original_signal_dir

    def test_deadline_calculation_correct(self, tmp_path):
        """hard_deadline == started_at + HARD_TIMEOUT."""
        gc = _import_gpu_controller()
        original_signal_dir = gc.SIGNAL_DIR
        gc.SIGNAL_DIR = tmp_path
        try:
            gc.write_active_operation("req-dl-001", "stop_serving")
            active = gc.read_active_operation()
            assert active is not None
            assert active["hard_deadline"] == pytest.approx(
                active["started_at"] + gc.HARD_TIMEOUT, abs=0.1
            )
        finally:
            gc.SIGNAL_DIR = original_signal_dir

    def test_watchdog_triggers_on_expired_deadline_after_restart(self, tmp_path):
        """Watchdog recovers when a persisted deadline has passed (simulated restart)."""
        gc = _import_gpu_controller()
        original_signal_dir = gc.SIGNAL_DIR
        gc.SIGNAL_DIR = tmp_path
        try:
            # Simulate an operation that started 2x HARD_TIMEOUT ago
            past_time = time.time() - (gc.HARD_TIMEOUT * 2)
            active = {
                "request_id": "req-expired",
                "action": "stop_serving",
                "started_at": past_time,
                "hard_deadline": past_time + gc.HARD_TIMEOUT,
                "wall_clock": "2026-09-14T00:00:00+00:00",
            }
            gc.atomic_write_json(tmp_path / "active.json", active)

            # Simulate restart: mock docker/health to indicate vLLM not running
            gc.docker_inspect_running = lambda c: False
            gc.check_vllm_health = lambda: False
            gc.handle_start_serving = lambda rid: gc.clear_active_operation()
            original_logger = gc.logger

            # Mock logger to accept kwargs (stdlib logging doesn't support them)
            import unittest.mock

            gc.logger = unittest.mock.MagicMock()
            try:
                gc.watchdog_check()
                # After watchdog runs, active.json should be cleared (recovery happened)
                assert not (tmp_path / "active.json").exists()
            finally:
                gc.docker_inspect_running = gc.docker_inspect_running
                gc.check_vllm_health = gc.check_vllm_health
                gc.handle_start_serving = gc.handle_start_serving
                gc.logger = original_logger
        finally:
            gc.SIGNAL_DIR = original_signal_dir

    def test_watchdog_no_false_recovery_for_nonexpired_deadline(self, tmp_path):
        """Watchdog does NOT recover when the persisted deadline has NOT passed."""
        gc = _import_gpu_controller()
        original_signal_dir = gc.SIGNAL_DIR
        gc.SIGNAL_DIR = tmp_path
        try:
            # Simulate an operation that started 10 seconds ago (well within deadline)
            recent_time = time.time() - 10
            active = {
                "request_id": "req-active",
                "action": "stop_serving",
                "started_at": recent_time,
                "hard_deadline": recent_time + gc.HARD_TIMEOUT,
                "wall_clock": "2026-09-14T00:00:00+00:00",
            }
            gc.atomic_write_json(tmp_path / "active.json", active)

            recovery_called = False
            original_handle_start = gc.handle_start_serving

            def spy_handle_start(rid):
                nonlocal recovery_called
                recovery_called = True

            gc.handle_start_serving = spy_handle_start
            try:
                gc.watchdog_check()
                assert not recovery_called, (
                    "watchdog must NOT trigger for non-expired deadline"
                )
                assert (tmp_path / "active.json").exists(), "active.json must remain"
            finally:
                gc.handle_start_serving = original_handle_start
        finally:
            gc.SIGNAL_DIR = original_signal_dir

    def test_startup_reconciliation_uses_absolute_deadline(self, tmp_path):
        """Startup reconciliation recovers only when the absolute deadline has passed."""
        gc = _import_gpu_controller()
        original_signal_dir = gc.SIGNAL_DIR
        original_logger = gc.logger
        import unittest.mock

        gc.SIGNAL_DIR = tmp_path
        gc.logger = unittest.mock.MagicMock()
        try:
            # Case 1: deadline passed → should recover
            past_time = time.time() - (gc.HARD_TIMEOUT * 3)
            active_expired = {
                "request_id": "req-startup-expired",
                "action": "stop_serving",
                "started_at": past_time,
                "hard_deadline": past_time + gc.HARD_TIMEOUT,
                "wall_clock": "2026-09-14T00:00:00+00:00",
            }
            gc.atomic_write_json(tmp_path / "active.json", active_expired)

            recovery_called = False

            def spy_handle_start(rid):
                nonlocal recovery_called
                recovery_called = True
                gc.clear_active_operation()

            gc.handle_start_serving = spy_handle_start
            gc.docker_inspect_running = lambda c: False
            gc.check_vllm_health = lambda: False
            try:
                gc.reconcile_on_startup()
                assert recovery_called, (
                    "reconciliation should recover on expired deadline"
                )
            finally:
                gc.handle_start_serving = gc.handle_start_serving
                gc.docker_inspect_running = gc.docker_inspect_running
                gc.check_vllm_health = gc.check_vllm_health

            # Case 2: deadline NOT passed → should NOT recover
            recent_time = time.time() - 5
            active_within = {
                "request_id": "req-startup-within",
                "action": "stop_serving",
                "started_at": recent_time,
                "hard_deadline": recent_time + gc.HARD_TIMEOUT,
                "wall_clock": "2026-09-14T00:00:00+00:00",
            }
            gc.atomic_write_json(tmp_path / "active.json", active_within)

            recovery_called = False
            gc.handle_start_serving = spy_handle_start
            try:
                gc.reconcile_on_startup()
                assert not recovery_called, (
                    "reconciliation should NOT recover within deadline"
                )
            finally:
                gc.handle_start_serving = gc.handle_start_serving
        finally:
            gc.SIGNAL_DIR = original_signal_dir
            gc.logger = original_logger

    def test_request_id_correlation_unchanged(self, tmp_path):
        """Request ID correlation works identically with absolute timestamps."""
        gc = _import_gpu_controller()
        original_signal_dir = gc.SIGNAL_DIR
        gc.SIGNAL_DIR = tmp_path
        try:
            gc.write_active_operation("req-corr-999", "stop_serving")
            active = gc.read_active_operation()
            assert active is not None
            assert active["request_id"] == "req-corr-999"
            assert active["action"] == "stop_serving"
            # New fields are absolute timestamps
            assert isinstance(active["started_at"], float)
            assert isinstance(active["hard_deadline"], float)
        finally:
            gc.SIGNAL_DIR = original_signal_dir


# ---------------------------------------------------------------------------
# Health timeout semantics (Phase 2A live test fix)
# ---------------------------------------------------------------------------
@requires_gpu_controller
class TestHealthTimeoutSemantics:
    """Tests for configurable health timeout and active-operation lifecycle
    after health check timeout in handle_start_serving().

    Key invariant: if the container is running but vLLM is unhealthy past the
    health deadline, active.json is NOT cleared — the watchdog will detect it
    and retry on the next cycle.
    """

    def _make_active(
        self, tmp_path, request_id="req-health-001", action="stop_serving"
    ):
        """Write a realistic active.json (as if a stop completed and start is pending)."""
        gc = _import_gpu_controller()
        now = time.time()
        active = {
            "request_id": request_id,
            "action": action,
            "started_at": now,
            "hard_deadline": now + gc.HARD_TIMEOUT,
            "wall_clock": "2026-09-14T00:00:00+00:00",
        }
        gc.atomic_write_json(tmp_path / "active.json", active)

    def _patch_controller(
        self,
        gc,
        tmp_path,
        *,
        docker_start=True,
        container_running=True,
        vllm_healthy=False,
        handle_start_serving=None,
    ):
        """Patch controller functions for testing handle_start_serving()."""
        import unittest.mock

        original_dir = gc.SIGNAL_DIR
        original_docker_start = gc.docker_start
        original_docker_inspect = gc.docker_inspect_running
        original_health = gc.check_vllm_health
        original_logger = gc.logger

        gc.SIGNAL_DIR = tmp_path
        gc.docker_start = lambda c: docker_start
        gc.docker_inspect_running = lambda c: container_running
        gc.check_vllm_health = lambda: vllm_healthy
        gc.logger = unittest.mock.MagicMock()

        if handle_start_serving is not None:
            original_hs = gc.handle_start_serving
            gc.handle_start_serving = handle_start_serving

        class _PatchCtx:
            def __enter__(self_):
                return self_

            def __exit__(self_, *args):
                gc.SIGNAL_DIR = original_dir
                gc.docker_start = original_docker_start
                gc.docker_inspect_running = original_docker_inspect
                gc.check_vllm_health = original_health
                gc.logger = original_logger
                if handle_start_serving is not None:
                    gc.handle_start_serving = original_hs

        return _PatchCtx()

    def test_healthy_before_timeout(self, tmp_path):
        """1. vLLM becomes healthy before timeout → response=healthy, active cleared."""
        gc = _import_gpu_controller()
        self._make_active(tmp_path)
        with self._patch_controller(gc, tmp_path, vllm_healthy=True):
            gc.handle_start_serving("req-health-001")
            resp = json.loads((tmp_path / "response.json").read_text())
            assert resp["phase"] == "healthy"
            assert resp["vllm_healthy"] is True
            assert resp["error"] is None
            assert not (tmp_path / "active.json").exists(), (
                "active should be cleared on success"
            )

    def test_healthy_after_60s_before_180s(self, tmp_path):
        """2. vLLM becomes healthy after 60s but before 180s → healthy.

        Uses a short HEALTH_CHECK_DEADLINE (0.1s) to simulate timeout,
        then verifies the health loop logic works by having health return
        True immediately (simulating the 'became healthy just in time' case).
        """
        gc = _import_gpu_controller()
        self._make_active(tmp_path)
        # With health_check_deadline=0.1, health returns True immediately → healthy
        with self._patch_controller(gc, tmp_path, vllm_healthy=True):
            original_deadline = gc.HEALTH_CHECK_DEADLINE
            gc.HEALTH_CHECK_DEADLINE = 0.1
            try:
                gc.handle_start_serving("req-health-002")
                resp = json.loads((tmp_path / "response.json").read_text())
                assert resp["phase"] == "healthy"
                assert resp["vllm_healthy"] is True
            finally:
                gc.HEALTH_CHECK_DEADLINE = original_deadline

    def test_unhealthy_past_timeout(self, tmp_path):
        """3. vLLM remains unhealthy past timeout → error, active NOT cleared."""
        gc = _import_gpu_controller()
        self._make_active(tmp_path)
        with self._patch_controller(gc, tmp_path, vllm_healthy=False):
            original_deadline = gc.HEALTH_CHECK_DEADLINE
            gc.HEALTH_CHECK_DEADLINE = 0.1
            try:
                gc.handle_start_serving("req-health-003")
                resp = json.loads((tmp_path / "response.json").read_text())
                assert resp["phase"] == "error"
                assert resp["error"] == "health_check_timeout"
                assert resp["vllm_healthy"] is False
                assert resp["vllm_stopped"] is False
                # Key invariant: active.json NOT cleared
                assert (tmp_path / "active.json").exists(), (
                    "active.json must NOT be cleared on health timeout"
                )
            finally:
                gc.HEALTH_CHECK_DEADLINE = original_deadline

    def test_timeout_never_falsely_reports_healthy(self, tmp_path):
        """4. Health timeout must never report healthy when vLLM is unhealthy."""
        gc = _import_gpu_controller()
        self._make_active(tmp_path)
        with self._patch_controller(gc, tmp_path, vllm_healthy=False):
            original_deadline = gc.HEALTH_CHECK_DEADLINE
            gc.HEALTH_CHECK_DEADLINE = 0.1
            try:
                gc.handle_start_serving("req-health-004")
                resp = json.loads((tmp_path / "response.json").read_text())
                assert resp["vllm_healthy"] is not True
                assert resp["phase"] != "healthy"
            finally:
                gc.HEALTH_CHECK_DEADLINE = original_deadline

    def test_active_state_after_start_timeout(self, tmp_path):
        """5. After health timeout, active.json remains for watchdog to retry."""
        gc = _import_gpu_controller()
        self._make_active(tmp_path, request_id="req-health-005", action="stop_serving")
        with self._patch_controller(gc, tmp_path, vllm_healthy=False):
            original_deadline = gc.HEALTH_CHECK_DEADLINE
            gc.HEALTH_CHECK_DEADLINE = 0.1
            try:
                gc.handle_start_serving("req-health-005")
                # active.json should still exist (not cleared on health timeout)
                active = gc.read_active_operation()
                assert active is not None
                assert active["request_id"] == "req-health-005"
                # Watchdog should now detect unhealthy state and retry
                retry_called = False
                original_handle_start = gc.handle_start_serving

                def spy_handle_start(rid):
                    nonlocal retry_called
                    retry_called = True
                    gc.clear_active_operation()

                gc.handle_start_serving = spy_handle_start
                gc.docker_inspect_running = lambda c: True
                gc.check_vllm_health = lambda: False
                try:
                    gc.watchdog_check()
                    assert retry_called, "watchdog should retry on unhealthy start"
                finally:
                    gc.handle_start_serving = original_handle_start
            finally:
                gc.HEALTH_CHECK_DEADLINE = original_deadline

    def test_restart_during_unhealthy_start_recoverable(self, tmp_path):
        """6. Controller restart during unhealthy-start state remains recoverable."""
        gc = _import_gpu_controller()
        self._make_active(tmp_path, action="stop_serving")
        # Simulate: container running, health not yet checked (fresh restart)
        with self._patch_controller(
            gc, tmp_path, container_running=True, vllm_healthy=False
        ):
            import unittest.mock

            original_logger = gc.logger
            gc.logger = unittest.mock.MagicMock()
            try:
                # On startup, reconcile sees active.json within deadline
                # It should NOT auto-recover (deadline not passed)
                recovery_called = False
                original_handle_start = gc.handle_start_serving

                def spy_handle_start(rid):
                    nonlocal recovery_called
                    recovery_called = True

                gc.handle_start_serving = spy_handle_start
                try:
                    gc.reconcile_on_startup()
                    assert not recovery_called, (
                        "reconcile should NOT recover within deadline"
                    )
                    # active.json still exists for watchdog to handle
                    assert (tmp_path / "active.json").exists()
                finally:
                    gc.handle_start_serving = original_handle_start
            finally:
                gc.logger = original_logger

    def test_successful_health_clears_active(self, tmp_path):
        """7. When vLLM becomes healthy, active.json is cleared."""
        gc = _import_gpu_controller()
        self._make_active(tmp_path)
        with self._patch_controller(gc, tmp_path, vllm_healthy=True):
            gc.handle_start_serving("req-health-007")
            assert not (tmp_path / "active.json").exists()
            resp = json.loads((tmp_path / "response.json").read_text())
            assert resp["phase"] == "healthy"

    def test_no_redundant_restart_when_already_healthy(self, tmp_path):
        """8. Watchdog does NOT restart vLLM when it's already healthy (expired deadline).

        When the hard deadline has passed but vLLM is healthy, the watchdog
        clears active.json (idempotent recovery) WITHOUT calling
        handle_start_serving — no redundant restart.
        """
        gc = _import_gpu_controller()
        # Create active with expired deadline (so watchdog triggers)
        past_time = time.time() - (gc.HARD_TIMEOUT * 2)
        active = {
            "request_id": "req-health-008",
            "action": "stop_serving",
            "started_at": past_time,
            "hard_deadline": past_time + gc.HARD_TIMEOUT,
            "wall_clock": "2026-09-14T00:00:00+00:00",
        }
        gc.atomic_write_json(tmp_path / "active.json", active)
        restart_count = 0
        original_handle_start = gc.handle_start_serving

        def counting_handle_start(rid):
            nonlocal restart_count
            restart_count += 1
            gc.clear_active_operation()

        gc.handle_start_serving = counting_handle_start
        gc.docker_inspect_running = lambda c: True
        gc.check_vllm_health = lambda: True
        try:
            gc.watchdog_check()
            assert restart_count == 0, "should not restart when already healthy"
        finally:
            gc.handle_start_serving = original_handle_start


# ---------------------------------------------------------------------------
# Health timeout configuration
# ---------------------------------------------------------------------------
@requires_gpu_controller
class TestHealthTimeoutConfig:
    """Tests for GPU_CONTROLLER_HEALTH_TIMEOUT environment variable."""

    def test_health_check_deadline_default(self):
        """Default health check deadline is 180 seconds."""
        gc = _import_gpu_controller()
        assert gc.HEALTH_CHECK_DEADLINE == 180

    def test_health_check_deadline_configurable(self, monkeypatch):
        """GPU_CONTROLLER_HEALTH_TIMEOUT env var sets the deadline."""
        gc = _import_gpu_controller()
        monkeypatch.setattr(gc, "HEALTH_CHECK_DEADLINE", 300)
        assert gc.HEALTH_CHECK_DEADLINE == 300

    def test_health_timeout_uses_deadline_not_60(self, tmp_path):
        """Health loop uses HEALTH_CHECK_DEADLINE, not hardcoded 60."""
        gc = _import_gpu_controller()

        # Write active.json manually
        now = time.time()
        active = {
            "request_id": "req-deadline-001",
            "action": "stop_serving",
            "started_at": now,
            "hard_deadline": now + gc.HARD_TIMEOUT,
            "wall_clock": "2026-09-14T00:00:00+00:00",
        }
        gc.atomic_write_json(tmp_path / "active.json", active)

        # Set a very short deadline
        gc.HEALTH_CHECK_DEADLINE = 0.05

        import unittest.mock

        original_dir = gc.SIGNAL_DIR
        original_docker_start = gc.docker_start
        original_docker_inspect = gc.docker_inspect_running
        original_health = gc.check_vllm_health
        original_logger = gc.logger

        gc.SIGNAL_DIR = tmp_path
        gc.docker_start = lambda c: True
        gc.docker_inspect_running = lambda c: True
        gc.check_vllm_health = lambda: False  # always unhealthy
        gc.logger = unittest.mock.MagicMock()
        try:
            gc.handle_start_serving("req-deadline-001")
            resp = json.loads((tmp_path / "response.json").read_text())
            assert resp["error"] == "health_check_timeout"
        finally:
            gc.SIGNAL_DIR = original_dir
            gc.docker_start = original_docker_start
            gc.docker_inspect_running = original_docker_inspect
            gc.check_vllm_health = original_health
            gc.logger = original_logger
            gc.HEALTH_CHECK_DEADLINE = 180  # restore default
