# Phase 2A Implementation Plan — Option B1 File-Based Signaling

**Status:** PLAN — awaiting approval to implement
**Scope:** Phase 2A only. No real training. No defnex-vllm modifications.

---

## 1. Files Changed

### Backend codebase (`ml-close-loop-be/`)

| File | Change Type | What |
|------|------------|------|
| `app/config.py` | MODIFY | Add `"file_signal"` to `serving_control` Literal; add `gpu_control_dir`, `gpu_control_timeout`, `gpu_control_poll` settings; add `_validate_file_signal_coordination` validator |
| `app/workers/gpu_orchestrator.py` | MODIFY | Add `FileSignalingServingControl` class; add `FileSignalingVRAMReader` class; update `make_coordinator()` for `file_signal` branch |
| `tests/test_file_signaling.py` | CREATE | Unit tests for file-based signaling protocol, VRAM reader, config validation, watchdog |

### Host-side files (outside repo)

| File | Type | What |
|------|------|------|
| `/home/ubuntu/defnex-mlops-experiment/gpu_controller/gpu_controller.py` | CREATE | Host-side daemon: watches `.gpu-control/`, executes docker/nvidia-smi, writes responses, watchdog recovery |
| `/etc/systemd/system/gpu-controller.service` | CREATE | systemd unit file |

### Signal directory (created at runtime)

| Path | Purpose |
|------|---------|
| `/home/ubuntu/defnex-mlops-experiment/outputs/.gpu-control/` | Shared signal directory visible to both worker and host |

---

## 2. Host Files — Exact Content

### `gpu_controller.py` — Host-side daemon

```python
#!/usr/bin/env python3
"""Host-side GPU lifecycle controller for DEFNEX MLOps.

Watches /home/ubuntu/defnex-mlops-experiment/outputs/.gpu-control/ for
JSON request files written by the worker container, executes Docker
stop/start and nvidia-smi commands, and writes JSON response files.

Security: Only supports exactly these actions:
- stop_serving: docker stop defnex-vllm
- start_serving: docker start defnex-vllm
- No arbitrary shell commands. No PID-based management.

Watchdog: Tracks active stop-serving operations. If an operation exceeds
the hard timeout (default 1800s / 30 min), auto-restarts vLLM.
"""

import json
import os
import signal
import sys
import tempfile
import time
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

# Configuration
SIGNAL_DIR = Path("/home/ubuntu/defnex-mlops-experiment/outputs/.gpu-control")
CONTAINER_NAME = "defnex-vllm"
HARD_TIMEOUT = 1800  # 30 min max for a single stop-serving operation
POLL_INTERVAL = 0.5  # seconds between polling for new requests
HEALTH_URL = "http://localhost:8001/health"
HEALTH_TIMEOUT = 5  # seconds for health check HTTP request
STOP_TIMEOUT = 30  # seconds to wait for container to stop
START_TIMEOUT = 30  # seconds to wait for container to start

# State
active_operation = None  # {"request_id": str, "started_at": float}
running = True


def now_iso():
    return datetime.now(timezone.utc).isoformat()


def atomic_write_json(path: Path, data: dict):
    """Write JSON atomically: write to temp file, then rename."""
    tmp_path = path.with_suffix(".tmp")
    with open(tmp_path, "w") as f:
        json.dump(data, f, indent=2)
        f.flush()
        os.fsync(f.fileno())
    tmp_path.rename(path)


def read_json(path: Path) -> dict | None:
    """Read JSON file, return None on any error."""
    try:
        with open(path) as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError, OSError):
        return None


def validate_request(data: dict) -> bool:
    """Validate request schema. Only stop_serving and start_serving are allowed."""
    if not isinstance(data, dict):
        return False
    required = {"request_id", "action", "timestamp"}
    if not required.issubset(data.keys()):
        return False
    if data["action"] not in ("stop_serving", "start_serving"):
        return False
    if not isinstance(data["request_id"], str) or not data["request_id"]:
        return False
    return True


def docker_stop(container: str) -> bool:
    """Stop a Docker container by name. Returns True if stopped."""
    import subprocess
    result = subprocess.run(
        ["docker", "stop", container],
        capture_output=True, timeout=STOP_TIMEOUT + 5
    )
    return result.returncode == 0


def docker_inspect_running(container: str) -> bool | None:
    """Check if container is running. Returns None on error."""
    import subprocess
    result = subprocess.run(
        ["docker", "inspect", "--format", "{{.State.Running}}", container],
        capture_output=True, text=True, timeout=10
    )
    if result.returncode != 0:
        return None
    return result.stdout.strip().lower() == "true"


def docker_start(container: str) -> bool:
    """Start a Docker container by name. Returns True if started."""
    import subprocess
    result = subprocess.run(
        ["docker", "start", container],
        capture_output=True, timeout=START_TIMEOUT + 5
    )
    return result.returncode == 0


def query_vram_free_mb() -> int | None:
    """Query nvidia-smi for free GPU memory. Returns None on error."""
    import subprocess
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.free", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=30, check=True
        )
        line = result.stdout.strip().splitlines()[0].strip()
        return int(line)
    except Exception:
        return None


def check_vllm_health() -> bool:
    """Check if vLLM is healthy via HTTP."""
    try:
        req = urllib.request.Request(HEALTH_URL)
        with urllib.request.urlopen(req, timeout=HEALTH_TIMEOUT) as resp:
            return resp.status == 200
    except Exception:
        return False


def write_response(request_id: str, phase: str, **kwargs):
    """Write response.json atomically."""
    response = {
        "request_id": request_id,
        "phase": phase,
        "vllm_stopped": kwargs.get("vllm_stopped"),
        "vram_free_mb": kwargs.get("vram_free_mb"),
        "vllm_healthy": kwargs.get("vllm_healthy"),
        "timestamp": now_iso(),
        "error": kwargs.get("error"),
    }
    atomic_write_json(SIGNAL_DIR / "response.json", response)


def handle_stop_serving(request_id: str):
    """Handle stop_serving request: stop vLLM, wait, query VRAM."""
    global active_operation

    write_response(request_id, "stopping")

    # Stop container
    if not docker_stop(CONTAINER_NAME):
        write_response(request_id, "error", error="docker_stop_failed")
        return

    # Wait until container is actually stopped
    deadline = time.monotonic() + STOP_TIMEOUT
    while time.monotonic() < deadline:
        running = docker_inspect_running(CONTAINER_NAME)
        if running is False:
            break
        time.sleep(1)
    else:
        write_response(request_id, "error", error="container_did_not_stop")
        return

    # Record active operation for watchdog
    active_operation = {
        "request_id": request_id,
        "started_at": time.monotonic(),
    }

    # Query VRAM
    vram_free = query_vram_free_mb()
    if vram_free is None:
        write_response(request_id, "error", error="nvidia_smi_failed")
        active_operation = None
        return

    write_response(
        request_id, "vram_checked",
        vllm_stopped=True,
        vram_free_mb=vram_free,
    )


def handle_start_serving(request_id: str):
    """Handle start_serving request: start vLLM, wait, health check."""
    global active_operation

    write_response(request_id, "starting")

    # Start container
    if not docker_start(CONTAINER_NAME):
        write_response(request_id, "error", error="docker_start_failed")
        active_operation = None
        return

    # Wait until container is running
    deadline = time.monotonic() + START_TIMEOUT
    while time.monotonic() < deadline:
        running_state = docker_inspect_running(CONTAINER_NAME)
        if running_state is True:
            break
        time.sleep(1)
    else:
        write_response(request_id, "error", error="container_did_not_start")
        active_operation = None
        return

    # Health check
    health_deadline = time.monotonic() + 60  # up to 60s for vLLM to become healthy
    while time.monotonic() < health_deadline:
        if check_vllm_health():
            break
        time.sleep(2)
    else:
        write_response(request_id, "error", error="health_check_timeout",
                       vllm_stopped=False, vllm_healthy=False)
        active_operation = None
        return

    active_operation = None
    write_response(request_id, "healthy", vllm_stopped=False, vllm_healthy=True)


def watchdog_check():
    """If an active stop operation has exceeded HARD_TIMEOUT, auto-restart vLLM."""
    global active_operation

    if active_operation is None:
        return

    elapsed = time.monotonic() - active_operation["started_at"]
    if elapsed < HARD_TIMEOUT:
        return

    # Hard timeout exceeded — auto-restart
    request_id = active_operation["request_id"]
    active_operation = None
    handle_start_serving(request_id)


def process_request():
    """Read and process a single request if present."""
    request_path = SIGNAL_DIR / "request.json"
    data = read_json(request_path)

    if data is None:
        return

    if not validate_request(data):
        # Invalid request — write error response and delete
        write_response(
            data.get("request_id", "unknown"), "error",
            error="invalid_request"
        )
        try:
            request_path.unlink()
        except OSError:
            pass
        return

    request_id = data["request_id"]
    action = data["action"]

    # Delete request file before processing (prevents re-processing)
    try:
        request_path.unlink()
    except OSError:
        pass

    if action == "stop_serving":
        handle_stop_serving(request_id)
    elif action == "start_serving":
        handle_start_serving(request_id)


def write_pid_file():
    """Write PID file for watchdog/monitoring."""
    pid_path = SIGNAL_DIR / "controller.pid"
    atomic_write_json(pid_path, {"pid": os.getpid(), "started_at": now_iso()})


def cleanup(signum, frame):
    """Graceful shutdown."""
    global running
    running = False


def main():
    global running

    signal.signal(signal.SIGTERM, cleanup)
    signal.signal(signal.SIGINT, cleanup)

    SIGNAL_DIR.mkdir(parents=True, exist_ok=True)
    write_pid_file()

    while running:
        try:
            process_request()
            watchdog_check()
        except Exception:
            pass  # Never crash the daemon
        time.sleep(POLL_INTERVAL)


if __name__ == "__main__":
    main()
```

### `gpu-controller.service` — systemd unit

```ini
[Unit]
Description=DEFNEX MLOps GPU Lifecycle Controller
After=docker.service
Requires=docker.service

[Service]
Type=simple
User=ubuntu
Group=ubuntu
ExecStart=/usr/bin/python3 /home/ubuntu/defnex-mlops-experiment/gpu_controller/gpu_controller.py
Restart=on-failure
RestartSec=5
StandardOutput=journal
StandardError=journal

# Security hardening
NoNewPrivileges=yes
ProtectSystem=strict
ReadWritePaths=/home/ubuntu/defnex-mlops-experiment/outputs
PrivateTmp=yes

[Install]
WantedBy=multi-user.target
```

---

## 3. Protocol / State Machine

```
REQUEST (worker writes):
{
    "request_id": "<uuid>",
    "action": "stop_serving" | "start_serving",
    "timestamp": "<ISO8601>"
}

RESPONSE (host controller writes):
{
    "request_id": "<same uuid>",
    "phase": "stopping" | "stopped" | "vram_checked" | "starting" | "started" | "healthy" | "error",
    "vllm_stopped": true | false | null,
    "vram_free_mb": <int> | null,
    "vllm_healthy": true | false | null,
    "timestamp": "<ISO8601>",
    "error": null | <string>
}
```

### Phase Transitions

```
stop_serving request:
  stopping → (docker stop) → (inspect) → vram_checked (success)
                                        → error (failure)

start_serving request:
  starting → (docker start) → (inspect) → (health check) → healthy (success)
                                                          → error (failure)
```

### Watchdog State Machine

```
Active operation recorded when: stop_serving succeeds (phase=vram_checked)
Active operation cleared when: start_serving succeeds (phase=healthy)
Hard timeout trigger: elapsed > HARD_TIMEOUT since active_operation.started_at
Recovery action: execute start_serving automatically
Recovery is idempotent: if vLLM is already running, health check passes immediately
```

---

## 4. Backend Code Changes — Exact Edits

### `app/config.py`

1. Add `"file_signal"` to `serving_control` Literal:
```python
serving_control: Literal["mock", "shell", "file_signal"] = "mock"
```

2. Add new settings after `vram_check_timeout`:
```python
# File-based GPU signaling (Option B1). Worker writes JSON requests to a
# shared mount; a host-side systemd service reads them and executes
# Docker stop/start + nvidia-smi. No docker.sock in any container.
gpu_control_dir: str = "/models/.gpu-control"
gpu_control_timeout: float = 120.0  # seconds to wait for host controller response
gpu_control_poll: float = 1.0  # seconds between polling response.json
```

3. Add validator for `file_signal` mode (after existing `_validate_serving_coordination`):
```python
@model_validator(mode="after")
def _validate_file_signal_coordination(self) -> Self:
    if self.serving_control != "file_signal":
        return self
    if not self.gpu_control_dir:
        raise ValueError(
            "SERVING_CONTROL=file_signal requires GPU_CONTROL_DIR to be set"
        )
    if "vram_free_threshold_mb" not in self.model_fields_set:
        raise ValueError(
            "SERVING_CONTROL=file_signal requires VRAM_FREE_THRESHOLD_MB "
            "to be set explicitly"
        )
    if self.vram_free_threshold_mb <= 0:
        raise ValueError("VRAM_FREE_THRESHOLD_MB must be positive")
    return self
```

### `app/workers/gpu_orchestrator.py`

1. Add imports at top:
```python
import json
import os
import uuid
from datetime import datetime, timezone
from pathlib import Path
```

2. Add `FileSignalingServingControl` class (after `ShellServingControl`):
```python
class FileSignalingServingControl:
    """File-based serving control: writes JSON requests to a shared mount,
    polls JSON responses from the host-side controller.

    Active only when `SERVING_CONTROL=file_signal`. The worker writes
    request.json to gpu_control_dir, then polls response.json until
    the host controller writes a response matching the request_id.
    """

    def __init__(
        self,
        control_dir: str,
        timeout: float = 120.0,
        poll: float = 1.0,
    ) -> None:
        self._dir = Path(control_dir)
        self._timeout = timeout
        self._poll = poll
        self._last_vram_free_mb: int | None = None

    def _write_request(self, action: str) -> str:
        """Write request.json atomically. Returns request_id."""
        request_id = str(uuid.uuid4())
        request = {
            "request_id": request_id,
            "action": action,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }
        tmp_path = self._dir / "request.json.tmp"
        self._dir.mkdir(parents=True, exist_ok=True)
        with open(tmp_path, "w") as f:
            json.dump(request, f, indent=2)
            f.flush()
            os.fsync(f.fileno())
        tmp_path.rename(self._dir / "request.json")
        return request_id

    def _poll_response(self, request_id: str) -> dict:
        """Poll response.json until request_id matches or timeout."""
        deadline = time.monotonic() + self._timeout
        response_path = self._dir / "response.json"
        while time.monotonic() < deadline:
            try:
                with open(response_path) as f:
                    response = json.load(f)
                if response.get("request_id") == request_id:
                    return response
            except (FileNotFoundError, json.JSONDecodeError, OSError):
                pass
            time.sleep(self._poll)
        raise TimeoutError(
            f"No response matching request_id {request_id} within {self._timeout}s"
        )

    def stop(self) -> None:
        request_id = self._write_request("stop_serving")
        response = self._poll_response(request_id)
        phase = response.get("phase")
        error = response.get("error")
        if phase == "error":
            raise ServingStopFailed(f"Host controller error: {error}")
        if not response.get("vllm_stopped"):
            raise ServingStopFailed(
                f"Host controller did not confirm vLLM stopped: phase={phase}"
            )
        self._last_vram_free_mb = response.get("vram_free_mb")

    def start(self) -> None:
        request_id = self._write_request("start_serving")
        response = self._poll_response(request_id)
        phase = response.get("phase")
        error = response.get("error")
        if phase == "error":
            raise ServingStartFailed(f"Host controller error: {error}")
        if phase not in ("healthy", "started"):
            raise ServingStartFailed(
                f"Host controller did not confirm vLLM healthy: phase={phase}"
            )

    def health_check(self) -> bool:
        # Check the last response for health status
        try:
            response_path = self._dir / "response.json"
            with open(response_path) as f:
                response = json.load(f)
            return response.get("vllm_healthy") is True
        except (FileNotFoundError, json.JSONDecodeError, OSError):
            return False
```

3. Add `FileSignalingVRAMReader` class:
```python
class FileSignalingVRAMReader:
    """Reads VRAM from the last file-signaling response.

    Used alongside FileSignalingServingControl: after the host controller
    stops vLLM and queries nvidia-smi, it writes vram_free_mb to the
    response. This reader returns that value instead of running nvidia-smi
    directly (which may not be available inside the container).
    """

    def __init__(self, control: FileSignalingServingControl) -> None:
        self._control = control

    def free_mb(self) -> int:
        vram = self._control._last_vram_free_mb
        if vram is None:
            raise RuntimeError(
                "VRAM not available: no stop_serving response has been received yet"
            )
        return vram
```

4. Update `make_coordinator()`:
```python
def make_coordinator() -> ServingCoordinator:
    if settings.serving_control == "shell":
        # ... existing shell code ...
    if settings.serving_control == "file_signal":
        file_control = FileSignalingServingControl(
            control_dir=settings.gpu_control_dir,
            timeout=settings.gpu_control_timeout,
            poll=settings.gpu_control_poll,
        )
        vram: VRAMReader = FileSignalingVRAMReader(file_control)
        return RealServingCoordinator(
            control=file_control,
            vram=vram,
            threshold_mb=settings.vram_free_threshold_mb,
            timeout=settings.vram_check_timeout,
            poll=settings.vram_check_poll,
        )
    return NoopServingCoordinator()
```

---

## 5. Unit Tests — `tests/test_file_signaling.py`

### Test categories

| Category | Tests |
|----------|-------|
| Request serialization | Valid request, missing fields, extra fields, invalid action |
| Response correlation | Matching request_id, stale response rejection |
| Atomic writes | No partial files, temp file cleanup |
| Invalid action rejection | Unknown action, empty action, shell injection attempt |
| Stop/start state handling | stop succeeds, stop fails, start succeeds, start fails |
| VRAM threshold handling | Above threshold, below threshold, threshold exactly met |
| Timeout handling | Response timeout, slow response |
| Watchdog recovery | Normal operation, hard timeout exceeded, idempotent recovery |
| Command injection | Request with malicious action field |
| Config validation | file_signal mode requires gpu_control_dir and threshold |

### Mock strategy

- Mock `docker stop/start/inspect` via subprocess mock
- Mock `nvidia-smi` via subprocess mock
- Mock file I/O for unit isolation (use tmp_path fixture)
- Mock `urllib.request.urlopen` for health checks

---

## 6. Tests Passed

Pending implementation. Will report after running `pytest tests/test_file_signaling.py -v`.

---

## 7. What Remains Before Live Signaling Test

After Phase 2A implementation:

1. **Enable systemd service** — `sudo systemctl enable --now gpu-controller`
2. **Set worker env vars** — `SERVING_CONTROL=file_signal`, `GPU_CONTROL_DIR=/models/.gpu-control`, `VRAM_FREE_THRESHOLD_MB=8000`
3. **Recreate worker** — `docker compose up -d --build worker`
4. **Create test training run** — verify signaling works (stop → VRAM check → start)
5. **Do NOT run real training** — just verify the protocol

---

## 8. Exact Human Approval Required

| # | Action | Risk |
|---|--------|------|
| A1 | Create `/home/ubuntu/defnex-mlops-experiment/gpu_controller/gpu_controller.py` | Host-side Python script |
| A2 | Create `/etc/systemd/system/gpu-controller.service` | systemd service unit |
| A3 | `sudo systemctl enable gpu-controller` | Auto-starts on boot |
| A4 | `sudo systemctl start gpu-controller` | Starts the daemon |
| A5 | Set `SERVING_CONTROL=file_signal` in worker env | Worker restart |
| A6 | `docker compose up -d --build worker` | Worker restart with new config |
| A7 | Create test training run for signaling verification | vLLM stop/start cycle |

---

## 9. Testing Plan

### Phase 2A tests (no real GPU)

1. Unit tests in `tests/test_file_signaling.py` — all mocked, no real Docker/GPU
2. Config validation tests — verify `file_signal` mode requires correct settings
3. Existing tests must still pass — `make_coordinator()` default path unchanged

### Live signaling verification (separate approval)

1. Start gpu-controller service
2. Restart worker with `SERVING_CONTROL=file_signal`
3. Create a test training run
4. Observe: worker writes request.json, controller reads it, stops vLLM, queries VRAM, writes response.json
5. Observe: worker reads response, checks threshold
6. Observe: controller restarts vLLM after training
7. Verify: vLLM is healthy after restart
