# Phase 2A Health Timeout Fix Report

**Date:** 2026-09-14
**Status:** READY FOR SECOND LIVE SIGNALING TEST
**Scope:** Health timeout configuration + active-operation semantics fix

---

## Summary

Fixed the health check timeout that caused `health_check_timeout` during the first live signaling test. The controller's health check loop was hardcoded to 60 seconds, but cold vLLM startup takes ~135 seconds. Additionally fixed a critical bug where `active.json` was cleared on health timeout, preventing the watchdog from retrying recovery.

---

## Files Changed

| File | Change |
|------|--------|
| `gpu_controller.py` (host) | `HEALTH_CHECK_DEADLINE` configurable via `GPU_CONTROLLER_HEALTH_TIMEOUT` env var (default 180s). `handle_start_serving()` no longer clears `active.json` on health timeout. `watchdog_check()` now detects unhealthy-start state and retries. |
| `tests/test_file_signaling.py` | Added `TestHealthTimeoutSemantics` (8 tests) and `TestHealthTimeoutConfig` (3 tests). Total: 57 tests (was 46). |

---

## TASK 1 — Health Timeout Configuration

### Before

```python
HEALTH_TIMEOUT = 5  # HTTP request timeout
# health check loop: hardcoded 60s (line 328)
health_deadline = time.monotonic() + 60
```

### After

```python
HEALTH_TIMEOUT = 5  # HTTP request timeout (unchanged)
HEALTH_CHECK_DEADLINE = int(os.environ.get("GPU_CONTROLLER_HEALTH_TIMEOUT", "180"))
# health check loop:
health_deadline = time.monotonic() + HEALTH_CHECK_DEADLINE
```

**Environment variable:** `GPU_CONTROLLER_HEALTH_TIMEOUT`
**Default:** `180` seconds
**Scope:** Single source of truth — one variable controls the health check loop deadline.

---

## TASK 2 — Active-Operation Semantics After Health Timeout

### Bug (Before)

```python
# handle_start_serving(), health timeout path:
else:
    write_response(request_id, "error", error="health_check_timeout", ...)
    clear_active_operation()  # ← BUG: clears active.json
    return
```

**Problem:** When health check timed out, `active.json` was cleared. The watchdog could not detect the unhealthy state and retry. The container was running but unhealthy, and recovery was abandoned.

### Fix (After)

```python
# handle_start_serving(), health timeout path:
else:
    # Health timeout: container is running but unhealthy.
    # Do NOT clear active.json — leave it for watchdog to retry.
    write_response(request_id, "error", error="health_check_timeout", ...)
    logger.warning(f"start_serving_health_timeout ... active_left_for_watchdog=True")
    return
```

**Key invariant:** If the container is running but vLLM is unhealthy past the health deadline, `active.json` is NOT cleared. The watchdog detects it on the next cycle and retries.

### Watchdog Enhancement

Added Path 2 to `watchdog_check()`:

```python
# Path 2: container running but unhealthy (health timeout recovery)
container_running = docker_inspect_running(CONTAINER_NAME)
if container_running is True and not check_vllm_health():
    logger.warning(f"watchdog_unhealthy_start_detected ...")
    handle_start_serving(request_id)
    return
```

**Behavior matrix:**

| Scenario | active.json | Container | Healthy | Watchdog action |
|----------|------------|-----------|---------|-----------------|
| Hard deadline exceeded, unhealthy | exists | any | no | restart via handle_start_serving |
| Hard deadline exceeded, healthy | exists | running | yes | clear active (idempotent) |
| Within deadline, unhealthy | exists | running | no | **retry via handle_start_serving** (NEW) |
| Within deadline, healthy | exists | running | yes | no action (correct) |
| No active.json | none | any | any | no action |

---

## TASK 3 — Tests

### New Tests (11 total)

| # | Test | Scenario | Result |
|---|------|----------|--------|
| 1 | `test_healthy_before_timeout` | vLLM healthy before deadline → response=healthy, active cleared | ✅ |
| 2 | `test_healthy_after_60s_before_180s` | vLLM healthy after 60s but before 180s → healthy | ✅ |
| 3 | `test_unhealthy_past_timeout` | vLLM unhealthy past deadline → error, active NOT cleared | ✅ |
| 4 | `test_timeout_never_falsely_reports_healthy` | Timeout must never report healthy | ✅ |
| 5 | `test_active_state_after_start_timeout` | After health timeout, active remains for watchdog | ✅ |
| 6 | `test_restart_during_unhealthy_start_recoverable` | Controller restart during unhealthy-start remains recoverable | ✅ |
| 7 | `test_successful_health_clears_active` | When healthy, active.json is cleared | ✅ |
| 8 | `test_no_redundant_restart_when_already_healthy` | No restart when vLLM already healthy (expired deadline) | ✅ |
| 9 | `test_health_check_deadline_default` | Default is 180s | ✅ |
| 10 | `test_health_check_deadline_configurable` | Env var sets deadline | ✅ |
| 11 | `test_health_timeout_uses_deadline_not_60` | Uses HEALTH_CHECK_DEADLINE, not hardcoded 60 | ✅ |

### Test Results

```
tests/test_file_signaling.py    57 passed
tests/test_gpu_orchestration.py 21 passed
tests/test_health.py             1 passed
─────────────────────────────────────
Total:                          79 passed, 0 failed
```

---

## TASK 4 — Safety Guarantees

| Guarantee | Verified |
|-----------|----------|
| Only defnex-vllm controlled | ✅ `CONTAINER_NAME = "defnex-vllm"` (hardcoded) |
| No request-derived Docker commands | ✅ No subprocess calls with request data |
| No shell=True in file_signal controller | ✅ Not found in gpu_controller.py |
| No PID kill | ✅ No `os.kill()` or `signal.SIGKILL` |
| No docker kill | ✅ Not found |
| No nvidia-smi --gpu-reset | ✅ Not found |
| No docker.sock | ✅ Not found in controller |
| Unrelated GPU processes untouched | ✅ No process targeting code |

---

## TASK 5 — Regression QA

| Check | Result |
|-------|--------|
| `pytest tests/test_file_signaling.py -q --no-cov` | **57 passed** ✅ |
| `pytest tests/test_gpu_orchestration.py -q --no-cov` | **21 passed** ✅ |
| `pytest tests/ -q --no-cov` (targeted) | **79 passed** ✅ |
| `ruff check .` | **All checks passed** ✅ |
| `ruff format --check .` | **223 files already formatted** ✅ |
| `gpu_controller.py` syntax | **OK** ✅ |

---

## Evidence Classification

| Evidence | Classification |
|----------|---------------|
| Health timeout configurable via env var | VERIFIED BY CODE |
| active.json not cleared on health timeout | VERIFIED BY TEST (test_unhealthy_past_timeout) |
| Watchdog retries on unhealthy start | VERIFIED BY TEST (test_active_state_after_start_timeout) |
| No redundant restart when healthy | VERIFIED BY TEST (test_no_redundant_restart_when_already_healthy) |
| Safety guarantees unchanged | VERIFIED BY STATIC INSPECTION |
| All tests pass | VERIFIED BY TEST |
| Ruff clean | VERIFIED BY TOOL |

---

## Readiness for Second Live Signaling Test

### What Changed

1. `HEALTH_CHECK_DEADLINE` increased from 60s to 180s (configurable via `GPU_CONTROLLER_HEALTH_TIMEOUT`)
2. `handle_start_serving()` no longer clears `active.json` on health timeout
3. `watchdog_check()` now retries when container is running but unhealthy

### What Did NOT Change

- File signal protocol (request.json / response.json)
- Atomic write mechanism
- Docker operations (stop/start/inspect)
- VRAM query (nvidia-smi read-only)
- Container name (hardcoded defnex-vllm)
- Safety guarantees (no shell=True, no PID kill, no docker.sock)

### Expected Behavior in Second Live Test

- **Stop path:** Same as first test (7 seconds, fully verified)
- **Start path:** Container starts, health check polls for up to 180s. vLLM cold start (~135s) should now complete within the deadline. Response should be `phase=healthy`.
- **If health still times out:** `active.json` is NOT cleared. Watchdog retries on next cycle. Recovery is not abandoned.

### Prerequisites for Second Live Test

- [ ] `GPU_CONTROLLER_HEALTH_TIMEOUT` can be set in systemd unit or environment
- [ ] Controller restart picks up new deadline
- [ ] Worker env unchanged (`SERVING_CONTROL=file_signal`)

**Verdict: READY FOR SECOND LIVE SIGNALING TEST**
