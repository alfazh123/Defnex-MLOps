# Phase 2A Second Live Signaling Test — Read-Only Preflight

**Date:** 2026-09-15
**Status:** READY
**Mode:** READ-ONLY — no files modified, no containers started/stopped

---

## 1. Effective Health Timeout

| Item | Value | Source |
|------|-------|--------|
| Code | `HEALTH_CHECK_DEADLINE = int(os.environ.get("GPU_CONTROLLER_HEALTH_TIMEOUT", "180"))` | `gpu_controller.py:41` |
| systemd unit | No `GPU_CONTROLLER_HEALTH_TIMEOUT` set | `/etc/systemd/system/gpu-controller.service` |
| Effective value | **180 seconds** | Fallback default |

**The controller will use 180 seconds for the health check loop deadline.**

---

## 2. Systemd Unit — No Explicit Value

The installed systemd unit (`/etc/systemd/system/gpu-controller.service`) contains **no `Environment=` directive** for `GPU_CONTROLLER_HEALTH_TIMEOUT`. The controller process inherits the default fallback of `180` from the code.

This is correct — the default is sufficient.

---

## 3. Fallback Confirmation

```python
HEALTH_CHECK_DEADLINE = int(os.environ.get("GPU_CONTROLLER_HEALTH_TIMEOUT", "180"))
```

If `GPU_CONTROLLER_HEALTH_TIMEOUT` is not set (as confirmed), `os.environ.get()` returns `"180"`, `int()` converts to `180`. **Confirmed.**

---

## 4. Systemd Unit Consistency

| Check | Result |
|-------|--------|
| Installed unit matches repo docs | ✅ IDENTICAL (`diff` exit 0) |
| Files compared | `/etc/systemd/system/gpu-controller.service` vs `docs/gpu-controller.service` |

---

## 5. Controller Code Verification

**Line 41:**
```python
HEALTH_CHECK_DEADLINE = int(os.environ.get("GPU_CONTROLLER_HEALTH_TIMEOUT", "180"))
```

**Line 344 (health check loop):**
```python
health_deadline = time.monotonic() + HEALTH_CHECK_DEADLINE
```

**Line 362 (health timeout log):**
```python
f"deadline={HEALTH_CHECK_DEADLINE}s, active_left_for_watchdog=True"
```

**Line 523 (startup log):**
```python
f"hard_timeout={HARD_TIMEOUT}, health_check_deadline={HEALTH_CHECK_DEADLINE}s"
```

All references use the same `HEALTH_CHECK_DEADLINE` variable. **No overrides found.**

---

## 6. No Other Health Timeout Constants

| Constant | Value | Used for | Overrides HEALTH_CHECK_DEADLINE? |
|----------|-------|----------|----------------------------------|
| `HEALTH_TIMEOUT` | 5 | Individual HTTP request timeout (`urlopen`) | NO |
| `HEALTH_CHECK_DEADLINE` | 180 | Health check loop deadline | THIS IS THE ONE |
| `HARD_TIMEOUT` | 1800 | Max time for entire stop-serving operation | NO |
| `STOP_TIMEOUT` | 30 | Wait for container to stop | NO |
| `START_TIMEOUT` | 30 | Wait for container to start running | NO |

**No other constant overrides the health check loop deadline.**

---

## 7. Worker File Signal Configuration

```
SERVING_CONTROL=file_signal
GPU_CONTROL_DIR=/models/.gpu-control
VRAM_FREE_THRESHOLD_MB=8000
```

**All three env vars confirmed correct.**

---

## 8. Controller Executable Path

| Item | Value | Correct? |
|------|-------|----------|
| ExecStart | `/usr/bin/python3 /home/ubuntu/defnex-mlops-experiment/gpu_controller/gpu_controller.py` | ✅ |
| `/usr/bin/python3` | Symlink → `python3.12` (Python 3.12.3) | ✅ |
| `gpu_controller.py` | Exists, last modified 2026-09-15 02:36 UTC | ✅ |

---

## Runtime State (READ-ONLY)

| Check | Status | Detail |
|-------|--------|--------|
| defnex-vllm running | ✅ | `docker inspect` → `true` |
| port 8001 healthy | ✅ | HTTP 200 |
| gpu-controller service | ✅ | `inactive (dead)` — stopped after first test, ready to start |
| stale request.json | ✅ | None (file does not exist) |
| active.json | ✅ | None (file does not exist) |
| stale response.json | ✅ | Present (from first test, `health_check_timeout`) — harmless, will be overwritten |
| controller.pid | ✅ | Present (from first test) — harmless |
| GPU process count | ✅ | 14 (same as baseline) |

---

## Preflight Checklist

| # | Check | Result |
|---|-------|--------|
| 1 | Effective health timeout | **180 seconds** |
| 2 | Systemd provides explicit value | **No** — uses code default |
| 3 | Fallback to 180s confirmed | **Yes** |
| 4 | Installed unit = repo docs | **Identical** |
| 5 | Code: `HEALTH_CHECK_DEADLINE = int(os.environ.get("GPU_CONTROLLER_HEALTH_TIMEOUT", "180"))` | **Confirmed** |
| 6 | No other health timeout constant overrides | **Confirmed** |
| 7 | Worker file_signal config correct | **Confirmed** |
| 8 | Controller executable path correct | **Confirmed** |

---

## Decision

| Item | Value |
|------|-------|
| Effective health timeout | **180 seconds** |
| Controller restart required to pick up config | **YES** — controller is currently `inactive (dead)`. Must `sudo systemctl start gpu-controller.service` before test. New process will read the updated code with 180s default. |
| READY for second live signaling test | **YES** |
| Blockers | **NONE** |
