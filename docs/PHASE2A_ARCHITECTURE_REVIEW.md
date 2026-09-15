# Phase 2A Architecture Review — Security & GPU Safety

**Date:** 2026-09-14
**Status:** READ-ONLY — NO files modified, NO containers started/stopped/recreated
**Constraint:** docker.sock NOT approved. Security and shared-GPU safety are first-class.

---

## 1. Executive Summary

**Q1: Is docker.sock really necessary for this prototype?**
**No.** The worker and defnex-vllm share a host mount at `/home/ubuntu/defnex-mlops-experiment/outputs` (mapped to `/models` in both containers). A host-side daemon can watch this path for signal files, eliminating the need for docker.sock inside any container.

**Q2: Can we implement host-side GPU lifecycle control instead?**
**Yes.** A host-side systemd service (runs on the VM, not in Docker) handles all privileged operations: stop/start defnex-vllm, query nvidia-smi, verify VRAM. The worker communicates via file-based JSON signals written to the shared mount.

**Recommended architecture: Option B1 — File-based signaling via shared mount + host systemd service.**

---

## 2. Architecture Options Compared

### Option A: worker → docker.sock → Docker daemon

```
┌─────────────┐     /var/run/docker.sock     ┌──────────────┐
│   worker     │ ──────────────────────────→ │ Docker daemon │
│  (container) │   docker stop/start          │   (host)     │
└─────────────┘                               └──────────────┘
```

| Criterion | Assessment |
|-----------|-----------|
| Security | **POOR** — docker.sock = root on host. Any process in the container can start/stop any container, mount host filesystem, execute arbitrary commands as root |
| Complexity | Low — existing `ShellServingControl` works as-is |
| Prototype-safe | Marginal — single-user dev VM only |
| Production-safe | **NO** |
| User approved | **NO** |

**Verdict: REJECTED.** docker.sock grants root-equivalent access. Violates least privilege.

### Option B1: worker → file signal → host-side controller → Docker daemon (RECOMMENDED)

```
┌─────────────┐   shared mount /models    ┌──────────────────┐   docker CLI   ┌──────────────┐
│   worker     │ ── .gpu-control/ ──────→ │  gpu-controller   │ ────────────→ │ Docker daemon │
│  (container) │   write JSON request      │  (systemd service)│               │   (host)     │
│              │ ← read JSON response ─── │                   │ ← nvidia-smi ─│              │
└─────────────┘                           └──────────────────┘               └──────────────┘
```

| Criterion | Assessment |
|-----------|-----------|
| Security | **GOOD** — worker never gets docker.sock. Host controller runs as `ubuntu` user with docker group membership only |
| Complexity | Medium — requires host daemon + file protocol |
| Prototype-safe | **YES** |
| Production-safe | **YES** — same pattern works with Kubernetes sidecar |
| Shared mount verified | **YES** — live tested: worker writes to `/models/.gpu-control/`, host sees at `/home/ubuntu/defnex-mlops-experiment/outputs/.gpu-control/` |

**Verdict: RECOMMENDED.** Clean separation. Worker is unprivileged. Host retains full control.

### Option C: worker → backend control API → host-side controller → Docker daemon

```
┌─────────────┐   HTTP    ┌─────────────┐   HTTP    ┌──────────────────┐   docker   ┌──────────────┐
│   worker     │ ───────→ │  backend     │ ───────→ │  host controller  │ ─────────→ │ Docker daemon │
│  (container) │          │  (container) │          │  (systemd service)│            │   (host)     │
└─────────────┘          └─────────────┘          └──────────────────┘            └──────────────┘
```

| Criterion | Assessment |
|-----------|-----------|
| Security | **GOOD** — same as B1 |
| Complexity | **HIGH** — three hops, two network boundaries, backend is also in Docker and can't reach docker.sock either |
| Prototype-safe | Over-engineered |
| Additional issues | Backend just proxies — no security benefit over B1. Adds latency and failure surface |

**Verdict: REJECTED.** Unnecessary indirection. Backend can't execute Docker commands itself.

### Option D: Separate GPU training node

| Criterion | Assessment |
|-----------|-----------|
| Security | **EXCELLENT** — complete isolation |
| Complexity | N/A |
| Prototype-safe | N/A — requires second GPU/VM |
| Production-safe | **IDEAL** |

**Verdict: NOT APPLICABLE.** Single GPU, single VM. Future target per PRD v2.

### Summary Table

| Option | Security | Complexity | Prototype | Production | Verdict |
|--------|----------|------------|-----------|------------|---------|
| A: docker.sock | Poor | Low | Marginal | No | **REJECTED** |
| **B1: file signal** | **Good** | **Medium** | **YES** | **YES** | **RECOMMENDED** |
| C: backend proxy | Good | High | Overkill | Yes | REJECTED |
| D: separate GPU | Excellent | N/A | N/A | Ideal | FUTURE |

---

## 3. Recommended Architecture: Option B1

### Component Diagram

```
HOST (ubuntu, docker group)
├── gpu-controller.service (systemd)
│   ├── Watches: /home/ubuntu/defnex-mlops-experiment/outputs/.gpu-control/
│   ├── Executes: docker stop/start defnex-vllm
│   ├── Executes: nvidia-smi --query-gpu=memory.free
│   └── Writes: response JSON to same directory
│
├── defnex-vllm (Docker, CDI GPU, port 8001)
│   └── Mounts: /home/ubuntu/defnex-mlops-experiment/outputs → /models (ro)
│
├── ml-close-loop-be-worker-1 (Docker, NO GPU, NO docker.sock)
│   ├── Mounts: /home/ubuntu/defnex-mlops-experiment/outputs → /models
│   ├── Mounts: /home/ubuntu/Defnex-MLOps/ml-close-loop-be/data → /app/data
│   └── Writes: /models/.gpu-control/request.json
│
└── ml-close-loop-be-backend-1 (Docker, NO GPU)
    └── Port 8000 → 8000
```

### Communication Path

```
Worker (inside container)                    Host
─────────────────────────────               ─────────────────────────
1. Acquire GPU lock (flock)
2. Write request.json:
   {"action":"stop_serving",
    "request_id":"<uuid>",
    "timestamp":"<iso8601>"}
                                           3. Inotify/poll detects request.json
                                           4. Execute: docker stop defnex-vllm
                                           5. Wait for container exit (poll)
                                           6. Execute: nvidia-smi --query-gpu=memory.free
                                           7. Write response.json:
                                              {"request_id":"<uuid>",
                                               "phase":"vram_checked",
                                               "vllm_stopped":true,
                                               "vram_free_mb":14350,
                                               "timestamp":"<iso8601>"}
8. Poll response.json
9. Check: vram_free_mb >= threshold?
   YES → proceed to training
   NO  → write start_serving request, abort
10. Run training subprocess
11. Write request.json:
    {"action":"start_serving",
     "request_id":"<uuid>"}
                                           12. Execute: docker start defnex-vllm
                                           13. Health check: HTTP GET :8001/health
                                           14. Write response.json:
                                               {"phase":"healthy",
                                                "vllm_healthy":true}
15. Poll response.json
16. Release GPU lock
```

### Why This Is Safest for This VM

1. **Zero privilege escalation** — Worker never gets docker.sock, never gets host access
2. **Host retains full control** — Only the host's systemd service can stop/start containers
3. **Audit trail** — Every request/response is a JSON file on disk with timestamps
4. **Existing infrastructure** — Uses the already-mounted shared volume (`/models`)
5. **Clean abstraction** — Implements existing `ServingControl` protocol, no code restructuring needed
6. **Crash-safe** — If worker crashes mid-cycle, host controller detects timeout and restarts vLLM
7. **No new network exposure** — No new ports, no new Docker networks

---

## 4. GPU Ownership Analysis

### Current GPU State (live, read-only)

```
GPU: NVIDIA H100 PCIe 80GB
Total: 81,559 MiB
Used:  78,964 MiB (96.8%)
Free:   2,116 MiB
Util:   0% (compute idle, memory occupied)
```

### GPU Processes (14 total)

| PID | Memory (MiB) | Likely Owner |
|-----|-------------|-------------|
| 982687 | 15,150 | Unknown |
| 21233 | 14,198 | Unknown |
| 22701 | 11,034 | Unknown |
| 2170057 | 7,594 | Unknown |
| 22409 | 5,488 | Unknown |
| 22297 | 5,470 | Unknown |
| 22279 | 5,466 | Unknown |
| 18147 | 3,466 | Unknown |
| 18159 | 3,464 | Unknown |
| 2955651 | 2,056 | Unknown |
| 24059 | 1,740 | Unknown |
| 23079 | 1,446 | Unknown |
| 23085 | 1,338 | Unknown |
| 23150 | 964 | Unknown |

### Critical Observation

**defnex-vllm is NOT the only GPU consumer.** There are 14 processes using GPU memory. defnex-vllm's `--gpu-memory-utilization 0.15` allocates ~12,234 MiB. The remaining ~66,730 MiB is used by other processes.

**Stopping defnex-vllm does NOT guarantee a specific amount of free VRAM.** The 14 other processes may also change between now and training time. Some may exit; others may start.

### Safe Pre-Training VRAM Estimation

```
Scenario analysis (worst case):

Current free:                  2,116 MiB
vLLM stop frees:            ~12,234 MiB (15% of 81,559)
Theoretical max free:       ~14,350 MiB
Other processes may change:  ±5,000 MiB

Conservative estimate:       ~9,000 MiB free after vLLM stop
Qwen2.5-0.5B training need: ~2,000-4,000 MiB (LoRA, batch=1)

Threshold recommendation:    8,000 MiB (conservative)
```

**The pre-training gate MUST verify actual free memory AFTER vLLM stops, not assume it.**

---

## 5. Pre-Training Gate Design

### Gate Requirements (from user)

1. Stop serving
2. Wait until vLLM process/container is actually stopped
3. Query nvidia-smi
4. Verify minimum free-memory threshold
5. Abort training if threshold is not reached
6. Never kill unrelated GPU processes

### Gate Implementation (host-side controller)

```python
# Host controller pre-training gate logic (pseudocode)

def handle_stop_serving(request):
    # Step 1: Stop vLLM
    subprocess.run(["docker", "stop", "defnex-vllm"], timeout=30)

    # Step 2: Wait until container is actually stopped
    for _ in range(30):
        result = subprocess.run(
            ["docker", "inspect", "--format", "{{.State.Running}}", "defnex-vllm"],
            capture_output=True, text=True
        )
        if result.stdout.strip() == "false":
            break
        time.sleep(1)
    else:
        # Container did not stop within 30s — ABORT
        write_response(phase="error", error="vllm_did_not_stop")
        return

    # Step 3: Query nvidia-smi
    result = subprocess.run(
        ["nvidia-smi", "--query-gpu=memory.free", "--format=csv,noheader,nounits"],
        capture_output=True, text=True, timeout=30
    )
    vram_free_mb = int(result.stdout.strip().splitlines()[0].strip())

    # Step 4: Write VRAM status (worker decides threshold check)
    write_response(
        phase="vram_checked",
        vllm_stopped=True,
        vram_free_mb=vram_free_mb,
        timestamp=now()
    )

    # NEVER kill unrelated GPU processes — that's a hard rule
```

### Gate Decision (worker-side)

```python
# Worker gate decision logic (pseudocode)

def pre_training_gate(coordinator, threshold_mb):
    # Coordinator writes stop request, polls response
    response = coordinator.request_stop_serving()

    # Verify vLLM actually stopped
    if not response.get("vllm_stopped"):
        raise ServingStopFailed("Host controller reports vLLM not stopped")

    # Verify VRAM meets threshold
    vram_free = response.get("vram_free_mb", 0)
    if vram_free < threshold_mb:
        # ABORT: not enough VRAM
        # MUST restart vLLM before releasing lock
        coordinator.request_start_serving()
        raise VRAMNotFree(f"VRAM {vram_free} MiB < threshold {threshold_mb} MiB")

    # Proceed with training
    return vram_free
```

### Gate Safety Properties

| Property | Guarantee |
|----------|-----------|
| Atomicity | Worker holds GPU lock throughout — no concurrent training |
| VRAM verification | Always measured AFTER vLLM stops, never assumed |
| Threshold enforcement | Worker-side decision — host just reports numbers |
| Abort path | If threshold not met, vLLM is restarted before lock release |
| No kill | Host controller NEVER runs `kill` on any GPU process |
| Timeout | If vLLM doesn't stop in 30s, gate fails safely |
| Crash recovery | If worker crashes, host controller has watchdog to restart vLLM |

---

## 6. Status Reporting Protocol

### Request Format (worker → host)

```json
{
    "action": "stop_serving",
    "request_id": "550e8400-e29b-41d4-a716-446655440000",
    "timestamp": "2026-09-14T04:30:00Z"
}
```

### Response Format (host → worker)

```json
{
    "request_id": "550e8400-e29b-41d4-a716-446655440000",
    "phase": "vram_checked",
    "vllm_stopped": true,
    "vram_free_mb": 14350,
    "timestamp": "2026-09-14T04:30:05Z",
    "error": null
}
```

### Phase Values

| Phase | Meaning | Who writes | When |
|-------|---------|-----------|------|
| `stop_requested` | Worker sent stop command | Worker | Before polling |
| `stopping` | Host executing docker stop | Host | During stop |
| `stopped` | vLLM container confirmed stopped | Host | After docker inspect confirms |
| `vram_checked` | nvidia-smi queried, free memory recorded | Host | After nvidia-smi |
| `start_requested` | Worker sent start command | Worker | After training |
| `starting` | Host executing docker start | Host | During start |
| `started` | vLLM container confirmed running | Host | After docker inspect confirms |
| `healthy` | vLLM health check passed | Host | After HTTP health check |
| `error` | Something went wrong | Host | On any failure |
| `timeout` | Operation did not complete in time | Host | On deadline |

### File Locations

```
/home/ubuntu/defnex-mlops-experiment/outputs/.gpu-control/
├── request.json          # Worker writes, host reads
├── response.json         # Host writes, worker reads
└── controller.pid        # Host controller PID (for watchdog)
```

### Status Lifecycle Diagram

```
Worker                           Host Controller
──────                           ───────────────

write request.json ─────────→   inotify detects
  {action: stop_serving}         │
                                 ├─ docker stop defnex-vllm
                                 │   (write response: phase=stopping)
                                 │
                                 ├─ docker inspect (poll until stopped)
                                 │   (write response: phase=stopped)
                                 │
                                 ├─ nvidia-smi --query-gpu=memory.free
                                 │   (write response: phase=vram_checked,
                                 │    vram_free_mb=14350)
                                 │
poll response.json ←────────     │
  {phase: vram_checked}          │
                                 │
[training runs]                  │
                                 │
write request.json ─────────→   inotify detects
  {action: start_serving}        │
                                 ├─ docker start defnex-vllm
                                 │   (write response: phase=starting)
                                 │
                                 ├─ docker inspect (poll until running)
                                 │   (write response: phase=started)
                                 │
                                 ├─ HTTP GET localhost:8001/health
                                 │   (write response: phase=healthy)
                                 │
poll response.json ←────────     │
  {phase: healthy}               │
```

---

## 7. Exact Files/Components Needed

### New Files (host side)

| File | Purpose | Location |
|------|---------|----------|
| `gpu_controller.py` | Host-side daemon: watches signal files, executes docker/nvidia-smi, writes responses | `/home/ubuntu/defnex-mlops-experiment/gpu_controller/gpu_controller.py` |
| `gpu-controller.service` | systemd unit file for the daemon | `/etc/systemd/system/gpu-controller.service` |
| `request.json` | Worker writes requests here | `/home/ubuntu/defnex-mlops-experiment/outputs/.gpu-control/request.json` |
| `response.json` | Host writes responses here | `/home/ubuntu/defnex-mlops-experiment/outputs/.gpu-control/response.json` |

### Modified Files (backend codebase)

| File | Change | Purpose |
|------|--------|---------|
| `app/workers/gpu_orchestrator.py` | Add `FileSignalingServingControl` class | Implements `ServingControl` protocol via file-based signaling instead of shell commands |
| `app/workers/gpu_orchestrator.py` | Add `FileSignalingVRAMReader` class | Reads VRAM from response.json instead of running nvidia-smi directly |
| `app/workers/gpu_orchestrator.py` | Update `make_coordinator()` | Add `SERVING_CONTROL=file_signal` option |
| `app/config.py` | Add `file_signal` to `serving_control` Literal | Allow file-based signaling mode |
| `app/config.py` | Add `gpu_control_dir` setting | Path to `.gpu-control/` directory |

### Files NOT Modified

| File | Reason |
|------|--------|
| `Dockerfile` | No docker CLI needed in worker image |
| `docker-compose.yml` | No docker.sock mount, no new env vars for this option |
| `docker-worker-entrypoint.sh` | No change |
| `app/workers/training_worker.py` | No change — uses `ServingCoordinator` protocol, agnostic to implementation |

### Existing Abstractions Reused

| Abstraction | Location | How B1 implements it |
|-------------|----------|---------------------|
| `ServingControl` Protocol | `gpu_orchestrator.py:40-57` | `FileSignalingServingControl.stop()` writes request.json, polls response.json |
| `VRAMReader` Protocol | `gpu_orchestrator.py:60-68` | `FileSignalingVRAMReader.free_mb()` reads `vram_free_mb` from response.json |
| `ServingCoordinator` Protocol | `gpu_orchestrator.py:71-77` | `RealServingCoordinator` unchanged — just wires different control + vram implementations |

---

## 8. Security Implications

### Option B1 Security Model

```
PRIVILEGE MAP:

┌─────────────────────────────┬──────────────────────────────────────────────┐
│ Component                   │ Privileges                                    │
├─────────────────────────────┼──────────────────────────────────────────────┤
│ worker container            │ Write to /models/.gpu-control/ (shared vol)   │
│                             │ Read from /models/.gpu-control/               │
│                             │ GPU access (CDI) for training subprocess      │
│                             │ NO docker CLI                                 │
│                             │ NO docker.sock                                │
│                             │ NO host filesystem access                     │
├─────────────────────────────┼──────────────────────────────────────────────┤
│ gpu-controller (host)       │ docker stop/start defnex-vllm                 │
│                             │ nvidia-smi query                              │
│                             │ Read/write /home/ubuntu/.../outputs/.gpu-control/ │
│                             │ Runs as ubuntu user (docker group)            │
├─────────────────────────────┼──────────────────────────────────────────────┤
│ defnex-vllm container       │ GPU access (CDI)                              │
│                             │ Read /models (ro)                             │
│                             │ No write access to shared volume              │
└─────────────────────────────┴──────────────────────────────────────────────┘
```

### Attack Surface Comparison

| Attack Vector | Option A (docker.sock) | Option B1 (file signal) |
|--------------|----------------------|------------------------|
| Container escape to host | **CRITICAL** — docker.sock gives root | **LOW** — only file write to shared mount |
| Malicious container start | **YES** — worker can start any container | **NO** — worker can only signal controller |
| Host filesystem mount | **YES** — worker can mount via docker API | **NO** — worker has no docker access |
| Process injection | **YES** — worker can exec into any container | **NO** |
| GPU process kill | **YES** — worker can kill via docker exec | **NO** — controller never kills GPU processes |
| Audit trail | **NONE** — docker API calls not logged by default | **YES** — all requests/responses are files on disk |

### Remaining Risks (B1)

| Risk | Mitigation |
|------|-----------|
| Worker writes malicious request.json | Controller validates JSON schema before acting |
| Worker floods controller with requests | Controller rate-limits (1 request per 60s) |
| Worker reads other files on shared mount | Worker already has this mount — no new exposure |
| Controller crashes mid-cycle | Watchdog restarts vLLM after timeout |
| File permissions (worker writes as root, host reads as ubuntu) | Controller runs with `sudo` or uses `chmod` on signal dir |

---

## 9. GPU Safety Guarantees

### Guarantee 1: No VRAM Assumptions

```
WRONG (old design):
  "Stopping defnex-vllm frees ~12 GB"
  → Assumes other processes don't change

RIGHT (B1 design):
  "Stop vLLM, THEN measure free VRAM with nvidia-smi"
  → Actual measurement, not assumption
```

### Guarantee 2: Threshold Enforcement

```
vram_free_mb is measured by host controller after vLLM stops.
Worker checks: vram_free_mb >= VRAM_FREE_THRESHOLD_MB
If not met → abort, restart vLLM, raise VRAMNotFree.
Training NEVER starts without verified free VRAM.
```

### Guarantee 3: No Unrelated Process Kills

```
HARD RULE: The host controller NEVER executes:
  - kill <pid>
  - nvidia-smi --gpu-reset
  - docker kill <container>
  - Any command targeting a process by PID

The controller ONLY:
  - docker stop defnex-vllm (by container name)
  - docker start defnex-vllm (by container name)
  - nvidia-smi --query-gpu=memory.free (read-only query)
```

### Guarantee 4: Crash Recovery

```
If worker crashes during training:
  1. GPU lock is released (kernel releases flock on process exit)
  2. Host controller has a watchdog timer
  3. If no "start_serving" request within TIMEOUT (e.g., 30 min):
     → Host controller automatically restarts vLLM
     → Writes response: {phase: "error", error: "worker_timeout"}
  4. vLLM is never left stopped indefinitely
```

### Guarantee 5: Atomic Cycle

```
The entire cycle (stop → VRAM check → train → restart) happens
inside the GPU lock. No other training worker can interleave.
The host controller is the ONLY entity that can stop/start vLLM.
```

---

## 10. Exact Phase 2A Implementation Plan

### Step 1: Create host-side controller (no container changes)

**Files:**
- `/home/ubuntu/defnex-mlops-experiment/gpu_controller/gpu_controller.py`
- `/etc/systemd/system/gpu-controller.service`

**What it does:**
- Watches `/home/ubuntu/defnex-mlops-experiment/outputs/.gpu-control/` via polling (inotify)
- On `request.json` with `action: stop_serving`:
  - Executes `docker stop defnex-vllm`
  - Polls `docker inspect` until container is stopped (30s timeout)
  - Queries `nvidia-smi --query-gpu=memory.free`
  - Writes `response.json` with `phase: vram_checked, vram_free_mb: <value>`
- On `request.json` with `action: start_serving`:
  - Executes `docker start defnex-vllm`
  - Polls `docker inspect` until container is running (30s timeout)
  - Queries `curl localhost:8001/health` (vLLM health check)
  - Writes `response.json` with `phase: healthy`
- Watchdog: if no start_serving request within 1800s (30 min), auto-restart vLLM

**Docker/containers touched:** NONE
**Human approval required:** YES (creating systemd service)

### Step 2: Add FileSignalingServingControl to backend codebase

**Files:**
- `app/workers/gpu_orchestrator.py` — add `FileSignalingServingControl` class
- `app/config.py` — add `file_signal` to `serving_control` Literal, add `gpu_control_dir` setting

**What it does:**
- Implements `ServingControl` protocol
- `stop()`: writes request.json with `action: stop_serving`, polls response.json for `phase: vram_checked`
- `start()`: writes request.json with `action: start_serving`, polls response.json for `phase: healthy`
- `health_check()`: checks `response.json` has `phase: healthy`

**Docker/containers touched:** NONE (code change only)
**Human approval required:** YES (code change)

### Step 3: Set environment variables

**Files:**
- `docker-compose.yml` — worker service env vars only

**What to set:**
```yaml
SERVING_CONTROL: file_signal
GPU_CONTROL_DIR: /models/.gpu-control
VRAM_FREE_THRESHOLD_MB: 8000
```

**Docker/containers touched:** Worker restart only (no GPU, no docker.sock)
**Human approval required:** YES (worker restart)

### Step 4: Verify (no training yet)

1. Start host controller: `sudo systemctl start gpu-controller`
2. Restart worker: `docker compose up -d --build worker`
3. Create a test training run via API
4. Verify: controller receives stop request, stops vLLM, queries VRAM, writes response
5. Verify: worker reads response, checks threshold, proceeds or aborts
6. Verify: controller restarts vLLM after training
7. **Do NOT run actual training yet** — just verify the signaling works

**Docker/containers touched:** Worker restart only
**Human approval required:** YES (verification run)

### Step 5: First real training (separate approval)

Only after Step 4 is verified:
1. Create real training run
2. Worker acquires GPU lock
3. Controller stops vLLM, verifies VRAM
4. Worker runs training subprocess
5. Controller restarts vLLM
6. Verify: artifact at `/models/artifacts/...`

**Docker/containers touched:** vLLM stop/start (via controller)
**Human approval required:** YES (GPU training)

---

## 11. What Requires Human Approval

| # | Action | Risk | Approval |
|---|--------|------|----------|
| **A1** | Create systemd service for gpu-controller | Host-level service | YES |
| **A2** | Add `file_signal` mode to backend code | Code change | YES |
| **A3** | Set `SERVING_CONTROL=file_signal` in worker env | Worker restart | YES |
| **A4** | Verify signaling (no real training) | vLLM stop/start cycle | YES |
| **A5** | First real 0.5B training run | GPU usage + vLLM downtime | YES |
| **A6** | Recreate defnex-vllm with `--max-loras 4` + `VLLM_ALLOW_RUNTIME_LORA_UPDATING=true` | vLLM restart | YES (Phase 2B) |

---

## 12. What MUST NOT Be Automated

| # | Constraint | Rationale |
|---|-----------|-----------|
| **N1** | Never kill GPU processes by PID | Shared H100 with other tenants — killing unrelated processes is destructive |
| **N2** | Never run `nvidia-smi --gpu-reset` | Resets ALL GPU state, affects all tenants |
| **N3** | Never assume VRAM free after vLLM stop | Other processes may change; always measure |
| **N4** | Never bypass the pre-training gate | Every training run must go through stop → measure → check threshold |
| **N5** | Never leave vLLM stopped indefinitely | Host controller watchdog must auto-restart after timeout |
| **N6** | Never commit or log secrets | GPU controller has no secrets; keep it that way |
| **N7** | Never modify defnex-vllm without explicit approval | Existing serving container is off-limits until Phase 2B |
| **N8** | Never run `docker compose down` | Per user constraint — targeted restarts only |
| **N9** | Never mount docker.sock into containers | Per this architecture review — host-side controller is the boundary |
| **N10** | Never train without GPU lock | flock serialization is mandatory, not optional |

---

## Appendix: Network Topology (verified live)

```
Docker Networks:
  bridge (172.17.0.0/16)
  defnex-mlops-experiment_default (172.18.0.0/16)
  ml-close-loop-be_default (172.19.0.0/16)

Container Network Membership:
  defnex-vllm           → defnex-mlops-experiment_default (172.18.0.2)
  backend               → ml-close-loop-be_default (172.19.0.3)
  worker                → ml-close-loop-be_default (172.19.0.4)
  minio                 → ml-close-loop-be_default

Cross-network reachability:
  worker → 172.17.0.1:8001 (host bridge gateway)    : REACHABLE
  worker → 172.18.0.1:8001 (defnex-mlops gateway)   : REACHABLE
  worker → 172.19.0.1:8001 (ml-close-loop gateway)  : REACHABLE
  worker → 172.18.0.2:8000 (defnex-vllm direct)     : UNREACHABLE (different network)
  worker → host.docker.internal:8001                 : UNREACHABLE (not resolved)

Shared mount:
  Worker: /home/ubuntu/defnex-mlops-experiment/outputs → /models
  defnex-vllm: /home/ubuntu/defnex-mlops-experiment/outputs → /models (ro)
  Host: /home/ubuntu/defnex-mlops-experiment/outputs/

  Worker can write to /models/.gpu-control/: YES (verified live)
  Host sees the same path: YES (verified live)
```

---

## Appendix: Existing Code Abstractions

The B1 design works with the existing `ServingControl` / `VRAMReader` / `ServingCoordinator` protocol hierarchy in `gpu_orchestrator.py`. The `serving_cycle()` context manager (lines 154-234) is implementation-agnostic — it calls `control.stop()`, polls `vram.free_mb()`, yields, then `control.start()` + `control.health_check()` in the `finally` block. No changes needed to the cycle logic itself.

```
Current:
  make_coordinator() → SERVING_CONTROL=mock → NoopServingCoordinator
                     → SERVING_CONTROL=shell → RealServingCoordinator(ShellServingControl, NvidiaSmiVRAMReader)

Proposed:
  make_coordinator() → SERVING_CONTROL=mock → NoopServingCoordinator
                     → SERVING_CONTROL=shell → RealServingCoordinator(ShellServingControl, NvidiaSmiVRAMReader)
                     → SERVING_CONTROL=file_signal → RealServingCoordinator(FileSignalingServingControl, FileSignalingVRAMReader)
```

The `training_worker.process_next_job()` function (lines 126-235) is also implementation-agnostic — it just calls `coordinator.cycle()`. The only change needed is the `SERVING_CONTROL` setting.
