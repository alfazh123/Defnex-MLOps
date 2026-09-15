# Phase 2A GPU Handoff — Read-Only Preflight Report

**Date:** 2026-09-14
**Status:** READ-ONLY — NO files modified, NO containers started/stopped/recreated
**Scope:** Determine whether the proposed GPU handoff implementation is technically possible with the current Docker setup

---

## Table of Contents

1. [PRECHECK 1 — Docker GPU Support](#1-precheck-1--docker-gpu-support)
2. [PRECHECK 2 — Worker Image CUDA Compatibility](#2-precheck-2--worker-image-cuda-compatibility)
3. [PRECHECK 3 — Docker Daemon Control from Worker](#3-precheck-3--docker-daemon-control-from-worker)
4. [PRECHECK 4 — Current vLLM Identity](#4-precheck-4--current-vllm-identity)
5. [PRECHECK 5 — GPU Ownership](#5-precheck-5--gpu-ownership)
6. [PRECHECK 6 — Proposed Architecture Review](#6-precheck-6--proposed-architecture-review)
7. [Consolidated Findings](#7-consolidated-findings)
8. [Recommended Phase 2A Design](#8-recommended-phase-2a-design)
9. [Exact Changes Required (Not Implemented)](#9-exact-changes-required-not-implemented)
10. [Human Approvals Required](#10-human-approvals-required)
11. [First Real 0.5B Training Test Sequence](#11-first-real-05b-training-test-sequence)

---

## 1. PRECHECK 1 — Docker GPU Support

| Component | Version / Status | Source |
|-----------|-----------------|--------|
| Docker Server | 29.7.2 | `docker version --format '{{.Server.Version}}'` |
| Docker Compose | v5.5.0 | `docker compose version` |
| NVIDIA Container Toolkit | 1.20.0-1 | `dpkg -l` |
| nvidia-container-cli | 1.20.0 | `/usr/bin/nvidia-container-cli` |
| CDI available | YES | `/etc/cdi/nvidia.yaml` exists |
| CDI devices | `nvidia.com/gpu=0`, `nvidia.com/gpu=GPU-24367616-c99c-5a94-1b35-1cd629e4afbf`, `nvidia.com/gpu=all` | `nvidia-ctk cdi list` |

### defnex-vllm GPU mechanism (read-only inspect)

```
DeviceRequests: [
  {
    "Driver": "cdi",
    "Count": 0,
    "DeviceIDs": ["nvidia.com/gpu=all"],
    "Capabilities": null,
    "Options": null
  }
]
```

**Verdict:** defnex-vllm uses **CDI mode** (`nvidia.com/gpu=all`), not the legacy `--gpus all` Docker flag. The worker should use the **same CDI mechanism** for consistency.

---

## 2. PRECHECK 2 — Worker Image CUDA Compatibility

### Dockerfile base image

```dockerfile
FROM python:3.12-slim
```

- **No CUDA** in the base image — this is expected and correct
- Worker does NOT run GPU code itself — it spawns a subprocess using the mounted training venv

### Mounted training venv (read-only)

| Component | Version | CUDA |
|-----------|---------|------|
| Python | 3.12.3 | N/A |
| PyTorch | 2.11.0+cu130 | CUDA 13.0 |
| `torch.cuda.is_available()` | `True` | Host GPU detected |
| unsloth | 2026.9.2 | N/A |
| trl | 0.24.0 | N/A |
| transformers | 5.5.0 | N/A |
| datasets | 4.3.0 | N/A |

### Training entrypoint mechanism

```python
# app/providers/training_provider.py:93-273
# LocalSubprocessProvider spawns:
subprocess.run([
    "/opt/training-venv/bin/python",  # mounted read-only from host
    str(script_path),
    "--config", str(config_path),
    "--output_dir", str(output_dir),
])
```

**Key insight:** The worker container does NOT need CUDA itself. It spawns a subprocess using `/opt/training-venv/bin/python` which has PyTorch+CUDA. The subprocess needs GPU access via the NVIDIA Container Toolkit.

**Verdict:** CUDA compatibility is confirmed. PyTorch 2.11.0+cu130 in the mounted venv matches the host NVIDIA driver (575.x). No version conflict.

---

## 3. PRECHECK 3 — Docker Daemon Control from Worker

### Current state (live verified)

| Check | Result |
|-------|--------|
| `which docker` inside worker | **NOT FOUND** |
| `/var/run/docker.sock` mounted | **NOT MOUNTED** |
| Worker user | `root` (uid=0) |
| Worker capabilities | Default (no special caps) |
| Worker privileged mode | No |

### Implication

The proposed design:

```
SERVING_STOP_CMD="docker stop defnex-vllm"
SERVING_START_CMD="docker start defnex-vllm"
SERVING_HEALTH_CMD="docker exec defnex-vllm ..."
```

**CANNOT WORK** with the current worker container. The worker has:
1. No `docker` CLI binary
2. No access to `/var/run/docker.sock`
3. No way to control Docker daemon

### Options to resolve

| Option | Mechanism | Security | Prototype-safe | Production-safe |
|--------|-----------|----------|---------------|-----------------|
| **A: Mount docker.sock + install CLI** | Worker controls Docker directly | **LOW** (docker.sock = root on host) | YES | NO |
| **B: Host-side wrapper service** | File-based signal: worker writes flag file, host cron/systemd watches and runs docker commands | MEDIUM | YES | YES |
| **C: Sidecar container** | Separate container with docker.sock, communicates with worker via shared volume/IPC | MEDIUM | YES | YES |
| **D: Direct container management** | Worker uses `nvidia-container-cli` to manage GPU directly (bypass Docker) | LOW | NO | NO |

---

## 4. PRECHECK 4 — Current vLLM Identity

| Property | Value |
|----------|-------|
| Container name | `/defnex-vllm` |
| Image | `vllm/vllm-openai:latest` |
| Image tag | `vllm/vllm-openai:v0.28.0` |
| GPU mechanism | CDI: `nvidia.com/gpu=all` |
| Restart policy | `unless-stopped` |
| Ports | 8000 (container) → 8001 (host) |
| Network | `defnex-mlops-experiment_default` (172.18.0.2) |

### Mounts

| Source (host) | Destination | Mode |
|---------------|-------------|------|
| `/home/ubuntu/defnex-mlops-experiment/outputs` | `/models` | **ro** (read-only) |
| `/home/ubuntu/.cache/huggingface` | `/root/.cache/huggingface` | rw |

### Command

```
--model Qwen/Qwen2.5-0.5B-Instruct
--dtype bfloat16
--max-model-len 1024
--gpu-memory-utilization 0.15
--enable-lora
--max-loras 1
--max-lora-rank 16
--lora-modules mlops-lora=/models/qwen2.5-0.5b-mlops-lora
```

### Environment (filtered)

```
NVIDIA_REQUIRE_CUDA=cuda>=13.0 ...
NV_CUDA_CUDART_VERSION=13.0.96-1
CUDA_VERSION=13.0.2
VLLM_IMAGE_TAG=vllm/vllm-openai:v0.28.0
```

**Note:** `VLLM_ALLOW_RUNTIME_LORA_UPDATING` is **NOT SET**. `--max-loras 1` means only one adapter at a time.

**Verdict:** Proposed stop/start lifecycle would affect exactly this container. No other vLLM containers exist.

---

## 5. PRECHECK 5 — GPU Ownership

### GPU hardware

| Property | Value |
|----------|-------|
| GPU count | 1 |
| GPU name | NVIDIA H100 PCIe |
| GPU UUID | `GPU-24367616-c99c-5a94-1b35-1cd629e4afbf` |
| Total memory | 81,559 MiB |
| Used memory | 78,964 MiB |
| Free memory | 2,116 MiB |
| GPU utilization | 0% (compute idle, memory occupied) |

### Running GPU processes

| PID | Memory |
|-----|--------|
| 18147 | 3,466 MiB |
| 18159 | 3,464 MiB |
| 22297 | 5,470 MiB |
| 22279 | 5,466 MiB |
| 22409 | 5,488 MiB |
| 22701 | 11,034 MiB |
| 21233 | 14,198 MiB |
| 23079 | 1,446 MiB |
| 23150 | 964 MiB |
| 23085 | 1,338 MiB |
| 24059 | 1,740 MiB |
| 2170057 | 7,594 MiB |
| 982687 | 15,150 MiB |
| 2955651 | 2,056 MiB |

**Total:** 78,964 MiB used across 14 processes.

**Critical observation:** defnex-vllm is **NOT the only GPU consumer**. There are 14 processes using GPU memory. defnex-vllm's `--gpu-memory-utilization 0.15` allocates ~12,234 MiB (15% of 81,559). The remaining ~66,730 MiB is used by other processes.

**Implication:** Stopping defnex-vllm frees only ~12,234 MiB. For Qwen2.5-0.5B training, ~2-4 GB is needed. Net free after stop: ~14,350 MiB (12,234 from vLLM + 2,116 current free). This is sufficient for 0.5B training.

**Verdict:** defnex-vllm is the expected GPU consumer for inference. Other processes are unrelated workloads (likely other tenants or system processes).

---

## 6. PRECHECK 6 — Proposed Architecture Review

### Current proposed design

```
worker (Docker container)
  → docker stop defnex-vllm
  → verify VRAM free via nvidia-smi
  → Unsloth training (subprocess via mounted venv)
  → docker start defnex-vllm
  → load adapter via HTTP
```

### Option A: Worker controls Docker via docker.sock

**Mechanism:** Mount `/var/run/docker.sock` into worker, install `docker` CLI.

**Pros:**
- Simplest implementation
- Direct control
- Matches existing `ShellServingControl` code exactly

**Cons:**
- **docker.sock = root access on host** — any process in the container can:
  - Start/stop any container
  - Mount host filesystem
  - Install packages
  - Execute arbitrary commands as root
- Security risk even on dev VM
- Not acceptable for production

**Verdict:** Acceptable for **prototype only** on isolated dev VM. Not for production.

### Option B: Host-side wrapper service

**Mechanism:** Worker writes a signal file (e.g., `/models/.gpu_signal.stop`), host-level script watches and executes docker commands.

**Pros:**
- Worker never gets docker.sock access
- Host retains full control
- Can add authorization/logging
- Production-safe pattern

**Cons:**
- More complex implementation
- Requires host-side daemon or cron
- Latency: signal file polling interval adds delay

**Verdict:** Better security, slightly more complex. Good for production.

### Option C: Separate training/inference GPU nodes

**Mechanism:** Training runs on a dedicated GPU node, no stop/start handoff needed.

**Pros:**
- No downtime
- No GPU contention
- Clean separation

**Cons:**
- Requires additional hardware (second GPU or second VM)
- Not available in current environment
- Overkill for prototype

**Verdict:** Ideal but not applicable — single GPU environment.

### Recommendation for THIS VM (prototype)

**Option A (docker.sock) is the safest for the prototype** because:
1. Single-user dev VM — no multi-tenant security concern
2. Simplest to implement and debug
3. Matches existing `ShellServingControl` code exactly
4. User explicitly approved this approach in prior conversation
5. Production migration will use Option B or C regardless

---

## 7. Consolidated Findings

### 1. GPU access feasibility

| Check | Status |
|-------|--------|
| NVIDIA Container Toolkit installed | YES (v1.20.0-1) |
| CDI available | YES (`/etc/cdi/nvidia.yaml`) |
| CDI devices | `nvidia.com/gpu=all` available |
| defnex-vllm uses CDI | YES (confirmed via docker inspect) |
| Worker can use CDI | **YES** — same mechanism available |
| GPU memory sufficient for 0.5B training | YES (~14 GB free after vLLM stop) |

**Feasibility: CONFIRMED**

### 2. CUDA/Unsloth feasibility

| Check | Status |
|-------|--------|
| Training venv on host | YES (Python 3.12.3, PyTorch 2.11.0+cu130) |
| CUDA version match | YES (PyTorch cu130 matches host driver) |
| unsloth installed | YES (v2026.9.2) |
| torch.cuda.is_available() | True (on host) |
| Worker Dockerfile has CUDA | NO (expected — not needed) |
| Subprocess can use GPU | **YES** — via mounted venv + CDI passthrough |

**Feasibility: CONFIRMED**

### 3. Docker-control feasibility from worker

| Check | Status |
|-------|--------|
| Docker CLI in worker | **NOT FOUND** |
| docker.sock mounted | **NOT MOUNTED** |
| Worker can run docker commands | **NO** |

**Feasibility: NOT POSSIBLE without changes**

**BLOCKER:** Worker cannot execute `docker stop/start defnex-vllm` without:
1. Installing `docker` CLI in the worker image (add to Dockerfile)
2. Mounting `/var/run/docker.sock` (add to docker-compose.yml)

### 4. Exact blockers

| # | Blocker | Severity | Resolution |
|---|---------|----------|------------|
| **B1** | No `docker` CLI in worker container | **P0** | Add `apt-get install -y docker.io` to Dockerfile |
| **B2** | No `/var/run/docker.sock` mount | **P0** | Add volume mount to docker-compose.yml worker service |
| **B3** | `SERVING_CONTROL=mock` | **P0** | Set to `shell` in worker env |
| **B4** | `VRAM_READER=mock` | **P0** | Set to `nvidia_smi` in worker env |
| **B5** | `SERVING_STOP_CMD` empty | **P0** | Set to `docker stop defnex-vllm` |
| **B6** | `SERVING_START_CMD` empty | **P0** | Set to `docker start defnex-vllm` |
| **B7** | `SERVING_HEALTH_CMD` empty | **P0** | Set to health check command |
| **B8** | Worker has no GPU access | **P0** | Add CDI device request to docker-compose.yml |
| **B9** | `--max-loras 1` on defnex-vllm | **P0** | Recreate with `--max-loras 4` (Phase 2B) |
| **B10** | `VLLM_ALLOW_RUNTIME_LORA_UPDATING` not set | **P0** | Add to defnex-vllm env (Phase 2B) |

### 5. Safest minimal Phase 2A design

```
┌─────────────────────────────────────────────────────────────┐
│                    PROPOSED PHASE 2A                         │
│                                                              │
│  docker-compose.yml changes (worker service only):           │
│                                                              │
│  1. Add GPU access:                                          │
│     deploy:                                                  │
│       resources:                                             │
│         reservations:                                        │
│           devices:                                            │
│             - driver: nvidia                                  │
│               count: 1                                        │
│               capabilities: [gpu]                             │
│                                                              │
│  2. Add docker.sock mount:                                   │
│     volumes:                                                 │
│       - /var/run/docker.sock:/var/run/docker.sock             │
│                                                              │
│  3. Set environment:                                         │
│     SERVING_CONTROL: shell                                    │
│     VRAM_READER: nvidia_smi                                   │
│     SERVING_STOP_CMD: "docker stop defnex-vllm"              │
│     SERVING_START_CMD: "docker start defnex-vllm"            │
│     SERVING_HEALTH_CMD: "docker inspect defnex-vllm ..."     │
│                                                              │
│  4. Dockerfile change:                                       │
│     RUN apt-get update && apt-get install -y docker.io       │
│                                                              │
│  5. Recreate:                                                │
│     docker compose up -d --build worker                      │
│                                                              │
│  6. Verify:                                                  │
│     docker exec worker docker ps | grep defnex-vllm          │
│     docker exec worker nvidia-smi                            │
└─────────────────────────────────────────────────────────────┘
```

### 6. Whether docker.sock should be used

**YES for this prototype.** Rationale:

1. **Single-user dev VM** — no multi-tenant security concern
2. **User explicitly approved** — prior conversation confirmed docker.sock approach
3. **Matches existing code** — `ShellServingControl` is designed for shell-out to docker
4. **Simplest to implement** — 3 changes (Dockerfile, compose, env)
5. **Production will differ** — PRD v2 targets Kubernetes + dedicated GPU nodes, not docker.sock

**Production alternative:** Use Option B (host-side wrapper) or Option C (separate GPU nodes).

### 7. Exact changes required (NOT IMPLEMENTED)

#### File: `ml-close-loop-be/Dockerfile`

Add Docker CLI installation:

```dockerfile
# After existing RUN pip install ...
RUN apt-get update && \
    apt-get install -y --no-install-recommends docker.io && \
    rm -rf /var/lib/apt/lists/*
```

#### File: `ml-close-loop-be/docker-compose.yml`

Worker service additions:

```yaml
worker:
  build: .
  container_name: defnex-ml-close-loop-be-worker-1
  restart: unless-stopped
  env_file: .env
  environment:
    - SERVING_BACKEND=vllm
    - VLLM_URL=http://172.17.0.1:8001
    - SERVING_CONTROL=shell
    - VRAM_READER=nvidia_smi
    - SERVING_STOP_CMD=docker stop defnex-vllm
    - SERVING_START_CMD=docker start defnex-vllm
    - SERVING_HEALTH_CMD=docker inspect --format='{{.State.Running}}' defnex-vllm
    - TRAINING_PYTHON=/opt/training-venv/bin/python
    - TRAINING_VENV_PYTHON=/opt/training-venv/bin/python
    - VRAM_FREE_THRESHOLD_MB=8000
    - GPU_LOCK_FILE=/app/data/gpu.lock
    - GPU_LOCK_TIMEOUT=600
    - GPU_VRAM_READER=nvidia_smi
    - TRAINING_TIMEOUT=1800
  volumes:
    - ./data:/app/data
    - /home/ubuntu/defnex-mlops-experiment/venvs/training-venv:/opt/training-venv:ro
    - /home/ubuntu/defnex-mlops-experiment/outputs:/models
    - /var/run/docker.sock:/var/run/docker.sock
  deploy:
    resources:
      reservations:
        devices:
          - driver: nvidia
            count: 1
            capabilities: [gpu]
```

### 8. Human approvals required

| # | Approval | What it covers |
|---|----------|----------------|
| **A1** | GPU passthrough for worker | H100 shared — worker gets 1 GPU via CDI |
| **A2** | docker.sock mount | Worker gains root-level Docker control on host |
| **A3** | Docker CLI in worker image | Increases image size (~100MB) |
| **A4** | Stop/start defnex-vllm | ~30-120s downtime per training run |

### 9. First real 0.5B training test sequence

```
Phase 2A: Worker GPU + Docker control
──────────────────────────────────────
1. Add docker CLI to Dockerfile
2. Add docker.sock mount to docker-compose.yml
3. Add GPU access (CDI) to docker-compose.yml
4. Set SERVING_CONTROL=shell, VRAM_READER=nvidia_smi
5. Configure serving stop/start/health commands
6. docker compose up -d --build worker
7. Verify: docker exec worker docker ps | grep defnex-vllm
8. Verify: docker exec worker nvidia-smi

Phase 2B: Recreate defnex-vllm (separate, requires approval)
────────────────────────────────────────────────────────────
1. docker stop defnex-vllm
2. docker rm defnex-vllm
3. Recreate with: VLLM_ALLOW_RUNTIME_LORA_UPDATING=true, --max-loras 4
4. Verify: POST /v1/load_lora_adapter returns 200

Phase 2C: First training test
─────────────────────────────
1. Create training run via API
2. Worker acquires GPU lock
3. Worker stops defnex-vllm
4. Worker verifies VRAM free >= 8GB
5. Worker spawns training subprocess
6. Training completes
7. Worker restarts defnex-vllm
8. Verify: training status = COMPLETED
9. Verify: artifact at /models/artifacts/...
10. Deploy adapter via API
11. Run inference → verify non-mock response
```

---

## Appendix: Container Configuration Summary (Post-Phase 2A)

| Container | GPU Access | docker.sock | SERVING_CONTROL | Restart |
|-----------|-----------|-------------|-----------------|---------|
| backend | No | No | N/A | unless-stopped |
| worker | **YES (CDI)** | **YES** | **shell** | unless-stopped |
| defnex-vllm | **YES (CDI)** | N/A | N/A | unless-stopped |
| minio | No | No | N/A | unless-stopped |
