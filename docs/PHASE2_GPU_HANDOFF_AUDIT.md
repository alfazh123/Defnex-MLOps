# Phase 2 GPU Handoff Readiness Audit

**Date:** 2026-09-14
**Status:** READ-ONLY AUDIT COMPLETE
**Scope:** Determine whether the current architecture can safely hand off the shared H100 between vLLM serving and Unsloth training

---

## Table of Contents

1. [Current GPU Lifecycle Diagram](#1-current-gpu-lifecycle-diagram)
2. [Exact Current Worker GPU State](#2-exact-current-worker-gpu-state)
3. [Exact Current vLLM Lifecycle Control Capability](#3-exact-current-vllm-lifecycle-control-capability)
4. [P0 / P1 Blockers](#4-p0--p1-blockers)
5. [Minimal Changes for First Real 0.5B Training](#5-minimal-changes-required-for-first-real-05b-unsloth-training)
6. [Minimal Changes for Runtime LoRA Deployment](#6-minimal-changes-required-for-runtime-lora-deployment)
7. [Does First Training Require Stopping defnex-vllm?](#7-does-the-first-real-training-run-require-stopping-defnex-vllm)
8. [Can Backend Automatically Restore Serving?](#8-can-backend-automatically-restore-serving-afterward)
9. [Exact Human Approvals Required](#9-exact-human-approvals-required)
10. [Recommended Phase 2 Sequence](#10-recommended-phase-2a--2b--2c-sequence)
11. [Evidence Classification](#11-evidence-classification)

---

## 1. Current GPU Lifecycle Diagram

```
                    ┌─────────────────────────────────────────────┐
                    │          CURRENT STATE (no GPU access)       │
                    │                                              │
                    │  defnex-vllm (standalone, owns GPU)          │
                    │  ├── Port 8001:8000                          │
                    │  ├── --enable-lora --max-loras 1             │
                    │  ├── VLLM_ALLOW_RUNTIME_LORA_UPDATING: NOT SET│
                    │  ├── Mount: /models (outputs:ro)             │
                    │  └── Restart: unless-stopped                 │
                    │                                              │
                    │  worker (Docker, NO GPU)                     │
                    │  ├── ARTIFACT_STORAGE_DIR=/models/artifacts  │
                    │  ├── SERVING_CONTROL=mock (default)          │
                    │  ├── TRAINING_PYTHON=/opt/training-venv/bin/python │
                    │  └── GPU access: NONE                        │
                    │                                              │
                    │  backend (Docker, NO GPU)                    │
                    │  ├── SERVING_BACKEND=vllm                    │
                    │  └── VLLM_URL=http://172.17.0.1:8001         │
                    └─────────────────────────────────────────────┘

                    ┌─────────────────────────────────────────────┐
                    │          TARGET STATE (with GPU access)       │
                    │                                              │
                    │  step 1: stop vLLM                           │
                    │    └── docker stop defnex-vllm               │
                    │                                              │
                    │  step 2: verify VRAM free >= 8GB             │
                    │    └── nvidia-smi --query-gpu=memory.free    │
                    │                                              │
                    │  step 3: train (worker, GPU-passthrough)     │
                    │    └── /opt/training-venv/bin/python          │
                    │        run_training.py --config ...           │
                    │        → /models/artifacts/.../adapter.*      │
                    │                                              │
                    │  step 4: restart vLLM                        │
                    │    └── docker start defnex-vllm               │
                    │                                              │
                    │  step 5: load adapter into vLLM              │
                    │    └── POST /v1/load_lora_adapter             │
                    │        lora_name: {model_id}-v{version}      │
                    │        lora_path: /models/artifacts/.../      │
                    │                                              │
                    │  step 6: inference                            │
                    │    └── POST /v1/completions                   │
                    └─────────────────────────────────────────────┘
```

---

## 2. Exact Current Worker GPU State

| Property | Value | Source | Verified |
|----------|-------|--------|----------|
| GPU access in docker-compose | **NONE** | `docker-compose.yml` worker service — no `deploy.resources.reservations.devices` | **live** |
| GPU runtime | **NONE** | No `runtime: nvidia` | **live** |
| Dockerfile CUDA | **NONE** | `python:3.12-slim` base image | **code** |
| `nvidia-container-toolkit` on host | v1.20.0-1 | `/usr/bin/nvidia-container-cli` | **live** |
| Training venv CUDA | PyTorch 2.11.0+cu130, `torch.cuda.is_available()=True` | `/home/ubuntu/defnex-mlops-experiment/.venv` | **live** |
| Training subprocess Python | `/opt/training-venv/bin/python` (mounted read-only) | `docker-compose.yml` line 27 | **live** |
| `SERVING_CONTROL` | `mock` (default) — orchestrator is no-op | `config.py:100` | **code** |
| `VRAM_READER` | `mock` (default) — no nvidia-smi | `config.py:111` | **code** |
| `run_training.py` imports | torch/unsloth inside `_run_training()` only (lazy) | `app/training/run_training.py:49-62` | **code** |

---

## 3. Exact Current vLLM Lifecycle Control Capability

| Control | Configured | Implementation | Live |
|---------|-----------|----------------|------|
| Stop serving | `SERVING_STOP_CMD=""` (empty) | `ShellServingControl.stop()` runs shell cmd via `subprocess.run` | **No-op** (empty cmd = no-op) |
| Start serving | `SERVING_START_CMD=""` (empty) | `ShellServingControl.start()` runs shell cmd | **No-op** |
| Health check | `SERVING_HEALTH_CMD=""` (empty) | `ShellServingControl.health_check()` runs shell cmd | **No-op** |
| VRAM check | `VRAM_READER=mock` | `NvidiaSmiVRAMReader` runs `nvidia-smi` | **Not instantiated** |
| Serving control mode | `SERVING_CONTROL=mock` | `make_coordinator()` → `NoopServingCoordinator` (yield only) | **No-op** |
| GPU lock | `data/gpu.lock`, 300s timeout | `fcntl.flock(LOCK_EX)` | **Implemented** |
| Deploy adapter load | `POST /v1/load_lora_adapter` | `VLLMServingBackend.deploy()` | **Implemented** |
| Deploy adapter unload | `POST /v1/unload_lora_adapter` | `VLLMServingBackend.unload()` (404 = ok) | **Implemented** |
| Deploy idempotency | **None** — unconditional load call | `serving.py:206-233` | **Gap** |

---

## 4. P0 / P1 Blockers

| # | Blocker | Severity | Status | What blocks |
|---|---------|----------|--------|-------------|
| **P0-1** | Worker has **zero GPU access** in docker-compose | **P0** | **Not configured** | Any real training cannot run inside the worker container |
| **P0-2** | `VLLM_ALLOW_RUNTIME_LORA_UPDATING` **not set** on defnex-vllm | **P0** | **Not configured** | `POST /v1/load_lora_adapter` returns 404; runtime LoRA load/unload is disabled |
| **P0-3** | `--max-loras 1` on defnex-vllm | **P0** | **Not configured** | Cannot load new adapter before unloading old; deploy pattern requires >= 2 |
| **P0-4** | `SERVING_CONTROL=mock` — orchestrator is no-op | **P0** | **Not configured** | Worker won't stop vLLM before training; training and vLLM would fight for GPU |
| **P0-5** | `SERVING_STOP_CMD`, `SERVING_START_CMD`, `SERVING_HEALTH_CMD` all empty | **P0** | **Not configured** | Even if `SERVING_CONTROL=shell`, validator rejects empty commands |
| **P0-6** | `VRAM_READER=mock` — no nvidia-smi in worker | **P0** | **Not configured** | Cannot verify VRAM is free before training |
| **P1-1** | No idempotency guard on `deploy()` | **P1** | **Code gap** | Duplicate load calls may fail with non-404 errors |
| **P1-2** | `_adapter_path()` passes `s3://` URIs verbatim | **P1** | **Code gap** | vLLM cannot resolve S3 URIs (only affects S3 backend) |

---

## 5. Minimal Changes Required for First Real 0.5B Unsloth Training

| # | Change | Where | What |
|---|--------|-------|------|
| **C1** | Add GPU access to worker | `docker-compose.yml` worker service | Add `deploy.resources.reservations.devices` block with `driver: nvidia, count: 1, capabilities: [gpu]` |
| **C2** | Set `SERVING_CONTROL=shell` | `docker-compose.yml` worker env | `SERVING_CONTROL: shell` |
| **C3** | Set `VRAM_READER=nvidia_smi` | `docker-compose.yml` worker env | `VRAM_READER: nvidia_smi` |
| **C4** | Configure serving commands | `docker-compose.yml` worker env | See exact values below |
| **C5** | Recreate worker container | `docker compose up -d --build worker` | Apply new GPU config |

### Exact Serving Commands (C4)

```yaml
SERVING_STOP_CMD: "docker stop defnex-vllm"
SERVING_START_CMD: "docker start defnex-vllm"
SERVING_HEALTH_CMD: "docker exec defnex-vllm python3 -c \"import urllib.request; urllib.request.urlopen('http://localhost:8000/health', timeout=5)\""
```

---

## 6. Minimal Changes Required for Runtime LoRA Deployment

| # | Change | Where | What |
|---|--------|-------|------|
| **L1** | Recreate defnex-vllm with runtime LoRA flags | `docker-compose.inference.yml` (host) | Add `VLLM_ALLOW_RUNTIME_LORA_UPDATING=true` env var, change `--max-loras` from 1 to 4 |
| **L2** | Add deploy idempotency (optional, P1) | `app/services/serving.py` | Check if adapter already loaded before calling `load_lora_adapter`; or catch 409/already-loaded responses |
| **L3** | Rebuild backend (for deploy path) | Already done — `SERVING_BACKEND=vllm` is set | Backend already calls real vLLM deploy |

### New defnex-vllm docker-compose.inference.yml

```yaml
services:
  vllm:
    image: vllm/vllm-openai:latest
    container_name: defnex-vllm
    restart: unless-stopped
    ports:
      - "8001:8000"
    volumes:
      - ~/.cache/huggingface:/root/.cache/huggingface
      - ./outputs:/models:ro
    devices:
      - "nvidia.com/gpu=all"
    ipc: host
    environment:
      - VLLM_ALLOW_RUNTIME_LORA_UPDATING=true
    command:
      - --model
      - Qwen/Qwen2.5-0.5B-Instruct
      - --dtype
      - bfloat16
      - --max-model-len
      - "1024"
      - --gpu-memory-utilization
      - "0.15"
      - --enable-lora
      - --max-loras
      - "4"
      - --max-lora-rank
      - "64"
```

---

## 7. Does the First Real Training Run Require Stopping defnex-vllm?

**YES.** The GPU orchestrator (`serving_cycle()`) stops vLLM, waits for VRAM free, trains, then restarts vLLM — all inside the GPU flock. Without stopping vLLM, training and vLLM would compete for GPU memory on the shared H100, causing OOM.

### Training Sequence

1. Worker acquires GPU flock
2. `ShellServingControl.stop()` → `docker stop defnex-vllm`
3. `NvidiaSmiVRAMReader.free_mb()` → polls until free >= 8GB
4. `LocalSubprocessProvider.submit()` → spawns `/opt/training-venv/bin/python run_training.py`
5. Training completes → artifact written to `/models/artifacts/`
6. `ShellServingControl.start()` → `docker start defnex-vllm`
7. `ShellServingControl.health_check()` → verify vLLM is up
8. Worker releases GPU flock

### Downtime Estimate

- vLLM stop: ~2-5 seconds
- VRAM verification: ~1-5 seconds (polling)
- Training (Qwen2.5-0.5B, 2 steps): ~30-60 seconds
- vLLM restart: ~10-30 seconds (model load)
- **Total downtime: ~45-100 seconds**

---

## 8. Can Backend Automatically Restore Serving Afterward?

**YES.** The `serving_cycle()` `finally` block (`gpu_orchestrator.py:221-234`) always runs `control.start()` then `control.health_check()`, regardless of training outcome (success, failure, crash, SIGTERM). Failures are logged, never raised — so they never mask the training result.

However, the backend **does not reload adapters** — that's a separate operation via `VLLMServingBackend.deploy()` called from `deployment_service.deploy()`. After vLLM restarts, the previously loaded adapter must be re-deployed via the API.

### Recovery Matrix

| Scenario | vLLM restored? | Adapter restored? |
|----------|---------------|-------------------|
| Training success | **YES** (automatic) | **NO** (requires API call) |
| Training failure | **YES** (automatic) | **NO** (requires API call) |
| Training crash | **YES** (automatic) | **NO** (requires API call) |
| SIGTERM | **YES** (automatic) | **NO** (requires API call) |
| `control.start()` fails | **NO** (logged, best-effort) | **NO** |

---

## 9. Exact Human Approvals Required

| # | Approval | What it covers | Risk if skipped |
|---|----------|----------------|-----------------|
| **A1** | GPU passthrough for worker container | H100 shared with other tenants — worker gets 1 GPU | Worker may interfere with other GPU workloads |
| **A2** | Stop/start defnex-vllm during training | ~30-120s downtime per training run | No approval needed for dev; needed for production |
| **A3** | Recreate defnex-vllm with new flags | Container restart, brief serving interruption | Existing adapter `mlops-lora` temporarily unavailable during restart |
| **A4** | VRAM threshold >= 8GB | Confirmed safe for Qwen2.5-0.5B (needs ~1-2GB for training) | Training may fail if threshold too high or GPU too occupied |

---

## 10. Recommended Phase 2A / 2B / 2C Sequence

### Phase 2A — GPU Passthrough (worker only)

**Goal:** Verify worker can see GPU inside Docker container

1. Add GPU access to worker in `docker-compose.yml`
2. Set `SERVING_CONTROL=shell`, `VRAM_READER=nvidia_smi`
3. Configure serving commands (`docker stop/start/defnex-vllm`)
4. Recreate worker: `docker compose up -d --build worker`
5. Verify: `docker exec worker nvidia-smi` works
6. **No training yet** — just verify GPU visibility

**Files changed:** `docker-compose.yml` (worker service only)
**Downtime:** None (worker restart only)
**Risk:** Low — no training, no vLLM interaction

### Phase 2B — Recreate defnex-vllm with LoRA flags

**Goal:** Enable runtime adapter load/unload on vLLM

1. Back up current defnex-vllm config
2. Recreate with: `VLLM_ALLOW_RUNTIME_LORA_UPDATING=true`, `--max-loras 4`, `--max-lora-rank 64`
3. Verify: `POST /v1/load_lora_adapter` returns 200 (not 404)
4. Verify: existing adapter `mlops-lora` still loads correctly

**Files changed:** `docker-compose.inference.yml` (host, outside this repo)
**Downtime:** ~30-60 seconds (vLLM restart)
**Risk:** Medium — existing adapter temporarily unavailable
**Human approval required:** YES

### Phase 2C — First Real Training Run

**Goal:** Complete end-to-end training → artifact → adapter load → inference

1. Create training run via API
2. Worker acquires GPU lock → stops vLLM → verifies VRAM free → trains → restarts vLLM
3. Verify: training run status = COMPLETED
4. Verify: artifact at `/models/artifacts/...` visible from defnex-vllm
5. Deploy adapter via API → `POST /v1/load_lora_adapter`
6. Run inference → verify non-mock response

**Files changed:** None (runtime only)
**Downtime:** ~45-100 seconds (full training cycle)
**Risk:** High — GPU usage + vLLM downtime + first real training
**Human approval required:** YES (GPU usage + vLLM downtime)

---

## 11. Evidence Classification

| Finding | Evidence level |
|---------|---------------|
| Worker has zero GPU access | **Verified live** (docker inspect, docker-compose.yml) |
| `VLLM_ALLOW_RUNTIME_LORA_UPDATING` not set | **Verified live** (docker inspect defnex-vllm env) |
| `--max-loras 1` | **Verified live** (docker inspect defnex-vllm cmd) |
| `SERVING_CONTROL=mock` | **Verified live** (printenv in worker container) |
| All serving commands empty | **Verified live** (.env file + printenv) |
| GPU orchestrator is no-op | **Verified in code** (gpu_orchestrator.py `make_coordinator`) |
| GPU lock uses flock | **Verified in code** (gpu_lock.py) |
| `serving_cycle` always restarts | **Verified in code** (gpu_orchestrator.py:221-234) |
| `deploy()` has no idempotency | **Verified in code** (serving.py:206-233) |
| Artifact path visible from vLLM | **Verified in code** (both mount `/models`) |
| Training venv has CUDA | **Verified live** (`torch.cuda.is_available()=True`) |
| H100 at 96.8% utilization | **Verified live** (nvidia-smi) |
| `nvidia-container-toolkit` installed | **Verified live** (v1.20.0-1) |
| defnex-vllm restart policy = unless-stopped | **Verified live** (docker inspect) |
| defnex-vllm uses CDI GPU mode | **Verified live** (docker inspect DeviceRequests) |

---

## Appendix A: Mount Topology

```
Host                                        Backend            Worker             defnex-vllm
─────────────────────────────────────────────────────────────────────────────────────────────
/home/ubuntu/Defnex-MLOps/ml-close-loop-be/data
                                            → /app/data        → /app/data
/home/ubuntu/defnex-mlops-experiment/outputs
                                                                → /models          → /models
```

Both `worker` and `defnex-vllm` mount the **same host directory** at `/models`. Backend does **not** have this mount.

---

## Appendix B: Container Configuration Summary

| Container | GPU Access | Mount: /models | SERVING_CONTROL | Restart Policy |
|-----------|-----------|----------------|-----------------|----------------|
| backend | No | No | N/A | unless-stopped |
| worker | **No** (needs C1) | **Yes** | **mock** (needs C2) | unless-stopped |
| defnex-vllm | **Yes** (CDI, all GPUs) | **Yes** (outputs:ro) | N/A | unless-stopped |
| minio | No | No | N/A | unless-stopped |

---

## Appendix C: Key Code References

| Component | File | Lines | Purpose |
|-----------|------|-------|---------|
| GPU lock | `app/workers/gpu_lock.py` | 27-57 | `fcntl.flock(LOCK_EX)` exclusive lock |
| Serving cycle | `app/workers/gpu_orchestrator.py` | 154-234 | Stop → VRAM check → yield → restart |
| Shell control | `app/workers/gpu_orchestrator.py` | 237-277 | Shell out to docker stop/start |
| VRAM reader | `app/workers/gpu_orchestrator.py` | 280-296 | `nvidia-smi --query-gpu=memory.free` |
| Coordinator factory | `app/workers/gpu_orchestrator.py` | 328-358 | `make_coordinator()` — mock or shell |
| Training worker | `app/workers/training_worker.py` | 126-235 | `process_next_job()` — GPU lock + cycle |
| Training provider | `app/providers/training_provider.py` | 93-273 | `LocalSubprocessProvider` — subprocess spawn |
| Training script | `app/training/run_training.py` | 49-62 | Lazy torch/unsloth imports |
| Serving backend | `app/services/serving.py` | 179-319 | `VLLMServingBackend` — load/unload/generate |
| Adapter path | `app/services/serving.py` | 134-147 | `_adapter_path()` — URI to filesystem |
| Artifact storage | `app/services/artifact_storage.py` | 82-153 | `LocalFilesystemArtifactStorage` |
| Model registration | `app/services/model_service.py` | 145-201 | `register_model_version()` → finalize |
| Config | `app/config.py` | 80-114, 136-178 | All GPU/serving config fields + validator |
