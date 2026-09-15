# DEFNEX MLOps — Current Runtime Architecture

**Date:** 2026-09-15
**Status:** PROTOTYPE — Single Server

---

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│ HOST: ASTACITA-AI                                                              │
│ NVIDIA H100 PCIe 80GB (SHARED — ~65GB used by external/unknown workloads)      │
│                                                                                 │
│ ┌──────────────────────────────────────────────────────────────────────────┐   │
│ │ Docker Compose Project: ml-close-loop-be                                │   │
│ │ Network: ml-close-loop-be_default (172.19.0.0/16)                       │   │
│ │                                                                         │   │
│ │  ┌─────────────────┐   ┌─────────────────┐   ┌─────────────────┐       │   │
│ │  │    BACKEND      │   │     WORKER      │   │     MINIO       │       │   │
│ │  │  ml-close-loop- │   │  ml-close-loop- │   │  ml-close-loop- │       │   │
│ │  │  be-backend-1   │   │  be-worker-1    │   │  be-minio-1     │       │   │
│ │  │                 │   │                 │   │                 │       │   │
│ │  │  FastAPI        │   │  Training       │   │  Object Storage │       │   │
│ │  │  :8000          │   │  Worker         │   │  :9000 API      │       │   │
│ │  │                 │   │  (polling)      │   │  :9001 Console  │       │   │
│ │  │  SERVING_BACKEND│   │                 │   │                 │       │   │
│ │  │  =vllm          │   │  SERVING_CONTROL│   │  Status:        │       │   │
│ │  │                 │   │  =file_signal   │   │  HEALTHY        │       │   │
│ │  │  VLLM_URL=      │   │                 │   │                 │       │   │
│ │  │  http://172.17  │   │  GPU_ACCESS=YES │   │  (not wired to  │       │   │
│ │  │  .0.1:8001      │   │  CUDA=YES       │   │  backend yet)   │       │   │
│ │  └────────┬────────┘   └────────┬────────┘   └─────────────────┘       │   │
│ │           │                     │                                       │   │
│ │           │  ┌──────────────────┘                                       │   │
│ │           │  │                                                           │   │
│ │           │  │  VOLUME MOUNTS:                                          │   │
│ │           │  │  ./app:/app/app (source code)                            │   │
│ │           │  │  ./data:/app/data (DB + artifacts)                       │   │
│ │           │  │  /models → experiment/outputs (shared with vLLM)         │   │
│ │           │  │  /opt/training-venv → experiment/.venv (Unsloth)         │   │
│ │           │  │                                                           │   │
│ │           │  │  TRAINING EXECUTION:                                     │   │
│ │           │  │  ┌─────────────────────────────────────────────┐         │   │
│ │           │  │  │ /opt/training-venv/bin/python               │         │   │
│ │           │  │  │   → app/training/run_training.py            │         │   │
│ │           │  │  │     → Unsloth + SFTTrainer                  │         │   │
│ │           │  │  │       → GPU (H100)                          │         │   │
│ │           │  │  │                                               │         │   │
│ │           │  │  │ STATUS: BROKEN                               │         │   │
│ │           │  │  │ (API incompatibility — all runs fail)        │         │   │
│ │           │  │  └─────────────────────────────────────────────┘         │   │
│ │           │  │                                                           │   │
│ │           │  │  GPU SIGNALING PATH:                                      │   │
│ │           │  │  ┌─────────────────────────────────────────────┐         │   │
│ │           │  │  │ Worker writes request.json                  │         │   │
│ │           │  │  │   → /models/.gpu-control/request.json       │         │   │
│ │           │  │  │     → Host GPU controller reads             │         │   │
│ │           │  │  │       → Docker stop/start defnex-vllm       │         │   │
│ │           │  │  │         → nvidia-smi VRAM check             │         │   │
│ │           │  │  │           → response.json written           │         │   │
│ │           │  │  │             → Worker reads response          │         │   │
│ │           │  │  └─────────────────────────────────────────────┘         │   │
│ │           │  │                                                           │   │
│ └───────────┼──┼───────────────────────────────────────────────────────────┘   │
│             │  │                                                               │
│             │  │  VLLM_URL: http://172.17.0.1:8001                            │
│             │  │  (Docker bridge gateway)                                      │
│             │  │                                                               │
│ ┌───────────┼──┼───────────────────────────────────────────────────────────┐   │
│             │  │  Docker Compose Project: defnex-mlops-experiment          │   │
│             │  │  Network: defnex-mlops-experiment_default (172.18.0.0/16) │   │
│             │  │                                                           │   │
│             │  │  ┌─────────────────────────────────────────────────┐     │   │
│             │  └──│►  defnex-vllm                                    │     │   │
│                └──│►  vllm/vllm-openai:latest                        │     │   │
│                   │                                                 │     │   │
│                   │  Model: Qwen/Qwen2.5-0.5B-Instruct             │     │   │
│                   │  LoRA: mlops-lora (rank=16)                     │     │   │
│                   │  Port: 8001:8000                                │     │   │
│                   │  GPU Memory: 15% utilization                    │     │   │
│                   │                                                 │     │   │
│                   │  Mount:                                         │     │   │
│                   │  /models ← experiment/outputs (read-only)       │     │   │
│                   │  /root/.cache/huggingface ← ~/.cache/hf         │     │   │
│                   │                                                 │     │   │
│                   │  STATUS: MANAGED BY GPU CONTROLLER              │     │   │
│                   │  (currently STOPPED — was running earlier)      │     │   │
│                   └─────────────────────────────────────────────────┘     │   │
│                                                                           │   │
│ └──────────────────────────────────────────────────────────────────────────┘   │
│                                                                                 │
│ ┌──────────────────────────────────────────────────────────────────────────┐   │
│ │ HOST-SIDE PROCESS                                                        │   │
│ │                                                                          │   │
│ │  gpu_controller.py (PID 485335)                                          │   │
│ │  Location: /home/ubuntu/defnex-mlops-experiment/gpu_controller/          │   │
│ │  Signal Dir: /home/ubuntu/defnex-mlops-experiment/outputs/.gpu-control/  │   │
│ │                                                                          │   │
│ │  Capabilities:                                                           │   │
│ │  • stop_serving → docker stop defnex-vllm                               │   │
│ │  • start_serving → docker start defnex-vllm                             │   │
│ │  • nvidia-smi VRAM query                                                │   │
│ │  • Health check (localhost:8001/health)                                  │   │
│ │  • Crash recovery via active.json                                       │   │
│ │  • 30-minute hard timeout                                               │   │
│ │                                                                          │   │
│ │  Restriction: ONLY controls "defnex-vllm" container                     │   │
│ │  NOT a systemd service (runs as background process)                     │   │
│ └──────────────────────────────────────────────────────────────────────────┘   │
│                                                                                 │
│ ┌──────────────────────────────────────────────────────────────────────────┐   │
│ │ EXTERNAL GPU WORKLOADS (NOT controlled by DEFNEX)                        │   │
│ │                                                                          │   │
│ │  14 processes using ~65GB VRAM                                          │   │
│ │  PIDs: 18147, 18159, 21233, 22279, 22297, 22409, 22701, 23079,        │   │
│ │        23085, 23150, 24059, 2802940, 2955651, 2170057                   │   │
│ │  Owner: UNKNOWN (not DEFNEX-related)                                    │   │
│ │                                                                          │   │
│ │  ⚠ NEVER kill, reset, or interfere with these processes                 │   │
│ └──────────────────────────────────────────────────────────────────────────┘   │
│                                                                                 │
│ DATABASE: /home/ubuntu/Defnex-MLOps/ml-close-loop-be/data/app.db (SQLite)      │
│ ARTIFACTS: /home/ubuntu/Defnex-MLOps/ml-close-loop-be/data/artifacts/ (local)  │
│ DATASETS: /home/ubuntu/Defnex-MLOps/ml-close-loop-be/data/datasets/            │
└─────────────────────────────────────────────────────────────────────────────────┘
```

---

## Component Ownership

| Component | Repository | Runtime Owner | Actually Running? | Legacy Dependency? |
|-----------|------------|---------------|-------------------|-------------------|
| Backend | ml-close-loop-be | ml-close-loop-be Compose | YES | NO |
| Worker | ml-close-loop-be | ml-close-loop-be Compose | YES | YES (.venv + outputs) |
| MinIO | ml-close-loop-be | ml-close-loop-be Compose | YES | NO |
| SQLite DB | ml-close-loop-be | Backend container | YES | NO |
| defnex-vllm | defnex-mlops-experiment | Standalone container | MANAGED (stopped) | YES (entire runtime) |
| GPU Controller | defnex-mlops-experiment | Host process | YES | YES (entire runtime) |
| Training venv | defnex-mlops-experiment | Worker mount | YES (mounted) | YES |
| LoRA adapter | defnex-mlops-experiment | Worker/vLLM mount | YES (mounted) | YES |
| Compose "serving" | ml-close-loop-be | NOT STARTED | NO | N/A |

---

## Data Flow: Training Attempt (Current — Failing)

```
1. API: POST /api/v1/training-runs (creates PENDING run)
         │
2. Worker: Polls DB, finds PENDING run
         │
3. Worker: Acquires GPU lock (flock)
         │
4. Worker: File signal → stop_serving
         │
5. GPU Controller: docker stop defnex-vllm
         │
6. GPU Controller: nvidia-smi → VRAM free check
         │
7. Worker: Reads response → vLLM stopped, VRAM OK
         │
8. Worker: Spawns /opt/training-venv/bin/python run_training.py
         │
9. Training script: Loads Unsloth + model
         │
10. Training script: ✗ FAILS (API incompatibility)
         │
11. Worker: Marks run FAILED
         │
12. Worker: File signal → start_serving
         │
13. GPU Controller: docker start defnex-vllm
         │
14. GPU Controller: Health check → vLLM ready
```

---

## Data Flow: Target (When Training Works)

```
1. API: POST /api/v1/training-runs
         │
2. Worker: GPU lock → stop vLLM → verify VRAM
         │
3. Worker: Run training → produce adapter
         │
4. Worker: Register artifact (immutable per-version)
         │
5. Worker: Register ModelVersion in DB
         │
6. Worker: Start vLLM
         │
7. API: POST /api/v1/deployments (deploy model version)
         │
8. Deployment service: Load adapter via vLLM API
         │
9. Deployment service: Smoke test inference
         │
10. Deployment service: Move pointer to new version
         │
11. API: POST /api/v1/inference (real inference)
         │
12. vLLM: Generate with loaded adapter
```

---

## Network Topology

```
Host Network
├── :8000 → Backend (ml-close-loop-be)
├── :8001 → defnex-vllm (when running)
├── :9000 → MinIO API
├── :9001 → MinIO Console
│
Docker Bridge: ml-close-loop-be_default (172.19.0.0/16)
├── 172.19.0.x → Backend
├── 172.19.0.x → Worker
├── 172.19.0.x → MinIO
├── 172.17.0.1 → Host gateway (reaches defnex-vllm on :8001)
│
Docker Bridge: defnex-mlops-experiment_default (172.18.0.0/16)
├── 172.18.0.2 → defnex-vllm
```

---

## Key Configuration (Effective Runtime)

| Setting | Backend | Worker | Source |
|---------|---------|--------|--------|
| SERVING_BACKEND | vllm | vllm | docker-compose.yml override |
| VLLM_URL | http://172.17.0.1:8001 | http://172.17.0.1:8001 | docker-compose.yml override |
| SERVED_BASE_MODEL | Qwen/Qwen2.5-0.5B-Instruct | Qwen/Qwen2.5-0.5B-Instruct | docker-compose.yml override |
| SERVING_CONTROL | mock | file_signal | .env (backend) / docker-compose.yml (worker) |
| GPU_CONTROL_DIR | (unused) | /models/.gpu-control | docker-compose.yml override |
| VRAM_FREE_THRESHOLD_MB | (unused) | 8000 | docker-compose.yml override |
| TRAINING_PYTHON | python3 | /opt/training-venv/bin/python | .env (backend) / docker-compose.yml (worker) |
| ARTIFACT_STORAGE_DIR | data/artifacts | /models/artifacts | .env (backend) / docker-compose.yml (worker) |
| ARTIFACT_BACKEND | local | local | .env |
| MINIO_ENDPOINT | minio:9000 | minio:9000 | docker-compose.yml override |
| DATABASE_URL | sqlite:///./data/app.db | sqlite:///./data/app.db | .env |

---

*Architecture documented: 2026-09-15 05:15 UTC*
