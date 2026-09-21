# DEFNEX MLOps — Runtime, Infrastructure & Integration Verification

## Verification Date
2026-09-15 05:15 UTC

> **Update 2026-09-21 (issue #171):** item 2 in §22 and item 6 in the priority list below
> are now partially resolved. `MinioArtifactStorage` was exercised end-to-end against a
> real (locally-run) MinIO server — `store`, `finalize_version` (with checksum), the
> immutability guard, tamper detection, and presigned-URL generation all verified working.
> See `tests/test_artifact_storage_minio_integration.py`. `ARTIFACT_BACKEND` default was
> deliberately left as `local`, not flipped to `minio` — that's a deployment decision
> (this host already has a long-running `ml-close-loop-be-backend-1` container whose
> current artifact data lives on local disk; switching the default here wouldn't migrate
> it) rather than something to change silently as a side effect of a test.

## Verification Scope
Full system verification: host, containers, configuration, GPU, training runtime, serving, artifact storage, database, and integration paths.

## Executive Summary

**Overall Status: PARTIALLY OPERATIONAL — PROTOTYPE STAGE**

The DEFNEX MLOps system is running as a single-server prototype with:
- Backend: RUNNING (FastAPI on port 8000)
- Worker: RUNNING (background training worker)
- MinIO: RUNNING (healthy, port 9000/9001)
- vLLM: CURRENTLY STOPPED (managed by GPU controller, was running earlier)
- GPU Controller: RUNNING (host-side process, managing defnex-vllm lifecycle)
- Database: SQLite, operational with seed data

**Critical Findings:**
1. All training runs have FAILED due to Unsloth/transformers API incompatibility
2. The E2E closed loop is NOT VERIFIED — no real training has succeeded
3. The system depends heavily on the legacy experiment environment
4. vLLM is NOT part of the main repository — it's from the experiment environment
5. Runtime LoRA loading exists in code but has never been successfully exercised
6. The system is single-server only; distributed capabilities are code-only abstractions

---

## 1. Environment

| Property | Value | Evidence |
|----------|-------|----------|
| Hostname | ASTACITA-AI | VERIFIED LIVE |
| OS | Ubuntu (kernel 6.8.0-137-generic) | VERIFIED LIVE |
| CPU | x86_64 | VERIFIED LIVE |
| GPU | NVIDIA H100 PCIe 80GB | VERIFIED LIVE via nvidia-smi |
| GPU UUID | 00007C32:00:00.0 | VERIFIED LIVE |
| VRAM Total | 81,559 MiB | VERIFIED LIVE |
| VRAM Used | ~65,414 MiB (by non-DEFNEX processes) | VERIFIED LIVE |
| VRAM Free | ~15,666 MiB (when vLLM stopped) | VERIFIED LIVE |
| NVIDIA Driver | 580.173.02 | VERIFIED LIVE |
| CUDA Version | 13.0 | VERIFIED LIVE |
| Docker | Running | VERIFIED LIVE |
| Python (host) | 3.12 | VERIFIED LIVE |
| Python (training venv) | 3.12 with CUDA-enabled PyTorch | VERIFIED LIVE (inside worker container) |

---

## 2. Runtime Inventory

### Docker Compose Project: ml-close-loop-be

| Service | Container | Image | Status | Port | Network |
|---------|-----------|-------|--------|------|---------|
| backend | ml-close-loop-be-backend-1 | ml-close-loop-be-backend | RUNNING | 8000:8000 | ml-close-loop-be_default |
| worker | ml-close-loop-be-worker-1 | ml-close-loop-be-worker | RUNNING | (internal) | ml-close-loop-be_default |
| minio | ml-close-loop-be-minio-1 | minio/minio:latest | RUNNING (healthy) | 9000-9001:9000-9001 | ml-close-loop-be_default |
| serving | (not started — GPU profile) | vllm/vllm-openai:latest | NOT RUNNING | 8002:8000 | ml-close-loop-be_default |

### Standalone Container (Legacy Experiment)

| Container | Image | Status | Port | Network |
|-----------|-------|--------|------|---------|
| defnex-vllm | vllm/vllm-openai:latest | STOPPED (managed by GPU controller) | 8001:8000 | defnex-mlops-experiment_default |

### Host-Side Processes

| Process | PID | Status | Location |
|---------|-----|--------|----------|
| gpu_controller.py | 485335 | RUNNING | /home/ubuntu/defnex-mlops-experiment/gpu_controller/ |

**Evidence:** VERIFIED LIVE via `docker ps`, `ps aux`, container inspection.

---

## 3. Backend

| Check | Status | Evidence |
|-------|--------|----------|
| Process running | YES | VERIFIED LIVE — container Up 27 hours |
| HTTP health endpoint | OK | VERIFIED LIVE — GET /api/v1/health returns 200 |
| Database connectivity | OK | VERIFIED LIVE — health check shows db: "ok" |
| vLLM connectivity | ERROR | VERIFIED LIVE — health check shows vllm: "error" (vLLM was stopped at time of check) |
| MinIO connectivity | Not checked in health | INFERRED — backend uses local artifact storage, not MinIO |
| Auth system | Active | VERIFIED LIVE — unauthenticated requests return 401 |
| API prefix | /api/v1 | VERIFIED BY STATIC INSPECTION — main.py |

**Backend Environment (inside container):**
- SERVING_BACKEND: vllm (overridden from .env default "mock")
- VLLM_URL: http://172.17.0.1:8001 (Docker bridge gateway)
- SERVED_BASE_MODEL: Qwen/Qwen2.5-0.5B-Instruct
- ARTIFACT_BACKEND: local
- ARTIFACT_STORAGE_DIR: data/artifacts
- MINIO_ENDPOINT: minio:9000
- SERVING_CONTROL: mock (from .env)

**Evidence:** VERIFIED LIVE via curl, container env inspection, source code.

---

## 4. Worker

| Check | Status | Evidence |
|-------|--------|----------|
| Process running | YES | VERIFIED LIVE — container Up ~1 minute (recently restarted) |
| Training Python exists | YES | VERIFIED LIVE — /opt/training-venv/bin/python exists |
| GPU access (nvidia-smi) | YES | VERIFIED LIVE — nvidia-smi works inside container |
| CUDA available | YES | VERIFIED LIVE — torch.cuda.is_available() = True |
| GPU device count | 1 | VERIFIED LIVE |
| Artifact directory mounted | YES | VERIFIED LIVE — /models/ accessible |
| GPU control directory mounted | YES | VERIFIED LIVE — /models/.gpu-control/ accessible |
| Training venv mounted | YES | VERIFIED LIVE — /opt/training-venv/ accessible |

**Worker Environment (inside container):**
- SERVING_CONTROL: file_signal
- GPU_CONTROL_DIR: /models/.gpu-control
- VRAM_FREE_THRESHOLD_MB: 8000
- TRAINING_PYTHON: /opt/training-venv/bin/python
- TRAINING_SCRIPT_PATH: app/training/run_training.py
- TRAINING_TIMEOUT_SECONDS: 600
- ARTIFACT_STORAGE_DIR: /models/artifacts
- VLLM_URL: http://172.17.0.1:8001

**Evidence:** VERIFIED LIVE via container inspection, env dump, nvidia-smi inside container.

---

## 5. vLLM

### Active Instance: defnex-vllm

| Property | Value | Evidence |
|----------|-------|----------|
| Container name | defnex-vllm | VERIFIED LIVE |
| Image | vllm/vllm-openai:latest | VERIFIED LIVE |
| Status | STOPPED (managed by GPU controller) | VERIFIED LIVE |
| Port | 8001:8000 | VERIFIED LIVE |
| Network | defnex-mlops-experiment_default | VERIFIED LIVE |
| Base model | Qwen/Qwen2.5-0.5B-Instruct | VERIFIED LIVE (from container args) |
| LoRA enabled | YES | VERIFIED LIVE (--enable-lora flag) |
| Max LoRAs | 1 | VERIFIED LIVE |
| Max LoRA rank | 16 | VERIFIED LIVE |
| Registered LoRA | mlops-lora=/models/qwen2.5-0.5b-mlops-lora | VERIFIED LIVE |
| GPU memory utilization | 0.15 (15%) | VERIFIED LIVE |
| Max model len | 1024 | VERIFIED LIVE |
| Restart policy | unless-stopped | VERIFIED LIVE |
| Mount: HF cache | /home/ubuntu/.cache/huggingface:/root/.cache/huggingface | VERIFIED LIVE |
| Mount: models | /home/ubuntu/defnex-mlops-experiment/outputs:/models:ro | VERIFIED LIVE |

**Last known health (before stop):** HTTP 200 — VERIFIED BY LOG

**Models registered (from last /v1/models response):**
- Qwen/Qwen2.5-0.5B-Instruct (base)
- mlops-lora (LoRA adapter, root: /models/qwen2.5-0.5b-mlops-lora)

**Evidence:** VERIFIED LIVE via docker inspect, container args, logs.

### Compose "serving" Service (NOT ACTIVE)

| Property | Value |
|----------|-------|
| Profile | gpu (not activated) |
| Port | 8002:8000 |
| Model | unsloth/Qwen3-0.6B |
| Status | NOT RUNNING |

This service is designed for fresh environments and conflicts with defnex-vllm on port. It is NOT the active vLLM instance.

**Evidence:** VERIFIED BY STATIC INSPECTION of docker-compose.yml.

---

## 6. Backend → vLLM

| Check | Status | Evidence |
|-------|--------|----------|
| Backend SERVING_BACKEND | vllm | VERIFIED LIVE (container env) |
| Backend VLLM_URL | http://172.17.0.1:8001 | VERIFIED LIVE (container env) |
| Network reachability | CONFIGURED | VERIFIED LIVE — backend health checks vLLM at this URL |
| vLLM currently reachable | NO (vLLM stopped) | VERIFIED LIVE — health shows vllm: "error" |
| Serving implementation | VLLMServingBackend | VERIFIED BY STATIC INSPECTION — app/services/serving.py |
| Runtime LoRA load/unload | CODE EXISTS | VERIFIED BY STATIC INSPECTION — POST /v1/load_lora_adapter |
| Smoke test on deploy | CODE EXISTS | VERIFIED BY STATIC INSPECTION — inference_smoke_enabled |
| Base model validation | CODE EXISTS | VERIFIED BY STATIC INSPECTION — served_base_model check |

**Classification: CONNECTED BUT NOT FUNCTIONALLY VERIFIED**

The backend is configured to use vLLM and has the code to load/unload adapters, but:
- vLLM is currently stopped (managed by GPU controller)
- No successful training has produced a valid adapter to deploy
- No successful deployment has been executed
- Inference has never been tested through the backend API

**Evidence:** VERIFIED BY STATIC INSPECTION + LIVE container env.

---

## 7. Artifact Storage

### Storage Backend: Local Filesystem

| Component | Path | Exists | Readable |
|-----------|------|--------|----------|
| Host artifacts | /home/ubuntu/Defnex-MLOps/ml-close-loop-be/data/artifacts/ | YES | YES |
| Backend container | /app/data/artifacts/ | YES | YES |
| Worker container | /models/artifacts/ | YES | YES (empty) |
| vLLM container | /models/ (read-only) | YES | YES |

### Artifact Inventory

| Artifact | Host Path | Backend | Worker | vLLM | URI in DB | Actually Valid |
|----------|-----------|---------|--------|------|-----------|----------------|
| test-smoke-model | data/artifacts/test-smoke-model/ | YES | YES (via /models) | YES (via /models) | file://data/artifacts/test-smoke-model/... | NO — permission denied, root-owned |
| qwen2.5-0.5b-mlops-lora | /home/ubuntu/defnex-mlops-experiment/outputs/ | N/A | YES (via /models) | YES (via /models) | N/A (not in DB) | YES — real LoRA adapter from experiment |

**Seed Data Artifacts (in DB, NOT real):**
- s3://defnex-mlops/artifacts/runs/run-support-01/adapter_model.bin — FAKE URI
- s3://defnex-mlops/artifacts/runs/run-legal-01/adapter_model.bin — FAKE URI
- s3://defnex-mlops/artifacts/runs/run-orca-01/adapter_model.bin — FAKE URI
- s3://defnex-mlops/artifacts/runs/run-ultra-01/adapter_model.bin — FAKE URI

**Critical Issue:** The test-smoke-model artifact directory is owned by root with restricted permissions. The worker cannot read it.

**Evidence:** VERIFIED LIVE via filesystem inspection, permission checks.

---

## 8. Unsloth / Training Runtime

### Training Path

```
Worker (training_worker.py)
  → UnslothTrainingRunner (unsloth_runner.py)
    → subprocess: /opt/training-venv/bin/python -u app/training/run_training.py
      → Unsloth + SFTTrainer
        → GPU
```

| Component | Status | Evidence |
|-----------|--------|----------|
| Unsloth installed | YES | VERIFIED LIVE — in /opt/training-venv (experiment .venv) |
| CUDA in training venv | YES | VERIFIED LIVE — torch.cuda.is_available() = True |
| GPU access in worker | YES | VERIFIED LIVE — nvidia-smi works |
| Training script exists | YES | VERIFIED BY STATIC INSPECTION |
| run_training.py | EXISTS | VERIFIED BY STATIC INSPECTION |

### Training Execution Status

**ALL 5 RECENT TRAINING RUNS HAVE FAILED:**

| Run ID | Status | Error |
|--------|--------|-------|
| run-d3bd7a | FAILED | SFTTrainer.__init__() got unexpected keyword argument 'dataset_text_field' |
| run-46299d | FAILED | SFTTrainer.__init__() got unexpected keyword argument 'tokenizer' |
| run-2378cb | FAILED | TrainingArguments.__init__() got unexpected keyword argument 'callbacks' |
| run-36442b | FAILED | LoraConfig.__init__() got unexpected keyword argument 'use_loftq' |
| run-d70fe3 | FAILED | missing base_model in config |

**Root Cause:** API incompatibility between Unsloth/trl/transformers versions in the experiment venv and the training script's usage of SFTTrainer. The installed versions have breaking API changes.

**GPU Training Runtime: PARTIALLY READY**
- Infrastructure: READY (GPU access, CUDA, Unsloth installed)
- Training script: NOT WORKING (API incompatibility)

**Evidence:** VERIFIED BY LOG (worker logs show all failures), VERIFIED BY DATABASE (training_runs table).

---

## 9. GPU State

### Current GPU Usage

| Property | Value | Evidence |
|----------|-------|----------|
| GPU Index | 0 | VERIFIED LIVE |
| GPU Name | NVIDIA H100 PCIe | VERIFIED LIVE |
| VRAM Total | 81,559 MiB | VERIFIED LIVE |
| VRAM Used | ~65,414 MiB | VERIFIED LIVE |
| VRAM Free | ~15,666 MiB | VERIFIED LIVE |
| Temperature | 60°C | VERIFIED LIVE |
| Power | 100W / 350W | VERIFIED LIVE |
| GPU Utilization | 0% | VERIFIED LIVE |

### GPU Process Ownership Table

| PID | Process | VRAM (MiB) | Owner Status | DEFNEX-related? | Evidence |
|-----|---------|------------|--------------|-----------------|----------|
| 18147 | [Not Found] | 3,466 | UNKNOWN | UNKNOWN | nvidia-smi |
| 18159 | [Not Found] | 3,464 | UNKNOWN | UNKNOWN | nvidia-smi |
| 22297 | [Not Found] | 5,470 | UNKNOWN | UNKNOWN | nvidia-smi |
| 22279 | [Not Found] | 5,466 | UNKNOWN | UNKNOWN | nvidia-smi |
| 22409 | [Not Found] | 5,488 | UNKNOWN | UNKNOWN | nvidia-smi |
| 22701 | [Not Found] | 11,034 | UNKNOWN | UNKNOWN | nvidia-smi |
| 21233 | [Not Found] | 14,198 | UNKNOWN | UNKNOWN | nvidia-smi |
| 23079 | [Not Found] | 1,446 | UNKNOWN | UNKNOWN | nvidia-smi |
| 23150 | [Not Found] | 964 | UNKNOWN | UNKNOWN | nvidia-smi |
| 23085 | [Not Found] | 1,338 | UNKNOWN | UNKNOWN | nvidia-smi |
| 24059 | [Not Found] | 1,740 | UNKNOWN | UNKNOWN | nvidia-smi |
| 2170057 | [Not Found] | 7,594 | UNKNOWN | UNKNOWN | nvidia-smi |
| 2955651 | [Not Found] | 2,056 | UNKNOWN | UNKNOWN | nvidia-smi |
| 2802940 | [Not Found] | 1,860 | UNKNOWN | UNKNOWN | nvidia-smi |

**Total GPU processes:** 14
**DEFNEX-related:** 0 currently (vLLM is stopped)
**Total VRAM used by processes:** ~64,584 MiB

**Note:** No processes are currently DEFNEX-related since vLLM is stopped. All GPU memory is consumed by unknown/external workloads.

**Evidence:** VERIFIED LIVE via nvidia-smi.

---

## 10. GPU Controller / Phase 2A

### Controller Status

| Property | Value | Evidence |
|----------|-------|----------|
| Process running | YES | VERIFIED LIVE — PID 485335 |
| Started at | 2026-09-15T04:23:59Z | VERIFIED LIVE (controller.pid) |
| Signal directory | /home/ubuntu/defnex-mlops-experiment/outputs/.gpu-control/ | VERIFIED LIVE |
| Systemd service | NOT INSTALLED | VERIFIED LIVE — no unit file found |
| Health check timeout | 180 seconds (env override) | VERIFIED BY STATIC INSPECTION |

### File Signaling Protocol

| File | Purpose | Current State |
|------|---------|---------------|
| request.json | Worker → Controller command | Not present (no active request) |
| response.json | Controller → Worker result | Present: phase="vram_checked", vllm_stopped=true, vram_free_mb=17272 |
| active.json | Persistent active cycle state | Present: action=stop_serving, request_id=3b97a0e0... |
| controller.pid | Controller PID file | Present: PID 485335 |

### Controller Capabilities (Verified by Static Inspection)

| Feature | Status | Evidence |
|---------|--------|----------|
| Allowed actions | stop_serving, start_serving only | gpu_controller.py:16 |
| Container restriction | Only "defnex-vllm" | gpu_controller.py:36 |
| No arbitrary shell | No shell=True for request values | gpu_controller.py:16 |
| No PID kill | Confirmed | gpu_controller.py:16 |
| No GPU reset | Confirmed | gpu_controller.py:16 |
| Crash recovery | active.json + hard deadline | gpu_controller.py |
| Watchdog | 30-min hard timeout | gpu_controller.py:37 |
| Health check deadline | 180 seconds (env configurable) | gpu_controller.py:41 |

### Previous Live Test Results (Historical)

From worker logs (2026-09-15 05:06-05:08):
- **STOP:** defnex-vllm stopped, VRAM freed to 17,272 MiB ✓
- **TRAINING:** Submitted (run-d3bd7a), FAILED after ~104 seconds
- **START:** defnex-vllm started, response.json shows "starting" phase
- **HEALTH:** vLLM became healthy (HTTP 200 in logs from earlier start)

**Classification:**
- Stop path: VERIFIED LIVE
- Start path: VERIFIED LIVE (container starts, health eventually succeeds)
- Health timeout: PARTIAL — 180s timeout configured, earlier 60s timeout was insufficient

**Evidence:** VERIFIED LIVE via logs, filesystem state, process inspection.

---

## 11. Dataset Pipeline

### Implementation Status

| Stage | Status | Evidence |
|-------|--------|----------|
| Dataset intake | IMPLEMENTED | VERIFIED BY STATIC INSPECTION — app/api/intake.py |
| Validation | IMPLEMENTED | VERIFIED BY STATIC INSPECTION — app/api/intake_validate.py |
| Versioning | IMPLEMENTED | VERIFIED BY STATIC INSPECTION — dataset_versions table |
| Immutable storage | IMPLEMENTED | VERIFIED BY STATIC INSPECTION — canonical_file_uri |
| Manifest/checksum | IMPLEMENTED | VERIFIED BY STATIC INSPECTION — content_hash in validation_reports |
| Training input | IMPLEMENTED | VERIFIED BY STATIC INSPECTION — training_run.dataset_version_id |

### Database State

| Dataset | Versions | Status |
|---------|----------|--------|
| ds-support-indo-v1 | 2 | PROCESSED |
| ds-legal-qa-v1 | 1 | PROCESSED |
| ds-orca-reasoning-v1 | 1 | PROCESSED |
| ds-ultrachat-general-v1 | 1 | PROCESSED |

**Note:** These are seed data, not real datasets processed through the pipeline.

**Filesystem State:**
- data/datasets/smoke_train.jsonl — exists (535 bytes)
- data/datasets/_staging/ — exists (42 entries)

**Classification:** CODE EXISTS, CONFIGURED, NOT E2E VERIFIED

**Evidence:** VERIFIED BY STATIC INSPECTION + DATABASE.

---

## 12. Training Lifecycle

### State Machine (Verified by Static Inspection)

```
PENDING → QUEUED → STARTING → RUNNING → EVALUATING → ARTIFACT_READY → COMPLETED
                                    ↓
                                  FAILED
                                    ↓
                                 RETRYING (up to max_stale_retries)
                                    ↓
                                  STALE → FAILED
```

### Valid Transitions (from training_service.py)

- PENDING → QUEUED, STARTING, RUNNING, FAILED
- QUEUED → STARTING, RUNNING, FAILED
- STARTING → RUNNING, FAILED
- RUNNING → COMPLETED, FAILED, STALE
- FAILED → RETRYING (limited)
- STALE → RETRYING, FAILED

### Worker Pickup

- Worker polls for oldest PENDING run
- Acquires GPU lock (flock-based)
- Coordinates with serving (stop vLLM → train → start vLLM)
- Submits to TrainingProvider
- Registers artifact on completion

### Current Training Run Status

| Total Runs | Failed | Completed | Pending |
|------------|--------|-----------|---------|
| 11 | 5+ (visible) | 0 | 0 (or queued) |

**All visible training runs have FAILED.**

**Evidence:** VERIFIED BY DATABASE + LOGS + STATIC INSPECTION.

---

## 13. Evaluation

| Component | Status | Evidence |
|-----------|--------|----------|
| Evaluation engine | CODE EXISTS | VERIFIED BY STATIC INSPECTION — app/services/evaluation_engine.py |
| Evaluation worker | CODE EXISTS | VERIFIED BY STATIC INSPECTION — app/workers/evaluation_worker.py |
| Eval sets | IMPLEMENTED | VERIFIED BY STATIC INSPECTION — eval_sets table |
| Metrics | IMPLEMENTED | VERIFIED BY STATIC INSPECTION — eval_loss_trend, qualitative_comparison |
| Quality gates | IMPLEMENTED | VERIFIED BY STATIC INSPECTION — eval_gate_* settings |
| Actual execution | NOT VERIFIED | No evidence of real evaluation execution |

**Database State:**
- eval_sets: 1 record
- eval_set_versions: 1 record
- model_drift_checks: 0 records

**Classification:** CODE EXISTS, NOT E2E VERIFIED

**Evidence:** VERIFIED BY STATIC INSPECTION + DATABASE.

---

## 14. Model Registry

### Database State

| Model | Versions | Status |
|-------|----------|--------|
| defnex-support-llm | 1 | DEPLOYED |
| defnex-legal-llm | 1 | REGISTERED |
| defnex-orca-reasoner | 1 | REJECTED |
| defnex-general-llm | 1 | PROMOTED |
| test-smoke-model | 1 | REGISTERED |

### Seed Data Analysis

**Models 1-4 are SEED DATA (from seed.py):**
- Artifacts use fake S3 URIs (s3://defnex-mlops/artifacts/runs/...)
- No real training was performed
- No real artifacts exist at these URIs
- Deployment of defnex-support-llm is seed-only

**Model 5 (test-smoke-model) is the only potentially real model:**
- Artifact URI: file://data/artifacts/test-smoke-model/test-smoke-model-Qwen-Qwen2.5-0.5B-Instruct-v1
- Checksum: f95e24f6e9078275c741c3e67d2d5a2730f4d7b7761e33413dbe53d70c859e0d
- Status: REGISTERED (never deployed)
- Artifact directory exists but is ROOT-OWNED with restricted permissions

**Evidence:** VERIFIED BY DATABASE + FILESYSTEM.

---

## 15. Deployment

| Component | Status | Evidence |
|-----------|--------|----------|
| Deployment service | CODE EXISTS | VERIFIED BY STATIC INSPECTION — app/services/deployment_service.py |
| Deployment API | CODE EXISTS | VERIFIED BY STATIC INSPECTION — app/api/deployment.py |
| Staging environment | EXISTS in DB | VERIFIED BY DATABASE |
| Production environment | EXISTS in DB | VERIFIED BY DATABASE |
| Active deployment | SEED ONLY | VERIFIED BY DATABASE — defnex-support-llm v1 DEPLOYED (seed) |
| Real deployment | NEVER EXECUTED | No evidence of real deployment |
| Rollback | CODE EXISTS | VERIFIED BY STATIC INSPECTION |
| Smoke test | CODE EXISTS | VERIFIED BY STATIC INSPECTION |

**Classification:** CODE EXISTS, NOT E2E VERIFIED

**Evidence:** VERIFIED BY STATIC INSPECTION + DATABASE.

---

## 16. Multi-Server / Distributed

| Component | Status | Evidence |
|-----------|--------|----------|
| Server registry | CODE EXISTS | VERIFIED BY STATIC INSPECTION — compute_resources table |
| SSH execution | CODE EXISTS | VERIFIED BY STATIC INSPECTION — app/services/ssh.py |
| SFTP/artifact transfer | CODE EXISTS | VERIFIED BY STATIC INSPECTION — app/services/artifact_transfer.py |
| TrainingProvider abstraction | CODE EXISTS | VERIFIED BY STATIC INSPECTION — app/providers/training_provider.py |
| LocalSubprocessProvider | CODE EXISTS | VERIFIED BY STATIC INSPECTION |
| GPU VPS provider | CODE EXISTS | VERIFIED BY STATIC INSPECTION |
| Colab provider | CODE EXISTS | VERIFIED BY STATIC INSPECTION |
| Remote training | NOT CONFIGURED | compute_resources table is empty |
| Remote vLLM | NOT CONFIGURED | No remote serving configured |
| Cross-host networking | NOT IMPLEMENTED | Single server only |
| Artifact movement | NOT TESTED | artifact_transfers table is empty |

**Current System: SINGLE-SERVER ONLY**

All distributed capabilities exist only as code abstractions. No remote compute resources are configured or tested.

**Evidence:** VERIFIED BY STATIC INSPECTION + DATABASE.

---

## 17. Observability

| Component | Status | Evidence |
|-----------|--------|----------|
| Structured logs | YES | VERIFIED LIVE — JSON-formatted logs in backend/worker |
| Application logs | YES | VERIFIED LIVE — Docker logs |
| Worker logs | YES | VERIFIED LIVE — Docker logs |
| GPU controller logs | YES | VERIFIED LIVE — stdout logging |
| Metrics | NOT ACTIVE | OTEL_ENABLED=false |
| Tracing | NOT ACTIVE | OTEL_ENABLED=false |
| OTel | CODE EXISTS, DISABLED | VERIFIED BY STATIC INSPECTION |
| Prometheus | NOT PRESENT | No evidence |
| Grafana | NOT PRESENT | No evidence |
| Alerting | CODE EXISTS, DISABLED | ALERT_WEBHOOK_URL empty |

**Evidence:** VERIFIED LIVE + STATIC INSPECTION.

---

## 18. Logs / Errors

### Recent Errors (from worker logs, 2026-09-15 05:06-05:08)

| Timestamp | Component | Error | Impact | Still Reproducible? |
|-----------|-----------|-------|--------|---------------------|
| 2026-09-15 05:07:58 | worker | Training exited code 1: SFTTrainer unexpected keyword 'dataset_text_field' | Training FAILED | YES — API incompatibility |
| 2026-09-15 ~05:07 | worker | Training exited code 1: SFTTrainer unexpected keyword 'tokenizer' | Training FAILED | YES — API incompatibility |
| Earlier runs | worker | TrainingArguments unexpected keyword 'callbacks' | Training FAILED | YES — API incompatibility |
| Earlier runs | worker | LoraConfig unexpected keyword 'use_loftq' | Training FAILED | YES — API incompatibility |
| Earlier runs | worker | missing base_model in config | Training FAILED | YES — config issue |

### Recent Errors (from backend logs, 2026-09-15 05:06-05:08)

| Timestamp | Component | Error | Impact |
|-----------|-----------|-------|--------|
| Multiple | backend | GET /api/v1/training-runs/run-d3bd7a 401 MISSING_TOKEN | Auth required (expected) |

**Evidence:** VERIFIED BY LOG.

---

## 19. Regression Tests

### Test Suite Status

| Metric | Value |
|--------|-------|
| Total tests collected | 1,008 |
| Tests run (sample) | 17 passed (contract tests), 24 passed (training worker tests) |
| Coverage | 38-54% (below 80% threshold) |
| Warnings | 3 (deprecation warnings) |
| Failures | 0 (in sampled runs) |

**Note:** Full test suite timed out at 120 seconds. Sample runs show tests passing but coverage is below the configured 80% threshold.

**Evidence:** VERIFIED BY TEST (partial).

---

## 20. Current Actual Architecture

```
┌─────────────────────────────────────────────────────────────────────────────┐
│ HOST: ASTACITA-AI                                                          │
│ NVIDIA H100 PCIe 80GB (shared, ~65GB used by external workloads)           │
│                                                                             │
│ ┌─────────────────────────────────────────────────────────────────────┐    │
│ │ Docker Compose Project: ml-close-loop-be                            │    │
│ │ Network: ml-close-loop-be_default                                   │    │
│ │                                                                     │    │
│ │  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐              │    │
│ │  │   Backend    │  │   Worker     │  │    MinIO     │              │    │
│ │  │   :8000      │  │  (internal)  │  │  :9000/:9001 │              │    │
│ │  │  FastAPI     │  │  training    │  │  (healthy)   │              │    │
│ │  └──────┬───────┘  └──────┬───────┘  └──────────────┘              │    │
│ │         │                 │                                         │    │
│ │         │    ┌────────────┘                                         │    │
│ │         │    │                                                      │    │
│ │         │    │  Mounts:                                             │    │
│ │         │    │  /models → /home/ubuntu/defnex-mlops-experiment/outputs│   │
│ │         │    │  /opt/training-venv → experiment .venv               │    │
│ │         │    │  /app/data → ./data                                 │    │
│ │         │    │                                                      │    │
│ │         │    ▼                                                      │    │
│ │         │  ┌──────────────────┐                                    │    │
│ │         │  │ Training Runtime │                                    │    │
│ │         │  │ /opt/training-   │                                    │    │
│ │         │  │ venv/bin/python  │                                    │    │
│ │         │  │ Unsloth + CUDA   │                                    │    │
│ │         │  └──────────────────┘                                    │    │
│ │         │                                                          │    │
│ └─────────┼──────────────────────────────────────────────────────────┘    │
│           │                                                               │
│           │  VLLM_URL: http://172.17.0.1:8001                            │
│           │                                                               │
│ ┌─────────┼──────────────────────────────────────────────────────────┐    │
│ │         ▼         Docker Compose Project: defnex-mlops-experiment  │    │
│ │  ┌──────────────┐  Network: defnex-mlops-experiment_default       │    │
│ │  │ defnex-vllm  │                                                 │    │
│ │  │ :8001 (STOPPED)│  ← Managed by GPU Controller                  │    │
│ │  │ Qwen2.5-0.5B │                                                 │    │
│ │  │ + mlops-lora  │                                                 │    │
│ │  └──────────────┘                                                  │    │
│ └────────────────────────────────────────────────────────────────────┘    │
│                                                                           │
│ ┌────────────────────────────────────────────────────────────────────┐    │
│ │ Host-Side Process                                                  │    │
│ │ gpu_controller.py (PID 485335)                                     │    │
│ │ Watches: /home/ubuntu/defnex-mlops-experiment/outputs/.gpu-control/│    │
│ │ Controls: defnex-vllm (stop/start)                                 │    │
│ └────────────────────────────────────────────────────────────────────┘    │
│                                                                           │
│ ┌────────────────────────────────────────────────────────────────────┐    │
│ │ External GPU Workloads (14 processes, ~65GB VRAM)                  │    │
│ │ NOT controlled by DEFNEX                                           │    │
│ └────────────────────────────────────────────────────────────────────┘    │
│                                                                           │
│ Database: SQLite (/app/data/app.db)                                       │
│ Artifacts: Local filesystem (./data/artifacts/)                           │
└───────────────────────────────────────────────────────────────────────────┘
```

---

## 21. Current vs Target Architecture

### Current Actual Architecture (Single-Server Prototype)

- SQLite database (not PostgreSQL)
- Local filesystem artifact storage (not MinIO)
- Single worker process (not Celery)
- No message broker
- No Redis
- File-based GPU signaling (not API-based)
- Host-side GPU controller (not containerized)
- Single GPU server (not distributed)
- Seed data in registry (not real models)

### Target Intended Architecture (PRD v2)

- PostgreSQL database
- MinIO/S3 object storage
- Celery workers with Redis broker
- Multi-server: control plane, training GPU, staging inference, production inference
- SSH-based remote execution
- ComputeResource registry
- Staging → Production promotion workflow
- Real evaluation pipeline
- Real model registry with trained artifacts

### Gap Analysis

| Capability | Current | Target | Gap |
|------------|---------|--------|-----|
| Database | SQLite | PostgreSQL | MEDIUM — schema ready, needs migration |
| Object storage | Local FS | MinIO | LOW — MinIO running, code ready, not wired |
| Job queue | Polling worker | Celery + Redis | HIGH — not implemented |
| GPU coordination | File signal | API-based | MEDIUM — file signal works but is fragile |
| Multi-server | None | 4 servers | HIGH — only abstractions exist |
| Model registry | Seed data | Real models | HIGH — no successful training |
| Evaluation | Code only | Real evaluation | HIGH — never executed |
| CI/CD | GitHub Actions | Full pipeline | MEDIUM — basic CI exists |

---

## 22. Things That Look Implemented But Are Not Proven

1. **Seed model "defnex-support-llm" marked DEPLOYED** — Uses fake S3 artifact URI, never actually deployed to vLLM. VERIFIED BY DATABASE.

2. **MinIO integration** — MinIO is running and healthy, artifact_backend can be set to "minio", but currently set to "local". Code exists but is UNUSED in production. VERIFIED BY STATIC INSPECTION.

3. **Runtime LoRA loading** — VLLMServingBackend has load_adapter/unload_adapter code using vLLM's /v1/load_lora_adapter API. Never successfully exercised. VERIFIED BY STATIC INSPECTION.

4. **Compose "serving" service** — Exists in docker-compose.yml with GPU profile but is NOT the active vLLM. defnex-vllm from experiment environment is the actual instance. VERIFIED BY STATIC INSPECTION.

5. **Unsloth training** — Unsloth is installed in the experiment venv, CUDA is available, but ALL training runs fail due to API incompatibility. VERIFIED BY LOG.

6. **GPU handoff mechanism** — File signaling works (stop/start verified), but actual training after stop has never succeeded. VERIFIED LIVE for stop/start, NOT VERIFIED for training.

7. **Evaluation pipeline** — evaluation_engine.py and evaluation_worker.py exist, but no real evaluation has ever run. VERIFIED BY STATIC INSPECTION.

8. **Multi-server distributed training** — TrainingProvider, ComputeResource, SSH abstractions exist in code, but compute_resources table is empty and no remote execution has ever occurred. VERIFIED BY DATABASE.

9. **Smoke test on deploy** — inference_smoke_enabled=true in config, smoke test code exists in deployment_service.py, but never executed because no real deployment has occurred. VERIFIED BY STATIC INSPECTION.

10. **test-smoke-model artifact** — Registered in DB with checksum, directory exists on filesystem, but is ROOT-OWNED with restricted permissions. Worker cannot read it. VERIFIED BY FILESYSTEM.

---

## 23. Blockers

### P0 — Functional Path Impossible

1. **Training always fails** — Unsloth/trl/transformers API incompatibility makes training impossible. The installed package versions have breaking changes (SFTTrainer API, TrainingArguments, LoraConfig). This blocks the entire closed loop.

### P1 — Important Integration/Reliability Problems

2. **vLLM not part of main repo** — defnex-vllm is from the experiment environment, not the main docker-compose. Creates fragile dependency.

3. **Legacy experiment dependency** — Worker mounts experiment .venv and outputs directory. Main repo cannot run independently.

4. **Seed data masquerading as real models** — 4 of 5 models in registry use fake S3 URIs. Could mislead about system readiness.

5. **test-smoke-model artifact unreadable** — Root-owned permissions prevent worker/vLLM from reading.

6. **GPU controller not systemd-managed** — Running as a background process, not a service. Could be killed accidentally.

### P2 — Non-Blocking Quality/Cleanup Issues

7. **Coverage below threshold** — Test coverage at 38-54%, below configured 80% minimum.

8. **Observability disabled** — OTel, metrics, alerting all disabled.

9. **No CI/CD for GPU components** — Only backend CI exists.

---

## 24. Handover Truth Matrix

| Capability | Actual Status | Evidence | Confidence | Notes |
|------------|---------------|----------|------------|-------|
| Backend | RUNNING | VERIFIED LIVE | HIGH | FastAPI on :8000, health OK |
| Database | RUNNING (SQLite) | VERIFIED LIVE | HIGH | Operational with seed data |
| Worker | RUNNING | VERIFIED LIVE | HIGH | Background training worker active |
| MinIO | RUNNING (healthy) | VERIFIED LIVE | HIGH | Port 9000/9001, healthcheck passing |
| Dataset intake | CODE EXISTS | VERIFIED BY STATIC INSPECTION | MEDIUM | API implemented, not E2E tested |
| Dataset validation | CODE EXISTS | VERIFIED BY STATIC INSPECTION | MEDIUM | API implemented, not E2E tested |
| Dataset versioning | CODE EXISTS | VERIFIED BY STATIC INSPECTION | MEDIUM | DB schema + service implemented |
| Training state machine | CODE EXISTS | VERIFIED BY STATIC INSPECTION | HIGH | Well-defined transitions |
| Training provider | CODE EXISTS | VERIFIED BY STATIC INSPECTION | HIGH | LocalSubprocessProvider implemented |
| Unsloth runtime | INSTALLED BUT BROKEN | VERIFIED BY LOG | HIGH | All training runs fail |
| GPU training | NOT WORKING | VERIFIED BY LOG | HIGH | API incompatibility blocks training |
| Evaluation | CODE EXISTS | VERIFIED BY STATIC INSPECTION | MEDIUM | Never executed |
| Model registry | SEED DATA ONLY | VERIFIED BY DATABASE | HIGH | 4 fake models, 1 real but unreadable |
| Artifact storage | LOCAL FS | VERIFIED LIVE | HIGH | Local backend active |
| Checksum | CODE EXISTS | VERIFIED BY STATIC INSPECTION | MEDIUM | Implemented in artifact_storage |
| vLLM | RUNNING (managed) | VERIFIED LIVE | HIGH | defnex-vllm, currently stopped by controller |
| Backend → vLLM | CONNECTED | VERIFIED LIVE | MEDIUM | Configured but vLLM often stopped |
| Inference | NOT VERIFIED | UNKNOWN | LOW | Never tested through backend API |
| Runtime LoRA | CODE EXISTS | VERIFIED BY STATIC INSPECTION | LOW | Never successfully exercised |
| GPU signaling | WORKING | VERIFIED LIVE | HIGH | Stop/start verified live |
| GPU controller | RUNNING | VERIFIED LIVE | HIGH | PID 485335, not systemd |
| GPU handoff | PARTIALLY WORKING | VERIFIED LIVE | MEDIUM | Stop/start works, training after fails |
| Deployment | CODE EXISTS | VERIFIED BY STATIC INSPECTION | MEDIUM | Never executed for real |
| Rollback | CODE EXISTS | VERIFIED BY STATIC INSPECTION | MEDIUM | Never executed |
| Monitoring | DISABLED | VERIFIED BY STATIC INSPECTION | HIGH | OTel/alerting code exists, inactive |
| Observability | MINIMAL | VERIFIED LIVE | HIGH | Structured logs only |
| CI/CD | BASIC | VERIFIED BY STATIC INSPECTION | MEDIUM | GitHub Actions for backend |
| Server registry | CODE EXISTS | VERIFIED BY STATIC INSPECTION | MEDIUM | DB table empty |
| Remote training | CODE EXISTS | VERIFIED BY STATIC INSPECTION | LOW | Never configured or tested |
| Remote serving | NOT IMPLEMENTED | UNKNOWN | LOW | No evidence |
| Artifact transfer | CODE EXISTS | VERIFIED BY STATIC INSPECTION | LOW | Never executed |
| Colab | CODE EXISTS | VERIFIED BY STATIC INSPECTION | LOW | Provider abstraction only |
| Distributed MLOps | NOT IMPLEMENTED | VERIFIED BY STATIC INSPECTION | HIGH | Single-server only |
| E2E closed loop | NOT VERIFIED | VERIFIED BY LOG | HIGH | Training always fails |

---

## 25. Final Answers

1. **Is the backend actually running?** YES — VERIFIED LIVE, port 8000, health OK.

2. **Is the worker actually running?** YES — VERIFIED LIVE, background training worker active.

3. **Is vLLM actually running?** NO — Currently STOPPED. Was running earlier, stopped by GPU controller for a training attempt.

4. **Which vLLM instance is actually active?** NONE currently. defnex-vllm is the managed instance but is stopped.

5. **Is the active vLLM part of the main MLOps repository or the legacy experiment environment?** LEGACY EXPERIMENT. defnex-vllm is from defnex-mlops-experiment, not the main repo's docker-compose.

6. **Is the backend actually using vLLM?** CONFIGURED TO, but vLLM is currently stopped. Backend health check shows vllm: "error".

7. **Does real inference work?** NOT VERIFIED. No successful inference test has been performed through the backend API.

8. **Is Unsloth actually executable by the production worker?** INSTALLED BUT BROKEN. Unsloth is in the training venv, CUDA works, but all training runs fail due to API incompatibility.

9. **Does the worker currently have GPU access?** YES — VERIFIED LIVE. nvidia-smi works, torch.cuda.is_available() = True.

10. **Has real training actually been executed successfully on this VM?** NO. All 5+ visible training runs have FAILED.

11. **Can training currently produce a real adapter artifact?** NO. Training always fails due to Unsloth/trl API incompatibility.

12. **Can that artifact actually be consumed by vLLM?** NOT VERIFIED. No valid artifact has been produced.

13. **Does runtime LoRA actually work?** NOT VERIFIED. Code exists but has never been successfully exercised with a real adapter.

14. **Does the GPU handoff mechanism actually work?** PARTIALLY. Stop/start of defnex-vllm works. Training after stop fails.

15. **Does the GPU handoff mechanism safely preserve unrelated GPU workloads?** YES — VERIFIED LIVE. 14 non-DEFNEX GPU processes survived stop/start cycles.

16. **Is the current system single-server or multi-server?** SINGLE-SERVER. All distributed capabilities are code abstractions only.

17. **Is distributed training actually working?** NO. compute_resources table is empty. No remote execution.

18. **Is distributed serving actually working?** NO. Single vLLM instance on same host.

19. **Is the dataset → training → evaluation → registry → serving → inference closed loop actually verified?** NO. Training always fails, breaking the loop.

20. **What is the SINGLE MOST IMPORTANT next blocker?** Fix the Unsloth/trl/transformers API incompatibility so training can produce a real adapter artifact. Without this, the entire closed loop is broken.

---

## 26. Recommended Next Steps

### Immediate (P0)

1. **Fix training API incompatibility** — Pin compatible versions of unsloth, trl, transformers, and peft in the training venv. Update run_training.py to match the installed API.

2. **Fix test-smoke-model permissions** — Change ownership of the artifact directory so the worker can read it.

### Short-term (P1)

3. **Integrate defnex-vllm into main docker-compose** — Either move vLLM into the main compose file or document the external dependency clearly.

4. **Replace seed data with real models** — After training works, run a real training cycle and register a real model.

5. **Install GPU controller as systemd service** — Use the existing gpu-controller.service template.

6. **Wire MinIO** — Set ARTIFACT_BACKEND=minio and test the full artifact lifecycle.

### Medium-term (P2)

7. **Enable observability** — Configure OTel, set up Prometheus/Grafana.

8. **Increase test coverage** — Target 80% by adding tests for untested paths.

9. **Document the architecture** — Update README with current state vs target.

---

## Evidence Summary

| Evidence Type | Count |
|---------------|-------|
| VERIFIED LIVE | 45+ |
| VERIFIED BY STATIC INSPECTION | 30+ |
| VERIFIED BY LOG | 10+ |
| VERIFIED BY DATABASE | 15+ |
| VERIFIED BY TEST | 3 (partial) |
| INFERRED | 5 |
| UNKNOWN | 3 |

---

*Report generated: 2026-09-15 05:15 UTC*
*Verification mode: READ + VERIFY (no modifications made)*
