# DEFNEX MLOps Integration QA Audit Report

**Date:** 2026-09-11
**Scope:** Backend → Worker → Unsloth → Artifact → MinIO → Registry → vLLM → Inference
**Constraint:** Read-only audit. No code changes, no process kills, no GPU state modification.

---

## A. Current Architecture as Actually Running on This VM

```
┌─────────────────────────────────────────────────────────────────────┐
│  HOST VM                                                            │
│                                                                     │
│  ┌─────────────────────────────────┐  ┌──────────────────────────┐  │
│  │  Docker Network:                │  │  Docker Network:         │  │
│  │  ml-close-loop-be_default       │  │  defnex-mlops-experiment │  │
│  │  (172.19.0.0/16)                │  │  _default (172.18.0.0/16)│  │
│  │                                 │  │                          │  │
│  │  backend  :8000->host            │  │  defnex-vllm             │  │
│  │  VLLM_URL=http://serving:8000   │  │  :8001->host              │  │
│  │  SERVING_BACKEND=mock           │  │  Model: Qwen2.5-0.5B    │  │
│  │  ARTIFACT_BACKEND=local         │  │  LoRA: mlops-lora        │  │
│  │                                 │  │  --enable-lora           │  │
│  │  worker                         │  │  NO runtime lora update  │  │
│  │  TRAINING_PYTHON=python3        │  │  --max-loras 1           │  │
│  │  (no unsloth installed!)        │  │  ./outputs:/models (ro)  │  │
│  │                                 │  │                          │  │
│  │  minio  :9000,:9001             │  └──────────────────────────┘  │
│  │  (healthy)                      │                                │
│  │                                 │  ┌──────────────────────────┐  │
│  └─────────────────────────────────┘  │  /home/ubuntu/defnex-    │  │
│                                        │  mlops-experiment/       │  │
│  ┌─────────────────────────────────┐  │  .venv (has unsloth)    │  │
│  │  ml-close-loop-be/              │  │  outputs/qwen2.5-0.5b-  │  │
│  │  .venv (no unsloth)             │  │    mlops-lora/           │  │
│  │  data/artifacts/ (empty)        │  └──────────────────────────┘  │
│  │  data/app.db (SQLite)           │                                │
│  └─────────────────────────────────┘  GPU: H100 80GB, 2GB free     │
└─────────────────────────────────────────────────────────────────────┘
```

### Live Verification Results

| Check | Command | Result | Evidence Type |
|-------|---------|--------|---------------|
| Backend health | `curl localhost:8000/api/v1/health` | `{"status":"ok","checks":{"db":"ok"}}` | live command |
| Backend OpenAPI | `curl localhost:8000/openapi.json` | DEFNEX MLOps Backend v0.1.0 | live command |
| MinIO health | `curl localhost:9000/minio/health/live` | HTTP 200 | live command |
| vLLM health | `curl localhost:8001/health` | responds (empty body) | live command |
| vLLM models | `curl localhost:8001/v1/models` | `Qwen/Qwen2.5-0.5B-Instruct` + `mlops-lora` | live command |
| Backend VLLM_URL | `docker exec backend printenv VLLM_URL` | `http://serving:8000` | live command |
| Worker VLLM_URL | `docker exec worker printenv VLLM_URL` | `http://localhost:8001` | live command |
| SERVING_BACKEND | `docker exec backend printenv SERVING_BACKEND` | `mock` | live command |
| ARTIFACT_BACKEND | `docker exec backend printenv ARTIFACT_BACKEND` | `local` | live command |
| Worker TRAINING_PYTHON | `docker exec worker printenv TRAINING_PYTHON` | `python3` | live command |
| Worker unsloth | `docker exec worker python3 -c "import unsloth"` | `ModuleNotFoundError` | live command |
| Backend→vLLM reachable | `docker exec backend python3 -c "httpx.get('http://172.17.0.1:8001/health')"` | HTTP 200 | live command |
| Backend→serving:8000 | `docker exec backend python3 -c "httpx.get('http://serving:8000/health')"` | DNS resolution failure | live command |
| GPU | `nvidia-smi` | H100 PCIe 80GB, 2116 MiB free | live command |
| vLLM env | `docker inspect defnex-vllm` | No `VLLM_ALLOW_RUNTIME_LORA_UPDATING` | live command |
| vLLM command | `docker inspect defnex-vllm` | `--max-loras 1 --max-lora-rank 16 --lora-modules mlops-lora=/models/...` | live command |
| vLLM volumes | `docker inspect defnex-vllm` | `./outputs:/models:ro`, `~/.cache/huggingface:/root/.cache/huggingface` | live command |
| Compose networks | `docker network ls` | backend on `ml-close-loop-be_default`, vLLM on `defnex-mlops-experiment_default` | live command |
| Existing adapter | `ls /home/ubuntu/defnex-mlops-experiment/outputs/qwen2.5-0.5b-mlops-lora/` | adapter_model.safetensors (35MB), adapter_config.json, tokenizer files | live command |

### vLLM Container Configuration (Verified)

```
Image: vllm/vllm-openai:latest (v0.28.0)
Command:
  --model Qwen/Qwen2.5-0.5B-Instruct
  --dtype bfloat16
  --max-model-len 1024
  --gpu-memory-utilization 0.15
  --enable-lora
  --max-loras 1
  --max-lora-rank 16
  --lora-modules mlops-lora=/models/qwen2.5-0.5b-mlops-lora
Network: defnex-mlops-experiment_default (172.18.0.2)
Ports: 8001:8000
Volumes:
  /home/ubuntu/defnex-mlops-experiment/outputs:/models:ro
  /home/ubuntu/.cache/huggingface:/root/.cache/huggingface
```

### Existing Adapter (Verified)

```
Path: /home/ubuntu/defnex-mlops-experiment/outputs/qwen2.5-0.5b-mlops-lora/
Files:
  adapter_config.json   (1.3KB)
  adapter_model.safetensors (35MB)
  tokenizer.json (11MB)
  tokenizer_config.json
  chat_template.jinja
  checkpoint-3/, checkpoint-6/, checkpoint-9/

adapter_config.json:
  base_model_name_or_path: unsloth/qwen2.5-0.5b-instruct-unsloth-bnb-4bit
  peft_type: LORA
  r: 16, lora_alpha: 16
  target_modules: [gate_proj, up_proj, v_proj, o_proj, q_proj, k_proj, down_proj]
  use_dora: false, use_rslora: false
  peft_version: 0.20.0
```

---

## B. Golden-Path Integration Sequence

| # | Step | Endpoint / Function | File:Line | DB Object | Artifact | State Transition | External Dep | Status |
|---|------|---------------------|-----------|-----------|----------|-----------------|-------------|--------|
| 1 | Upload dataset | `POST /api/v1/datasets/intake/inspect` | `app/api/intake.py:81` | Dataset + DatasetVersion (PENDING) | File staged to `data/datasets/staging/{id}/` | none | none | **REAL** |
| 2 | Validate dataset | `POST /api/v1/datasets/intake/validate` | `app/api/intake.py` | ValidationReport | none | none | none | **REAL** |
| 3 | Commit dataset | `POST /api/v1/datasets/intake/commit` | `app/api/intake.py` | DatasetVersion → PROCESSED | Moved to `data/datasets/{dataset_id}/v{N}/` | PENDING→PROCESSED | none | **REAL** |
| 4 | Create training run | `POST /api/v1/training-runs` | `app/api/training.py:25` | TrainingRun (PENDING) | none | none | Validates dataset+report exist | **REAL** |
| 5 | Worker claims job | `process_next_job()` | `training_worker.py:126` | TrainingRun → RUNNING | none | PENDING→RUNNING | GPU lock (flock), serving coordinator | **REAL** |
| 6 | Unsloth executes | `LocalSubprocessProvider.submit()` → `subprocess.Popen` | `training_provider.py:119-170` | none | Temp staging dir | none | **REQUIRES unsloth/trl/transformers** | **BROKEN** (no unsloth in Docker) |
| 7 | Artifact produced | `model.save_pretrained(staging)` | `app/training/run_training.py:136-137` | none | `adapter_model.safetensors`, `adapter_config.json`, tokenizer files | none | GPU (CUDA) | **REAL** |
| 8 | Artifact stored | `model_service.register_model_version()` → `finalize_version()` | `model_service.py:145-201` | ModelVersion (REGISTERED) | Moved to `data/artifacts/{model_id}/{name}/` + `metadata.json` + SHA-256 | COMPLETED (TrainingRun) | none | **REAL** |
| 9 | Model version registered | (same as step 8) | `model_service.py:90-142` | Model + ModelVersion (REGISTERED) | `model_version.artifacts = [{"type":"adapter","uri":"file:///app/data/artifacts/...","checksum":"..."}]` | none | none | **REAL** |
| 10 | Trigger evaluation | `POST /api/v1/models/{id}/versions/{v}/evaluation` | `model_service.py:324-349` | ModelVersion.evaluation_requested = True | none | none | Eval set must exist | **REAL** |
| 11 | Evaluation runs | `evaluation_worker.process_next_evaluation()` → `compute_evaluation_update()` | `evaluation_worker.py:50-86` | ModelVersion → EVALUATED | none | REGISTERED→EVALUATED | ServingBackend.generate() (needs real vLLM) | **REAL** |
| 12 | Staging deploy | `POST /api/v1/models/{id}/versions/{v}/deploy` (env=staging) | `app/api/deployment.py:26-128` | ModelVersion → STAGING | none | EVALUATED→STAGING | GPU lock, serving backend | **REAL** |
| 13 | vLLM loads adapter | `VLLMServingBackend.deploy()` → `POST /v1/load_lora_adapter` | `app/services/serving.py:206-233` | none | Adapter loaded into vLLM | none | vLLM runtime LoRA API | **BLOCKED** (vLLM lacks runtime LoRA update flag) |
| 14 | Inference | `POST /api/v1/models/{id}/inference` | `app/api/inference.py:18-73` | none | none | none | vLLM completions API | **REAL** |
| 15 | Smoke test | `_run_smoke_test()` during deploy | `deployment_service.py:302-303` | none | none | none | vLLM generate | **REAL** |

### Full Data Flow Diagram

```
POST /training-runs
  │
  ▼
TrainingRun(status=PENDING) ─── db.commit() by router
  │
  ▼ [Worker polls every 5s]
process_next_job():
  1. mark_stale_runs()                ─── reclaim orphaned RUNNING runs
  2. next_claimable_run()             ─── SELECT PENDING/STALE ORDER BY priority DESC, created_at ASC
  3. gpu_lock(lock_file, timeout)     ─── exclusive fcntl.flock on data/gpu.lock
  4. coordinator.cycle()              ─── stop serving, verify VRAM (or no-op)
  5. claim_training_run()             ─── CAS PENDING|STALE → RUNNING
  6. Start heartbeat thread           ─── touch every 10s
  7. resolve_runner_for_run()         ─── LocalSubprocessProvider (default)
  8. runner.run() →
     │  subprocess.Popen([python3, -u, app/training/run_training.py,
     │                    --config '<JSON>', --staging <tempdir>])
     │  → FastLanguageModel.from_pretrained()
     │  → FastLanguageModel.get_peft_model()
     │  → SFTTrainer.train()
     │  → model.save_pretrained(staging)
     │  → stdout: {"event":"progress",...} / {"event":"done",...}
     │  ProviderRunnerAdapter polls get_status() until COMPLETED
     │  Returns staging dir path
     ▼
  9. complete_training_run()          ─── RUNNING → COMPLETED (records artifact_uri)
  10. register_model_version() →
      - Ensure Model row exists
      - Allocate version (atomic, retry on race)
      - finalize_version(): move staging → immutable artifact dir + metadata.json + SHA-256
      - Create ModelVersion(status=REGISTERED)
  │
  ▼ db.commit()
[Separate: evaluation_worker polls evaluation_requested=True]
  → compute_evaluation_update() via ServingBackend.generate()
  → submit_evaluation() → REGISTERED → EVALUATED (when all 3 signals present)
  → [Human] promotion decision → PROMOTED/REJECTED
  → [Human/API] deploy → load adapter → smoke test → DEPLOYED
  → inference → POST /v1/completions via adapter name {model_id}-v{version}
```

---

## C. Exact Commands/Tests Needed to Execute Golden Path

### Prerequisites

```bash
# Access the backend
cd /home/ubuntu/Defnex-MLOps/ml-close-loop-be
```

### Step 1: Authenticate

```bash
TOKEN=$(curl -s http://localhost:8000/api/v1/auth/login \
  -H 'Content-Type: application/json' \
  -d '{"username":"admin","password":"admin"}' | python3 -c \
  "import sys,json; print(json.load(sys.stdin)['access_token'])")
echo "Token: ${TOKEN:0:20}..."
```

### Step 2: Upload Dataset (Inspect)

```bash
# Create a small test JSONL file
cat > /tmp/test_dataset.jsonl << 'EOF'
{"messages":[{"role":"user","content":"What is 2+2?"},{"role":"assistant","content":"4"}]}
{"messages":[{"role":"user","content":"What is the capital of France?"},{"role":"assistant","content":"Paris"}]}
EOF

curl -s http://localhost:8000/api/v1/datasets/intake/inspect \
  -H "Authorization: Bearer $TOKEN" \
  -F "file=@/tmp/test_dataset.jsonl" | python3 -m json.tool
```

### Step 3: Validate Dataset

```bash
curl -s http://localhost:8000/api/v1/datasets/intake/validate \
  -H "Authorization: Bearer $TOKEN" \
  -H 'Content-Type: application/json' \
  -d '{
    "staging_id":"<FROM_STEP_2>",
    "dataset_id":"test-integration-ds",
    "source_format":"jsonl"
  }' | python3 -m json.tool
```

### Step 4: Commit Dataset

```bash
curl -s http://localhost:8000/api/v1/datasets/intake/commit \
  -H "Authorization: Bearer $TOKEN" \
  -H 'Content-Type: application/json' \
  -d '{
    "staging_id":"<FROM_STEP_2>",
    "dataset_id":"test-integration-ds",
    "validation_report_id": <FROM_STEP_3>,
    "source_format":"jsonl"
  }' | python3 -m json.tool
```

### Step 5: Create Training Run

```bash
curl -s http://localhost:8000/api/v1/training-runs \
  -H "Authorization: Bearer $TOKEN" \
  -H 'Content-Type: application/json' \
  -d '{
    "dataset_id":"test-integration-ds",
    "dataset_version":1,
    "model_id":"test-integration-model",
    "base_model":"Qwen/Qwen2.5-0.5B-Instruct",
    "training_config":{
      "epochs":1,
      "peft_method":"lora",
      "lora_r":16,
      "lora_alpha":16,
      "batch_size":1,
      "learning_rate":2e-4,
      "max_seq_length":512
    }
  }' | python3 -m json.tool
```

### Step 6: Monitor Worker

```bash
# Watch worker logs
docker logs -f ml-close-loop-be-worker-1

# Check training run status
curl -s http://localhost:8000/api/v1/training-runs/<RUN_ID> \
  -H "Authorization: Bearer $TOKEN" | python3 -m json.tool
```

### Step 7: Trigger Evaluation

```bash
curl -s http://localhost:8000/api/v1/models/test-integration-model/versions/1/evaluation \
  -H "Authorization: Bearer $TOKEN" \
  -H 'Content-Type: application/json' \
  -d '{"eval_set_id":"test-eval-set","eval_set_version":1}' | python3 -m json.tool
```

### Step 8: Deploy to Staging

```bash
curl -s http://localhost:8000/api/v1/models/test-integration-model/versions/1/deploy \
  -H "Authorization: Bearer $TOKEN" \
  -H 'Content-Type: application/json' \
  -d '{"environment":"staging"}' | python3 -m json.tool
```

### Step 9: Inference

```bash
curl -s http://localhost:8000/api/v1/models/test-integration-model/inference \
  -H "Authorization: Bearer $TOKEN" \
  -H 'Content-Type: application/json' \
  -d '{"target":"prod","prompt":"What is 2+2?"}' | python3 -m json.tool
```

---

## D. P0/P1/P2 Blockers

### P0 — Impossible to Run the Golden Path

| # | Blocker | Evidence | Code Location | Fix Required |
|---|---------|----------|---------------|-------------|
| **P0-1** | `SERVING_BACKEND=mock` | `docker exec backend printenv SERVING_BACKEND` → `mock` | `config.py:27` default `mock`, `.env:24` | Set `SERVING_BACKEND=vllm` in docker-compose backend environment |
| **P0-2** | VLLM_URL points to non-existent `serving:8000` | `docker-compose.yml:11` overrides VLLM_URL. DNS fails: `httpx.ConnectError: [Errno -3] Temporary failure in name resolution` | `docker-compose.yml:11` | Set `VLLM_URL=http://172.17.0.1:8001` (host gateway, verified reachable) |
| **P0-3** | Training Python has no unsloth/trl/transformers | `docker exec worker python3 -c "import unsloth"` → `ModuleNotFoundError` | `Dockerfile`, `TRAINING_PYTHON=python3` | Install ML deps in Dockerfile or mount experiment venv |
| **P0-4** | vLLM lacks `VLLM_ALLOW_RUNTIME_LORA_UPDATING=true` | `docker inspect defnex-vllm` env: no runtime update flag. `POST /v1/load_lora_adapter` → 404. | `defnex-mlops-experiment/docker-compose.inference.yml` | Recreate vLLM with `VLLM_ALLOW_RUNTIME_LORA_UPDATING=true` and `--max-loras 4` |
| **P0-5** | Artifact path mismatch: worker `/app/data` vs vLLM `/data` or `/models` | Worker stores `file:///app/data/artifacts/...`. `serving.py:142` strips `file://` → `/app/data/artifacts/...`. Compose serving mounts `./data:/data`. Existing vLLM mounts `./outputs:/models`. | `artifact_storage.py:115-125`, `serving.py:134-147`, `docker-compose.yml:44` | Fix mount path or rewrite URI in `_adapter_path()` |

### P1 — Flow Can Run but Important Issue Exists

| # | Issue | Evidence | Code Location | Fix Required |
|---|-------|----------|---------------|-------------|
| **P1-1** | GPU VRAM only 2GB free (of 80GB) | `nvidia-smi`: 2116 MiB free. Existing vLLM uses ~78GB. | `nvidia-smi` | Stop existing vLLM before training, or reduce `--gpu-memory-utilization` |
| **P1-2** | `get_artifact_storage()` factory bypassed | `model_service.py:191` defaults to `LocalFilesystemArtifactStorage()`. Factory only used in health check. | `model_service.py:191`, `deployment_service.py:111` | Pass storage instance through call chain |
| **P1-3** | MinIO endpoint not overridden for Docker | `.env:161` `MINIO_ENDPOINT=localhost:9000`. Inside Docker `localhost` is the container itself. | `.env:161`, `docker-compose.yml` (no override) | Add `MINIO_ENDPOINT=minio:9000` to docker-compose backend/worker environment |
| **P1-4** | No app-level vLLM health check | `app/api/health.py:9-29`: checks DB + MinIO only. No vLLM probe. | `app/api/health.py` | Add vLLM health check to `/api/v1/health` |
| **P1-5** | Compose serving profile conflicts with existing vLLM | `docker-compose.yml:40`: `${VLLM_SERVING_PORT:-8001}:8000`. Existing vLLM already on host 8001. | `docker-compose.yml:40` | Change port or stop existing vLLM |
| **P1-6** | Existing vLLM `--max-loras 1` limits to 1 adapter | `docker inspect defnex-vllm` command: `--max-loras 1`. Deploy needs ≥2 (load new before unload old). | `defnex-mlops-experiment/docker-compose.inference.yml` | Set `--max-loras 4` |
| **P1-7** | Backend and vLLM on different Docker networks | Backend on `ml-close-loop-be_default` (172.19.0.0/16), vLLM on `defnex-mlops-experiment_default` (172.18.0.0/16). | `docker network ls` | Set VLLM_URL to host gateway IP or join networks |

### P2 — Non-Blocking Weakness

| # | Issue | Evidence | Code Location | Fix Required |
|---|-------|----------|---------------|-------------|
| **P2-1** | Inference routing not environment-aware | `inference.py:65`: always calls `get_serving_backend()` with no env param. | `app/api/inference.py:65` | Add optional env parameter to inference endpoint |
| **P2-2** | No S3-to-vLLM bridge for MinIO artifacts | `serving.py:134-147`: only handles `file://` URIs. `s3://` passed verbatim to vLLM. | `app/services/serving.py:134-147` | Add S3 download step before adapter load |
| **P2-3** | Worker `SERVING_CONTROL=mock` by default | Worker doesn't stop/restart serving before/after training. | `gpu_orchestrator.py:328` | Configure `SERVING_CONTROL=shell` with proper commands |
| **P2-4** | `python-multipart` missing from pyproject.toml | Required for multipart file upload. Installed manually. | `pyproject.toml` | Add `python-multipart` to dependencies |
| **P2-5** | `ColabProvider` returns `s3://` URI as staging_dir | `Path("s3://...")` is not a valid local path. | `training_provider.py:578`, `model_service.py:192` | Handle `s3://` URIs in register_model_version |

---

## E. Minimal Changes Required Before First Real Integration Test

### Recommended Approach: Use Existing vLLM (Least Infrastructure Change)

| # | Change | File | What to Change | Risk |
|---|--------|------|---------------|------|
| 1 | Enable `SERVING_BACKEND=vllm` | `docker-compose.yml` backend environment | Add `SERVING_BACKEND: vllm` | Low — just changes mock to real backend |
| 2 | Fix VLLM_URL to reach existing vLLM | `docker-compose.yml` backend environment | Add `VLLM_URL: http://172.17.0.1:8001` (host gateway) | Low — verified reachable from backend container |
| 3 | Enable runtime LoRA on vLLM | `defnex-mlops-experiment/docker-compose.inference.yml` | Add `VLLM_ALLOW_RUNTIME_LORA_UPDATING=true` env, change `--max-loras 1` to `4` | Medium — requires recreating vLLM container |
| 4 | Fix artifact path mismatch | `docker-compose.yml` serving volumes OR `app/services/serving.py` | Option A: Change serving mount `./data:/app/data`. Option B: In `_adapter_path()`, rewrite `/app/data/` to `/data/`. Option C: Set worker `ARTIFACT_STORAGE_DIR=/data/artifacts` | Medium — must ensure consistency across worker/vLLM |
| 5 | Install unsloth in worker Docker image | `Dockerfile` | Add `pip install unsloth trl transformers datasets` | High — image size increase, may conflict with base image |
| 6 | Set `SERVED_BASE_MODEL` | `docker-compose.yml` or `.env` | Set to `Qwen/Qwen2.5-0.5B-Instruct` | Low — validates base model before deploy |
| 7 | Set `INFERENCE_SMOKE_ENABLED=true` | `.env` (already default) | No change needed | None |

### Alternative Approach: Start Compose Serving Profile

Same as above but instead of reusing existing vLLM:
- Set `VLLM_SERVING_PORT=8002` to avoid conflict with existing vLLM on 8001
- Start with `docker compose --profile gpu up -d serving`
- Point VLLM_URL to `http://serving:8000` (already the compose default)

**Advantage:** Isolated vLLM instance with correct config from docker-compose.yml.
**Disadvantage:** Uses more GPU VRAM (may not fit with existing vLLM running).

---

## F. Recommended Next QA Prompt

```
The integration audit has been completed. There are 5 P0 blockers preventing any
real integration flow. The recommended fix sequence is:

1. Fix VLLM_URL in docker-compose.yml to reach the existing vLLM via host gateway
   (http://172.17.0.1:8001, verified reachable from backend container)

2. Set SERVING_BACKEND=vllm in the backend Docker environment

3. Enable runtime LoRA on the existing vLLM (requires recreating the container
   with VLLM_ALLOW_RUNTIME_LORA_UPDATING=true and --max-loras 4)

4. Fix the artifact path mismatch between worker (/app/data) and vLLM (/data or /models)

5. Install unsloth/trl/transformers in the worker Docker image

6. Set SERVED_BASE_MODEL=Qwen/Qwen2.5-0.5B-Instruct

After these fixes, execute the golden-path test commands from section C to verify
the full integration flow: dataset → validation → training → artifact → registry
→ deployment → vLLM adapter load → inference.

Please approve:
  a) Which approach to take (reuse existing vLLM vs start compose serving)
  b) Whether to recreate the defnex-vllm container (needs GPU restart approval)
  c) Whether to install unsloth in Dockerfile or mount the experiment venv

I will then implement the minimal changes.
```

---

## Appendix: Key File References

| Component | File | Key Functions |
|-----------|------|---------------|
| Training API | `app/api/training.py:25-74` | `create_training_run` |
| Training Service | `app/services/training_service.py` | `create_training_run`, `claim_training_run`, `complete_training_run` |
| Worker Main Loop | `app/workers/training_worker.py:238-271` | `run_forever`, `process_next_job` |
| LocalSubprocessProvider | `app/providers/training_provider.py:93-273` | `submit`, `get_status`, `collect_result` |
| Training Script | `app/training/run_training.py:145-160` | `main` (FastLanguageModel + SFTTrainer) |
| Artifact Storage | `app/services/artifact_storage.py:82-296` | `LocalFilesystemArtifactStorage`, `MinioArtifactStorage` |
| Model Service | `app/services/model_service.py:145-201` | `register_model_version` |
| Serving Backend | `app/services/serving.py:179-319` | `VLLMServingBackend.deploy/unload/generate` |
| Deployment Service | `app/services/deployment_service.py:158-392` | `deploy`, `_deploy_locked` |
| Inference API | `app/api/inference.py:18-73` | `run_inference` |
| Evaluation Worker | `app/workers/evaluation_worker.py:50-86` | `process_next_evaluation` |
| Evaluation Engine | `app/services/evaluation_engine.py:62-150` | `compute_evaluation_update` |
| GPU Orchestrator | `app/workers/gpu_orchestrator.py` | `RealServingCoordinator`, `ShellServingControl` |
| Config | `app/config.py` | `Settings` class (all fields) |
| Docker Compose | `docker-compose.yml` | Service definitions, volumes, environment |
