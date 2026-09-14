# DEFNEX MLOps Integration Implementation Plan

**Date:** 2026-09-11
**Context:** Post-audit implementation plan based on the Integration QA Audit findings.
**Status:** Plan only. No code changes yet.

---

## Current Verified Blockers

### P0 — Impossible to run golden path

| # | Blocker | Verified Location |
|---|---------|-------------------|
| 1 | Backend `SERVING_BACKEND=mock` | `docker exec backend printenv SERVING_BACKEND` → `mock` |
| 2 | Backend `VLLM_URL=http://serving:8000` — container doesn't exist | `docker-compose.yml:11`, DNS fails |
| 3 | Worker has no unsloth/trl/transformers | `docker exec worker python3 -c "import unsloth"` → ModuleNotFoundError |
| 4 | Existing vLLM lacks `VLLM_ALLOW_RUNTIME_LORA_UPDATING` | `docker inspect defnex-vllm` env, `/v1/load_lora_adapter` → 404 |
| 5 | Artifact path mismatch: worker `/app/data` vs vLLM `/data` or `/models` | Worker URI `file:///app/data/artifacts/...`, vLLM mount `./outputs:/models` |

### P1/P2 — Important but not blocking

| # | Issue |
|---|-------|
| 6 | MinIO endpoint `localhost:9000` unreachable from Docker containers |
| 7 | `model_service.register_model_version` bypasses `get_artifact_storage()` factory |
| 8 | Existing vLLM `--max-loras 1` (deploy needs ≥2) |
| 9 | Inference not environment-aware |
| 10 | `python-multipart` now declared in `pyproject.toml` (already fixed) |

---

## Safety Constraints

- Do NOT restart, recreate, stop, remove, or modify `defnex-vllm`
- Do NOT kill GPU processes
- Do NOT reset GPU state
- Do NOT attempt Qwen3.8-27B
- Do NOT consume additional GPU resources
- Do NOT execute a real training job
- Do NOT edit code yet

---

## Section A: SAFE NOW — Changes Not Requiring GPU/VLLM Touch

### A1. Fix `docker-compose.yml` VLLM_URL for backend and worker

**File:** `docker-compose.yml`

**Current behavior:**
- Backend: `VLLM_URL: http://serving:8000` (line 11) — resolves to non-existent container
- Worker: inherits from `.env` which has `VLLM_URL=http://localhost:8001` — unreachable inside Docker

**Desired behavior:**
Both backend and worker reach the existing `defnex-vllm` via the Docker host gateway IP.

**Exact change:**
```yaml
# docker-compose.yml, backend service environment block (line 7-11)
environment:
  VLLM_URL: http://172.17.0.1:8001

# docker-compose.yml, add environment block to worker service (after line 19)
environment:
  VLLM_URL: http://172.17.0.1:8001
```

**Why `172.17.0.1` (bridge gateway) and not the compose gateway `172.19.0.1`:**
- Verified by live test: `docker exec backend python3 -c "import httpx; r=httpx.get('http://172.17.0.1:8001/health', timeout=5); print(r.status_code)"` → 200
- The `172.17.0.1` is the Docker bridge network gateway, which routes to the host. All containers on any bridge network can reach it.
- `defnex-vllm` publishes port 8001 on all host interfaces (`0.0.0.0:8001`), so bridge gateway routing works.

**API/contract impact:** None. Configuration only.

**Migration/config impact:** None. Docker Compose environment variable override.

**Tests required:**
- `docker exec backend python3 -c "import httpx; r=httpx.get('http://172.17.0.1:8001/health', timeout=5); assert r.status_code == 200"` (live connectivity)
- `curl http://localhost:8000/api/v1/health` (backend still boots)

**Risk:** Low. Only changes env var routing. If `172.17.0.1` is not routable (unlikely on standard Docker), deployment would fail with connection error (same as current state).

---

### A2. Set `SERVED_BASE_MODEL` in docker-compose

**File:** `docker-compose.yml`

**Current behavior:**
- `SERVED_BASE_MODEL` empty (base model validation disabled, `config.py:42`)

**Desired behavior:**
- `SERVED_BASE_MODEL=Qwen/Qwen2.5-0.5B-Instruct` — validates base model before deploy

**Exact change:**
```yaml
# docker-compose.yml, backend and worker environment
environment:
  VLLM_URL: http://172.17.0.1:8001
  SERVED_BASE_MODEL: Qwen/Qwen2.5-0.5B-Instruct
```

Note: `SERVING_BACKEND` stays as `mock` for now. It will be set to `vllm` only after worker + artifact path are verified (Phase 1C).

Note: `INFERENCE_SMOKE_ENABLED` defaults to `true` (`config.py:48`) and needs no override.

**API/contract impact:** None.

**Migration/config impact:** Deploy will reject artifacts whose `base_model` doesn't match the served model. For the integration test this is correct — both use Qwen2.5-0.5B-Instruct.

**Tests required:**
- Verify `SERVED_BASE_MODEL` validation blocks mismatched deploys

**Risk:** Low.

---

### A3. Artifact path alignment for standalone vLLM

**File:** `docker-compose.yml` (worker volumes + environment)

**Current behavior:**
- Worker volumes: `./app:/app/app`, `./data:/app/data` → worker writes to `/app/data/artifacts/...`
- vLLM volumes: `./outputs:/models` → vLLM reads from `/models/...`
- These are completely different host directories and container paths — artifact not found

**Desired behavior:**
Both worker and vLLM mount the same host directory (`/home/ubuntu/defnex-mlops-experiment/outputs`) at the same container path (`/models`).

**Exact change:**
```yaml
# docker-compose.yml, worker service
worker:
  volumes:
    - ./app:/app/app
    - /home/ubuntu/defnex-mlops-experiment/outputs:/models
  environment:
    VLLM_URL: http://172.17.0.1:8001
    SERVED_BASE_MODEL: Qwen/Qwen2.5-0.5B-Instruct
    ARTIFACT_STORAGE_DIR: /models/artifacts
```

**Why `/models`?** The standalone vLLM mounts `./outputs:/models`, so the worker must use the same container path. `_adapter_path()` returns `/models/artifacts/...` which vLLM can read directly.

**For the compose `serving` profile** (not used with standalone vLLM): the profile already mounts `./data:/data`, so use `ARTIFACT_STORAGE_DIR=/data/artifacts` instead. The two profiles use different `ARTIFACT_STORAGE_DIR` values to match their mount points.

**API/contract impact:** None. The URI stored in `model_version.artifacts` changes from `file:///app/data/artifacts/...` to `file:///models/artifacts/...`.

**Migration/config impact:** Old artifacts with `file:///app/data/artifacts/...` URIs would not be found. Acceptable for a fresh integration test (no existing artifacts).

**Tests required:**
- Worker can write to `/models/artifacts/` inside container
- `_adapter_path()` returns a path that vLLM can read

**Risk:** Low. Only changes volume mount and config for worker container.

---

### A4. Fix artifact path strategy (in `_adapter_path()`)

**File:** `app/services/serving.py`

**Current behavior (`serving.py:134-147`):**
```python
def _adapter_path(model_version: ModelVersion) -> str:
    for artifact in model_version.artifacts or []:
        if artifact.get("type") == "adapter":
            uri = artifact["uri"]
            return uri[len("file://"):] if uri.startswith("file://") else uri
    raise ServingError(...)
```
Strips `file://` prefix and returns the raw path. If worker stored `file:///models/artifacts/...`, vLLM receives `/models/artifacts/...`.

**Desired behavior:**
No code change needed. The function already works correctly for the aligned-mount strategy (A3). The worker stores `file:///models/artifacts/...` (via `ARTIFACT_STORAGE_DIR=/models/artifacts`), and `_adapter_path()` returns `/models/artifacts/...` which vLLM can read.

**API/contract impact:** None. The URI stored in `model_version.artifacts` changes from `file:///app/data/artifacts/...` to `file:///data/artifacts/...`.

**Migration/config impact:** Old artifacts with `file:///app/data/artifacts/...` URIs would not be found at the new path. This is acceptable for a fresh integration test (no existing artifacts).

**Tests required:**
- `test_vllm_serving.py` already tests `_adapter_path` with `file:///data/adapters/m-v1` URIs (line 97), confirming the expected format
- Add integration test that verifies the stored URI matches what vLLM can read

**Risk:** Low for new artifacts. Existing artifacts (if any) would have wrong paths.

---

### A5. Pass `get_artifact_storage()` through the call chain

**Files:**
- `app/services/model_service.py:191`
- `app/services/deployment_service.py:111`

**Current behavior:**
```python
# model_service.py:191
final_uri = (storage or LocalFilesystemArtifactStorage()).finalize_version(...)

# deployment_service.py:111
storage = storage or LocalFilesystemArtifactStorage()
```
Both default to `LocalFilesystemArtifactStorage()` regardless of `settings.artifact_backend`.

**Desired behavior:**
Use `get_artifact_storage()` as the default when no explicit storage is passed:

```python
# model_service.py:191
from app.services.artifact_storage import get_artifact_storage
final_uri = (storage or get_artifact_storage()).finalize_version(...)

# deployment_service.py:111
from app.services.artifact_storage import get_artifact_storage
storage = storage or get_artifact_storage()
```

**API/contract impact:** None. When `ARTIFACT_BACKEND=local`, `get_artifact_storage()` returns `LocalFilesystemArtifactStorage()` — same behavior. When `ARTIFACT_BACKEND=minio`, it returns `MinioArtifactStorage()` — desired.

**Migration/config impact:** None.

**Tests required:**
- Existing `test_artifact_storage.py` and `test_artifact_storage_minio.py` tests
- Add test verifying `get_artifact_storage()` is called when no explicit storage passed
- Add integration test with `ARTIFACT_BACKEND=minio` + MinIO service

**Risk:** Low. The factory returns the same type as the current default when `ARTIFACT_BACKEND=local`.

---

### A6. Fix MinIO endpoint for Docker networking

**File:** `docker-compose.yml`

**Current behavior:**
- `.env` has `MINIO_ENDPOINT=localhost:9000`
- Inside Docker containers, `localhost` refers to the container itself, not the `minio` service

**Desired behavior:**
Backend and worker reach MinIO at `minio:9000` (Docker Compose service name).

**Exact change:**
```yaml
# docker-compose.yml, backend and worker environment
environment:
  MINIO_ENDPOINT: minio:9000
```

**API/contract impact:** None.

**Migration/config impact:** Only affects Docker Compose. Host-based development continues using `localhost:9000`.

**Tests required:**
- `curl http://localhost:9000/minio/health/live` → 200 (MinIO is running)
- With `ARTIFACT_BACKEND=minio`: verify `MinioArtifactStorage` can list buckets from inside the backend container

**Risk:** Low. Only activates when `ARTIFACT_BACKEND=minio`.

---

### A7. Add `python-multipart` to pyproject.toml (already done)

**File:** `pyproject.toml`

**Current state:** `python-multipart>=0.0.12` is already in `dependencies` (line 21). The comment at line 18-21 explains why it's required.

**Status:** Already fixed. No action needed.

---

### A8. Add vLLM health check to backend `/health` endpoint

**File:** `app/api/health.py`

**Current behavior:**
```python
@router.get("/health")
def health(db: Session = Depends(get_db)) -> dict:
    checks: dict[str, str] = {"db": "ok"}
    # ... DB check ...
    # ... MinIO check ...
    return {"status": "ok", "checks": checks}
```
No vLLM health probe.

**Desired behavior:**
```python
@router.get("/health")
def health(db: Session = Depends(get_db)) -> dict:
    checks: dict[str, str] = {"db": "ok"}
    # ... DB check ...
    # ... MinIO check ...
    if settings.serving_backend == "vllm":
        try:
            r = httpx.get(f"{settings.vllm_url}/health", timeout=5)
            checks["vllm"] = "ok" if r.status_code == 200 else f"error:{r.status_code}"
        except Exception:
            checks["vllm"] = "error"
    return {"status": "ok", "checks": checks}
```

**API/contract impact:** Response gains a `vllm` key in `checks` when `SERVING_BACKEND=vllm`. API consumers that check `status == "ok"` are unaffected.

**Migration/config impact:** None.

**Tests required:**
- `test_health.py` — add test for vLLM health check when `serving_backend=vllm`
- Test with vLLM unreachable → `checks.vllm == "error"`

**Risk:** Low. Health check is informational, never blocks requests.

---

## Section B: GPU/INFRA APPROVAL REQUIRED — Changes Requiring vLLM Modification

### B1. Enable `VLLM_ALLOW_RUNTIME_LORA_UPDATING` on existing vLLM

**Target:** `defnex-vllm` container

**Current behavior:**
- `docker inspect defnex-vllm` env: no `VLLM_ALLOW_RUNTIME_LORA_UPDATING`
- `/v1/load_lora_adapter` returns 404
- vLLM started with `--lora-modules mlops-lora=/models/qwen2.5-0.5b-mlops-lora` (static only)

**Desired behavior:**
- Set `VLLM_ALLOW_RUNTIME_LORA_UPDATING=true`
- Change `--max-loras 1` to `--max-loras 4` (deploy loads new before unloading old, needs ≥2)

**Exact change:**
Recreate `defnex-vllm` with:
```bash
docker stop defnex-vllm && docker rm defnex-vllm

docker run -d --name defnex-vllm \
  --restart unless-stopped \
  -p 8001:8000 \
  -v ~/.cache/huggingface:/root/.cache/huggingface \
  -v /home/ubuntu/defnex-mlops-experiment/outputs:/models:ro \
  --gpus '"device=0"' \
  --ipc=host \
  -e VLLM_ALLOW_RUNTIME_LORA_UPDATING=true \
  vllm/vllm-openai:latest \
  --model Qwen/Qwen2.5-0.5B-Instruct \
  --dtype bfloat16 \
  --max-model-len 1024 \
  --gpu-memory-utilization 0.15 \
  --enable-lora \
  --max-loras 4 \
  --max-lora-rank 16 \
  --lora-modules mlops-lora=/models/qwen2.5-0.5b-mlops-lora
```

**API/contract impact:** None. vLLM API is unchanged; `VLLM_ALLOW_RUNTIME_LORA_UPDATING` is a vLLM server flag.

**Migration/config impact:** Container recreation. Existing adapter `mlops-lora` is preserved via volume mount.

**Tests required:**
- `curl -X POST http://localhost:8001/v1/load_lora_adapter -H 'Content-Type: application/json' -d '{"lora_name":"test","lora_path":"/models/qwen2.5-0.5b-mlops-lora"}'` → 200
- `curl http://localhost:8001/v1/models` → includes the loaded adapter

**Risk:** Medium. Requires stopping and recreating the vLLM container. Brief GPU downtime. No impact on other tenants (this is the only vLLM on this GPU).

---

### B2. Install unsloth/trl/transformers in worker Docker image

**Files:** `Dockerfile`, `docker-compose.yml`

**Current behavior:**
```dockerfile
RUN chmod +x docker-entrypoint.sh docker-worker-entrypoint.sh \
    && pip install --no-cache-dir "."
```
Only installs the app package. No ML dependencies.

**Desired behavior:**
Worker can execute `app/training/run_training.py` which imports `unsloth`, `trl`, `transformers`, `datasets`.

**Option 1 (recommended): Change Docker base to `python:3.12-slim`, then mount experiment venv**

The experiment venv at `/home/ubuntu/defnex-mlops-experiment/.venv` is **Python 3.12.3** (`pyvenv.cfg` shows `executable = /usr/bin/python3.12`). Mounting this venv into a Python 3.11-slim container will cause ABI incompatibility (shared libraries compiled for 3.12 won't load in 3.11).

```dockerfile
# Dockerfile: change base image
FROM python:3.12-slim AS base
```

Then mount the experiment venv in docker-compose:
```yaml
# docker-compose.yml, worker service
volumes:
  - ./app:/app/app
  - /home/ubuntu/defnex-mlops-experiment/outputs:/models
  - /home/ubuntu/defnex-mlops-experiment/.venv:/opt/training-venv:ro
```

Set `TRAINING_PYTHON` (Pydantic auto-maps `TRAINING_PYTHON` env var to `settings.training_python` field):
```yaml
environment:
  TRAINING_PYTHON: /opt/training-venv/bin/python
```

**Option 2: Install ML deps directly in the Dockerfile**

```dockerfile
FROM python:3.12-slim
RUN pip install --no-cache-dir "unsloth[cu124]" trl transformers datasets
```

Pros: no external venv dependency, self-contained image.
Cons: image size increases by ~5GB+, build time increases, CUDA toolkit needed during build.

**Option 1 is recommended** because:
- No image bloat
- Uses the already-working experiment venv
- Training and serving venvs stay separate (per project constraint)
- No GPU/CUDA dependency in the Docker build

**API/contract impact:** None.

**Migration/config impact:** The experiment venv must exist and have unsloth installed. Docker base image must be updated to Python 3.12-slim.

**Tests required:**
- `docker exec worker /opt/training-venv/bin/python -c "import unsloth; print('ok')"` → ok
- Worker logs show no import errors when processing a training run

**Risk:** Medium. Changing Docker base image may affect other dependencies. Verify all existing pip packages still install cleanly with Python 3.12-slim.

---

### B3. Compose `serving` profile port conflict

**File:** `docker-compose.yml:40`

**Current behavior:**
```yaml
ports:
  - "${VLLM_SERVING_PORT:-8001}:8000"
```
Default port 8001 conflicts with existing `defnex-vllm` on host port 8001.

**Desired behavior:**
If the compose `serving` profile is ever used, it should not conflict.

**Exact change:**
```yaml
ports:
  - "${VLLM_SERVING_PORT:-8002}:8000"
```

Or better: since the existing `defnex-vllm` is the real serving target, comment out or document that the compose `serving` profile is for fresh environments only and should not be used when an externally-managed vLLM exists.

**API/contract impact:** None.

**Migration/config impact:** If someone was using port 8001 from compose, they'd need to update.

**Tests required:** None (config only).

**Risk:** Low.

---

## Design Questions and Answers

### Q1: What is the cleanest artifact path strategy for the current architecture?

**Answer: Both containers mount the same host directory at the same container path.**

The `_adapter_path()` function (`serving.py:134-147`) strips `file://` and returns the raw path to vLLM. So the URI stored in the DB must already match the path vLLM sees inside its container. No code change needed in `_adapter_path`.

For the standalone vLLM (`defnex-vllm`), which mounts `./outputs:/models`:

1. Worker mounts the same host dir at `/models`: `-v /home/ubuntu/defnex-mlops-experiment/outputs:/models`
2. `ARTIFACT_STORAGE_DIR=/models/artifacts` (set in `docker-compose.yml`)
3. `LocalFilesystemArtifactStorage` writes to `/models/artifacts/{model_id}/{name}/`
4. `finalize_version` returns `file:///models/artifacts/{model_id}/{name}/`
5. `_adapter_path()` strips `file://` → `/models/artifacts/{model_id}/{name}/`
6. vLLM sees `/models/artifacts/{model_id}/{name}/` via its `./outputs:/models` mount ✓

For the compose `serving` profile (which mounts `./data:/data`), the same logic applies with `ARTIFACT_STORAGE_DIR=/data/artifacts`. The two profiles use different `ARTIFACT_STORAGE_DIR` values to match their respective mount points — this is a config difference, not a code difference.

**Why not symlinks?** A symlink from `./data/artifacts` → `./outputs/artifacts` would make the host directory visible, but the container path sent by `_adapter_path()` (`/data/artifacts/...`) would still not match what vLLM sees (`/models/artifacts/...`). The path must be consistent end-to-end.

---

### Q2: Should worker and vLLM share the same host-mounted directory, or should MinIO become the canonical artifact store now?

**Answer: For the current integration test, use host-shared directory. Plan MinIO for production.**

**Rationale:**
- MinIO is the PRD-intended production solution (PRD §13.1/§13.2)
- But `_adapter_path()` only handles `file://` URIs — `s3://` URIs would be passed verbatim to vLLM which can't read S3
- There's no S3-to-vLLM download bridge implemented
- MinIO adds operational complexity (another service, credential management)
- The host-shared directory works NOW with minimal changes

**Recommendation:**
1. **Current (integration test):** Host-shared directory with aligned mounts
2. **Next phase:** Implement S3-to-local download in `_adapter_path()` or `VLLMServingBackend.deploy()`, then switch to MinIO
3. **Production:** MinIO as canonical store, with download-before-load logic

---

### Q3: How should the existing standalone vLLM be represented as an InferenceTarget?

**Answer: As a `vllm_url_by_env` configuration entry.**

The existing `vllm_url_by_env` config (PRD §16.1) already supports per-environment vLLM URLs:
```
VLLM_URL_BY_ENV=staging:http://172.17.0.1:8001
```

The existing vLLM is the "staging" environment's serving target. A deploy to `environment=staging` would use `http://172.17.0.1:8001`.

**For the default environment (no explicit environment), the `VLLM_URL` setting is used.** Setting `VLLM_URL=http://172.17.0.1:8001` makes the default environment target the existing vLLM.

**No code changes needed.** This is already supported by the existing architecture.

---

### Q4: Should the Compose "serving" profile remain in the main compose file?

**Answer: Keep it, but document it as "for fresh environments only."**

**Rationale:**
- The compose `serving` profile is useful for fresh VMs without an existing vLLM
- On this VM, the existing `defnex-vllm` is the real serving target
- The two should not conflict (fixed by B3: change default port to 8002, already included in Phase 1)
- The `serving` profile is already gated behind `profiles: ["gpu"]` so it doesn't start by default

**Recommendation:** Add a comment in `docker-compose.yml`:
```yaml
# NOTE: This profile is for fresh environments. If an externally-managed vLLM
# already exists (e.g. defnex-vllm), do NOT enable this profile — it will
# conflict on port 8001. Use VLLM_URL to point at the existing vLLM instead.
```

---

### Q5: How to support both current standalone vLLM and future multi-server inference targets?

**Answer: The existing `vllm_url_by_env` + per-environment backends already solve this.**

The architecture is already designed for this:
1. `VLLM_URL` → default environment (current standalone vLLM)
2. `VLLM_URL_BY_ENV` → named environments (future staging/prod vLLM instances)
3. `get_serving_backend(environment)` → resolves to the correct backend per environment
4. Each backend is a separate `VLLMServingBackend` instance with its own `base_url`

**No code changes needed.** The infrastructure is in place. Future multi-server setups just need:
- New vLLM instances on different hosts/ports
- `VLLM_URL_BY_ENV=staging:http://staging-vllm:8001,production:http://prod-vllm:8001`

---

## Implementation Phases

### Phase 1 — Safe integration readiness (NO restart, NO GPU, NO SERVING_BACKEND=vllm)

**Goal:** Config + code changes that make the backend ready, without touching runtime state.

| Step | Change | File | Risk |
|------|--------|------|------|
| 1A.1 | Fix VLLM_URL for backend and worker | `docker-compose.yml` | Low |
| 1A.2 | Set SERVED_BASE_MODEL | `docker-compose.yml` | Low |
| 1A.3 | Worker artifact mount (`outputs:/models`), set `ARTIFACT_STORAGE_DIR=/models/artifacts` | `docker-compose.yml` | Low |
| 1A.4 | Fix MinIO endpoint for Docker | `docker-compose.yml` | Low |
| 1A.5 | Change Docker base to `python:3.12-slim` | `Dockerfile` | Medium |
| 1A.6 | Mount experiment venv, set `TRAINING_PYTHON` | `docker-compose.yml` | Medium |
| 1A.7 | Use `get_artifact_storage()` factory | `model_service.py`, `deployment_service.py` | Low |
| 1A.8 | Add vLLM health check | `app/api/health.py` | Low |
| 1A.9 | Change default serving port to 8002 | `docker-compose.yml` | Low |

**Explicitly NOT in this phase:**
- `SERVING_BACKEND` stays `mock` — real backend enabled only after worker verified
- No `docker compose down` — use targeted restarts only
- No `defnex-vllm` changes
- No training, no GPU

**Apply changes with targeted restart:**
```bash
docker compose up -d --build backend worker
```

**Tests after Phase 1:**
```bash
# Backend boots
curl http://localhost:8000/api/v1/health | python3 -m json.tool

# Backend logs (no startup errors)
docker compose logs backend | tail -10

# Existing tests still pass (SERVING_BACKEND=mock in test conftest)
.venv/bin/pytest tests/ -q --no-cov 2>&1 | tail -5
```

---

### Phase 1B — Verify worker runtime (no training yet)

**Goal:** Confirm ML dependencies are importable inside the worker container.

| Step | Check | Expected |
|------|-------|----------|
| 1B.1 | `docker exec worker /opt/training-venv/bin/python -c "import unsloth, trl, transformers, datasets; print('ML runtime OK')"` | `ML runtime OK` |
| 1B.2 | `docker exec worker /opt/training-venv/bin/python -c "from app.training.run_training import main; print('training entrypoint OK')"` | `training entrypoint OK` |
| 1B.3 | `docker exec worker ls /models/artifacts/` | directory exists or "not yet created" |

**No training is executed in this phase.** This is a plumbing check only.

---

### Phase 1C — Enable real backend

**Goal:** Switch `SERVING_BACKEND` from `mock` to `vllm` AFTER Phase 1 + 1B are verified.

| Step | Change | File | Risk |
|------|--------|------|------|
| 1C.1 | Set `SERVING_BACKEND: vllm` in backend and worker | `docker-compose.yml` | Medium |

**Why this is a separate phase:** Setting `SERVING_BACKEND=vllm` makes every deploy/inference call make real HTTP calls to vLLM. If the worker or artifact path isn't ready, deploys will fail with `ServingError`/`InferenceError` (502). We minimize half-ready states by enabling the real backend only after all plumbing is verified.

**Apply with targeted restart:**
```bash
docker compose up -d --build backend worker
```

**Tests after Phase 1C:**
```bash
# Backend boots with vllm setting
curl http://localhost:8000/api/v1/health | python3 -m json.tool
# Should now show "vllm": "ok" in checks

# Backend logs
docker compose logs backend | tail -10
```

---

### Phase 2 — vLLM approval (APPROVAL REQUIRED)

**Goal:** Recreate `defnex-vllm` with runtime LoRA support. Requires explicit human approval.

| Step | Change | Target | Risk |
|------|--------|--------|------|
| 2.1 | Recreate `defnex-vllm` with `VLLM_ALLOW_RUNTIME_LORA_UPDATING=true` | host | Medium |
| 2.2 | Change `--max-loras 1` → `--max-loras 4` | host | Low |

**Do NOT touch `defnex-vllm` until Phase 1 + 1B + 1C are verified and approved.**

**Tests after Phase 2:**
```bash
# Runtime LoRA API works
curl -s -X POST http://localhost:8001/v1/load_lora_adapter \
  -H 'Content-Type: application/json' \
  -d '{"lora_name":"test-adapter","lora_path":"/models/qwen2.5-0.5b-mlops-lora"}'

# Models endpoint lists loaded adapter
curl -s http://localhost:8001/v1/models | python3 -m json.tool

# Unload test adapter
curl -s -X POST http://localhost:8001/v1/unload_lora_adapter \
  -H 'Content-Type: application/json' \
  -d '{"lora_name":"test-adapter"}'
```

---

### Phase 3 — Integration smoke (broken into sub-tests)

**Goal:** Validate each link in the golden path independently before running the full loop.

| Test | What it validates | How |
|------|-------------------|-----|
| **A** | Backend → vLLM connectivity | `curl http://localhost:8000/api/v1/health` shows `vllm: ok` |
| **B** | Existing adapter → backend inference | `POST /api/v1/models/{model_id}/inference` with an already-deployed version |
| **C** | Worker → Unsloth → artifact | Create a training run with mock runner, verify artifact appears at `/models/artifacts/` |
| **D** | Artifact → registry | `model_service.register_model_version` stores correct URI in DB |
| **E** | Registry → vLLM runtime load | Deploy to staging, verify vLLM loads adapter via `/v1/load_lora_adapter` |
| **F** | **Full golden path** | Upload → Validate → Commit → Train → Evaluate → Deploy → Infer |

**Run tests A-E first.** Only proceed to F when all pass.

**Integration #1: LOCAL shared filesystem only (no MinIO).**

```text
worker → /models → vLLM
```

MinIO integration is **integration #2**, run only after the local golden path is proven.

---

### Phase 4 — Full golden path (integration #1)

**Goal:** End-to-end closed loop with LOCAL artifact storage.

| Step | Action |
|------|--------|
| 4.1 | Seed database: `.venv/bin/python seed.py --reset` |
| 4.2 | Upload test dataset via API |
| 4.3 | Validate + commit dataset |
| 4.4 | Create training run (real, small dataset) |
| 4.5 | Verify worker picks up job |
| 4.6 | Verify artifact appears at `/models/artifacts/` |
| 4.7 | Evaluate model version |
| 4.8 | Deploy to staging |
| 4.9 | Verify vLLM loads adapter |
| 4.10 | Run inference |

**Integration #2 (deferred):** MinIO as canonical artifact store, with S3→vLLM download bridge.

---

## Exact Files Likely to Change

| File | Change Type | Phase |
|------|-------------|-------|
| `docker-compose.yml` | Configuration (env vars, volumes, worker volumes) | 1, 1C |
| `Dockerfile` | Change base to `python:3.12-slim` | 1 |
| `.env` | Configuration (if not using compose env overrides) | 1 |
| `app/services/model_service.py:191` | Use `get_artifact_storage()` | 1 |
| `app/services/deployment_service.py:111` | Use `get_artifact_storage()` | 1 |
| `app/api/health.py` | Add vLLM health check | 1 |
| `tests/test_health.py` | Add vLLM health check test | 1 |
| `tests/test_vllm_serving.py` | Add integration tests | 3 |

---

## Decisions (Confirmed)

| # | Decision | Choice | Rationale |
|---|----------|--------|-----------|
| 1 | Base model for integration #1 | **Qwen2.5-0.5B-Instruct** | Already loaded in existing vLLM |
| 2 | Artifact storage for integration #1 | **LOCAL shared filesystem** | Prove ML loop first, MinIO deferred to integration #2 |
| 3 | Training for integration #1 | **REAL, small dataset** | Mock runner only for plumbing test (Phase 1B), real for golden path |
| 4 | Docker unsloth strategy | **python:3.12-slim + venv mount** | Less bloat, separate venvs, uses existing experiment venv |
| 5 | Compose serving profile | **Keep** (gated behind `profiles: ["gpu"]`) | Useful for fresh environments, documented as "for fresh envs only" |
| 6 | `SERVING_BACKEND` enable timing | **After Phase 1B verified** | Minimize half-ready states |

---

## Safety Constraints (Updated)

- Do NOT restart, recreate, stop, remove, or modify `defnex-vllm` until Phase 2
- Do NOT run `docker compose down` — use targeted restarts only (`docker compose up -d --build backend worker`)
- Do NOT kill GPU processes
- Do NOT reset GPU state
- Do NOT attempt Qwen3.8-27B
- Do NOT consume additional GPU resources
- Do NOT set `SERVING_BACKEND=vllm` until Phase 1B is verified
- MinIO integration is deferred to integration #2 — do not mix LOCAL and MINIO in the same test cycle
