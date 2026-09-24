# PHASE 2C APPLICATION-LAYER LIVE REHEARSAL REPORT

**Date:** 2026-09-23
**Session:** Fresh OpenCode session
**Issue:** #167

## Objective

Perform a single controlled live rehearsal of the REAL ml-close-loop-be application
promotion ladder using the already successful Phase 2B artifact, proving:

    ml-close-loop-be backend
         ↓
    deploy-staging
         ↓
    validate-staging
         ↓
    promote-production
         ↓
    application inference endpoint
         ↓
    newly trained adapter

## Issue #167 Acceptance Criteria

Issue #167 requires: "Rehearse the full ladder (evaluate → deploy-staging →
validate-staging → promote-production → inference) at least once against a freshly
trained model."

The second half (automated regression test) was completed in PR #184. This session
targets the first half: live rehearsal with real GPU/vLLM.

## Current Serving Before Test

| Component | State |
|-----------|-------|
| defnex-vllm | Running, port 8001, vLLM 0.28.0, static LoRA |
| Compose serving | Not running |
| Backend | Not running |
| Worker | Not running |

GPU: H100 80GB, 78964 MiB used / 81559 MiB total, 0% utilization.
14 tenant GPU processes active (unrelated, preserved throughout).

## Temporary Serving Target

| Attribute | Value |
|-----------|-------|
| Compose service | `serving` (profile `gpu`) |
| Image | `vllm/vllm-openai:latest` (vLLM 0.30.0) |
| Host port | 8002 |
| Base model | `Qwen/Qwen2.5-0.5B-Instruct` |
| Runtime LoRA | Enabled (`VLLM_ALLOW_RUNTIME_LORA_UPDATING=true`) |
| GPU memory utilization | 0.15 (default) |
| Backend VLLM_URL override | `http://172.17.0.1:8002` |

## Authentication

| Step | Result |
|------|--------|
| Register new admin | Blocked (subsequent registrations always get `user` role) |
| Password reset for existing `admin` | WORKED (via `/api/v1/auth/forgot-password` + `/api/v1/auth/reset-password`) |
| Login as admin | WORKED (JWT token obtained) |
| RBAC permissions | `admin` role has: `deployment:deploy`, `model:promote`, `model:validate` |

Note: Alembic migration `b7c8d9e0f1a2` (add `password_reset_tokens` table, issue #177)
was required before the password reset flow worked. Ran `alembic upgrade head` successfully.

## Existing ModelVersion

| Field | Value |
|-------|-------|
| model_id | `smoke-llm-v3` |
| version | 1 |
| status (pre-test) | `REGISTERED` |
| training_run_id | `run-9e66b6` |
| base_model | `Qwen/Qwen2.5-0.5B-Instruct` |
| artifact URI | `file:///models/artifacts/smoke-llm-v3/smoke-llm-v3-Qwen-Qwen2.5-0.5B-Instruct-v1` |
| artifact checksum | `37481825b045c0225c1d6b13350475c52e2c54eb4e95bc18fa201748555ce814` |
| artifact exists | YES (at `/home/ubuntu/defnex-mlops-experiment/outputs/artifacts/smoke-llm-v3/`) |

Pre-test preparation:
- Set evaluation signals via direct DB update (eval_loss_trend, qualitative_comparison,
  general_domain_regression_check, eval_set_id='golden-eval-v1', eval_set_version=1)
- Status moved from REGISTERED → EVALUATED (all three signals present)
- Created `/models` symlink to resolve `file:///models/artifacts/...` paths on host

## Deploy Staging

| Step | Result |
|------|--------|
| Endpoint | `POST /api/v1/models/smoke-llm-v3/versions/1/deploy-staging` |
| HTTP status | 409 Conflict |
| Error code | `STAGING_DEPLOY_NOT_ALLOWED` |
| Error message | `Expecting value: line 1 column 1 (char 0)` |

### Root Cause Analysis

The error is a `json.JSONDecodeError` (subclass of `ValueError`) raised inside
`request_sync_with_retry()` at `app/services/http_retry.py:132`.

**Detailed trace:**
1. `deploy_to_staging_endpoint` calls `promotion_service.deploy_to_staging()`
2. `deploy_to_staging` calls `stage_deploy()`
3. `stage_deploy` calls `deployment_service.deploy(environment="staging")`
4. `_deploy_locked` calls `backend.deploy(model_version)` via `VLLMServingBackend`
5. `VLLMServingBackend.deploy()` calls `request_sync_with_retry("POST", .../v1/load_lora_adapter, ...)`
6. `request_sync_with_retry` calls `resp.raise_for_status()` → passes (HTTP 200 OK)
7. `request_sync_with_retry` calls `resp.json()` → **FAILS**

**The vLLM 0.30.0 `/v1/load_lora_adapter` endpoint returns plain text:**
```
Success: LoRA adapter 'smoke-llm-v3-v1' added successfully.
```
**Content-Type:** not set (defaults to text/plain)
**HTTP status:** 200 OK

The adapter IS loaded successfully by vLLM (confirmed via direct curl test).
But `request_sync_with_retry` unconditionally calls `resp.json()` on success,
which raises `json.JSONDecodeError` because the response body is not JSON.

This `JSONDecodeError` is a subclass of `ValueError`, so it is caught by the
endpoint's `except ValueError as exc` handler and mapped to
`STAGING_DEPLOY_NOT_ALLOWED`.

**This is a BLOCKING BUG in the application layer.**

### Bug Location

| File | Line | Function | Issue |
|------|------|----------|-------|
| `app/services/http_retry.py` | 132 | `request_sync_with_retry` | `resp.json()` called unconditionally on all 2xx responses |
| `app/services/serving.py` | 230 | `VLLMServingBackend.deploy` | Uses `request_sync_with_retry` for `load_lora_adapter` which returns plain text |

### Fix Required

`request_sync_with_retry` needs to handle non-JSON response bodies. Options:

1. **Minimal**: Add a `parse_json: bool = True` parameter to `request_sync_with_retry`;
   `VLLMServingBackend.deploy` passes `parse_json=False`.
2. **Robust**: Catch `json.JSONDecodeError` in `request_sync_with_retry` and return
   the raw text as a fallback (changes return type to `dict | str`).
3. **Clean**: Split into two functions: `request_with_json_response` and
   `request_with_raw_response`.

Option 1 is recommended (minimal, explicit, no type changes).

## Validate Staging

**NOT REACHED** — blocked by deploy-staging failure.

## Promote Production

**NOT REACHED** — blocked by deploy-staging failure.

## Application Inference

**NOT REACHED** — blocked by deploy-staging failure.

## Evidence That New Adapter Was Used

**N/A** — deployment never completed.

Raw vLLM inference WAS confirmed working outside the application layer:
- `POST /v1/load_lora_adapter` → 200 OK, adapter loaded
- `POST /v1/completions` with `model: "smoke-llm-v3-v1"` → valid response with adapter output
- `POST /v1/unload_lora_adapter` → 200 OK, adapter removed

The adapter artifact is correct and functional. The barrier is purely in the
application code's HTTP response handling.

## GPU / Safety

| Check | Result |
|-------|--------|
| GPU reset performed | NO |
| Unrelated processes killed | NO |
| GPU memory baseline | 78964 MiB |
| GPU memory during test (serving running) | 76068 MiB |
| GPU memory after restore | 78964 MiB |
| GPU processes after restore | Same 14 tenant processes (unchanged) |

## Restoration

| Step | Result |
|------|--------|
| Unloaded adapter from compose serving | DONE |
| Stopped compose serving | DONE |
| Started legacy defnex-vllm via GPU controller | DONE (healthy, HTTP 200) |
| Verified legacy vLLM models | `Qwen/Qwen2.5-0.5B-Instruct` + `mlops-lora` |
| Stopped backend process | DONE |
| Reverted smoke-llm-v3 status to REGISTERED | DONE |
| Removed `/models` symlink | DONE |
| Git status | Clean (only pre-existing `unsloth_compiled_cache/`) |

## Configuration Changes

| Change | Scope | Reverted |
|--------|-------|----------|
| `/models` symlink → `/home/ubuntu/defnex-mlops-experiment/outputs` | Host filesystem | YES (removed) |
| `SERVING_BACKEND=vllm` env var | Backend process env | YES (process killed) |
| `VLLM_URL=http://172.17.0.1:8002` env var | Backend process env | YES (process killed) |
| smoke-llm-v3 status → EVALUATED | SQLite DB | YES (reverted to REGISTERED) |
| alembic migration `b7c8d9e0f1a2` | SQLite DB | NOT REVERTED (additive, harmless) |
| `admin` password reset | SQLite DB | NOT REVERTED (needed for future use) |
| `rehearsal-admin` user created | SQLite DB | NOT REVERTED (harmless) |

## Failures / Warnings

### BLOCKING: `json.JSONDecodeError` in `request_sync_with_retry`

- **Endpoint:** `POST /api/v1/models/smoke-llm-v3/versions/1/deploy-staging`
- **HTTP status:** 409
- **Error code:** `STAGING_DEPLOY_NOT_ALLOWED`
- **Root cause:** `request_sync_with_retry` calls `resp.json()` on vLLM's
  `/v1/load_lora_adapter` response, which is plain text, not JSON
- **Impact:** Blocks the entire deployment ladder at step 1
- **Severity:** BLOCKING for application-layer E2E

### WARNING: `password_reset_tokens` table missing

- `alembic upgrade head` was required before password reset worked
- Migration `b7c8d9e0f1a2` adds the table

### WARNING: Artifact permissions root-owned

- Artifact files in `/home/ubuntu/defnex-mlops-experiment/outputs/artifacts/smoke-llm-v3/`
  are owned by `root:root` (created by worker container)
- `sudo chown` was required to make them readable by the host user
- The existing `_make_world_readable` function in `artifact_storage.py` should handle
  this but was not invoked for this artifact (it was created before the fix was deployed)

## Final Classification

**APPLICATION LAYER NOT VERIFIED**

The deployment ladder is blocked at the first step (deploy-staging) by a
`json.JSONDecodeError` in `request_sync_with_retry` caused by vLLM 0.30.0
returning plain text from `/v1/load_lora_adapter` while the application code
expects JSON.

## ISSUE #167 EVIDENCE DELTA

### Acceptance Criteria Status

| Criterion | Status | Evidence |
|-----------|--------|----------|
| Register user | VERIFIED | `POST /api/v1/auth/register` → 201 |
| Login | VERIFIED | `POST /api/v1/auth/login` → 200, JWT obtained |
| Existing model version | VERIFIED | smoke-llm-v3 v1, REGISTERED, artifact exists |
| Deploy staging | **NOT VERIFIED** | 409 STAGING_DEPLOY_NOT_ALLOWED (json.JSONDecodeError) |
| Validate staging | NOT VERIFIED | Blocked by deploy-staging |
| Promote production | NOT VERIFIED | Blocked by deploy-staging |
| Backend inference | NOT VERIFIED | Blocked by deploy-staging |
| New adapter identity | NOT VERIFIED | Blocked by deploy-staging |
| Compose vLLM | VERIFIED | vLLM 0.30.0 healthy, runtime LoRA working |
| Runtime LoRA | VERIFIED | Raw vLLM API load/unload/inference confirmed |
| Original serving restored | VERIFIED | defnex-vllm healthy on port 8001 |
| GPU safety | VERIFIED | No GPU reset, no unrelated processes killed |

### Remaining Gaps

1. **BLOCKING BUG**: `http_retry.py:132` — `resp.json()` called unconditionally;
   vLLM `load_lora_adapter` returns plain text. Must be fixed before the ladder
   can be exercised through the application layer.

2. **Evaluation signal injection**: No automated evaluation worker ran; signals were
   injected via direct DB update. In production, the evaluation worker
   (`app/workers/evaluation_worker.py`) would need to be running and able to reach
   the serving backend.

3. **`/models` symlink**: The artifact URI in the DB uses `file:///models/artifacts/...`
   which only resolves inside Docker containers. For local backend execution, a host
   symlink is needed. Consider making the backend resolve artifact paths relative to
   its own `ARTIFACT_STORAGE_DIR` when the URI uses `file://` scheme.

4. **Alembic migration gap**: The `password_reset_tokens` table migration was missing
   from the local SQLite DB. This should be verified as applied in all environments.

### Evidence Files

- TrainingRun: `run-9e66b6` (COMPLETED)
- ModelVersion: `smoke-llm-v3` v1 (REGISTERED, artifact at
  `file:///models/artifacts/smoke-llm-v3/smoke-llm-v3-Qwen-Qwen2.5-0.5B-Instruct-v1`)
- Artifact: `/home/ubuntu/defnex-mlops-experiment/outputs/artifacts/smoke-llm-v3/smoke-llm-v3-Qwen-Qwen2.5-0.5B-Instruct-v1/`
- Backend logs: `/tmp/backend.log`
- GPU controller state: `/home/ubuntu/defnex-mlops-experiment/outputs/.gpu-control/response.json`
