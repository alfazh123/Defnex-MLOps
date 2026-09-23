# DEFNEX MLOps — Terminal QA Golden Path

**Baseline:** 2026-09-16 (backend/API facts re-checked 2026-09-18 against git log — no backend commits landed between those dates, only frontend work, so nothing here is stale on that account).
**Scope:** `Defnex-MLOps/ml-close-loop-be` only (the DEFNEX-MLOps Closed-Loop Backend). This document has nothing to do with the separate DEFNEX Core/Platform backend, which has no code.
**Status of this document as a whole:** `PARTIALLY VERIFIED — see the explicit split in each section`. Do not read a `curl` command's mere presence here as proof it has been run to completion; check its `Verification status` line.

---

## Purpose

A single, copy-pasteable, `curl`-only terminal flow to demonstrate the DEFNEX MLOps closed loop live in front of a mentor, from a clean shell on the VM, with **no dependency on Scalar/Swagger UI** (the demo environment cannot reliably reach them).

This is not a hypothetical "target" flow. Every endpoint, field, and payload below was extracted directly from the current source (`app/api/*.py`, `app/schemas/*.py`, `app/rbac.py`) on 2026-09-16, not from memory or from `openapi.yaml` alone. Where a step has **not** actually been exercised to completion as of this baseline, that is stated explicitly rather than implied.

---

## Current Runtime Assumptions

Read this before touching a keyboard in front of the mentor.

- **Repository:** `~/Defnex-MLOps` — **Backend:** `~/Defnex-MLOps/ml-close-loop-be` (adjust `cd` paths below if the VM's checkout lives elsewhere).
- **`SERVING_BACKEND` is live-configured as `vllm`** on this VM (`VLLM_URL=http://172.17.0.1:8001`) — this is a manual `.env` override on the VM, **not** the repo's own `.env.example` default (which is still `mock`). Do not assume this based on the repo files alone; confirm live with Step 0.2 below every time, since a fresh checkout or a reset `.env` would silently fall back to mock.
- **The real training→serving loop has already been proven once**, real, on this exact VM: `TrainingRun run-9e66b6` completed 5/5 SFT steps on `Qwen/Qwen2.5-0.5B-Instruct`, produced a real LoRA adapter, and registered `ModelVersion smoke-llm-v3 v1` (status `REGISTERED`). Source: `Defnex-MLOps/docs/PHASE2B_POST_RUN_VALIDATION_REPORT.md` and `PHASE2B_THIRD_SMOKE_TRAINING_REPORT.md`, both dated 2026-09-16.
- **`defnex-vllm`, the real serving container, is NOT part of the main `ml-close-loop-be` Docker Compose stack.** It belongs to a separate, external "legacy experiment" Compose project (`defnex-mlops-experiment`), reached by the backend via the Docker host bridge gateway (`172.17.0.1:8001`), not normal Compose service networking. The main stack's Compose file also defines its own `serving` service (GPU profile, port 8002) — **do not confuse the two**. `defnex-vllm` is the one actually in use; the Compose `serving` profile is not started and is not what `VLLM_URL` points at.
- **GPU is a shared H100 80GB.** As of the last check (2026-09-16), non-DEFNEX processes were already holding ~79GB of the 80GB, leaving only ~2GB free while `defnex-vllm` is running (and ~17GB free during the brief window it's stopped for training). This is normal, expected, and exactly what the proven run observed — but it means **there is no slack for a second concurrent training run or a mistake that leaves a process hanging.**
- **Evaluation, once the real evaluation worker starts computing signals, calls out to the configured `ServingBackend` (i.e., real vLLM if `SERVING_BACKEND=vllm`).** This has not been demonstrated end-to-end as of this baseline (see the explicit split below).
- **Everything from "Create TrainingRun" (Step 7) through "Verify ModelVersion" (Step 11) is a `VERIFIED GOLDEN PATH`** — it reproduces a real, already-successful run.
- **Everything from "Evaluation" through "Inference with the new adapter" (Steps 12–15) is `CODE-VERIFIED FROM SOURCE, NOT YET DEMONSTRATED END-TO-END`** as of this baseline. The endpoints and payloads below are real (extracted from `app/api/promotion.py`, `app/api/inference.py`, `app/schemas/*.py`), but no adapter has yet been hot-loaded into `defnex-vllm` and no inference call has yet been made through the backend API against a newly trained adapter. Rehearse this part before relying on it live.

---

## Preconditions

- SSH/terminal access to the VM, `cd`'d into `~/Defnex-MLOps/ml-close-loop-be` (or wherever this checkout lives).
- `docker compose ps` shows `backend`, `worker`, and `minio` as `Up`.
- `curl` and `python3` are available in the shell (this document uses `python3 -c "import json,sys; ..."` for JSON extraction rather than `jq`, matching what the prior version of this document already assumed was present — confirm `jq` separately with `jq --version` if you prefer it; do not assume it's installed without checking).
- You are not mid-way through someone else's training run. Check `curl -s http://localhost:8000/api/v1/training-runs -H "Authorization: Bearer $ACCESS_TOKEN" | python3 -m json.tool` (after Step 2) for any run in `PENDING`/`RUNNING` state before creating a new one.

---

## 0. Environment Variables

### STEP 0.1 — Base URL and health check

**Purpose:** confirm the backend is actually reachable before doing anything else.

**Command:**
```bash
cd ~/Defnex-MLOps/ml-close-loop-be   # sesuaikan path repo di VM ini

export BASE_URL="http://localhost:8000/api/v1"

curl -s "${BASE_URL}/health"
```

**Expected:** HTTP 200, JSON body with a `db` field showing `"ok"`. A `vllm` field showing `"error"` here is expected if `defnex-vllm` happens to be mid-cycle or stopped at this exact moment — re-check after a few seconds before assuming a real problem.

**Capture:** `BASE_URL` (used in every command below).

**Verification status:** `CODE-VERIFIED, RUNTIME-VERIFIED` — this exact health check was used live during the Phase 2B validation.

---

### STEP 0.2 — Confirm the live serving mode (do this every time, don't assume)

**Purpose:** the repo's own `.env.example` default is `SERVING_BACKEND=mock`; the live VM `.env` may or may not still be overridden to `vllm`. Never assume — check.

**Command:**
```bash
docker compose exec backend printenv SERVING_BACKEND
docker compose exec backend printenv VLLM_URL
```

**Expected:** `vllm` and `http://172.17.0.1:8001` respectively, per the 2026-09-16 baseline. If you get `mock` instead, the rest of this document downstream of training will behave differently — evaluation/inference will short-circuit to mock responses instead of calling real `defnex-vllm`. Note which mode you're actually in before the demo and adjust your narration accordingly; do not claim "real inference" if this comes back `mock`.

**Verification status:** `RUNTIME-VERIFIED` as of 2026-09-16 05:15 UTC (`RUNTIME_INFRASTRUCTURE_INTEGRATION_VERIFICATION.md` §3, §6). Re-check live — this is exactly the kind of thing that can silently change between baseline and demo day.

---

## 1. Register Demo User

### STEP 1 — Register

**Purpose:** create a demo identity with full permissions (admin holds every RBAC permission used below — `app/rbac.py:43`).

**Command:**
```bash
curl -s -X POST "${BASE_URL}/auth/register" \
  -H "Content-Type: application/json" \
  -d '{"username":"demo_admin","password":"Demo1234","role":"admin"}'
```

**Expected:** HTTP 201 with a `UserResponse` body (`id`, `username`, `role`, `created_at`), OR `409 USERNAME_TAKEN` if `demo_admin` already exists from a previous rehearsal — that's fine, skip straight to Step 2. Password must be ≥8 characters with at least one uppercase letter and one digit (`app/schemas/auth.py` — `Demo1234` satisfies this). Self-registration has no role restriction in the current code (`app/api/auth.py:100-115` — anyone can register as `admin`); this is a real, if permissive, characteristic of the current implementation, not something this document invented. Registration is rate-limited to 3/minute (`@limiter.limit("3/minute")`) — don't retry rapidly on a transient failure.

**Capture:** nothing (login in Step 2 is what matters).

**Verification status:** `CODE-VERIFIED FROM SOURCE` (`app/api/auth.py:100-115`, `app/schemas/auth.py`).

---

## 2. Login

### STEP 2 — Login and capture the access token

**Purpose:** obtain the bearer token used by every subsequent call.

**Command:**
```bash
ACCESS_TOKEN=$(curl -s -X POST "${BASE_URL}/auth/login" \
  -H "Content-Type: application/json" \
  -d '{"username":"demo_admin","password":"Demo1234"}' \
  | python3 -c "import json,sys; print(json.load(sys.stdin)['access_token'])")

echo "token acquired: ${ACCESS_TOKEN:0:12}... (truncated — never echo the full token)"
```

**Expected:** a non-empty truncated token printed. If this prints an empty string or a Python traceback, the login itself failed — re-run the raw `curl` without the `python3` pipe to see the actual error body (likely `401` with wrong credentials, or `429` if rate-limited).

**Capture:** `ACCESS_TOKEN` — used as `-H "Authorization: Bearer ${ACCESS_TOKEN}"` on every call from here on. **Never print `$ACCESS_TOKEN` in full, and never paste a real token into this file or into chat/screen-share logs.**

**Verification status:** `CODE-VERIFIED FROM SOURCE` (`app/schemas/auth.py` — `LoginRequest{username, password}` → `TokenResponse{access_token, refresh_token, token_type, user}`).

---

## 3. Prepare Demo Dataset

Two options — pick based on how much risk you can tolerate live in front of the mentor.

### Option A (recommended for the live demo) — reuse the already-proven dataset

The dataset that actually produced the successful `run-9e66b6` is already registered on this VM: `dataset_id=ds-smoke-test-v1`, `dataset_version=1`, backed by the existing repo fixture `data/datasets/smoke_train.jsonl` (10 examples, already present on disk — do not recreate it). This is the lowest-risk path because this exact combination is the one with real evidence of success.

```bash
DATASET_ID="ds-smoke-test-v1"
DATASET_VERSION=1

curl -s "${BASE_URL}/datasets/${DATASET_ID}/versions/${DATASET_VERSION}" \
  -H "Authorization: Bearer ${ACCESS_TOKEN}" | python3 -m json.tool
```

**Expected:** HTTP 200 confirming the dataset version still exists with `status: "PROCESSED"`. If it's gone (e.g., a fresh VM/DB reset since 2026-09-16), fall back to Option B.

**Verification status:** `RUNTIME-VERIFIED` — this is the literal dataset used by `run-9e66b6` (`PHASE2B_THIRD_SMOKE_TRAINING_REPORT.md`).

### Option B — demonstrate the full intake pipeline with a fresh tiny dataset

Use this if you specifically want to show the intake→validate→commit steps live (Sections 4–6 below), rather than skip straight to training. **Important:** the resulting dataset has NOT itself been proven to train successfully end-to-end as of this baseline — only `ds-smoke-test-v1` (Option A) has. If you use Option B's output for Step 7's training run, treat that combination as unverified (see Step 7).

```bash
cat > /tmp/demo-golden-path.jsonl << 'EOF'
{"id": "r1", "messages": [{"role": "user", "content": "Bagaimana cara reset password akun saya?"}, {"role": "assistant", "content": "Silakan buka halaman login, klik tautan lupa password, lalu ikuti instruksi yang dikirim ke email terdaftar Anda untuk membuat password baru dengan aman."}], "metadata": {"source_dataset": "demo-golden-path", "source_id": "demo-1"}}
{"id": "r2", "messages": [{"role": "user", "content": "Berapa lama waktu pengiriman standar?"}, {"role": "assistant", "content": "Waktu pengiriman standar biasanya membutuhkan tiga sampai lima hari kerja tergantung lokasi tujuan dan ketersediaan kurir di wilayah Anda pada saat ini."}], "metadata": {"source_dataset": "demo-golden-path", "source_id": "demo-2"}}
{"id": "r3", "messages": [{"role": "user", "content": "Apakah bisa membatalkan pesanan yang sudah dibayar?"}, {"role": "assistant", "content": "Pembatalan pesanan yang sudah dibayar dapat dilakukan selama status masih diproses, silakan hubungi layanan pelanggan kami secepatnya sebelum barang dikirim ke alamat tujuan."}], "metadata": {"source_dataset": "demo-golden-path", "source_id": "demo-3"}}
EOF
```

**Verification status:** `NOT RE-EXECUTED IN THIS UPDATE PASS`. This exact JSONL content is unchanged from the prior version of this document and follows the documented canonical ChatML schema (`docs/dataset/dataset-lifecycle-and-schema.md`), but it was not re-run against the live intake validator as part of writing this update — the intake/validate/commit *endpoints and request/response fields* below were verified from source (`app/api/intake.py`, `app/api/intake_validate.py`); the *content* passing validation is carried forward, not freshly re-proven. If it fails validation live, that is new information for this baseline and should be reported back, not silently worked around.

---

## 4. Dataset Intake / Upload

*(Skip this section entirely if you chose Option A in Step 3 — go straight to Step 7.)*

### STEP 4 — Inspect (upload) the file

**Purpose:** Step 1 of the intake wizard — stages the file and returns a `staging_id`. Admin-only (`app/api/intake.py:90-92`, `require_admin`).

**Command:**
```bash
INSPECT=$(curl -s -X POST "${BASE_URL}/datasets/intake/inspect" \
  -H "Authorization: Bearer ${ACCESS_TOKEN}" \
  -F "file=@/tmp/demo-golden-path.jsonl")
echo "$INSPECT" | python3 -m json.tool

STAGING_ID=$(echo "$INSPECT" | python3 -c "import json,sys; print(json.load(sys.stdin)['staging_id'])")
echo "staging_id: $STAGING_ID"
```

**Expected:** HTTP 200, `DatasetInspectResponse` with `staging_id`, `detected_format: "jsonl"`, `detected_sample_count: 3`, `checksum_sha256`, `parse_status`. A `400 UNSUPPORTED_FORMAT`/`EMPTY_FILE`/`PARSE_ERROR` means the file itself is the problem — check `/tmp/demo-golden-path.jsonl` was actually written by Step 3.

**Capture:** `STAGING_ID`.

**Verification status:** `CODE-VERIFIED FROM SOURCE` (`app/api/intake.py:81-138`).

---

## 5. Dataset Validation

### STEP 5 — Validate the staged file

**Purpose:** run H1–H9 structural validation against the staged upload (Step 4 of the intake wizard).

**Command:**
```bash
VALIDATE=$(curl -s -X POST "${BASE_URL}/datasets/intake/validate" \
  -H "Authorization: Bearer ${ACCESS_TOKEN}" -H "Content-Type: application/json" \
  -d "{\"staging_id\":\"$STAGING_ID\",\"dataset_id\":\"demo-golden-path\"}")
echo "$VALIDATE" | python3 -m json.tool

echo "$VALIDATE" | python3 -c "import json,sys; d=json.load(sys.stdin); print('status:', d['status'])"

REPORT_ID=$(echo "$VALIDATE" | python3 -c "import json,sys; print(json.load(sys.stdin)['validation_report_id'])")
echo "validation_report_id: $REPORT_ID"
```

**Expected:** `status` must NOT be `FAIL`. The response also includes `total_records`, `valid_records`, `warning_count`, `blocking_error_count`, `checks` (a list of named pass/fail checks), `checksum_sha256`. `schema_name` defaults to `defnex_scenario_v1` and `source_format` to `jsonl` if omitted, per `app/api/intake_validate.py:27-33` — this document relies on those defaults rather than passing them explicitly, matching the schema's own design.

**Capture:** `REPORT_ID` (an integer — used as `validation_report_id` in Step 6, not a string).

**Verification status:** `CODE-VERIFIED FROM SOURCE` (`app/api/intake_validate.py:44-105`). Do NOT treat an HTTP 200 here as "dataset ready" on its own — the `status` field inside the body is the actual gate, not the HTTP status code (per the task's own reminder and the code's `gate_decision` field).

---

## 6. Dataset Version / Commit

### STEP 6 — Commit as an immutable DatasetVersion

**Purpose:** Step 5 of the intake wizard — finalizes the staged, validated file into a real, immutable `DatasetVersion` row.

**Command:**
```bash
COMMIT=$(curl -s -X POST "${BASE_URL}/datasets/intake/commit" \
  -H "Authorization: Bearer ${ACCESS_TOKEN}" -H "Content-Type: application/json" \
  -d "{\"staging_id\":\"$STAGING_ID\",\"dataset_id\":\"demo-golden-path\",\"validation_report_id\":$REPORT_ID,\"display_name\":\"Demo Golden Path Dataset\"}")
echo "$COMMIT" | python3 -m json.tool

DATASET_ID="demo-golden-path"
DATASET_VERSION=$(echo "$COMMIT" | python3 -c "import json,sys; print(json.load(sys.stdin)['version'])")
echo "dataset_id: $DATASET_ID  dataset_version: $DATASET_VERSION"
```

**Expected:** HTTP 200, a `DatasetVersion` record with `status: "PROCESSED"` and the allocated `version` integer. `409 VALIDATION_FAILED` means the validation report's gate was `FAIL` — do not attempt to commit a failed validation. `409 CHECKSUM_MISMATCH` means the staged file changed between validate and commit — re-run Steps 4–5. You may optionally add `-H "X-Idempotency-Key: <any-unique-string>"` to make a retried commit safely replay the same result instead of erroring (`app/api/intake_validate.py:143-152` — a real, current feature, not required for the demo to work).

**Capture:** `DATASET_ID`, `DATASET_VERSION` (both now point at your freshly committed dataset — or, if you used Option A, these were already set to `ds-smoke-test-v1` / `1` in Step 3).

**Verification status:** `CODE-VERIFIED FROM SOURCE` (`app/api/intake_validate.py:145-232`).

---

## 7. Create TrainingRun

This is where Option A vs Option B (Step 3) matters for how much confidence you can have going in.

### STEP 7 — Create the training run

**Purpose:** trigger a real SFT training run via the worker-polled path — the same path that produced `run-9e66b6`.

**Command (Option A — reproduces the proven config exactly, recommended for the live demo):**
```bash
MODEL_ID="demo-golden-path"

TRAIN=$(curl -s -X POST "${BASE_URL}/training-runs" \
  -H "Authorization: Bearer ${ACCESS_TOKEN}" -H "Content-Type: application/json" \
  -d "{
    \"dataset_id\": \"ds-smoke-test-v1\",
    \"dataset_version\": 1,
    \"model_id\": \"${MODEL_ID}\",
    \"base_model\": \"Qwen/Qwen2.5-0.5B-Instruct\",
    \"training_config\": {
      \"peft_method\": \"lora\",
      \"format_type\": \"text\",
      \"hf_dataset\": \"/app/data/datasets/smoke_train.jsonl\",
      \"batch_size\": 1,
      \"gradient_accumulation_steps\": 1,
      \"max_seq_length\": 128,
      \"max_steps\": 5,
      \"lora_r\": 8,
      \"lora_alpha\": 8,
      \"lora_dropout\": 0.0,
      \"target_modules\": [\"q_proj\", \"v_proj\"],
      \"epochs\": 1,
      \"learning_rate\": 2e-5,
      \"optim\": \"adamw_8bit\",
      \"random_seed\": 42
    },
    \"triggered_by\": \"demo_admin\"
  }")
echo "$TRAIN" | python3 -m json.tool

RUN_ID=$(echo "$TRAIN" | python3 -c "import json,sys; print(json.load(sys.stdin)['training_run_id'])")
echo "run_id: $RUN_ID"
```

This is the **exact configuration** used by `run-9e66b6` (`Defnex-MLOps/docs/PHASE2B_THIRD_SMOKE_TRAINING_REPORT.md`, "Exact Config"), reproduced field-for-field against the current `TrainingConfig` schema (`app/schemas/training.py:34-92`). `model_id` is set to a fresh, distinct value (`demo-golden-path`, matches `[A-Za-z0-9][A-Za-z0-9._-]*` per `app/schemas/training.py:107-114`) so this run's `ModelVersion` doesn't collide with the existing `smoke-llm-v3`.

**Command (Option B — using your freshly committed dataset from Step 6; NOT yet proven to complete successfully):**
```bash
MODEL_ID="demo-golden-path"

TRAIN=$(curl -s -X POST "${BASE_URL}/training-runs" \
  -H "Authorization: Bearer ${ACCESS_TOKEN}" -H "Content-Type: application/json" \
  -d "{
    \"dataset_id\": \"${DATASET_ID}\",
    \"dataset_version\": ${DATASET_VERSION},
    \"model_id\": \"${MODEL_ID}\",
    \"base_model\": \"Qwen/Qwen2.5-0.5B-Instruct\",
    \"training_config\": {
      \"peft_method\": \"lora\", \"format_type\": \"chatml\",
      \"batch_size\": 1, \"gradient_accumulation_steps\": 1, \"max_seq_length\": 128,
      \"max_steps\": 5, \"lora_r\": 8, \"lora_alpha\": 8, \"lora_dropout\": 0.0,
      \"target_modules\": [\"q_proj\", \"v_proj\"], \"epochs\": 1,
      \"learning_rate\": 2e-5, \"optim\": \"adamw_8bit\", \"random_seed\": 42
    },
    \"triggered_by\": \"demo_admin\"
  }")
echo "$TRAIN" | python3 -m json.tool
RUN_ID=$(echo "$TRAIN" | python3 -c "import json,sys; print(json.load(sys.stdin)['training_run_id'])")
```
`format_type: "chatml"` is the schema default and matches the `messages`-array content committed in Step 6 — but this specific dataset-content + `chatml` combination has not itself been run to a `COMPLETED` state as of this baseline (only the raw-`text` + `ds-smoke-test-v1` combination in Option A has). If you use Option B for a live demo, rehearse it beforehand.

**Expected:** HTTP 201, a `TrainingRun` record with `status: "PENDING"`.

**Capture:** `RUN_ID`, `MODEL_ID`.

**GPU safety reminder:** creating this run will, via the proven GPU-controller cycle, briefly stop `defnex-vllm` (~3 minutes, mirroring `run-9e66b6`'s 185-second duration) and restart it automatically afterward. This is expected, not a failure — do not manually intervene with the container during this window.

**Verification status:** Option A = `RUNTIME-VERIFIED` (reproduces `run-9e66b6`'s config exactly; the *specific new run* is still a fresh execution, not a literal guarantee, but it is the closest thing to a known-good deterministic path this baseline has). Option B = `CODE-VERIFIED FROM SOURCE ONLY`.

---

## 8. Monitor TrainingRun

### STEP 8 — Poll for status (bounded, no infinite loop)

**Purpose:** watch the run progress without needing DB/log access. Uses the plain `GET` endpoint, **not** `GET /training-runs/{id}/progress` — that endpoint is a Server-Sent-Events stream tied to the *inline* `unsloth_client` execution path (`app/api/training.py:154-173`), not the worker-polled path this document uses, and would not reliably show live progress for a worker-processed run.

**Command:**
```bash
for i in $(seq 1 40); do
  STATUS_JSON=$(curl -s "${BASE_URL}/training-runs/${RUN_ID}" -H "Authorization: Bearer ${ACCESS_TOKEN}")
  STATUS=$(echo "$STATUS_JSON" | python3 -c "import json,sys; print(json.load(sys.stdin)['status'])")
  STEP=$(echo "$STATUS_JSON" | python3 -c "import json,sys; print(json.load(sys.stdin).get('current_step'))")
  echo "[$i] status=$STATUS current_step=$STEP"
  if [ "$STATUS" = "COMPLETED" ] || [ "$STATUS" = "FAILED" ]; then
    break
  fi
  sleep 15
done
```

**Expected:** status moves `PENDING` → `RUNNING` → `COMPLETED` within roughly 3–4 minutes (the proven run took ~185 seconds of actual training, plus queueing/GPU-handoff overhead either side). The loop above is bounded to 40 iterations × 15s = 10 minutes maximum — it will not hang forever. **Only one training run is created in this document** (`RUN_ID` from Step 7) — do not re-run Step 7 while this loop is still going.

**Capture:** nothing new; `RUN_ID` from Step 7 is still in scope.

**Verification status:** `CODE-VERIFIED FROM SOURCE` (`app/schemas/training.py:124-140` — `TrainingRun.status`/`current_step` fields exist on the plain GET response) + `RUNTIME-VERIFIED` timing (matches the observed ~185s duration in `PHASE2B_THIRD_SMOKE_TRAINING_REPORT.md`).

---

## 9. Verify Training Completion

### STEP 9 — Confirm COMPLETED and capture the model version

**Purpose:** don't just trust the last loop iteration — do one final explicit check.

**Command:**
```bash
FINAL=$(curl -s "${BASE_URL}/training-runs/${RUN_ID}" -H "Authorization: Bearer ${ACCESS_TOKEN}")
echo "$FINAL" | python3 -m json.tool

echo "$FINAL" | python3 -c "
import json,sys
d = json.load(sys.stdin)
print('status:', d['status'])
print('model_version:', d.get('model_version'))
print('dataset:', d['dataset_id'], 'v'+str(d['dataset_version']))
print('base_model:', d['base_model'])
"

MODEL_VERSION=$(echo "$FINAL" | python3 -c "import json,sys; print(json.load(sys.stdin)['model_version'])")
echo "model_version: $MODEL_VERSION"
```

**Expected:** `status: "COMPLETED"`, a non-null `model_version` integer. If `status: "FAILED"`, do NOT proceed — read the `Troubleshooting` section below before retrying.

**Capture:** `MODEL_VERSION`.

**Verification status:** `RUNTIME-VERIFIED` — `run-9e66b6` reached exactly this state (`PHASE2B_POST_RUN_VALIDATION_REPORT.md`, "Successful Run" table).

---

## 10. Verify Artifact

### STEP 10 — Confirm a real artifact exists via the registry, not just a status flag

**Purpose:** a `COMPLETED` status alone doesn't prove an artifact file exists — check the registry's own artifact record.

**Command:**
```bash
curl -s "${BASE_URL}/models/${MODEL_ID}/versions/${MODEL_VERSION}" \
  -H "Authorization: Bearer ${ACCESS_TOKEN}" \
  | python3 -c "
import json,sys
d = json.load(sys.stdin)
print('status:', d['status'])
print('training_run_id:', d['training_run_id'])
for a in d['artifacts']:
    print('artifact:', a['type'], a['uri'], a.get('size_bytes'), a.get('checksum'))
"
```

**Expected:** at least one entry in `artifacts` with `type: "adapter"` and a `uri` starting `file:///models/artifacts/...`, matching the pattern `file:///models/artifacts/smoke-llm-v3/smoke-llm-v3-Qwen-Qwen2.5-0.5B-Instruct-v1` observed for the proven run (your `model_id`/version will differ). `training_run_id` must equal `${RUN_ID}` — this is the lineage check, not an assumption.

**Verification status:** `CODE-VERIFIED FROM SOURCE` (`app/schemas/model.py:111-132` — `ModelRegistryRecord.artifacts: list[Artifact]`) + `RUNTIME-VERIFIED` for the proven run's exact artifact shape.

---

## 11. Verify ModelVersion / Registry

### STEP 11 — Full registry record

**Purpose:** show the complete lineage in one call: dataset → training run → model version → artifact.

**Command:**
```bash
curl -s "${BASE_URL}/models/${MODEL_ID}/versions/${MODEL_VERSION}" \
  -H "Authorization: Bearer ${ACCESS_TOKEN}" | python3 -m json.tool
```

**Expected:** HTTP 200, `status: "REGISTERED"` (not yet `EVALUATED`/`DEPLOYED` — that's the next, unverified section), plus `dataset_id`, `dataset_version`, `base_model`, `training_config`, `training_config_hash`, `created_at`, `created_by`. This is your **end of the verified golden path** — everything up to here reproduces real, already-proven behavior.

**Verification status:** `RUNTIME-VERIFIED` (`smoke-llm-v3` v1 reached exactly `status: REGISTERED` — `PHASE2B_POST_RUN_VALIDATION_REPORT.md`, "Model Registry" section).

---

## — Boundary: everything below this line is CODE-VERIFIED FROM SOURCE, NOT YET DEMONSTRATED END-TO-END as of 2026-09-16 —

---

## 12. Deployment / Serving

The lifecycle ladder is `REGISTERED → EVALUATED → (STAGING/decisions) → VALIDATED → PRODUCTION/DEPLOYED` (`app/schemas/model.py:8-18`, `ModelLifecycleStatus`). A model must be evaluated before it can be deployed to staging.

### STEP 12a — Register an eval set (skip if one already exists)

```bash
curl -s -X POST "${BASE_URL}/eval-sets/demo-eval-set/versions" \
  -H "Authorization: Bearer ${ACCESS_TOKEN}" -H "Content-Type: application/json" \
  -d '{"records": [{"messages": [{"role": "user", "content": "Bagaimana cara mengembalikan barang yang cacat?"}]}]}' \
  | python3 -m json.tool
```

### STEP 12b — Trigger evaluation

**Purpose:** trigger the evaluation worker to compute real signals via the configured `ServingBackend`.

**Command:**
```bash
curl -s -X POST "${BASE_URL}/models/${MODEL_ID}/versions/${MODEL_VERSION}/evaluation" \
  -H "Authorization: Bearer ${ACCESS_TOKEN}" -H "Content-Type: application/json" \
  -d '{"eval_set_id": "demo-eval-set", "eval_set_version": 1}' \
  | python3 -m json.tool
```

**Important:** as of `app/schemas/model.py:79-94` (issue #128), this endpoint is **trigger-only** — it no longer accepts caller-supplied evaluation numbers (`extra="forbid"` — a request shaped like an old-style manual payload will fail with `422`). Only `eval_set_id`/`eval_set_version` are accepted.

### STEP 12c — Run the evaluation worker (still not in docker-compose.yml — manual, in a separate terminal)

```bash
docker compose exec backend python -m app.workers.evaluation_worker
```

Leave this running in a separate terminal/pane; `Ctrl+C` once you see the version's status change (Step 12d).

### STEP 12d — Confirm EVALUATED

```bash
curl -s "${BASE_URL}/models/${MODEL_ID}/versions/${MODEL_VERSION}" \
  -H "Authorization: Bearer ${ACCESS_TOKEN}" | python3 -c "import json,sys; print(json.load(sys.stdin)['status'])"
```

**Expected:** `EVALUATED`.

**Verification status of Section 12 as a whole:** `CODE-VERIFIED FROM SOURCE` only. No evaluation has been run against `smoke-llm-v3` (or any freshly trained model) as of this baseline. Since `SERVING_BACKEND=vllm` is live, this worker will attempt to call **real** `defnex-vllm` for the evaluation signals — this has not been tried with a newly-registered adapter and may surface issues (see Troubleshooting).

---

## 13. Runtime Adapter Load

There is **no separate "load adapter" endpoint** the operator calls directly. Loading happens as an automatic side effect inside the deploy-staging/promote-production calls below (`app/api/promotion.py:102-104`: *"smoke test runs automatically as part of the deploy (PRD §38.4)"*). This section documents that ladder.

### STEP 13a — Deploy to staging

```bash
curl -s -X POST "${BASE_URL}/models/${MODEL_ID}/versions/${MODEL_VERSION}/deploy-staging" \
  -H "Authorization: Bearer ${ACCESS_TOKEN}" -H "Content-Type: application/json" \
  -d '{"decided_by": "demo_admin", "rationale": "Eval selesai, mau demo closed-loop."}' \
  | python3 -m json.tool
```

Requires `EVALUATED` status first (Step 12). Can fail `503 GPU_LOCK_TIMEOUT` (GPU busy) or `409 STAGING_DEPLOY_NOT_ALLOWED` (wrong status). This is the call that actually attempts to hot-load the new adapter into `defnex-vllm` via its runtime LoRA API — real GPU coordination, not a database-only status flip.

### STEP 13b — Validate staging

```bash
curl -s -X POST "${BASE_URL}/models/${MODEL_ID}/versions/${MODEL_VERSION}/validate-staging" \
  -H "Authorization: Bearer ${ACCESS_TOKEN}" -H "Content-Type: application/json" \
  -d '{"decided_by": "demo_admin", "rationale": "Sudah dicoba manual di staging."}' \
  | python3 -m json.tool
```

### STEP 13c — Promote to production

```bash
curl -s -X POST "${BASE_URL}/models/${MODEL_ID}/versions/${MODEL_VERSION}/promote-production" \
  -H "Authorization: Bearer ${ACCESS_TOKEN}" -H "Content-Type: application/json" \
  -d '{"decided_by": "demo_admin", "rationale": "Lolos validasi staging."}' \
  | python3 -m json.tool
```

### STEP 13d — Confirm deployment pointer

```bash
curl -s "${BASE_URL}/models/${MODEL_ID}/deployment" \
  -H "Authorization: Bearer ${ACCESS_TOKEN}" | python3 -m json.tool
```

**Expected:** `DeploymentStatus{model_id, current_deployed_version, deployed_at, status: "DEPLOYED"}` with `current_deployed_version` equal to `${MODEL_VERSION}`.

**Verification status of Section 13 as a whole:** `CODE-VERIFIED FROM SOURCE` (`app/api/promotion.py:85-198`, `app/api/deployment.py:145-155`). All three RBAC permissions (`DEPLOY`, `VALIDATE_STAGING`, `PROMOTE`) are held by the `admin` role (`app/rbac.py:43`), so the demo user can execute the whole ladder — but **no version has ever completed this ladder on this VM as of 2026-09-16.** Rehearse before the live demo.

---

## 14. Inference

### STEP 14 — Run inference against the production alias

**Command:**
```bash
curl -s -X POST "${BASE_URL}/models/${MODEL_ID}/inference" \
  -H "Authorization: Bearer ${ACCESS_TOKEN}" -H "Content-Type: application/json" \
  -d '{"target": "prod", "prompt": "Bagaimana cara reset password?"}' \
  | python3 -m json.tool
```

**Important, do not add fields the API doesn't have:** `InferenceRequest` (`app/schemas/inference.py`) has exactly two fields, `target` and `prompt`. **There is no `max_tokens` or `temperature` parameter exposed by this endpoint as of this baseline** — do not pass them; they will be silently ignored at best or rejected at worst depending on Pydantic's config. If sampling control is needed for the demo, that is a real, current API gap to name out loud, not something to work around by inventing a field.

**Expected:** HTTP 200, `InferenceResponse{model_id, version, generation}`. A `404 DEPLOYMENT_NOT_FOUND` means Section 13's ladder hasn't completed (nothing is deployed to `prod` yet) — this is the most likely failure mode given Section 13's unverified status. A `502 INFERENCE_FAILED` means the request reached real `defnex-vllm` but it errored.

**Which kind of inference is this?** Per the code comment in `app/api/inference.py:36-39`, `InferenceResponse.version` names "the concrete version whose adapter actually served the request" — **read this field, don't assume**. If `version` equals `${MODEL_VERSION}` (this run's freshly trained model), this is (C) newly trained adapter inference. If it equals some other version number, the `prod` alias was still pointing at a previously deployed model — that would be (B) existing adapter inference, and you should say so, not claim otherwise.

**Verification status:** `CODE-VERIFIED FROM SOURCE` only (`app/api/inference.py`). No inference call has been made through this endpoint against a freshly trained adapter as of this baseline.

---

## 15. Verify Newly Trained Adapter Was Used

Do not skip this — it is the difference between "we ran a demo" and "we proved the closed loop."

### STEP 15 — Cross-check the inference response against the training lineage

```bash
INFER_RESPONSE=$(curl -s -X POST "${BASE_URL}/models/${MODEL_ID}/inference" \
  -H "Authorization: Bearer ${ACCESS_TOKEN}" -H "Content-Type: application/json" \
  -d '{"target": "prod", "prompt": "Bagaimana cara reset password?"}')

echo "$INFER_RESPONSE" | python3 -c "
import json,sys
d = json.load(sys.stdin)
served_version = d['version']
expected_version = ${MODEL_VERSION}
print('served version:', served_version, '| expected (this run):', expected_version)
print('MATCH — this is genuinely the newly trained adapter' if served_version == expected_version else 'MISMATCH — prod is still serving a different, earlier version; do not claim this is the new adapter')
"
```

Only when this prints `MATCH` may the demo narration claim "the model we just trained is the one answering." Anything else — say so plainly.

The chain to keep in mind, and which step of it you've actually reached (per this baseline, before rehearsal):

```
artifact exists              → PROVEN (Step 10, run-9e66b6 real evidence)
artifact registered          → PROVEN (Step 11)
artifact visible to serving  → PROVEN (defnex-vllm can see /models mount — PHASE2B_POST_RUN_VALIDATION_REPORT.md §Serving)
adapter loaded                → NOT YET DEMONSTRATED (Step 13a's actual hot-load has not been exercised)
adapter selected by inference → NOT YET DEMONSTRATED
inference successful (new)    → NOT YET DEMONSTRATED
```

**Verification status:** the check itself is `CODE-VERIFIED FROM SOURCE`; the outcome is unknown until rehearsed.

---

## 16. Final Verification

### STEP 16 — One-shot summary of everything created this session

```bash
echo "=== Training Run ===" && curl -s "${BASE_URL}/training-runs/${RUN_ID}" -H "Authorization: Bearer ${ACCESS_TOKEN}" | python3 -m json.tool
echo "=== Model Version ===" && curl -s "${BASE_URL}/models/${MODEL_ID}/versions/${MODEL_VERSION}" -H "Authorization: Bearer ${ACCESS_TOKEN}" | python3 -m json.tool
echo "=== Deployment ===" && curl -s "${BASE_URL}/models/${MODEL_ID}/deployment" -H "Authorization: Bearer ${ACCESS_TOKEN}" | python3 -m json.tool
```

Save this output (`tee demo-run-$(date +%Y%m%d).log` on the whole block if you want a record) — concrete evidence beats "it worked, trust me" for a QA/mentor record.

---

## Troubleshooting

**GPU safety reminder (read before touching anything GPU-related): never run `nvidia-smi --gpu-reset`, `kill`, `kill -9`, `pkill`, or `docker kill` on this VM.** The H100 is shared with ~79GB already in use by processes unrelated to DEFNEX. The proven training cycle (Step 7→9) stops and restarts only `defnex-vllm`, automatically, via the existing file-signal GPU controller — never intervene manually mid-cycle.

- **Login fails (401):** re-check Step 1 actually returned 201/409, and that the password matches exactly (`Demo1234`).
- **Training run stuck in `PENDING`:** check the worker is actually up: `docker compose ps worker` should show `Up`, not `Exited`. `docker compose logs worker --tail 50` for the reason. Do not create a second training run to "try again" — investigate the first one.
- **Training `FAILED` with an `Unsloth`/`trl`/`transformers` API error (`unexpected keyword argument ...`):** this was the exact failure mode fixed in commit `573006a`/`b1901f6` (2026-09-15/16) — if it recurs, the training venv's package versions may have drifted since this baseline. Do not attempt to patch `run_training.py` live during a demo; fall back to narrating the proven `run-9e66b6` evidence instead.
- **`VRAM_FREE_THRESHOLD_MB` gate not passing / training stuck waiting on GPU:** the shared H100 may be more loaded than the 2026-09-16 baseline (79GB used). Check `nvidia-smi` (read-only) for current free VRAM; do not free it up by touching other processes.
- **`defnex-vllm` not healthy after training:** check `docker exec defnex-vllm curl -s localhost:8000/health` (or from the host, `curl -s http://172.17.0.1:8001/health`) — if it's still starting, wait; the proven cycle took a few minutes to fully recover. Do not restart it manually unless you understand the GPU controller's file-signal state machine (`Defnex-MLOps/docs/PHASE2A_*` docs) — an out-of-band restart can desync it from the worker's expectations.
- **Artifact not visible / permission denied reading it:** per `PHASE2B_POST_RUN_VALIDATION_REPORT.md` "Remaining Follow-Up" item 3, artifacts are created as root (UID 0) inside the worker container — reading them from the host may need `sudo`. This is a known, documented limitation, not a new bug.
- **Adapter not loadable / evaluation or deploy-staging fails against real vLLM:** this is exactly the unverified boundary (Sections 12–15). Have the `PHASE2B_POST_RUN_VALIDATION_REPORT.md` evidence ready as a fallback talking point if this fails live.
- **Inference returns non-200:** `404 DEPLOYMENT_NOT_FOUND` = nothing promoted to `prod` yet (most likely, given Section 13's status); `409 INFERENCE_NOT_ALLOWED` = you asked for a specific version that exists but isn't the currently-deployed one; `502 INFERENCE_FAILED` = real vLLM was reached but errored — check `docker logs defnex-vllm --tail 50` (read-only).
- **Rate limited (429):** registration is 3/minute; intake-validate/commit and training-run creation have their own configured limits (`app/config.py` `rate_limit_*` settings). Wait, don't hammer retries.

---

## Expected Outputs

| Step | Expected HTTP | Expected key field |
|---|---|---|
| 1. Register | 201 (or 409 if already exists) | `username` |
| 2. Login | 200 | `access_token` |
| 4–6. Intake (Option B only) | 200 each | `staging_id` → `validation_report_id` → `version` |
| 7. Create TrainingRun | 201 | `training_run_id`, `status: PENDING` |
| 8–9. Monitor/Verify | 200 | `status: COMPLETED`, `model_version` |
| 10–11. Registry | 200 | `artifacts[].uri`, `status: REGISTERED` |
| 12. Evaluation | 201 then worker-dependent | `status: EVALUATED` |
| 13. Deploy ladder | 201 each, or 503/409 | `status: DEPLOYED`, `current_deployed_version` |
| 14–15. Inference | 200, or 404/502 | `version` matching the new model |

---

## What This Demo Proves

- A real user can register, log in, and receive a working bearer token (real auth, not a stub).
- The dataset intake pipeline (inspect → validate → commit) is real, admin-gated, and produces immutable `DatasetVersion` rows with checksums.
- **A real SFT training run, on real GPU hardware (H100), with real Unsloth/SFTTrainer execution, can complete successfully and produce a real, checksummed LoRA adapter file** — this is proven, not simulated (`run-9e66b6`, 2026-09-16).
- The GPU handoff mechanism (stop serving → train → restart serving) works safely and does not disturb other tenants' GPU workloads sharing the same H100.
- The model registry correctly records full lineage from dataset version through training run to artifact.

## What This Demo Does NOT Prove

- That evaluation, staging deployment, production promotion, or inference against a *newly trained* adapter work end-to-end — these are code-verified from source but **not yet exercised to completion** on this VM as of this baseline.
- That the intake-uploaded chatml dataset (Option B) can itself be trained successfully — only the pre-existing `ds-smoke-test-v1` + raw-text combination has real evidence of success.
- That any of this works against a second, concurrent training run, a different base model, or a larger dataset than the 10-example smoke fixture.
- Sampling control over inference output (`max_tokens`, `temperature`) — the current API does not expose these parameters at all.
- Distributed/multi-server behavior of any kind — this entire document runs on one machine, one GPU, one `defnex-vllm` instance.
  