# DEFNEX MLOps Backend

![CI](https://github.com/alfazh123/Defnex-MLOps/actions/workflows/ci.yml/badge.svg)

FastAPI backend orchestrating the closed-loop MLOps pipeline:

**Dataset → Validation → Training → Evaluation → Model Registry → Promotion → Deployment**

Integrates with [Unsloth Studio](https://github.com/unslothai/unsloth) as the training/evaluation engine.
Static API spec: [`openapi.yaml`](openapi.yaml) · Live docs: `http://localhost:8000/docs`

## Features

- JWT authentication with admin/user RBAC (first user auto-becomes admin)
- 28 REST endpoints under `/api/v1/` (see [openapi.yaml](openapi.yaml))
- Pagination (`?page=&size=`) on list endpoints
- Query filtering (`?status=&search=&model=`)
- Versioned golden/eval sets (`POST /eval-sets/{id}/versions`, admin-only) kept
  disjoint from validated training data both ways (H8 leakage, `409 EVAL_SET_OVERLAP`)
- Promotion gate: a `PROMOTED` decision requires a recorded eval-set reference,
  a qualitative majority win, no general-domain regressions, and no eval-loss
  regression (human decision on the threshold; each check is an env toggle, all
  on by default, `409 PROMOTION_GATE_BLOCKED` when blocked)
- Rate limiting (5/min login, 3/min register) with `X-RateLimit-*` headers
- Request body size limit (1MB default, configurable)
- Structured logging (structlog, JSON)
- Graceful shutdown (SIGTERM → drain in-flight requests → close DB)
- Retry with exponential backoff on Unsloth API failures
- vLLM serving backend (issue #40): runtime LoRA adapter load/unload over the vLLM API,
  GPU profile in docker-compose; a mock backend keeps tests and no-GPU dev green
- Inference endpoint (issue #41): `POST /api/v1/models/{model_id}/inference` accepts a
  deployment alias (`prod`, resolved via the single `deployment_service.resolve_alias`
  function) or an explicit version number and returns the generation **plus the concrete
  version that served it**
- Deploy-time smoke test (issue #41): after a new adapter is loaded but *before* the
  production alias moves to it, the adapter must produce a generation at or above a
  configurable threshold; a failing smoke test aborts the deploy and the alias stays on the
  old version
- DB index optimization on foreign keys + connection pool tuning
- N+1 query prevention via eager loading
- pytest-cov coverage gate `--cov-fail-under=80` (currently 423 tests, 97% coverage)

## Quickstart — Docker

```bash
cp .env.example .env
docker compose up --build
curl http://localhost:8000/api/v1/health
```

Services:
- `backend` — FastAPI on `:8000`; `docker-entrypoint.sh` runs Alembic migrations then uvicorn (auto-reload)
- `worker` — `docker-worker-entrypoint.sh` runs `python -m app.workers.training_worker` (the single training-run executor)
- `unsloth-studio` — **optional**, Unsloth GPU image on `:8888` (NVIDIA GPU); commented out in `docker-compose.yml` — uncomment to enable

## GPU stack — vLLM serving (issue #40)

The `serving` service is a real [vLLM](https://docs.vllm.ai) server (NVIDIA GPU) that serves
fine-tuned LoRA adapters without restarting: adversarially-pushed versions are loaded with the
vLLM runtime API (`/v1/load_lora_adapter`), superseded ones unloaded (`/v1/unload_lora_adapter`).

```bash
# In .env: SERVING_BACKEND=vllm (the serving container is only started with the gpu profile)
docker compose --profile gpu up --build
```

Try it without a GPU first: keep `SERVING_BACKEND=mock` (the default) — the backend then uses
`MockServingBackend` and the full deploy/rollback lifecycle runs with no serving container at
all.

Serving details:

- Launch flags required for runtime LoRA work are already set in `docker-compose.yml`:
  `--enable-lora --max-lora-rank N` plus the env var `VLLM_ALLOW_RUNTIME_LORA_UPDATING=true`.
  Without `--enable-lora` (or with that env var unset) `load_lora_adapter` fails at runtime.
- Adapter artifacts must be reachable from the serving container: the worker writes them under
  `./data`, which `serving` mounts at `/data`, so a `lora_path` like `/data/adapters/...`
  (a `file:///data/...` artifact URI) resolves inside vLLM.
- Inside the compose network the backend reaches vLLM at `http://serving:8000`
  (`VLLM_URL` override); on bare-metal runs point `VLLM_URL` at `http://localhost:8001`
- If `SERVING_BACKEND=vllm` but `serving` is not running, `POST .../deploy` fails with
  `502 DEPLOY_FAILED` and the DB stays untouched (the load runs before any mutation).

## Inference & smoke test (issue #41)

```bash
# Generate from the deployed (prod) adapter - the response carries the concrete version:
curl -X POST http://localhost:8000/api/v1/models/qwen-sft-domain-x/inference \
  -H "Authorization: Bearer <token>" \
  -H "Content-Type: application/json" \
  -d '{"target": "prod", "prompt": "What is the capital of France?"}'
# → {"model_id": "qwen-sft-domain-x", "version": 2, "generation": "Paris."}

# Same, but address the deployed version by number instead of alias:
curl -X POST .../qwen-sft-domain-x/inference -d '{"target": "2", "prompt": "..."}'
```

- Alias resolution (`target: "prod"`) reuses the **single** resolution function from issue #36
  (`deployment_service.resolve_alias`); the endpoint contains no second alias-resolution logic.
  An explicit version number must be the currently `DEPLOYED` version (the only adapter
  guaranteed loaded and smoke-tested). Unknown alias / never-deployed model → `404`; explicit
  version that doesn't exist → `404 MODEL_NOT_FOUND`; explicit version that exists but isn't
  `DEPLOYED` → `409 INFERENCE_NOT_ALLOWED`; upstream generation failure →
  `502 INFERENCE_FAILED`.
- **Deploy-time smoke test** — after `deployment_service.deploy` loads a new adapter but
  *before* the `prod` alias/pointer moves (the `DEPLOYED` status is the single source of
  truth), the adapter must generate a completion at or above `INFERENCE_SMOKE_MIN_CHARS`
  chars for the prompt `INFERENCE_SMOKE_PROMPT`. On failure the adapter is unloaded again,
  the deploy aborts with the old version still serving, and the failure is logged
  (`smoke_test_failed`) and returned as `502 SMOKE_TEST_FAILED`. Because the pointer only
  moves after the smoke test passes (same transaction), an inference request that arrives
  mid-deploy always resolves to the old, still-`DEPLOYED` adapter. Each smoke-test knob is a
  config value, not a magic number; `INFERENCE_SMOKE_ENABLED=false` disables the gate.

## Quickstart — Local

Requires Python 3.11+.

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
cp .env.example .env
alembic upgrade head
uvicorn app.main:app --reload
# http://localhost:8000/docs
```

Or with [uv](https://docs.astral.sh/uv/):

```bash
uv sync
cp .env.example .env
uv run alembic upgrade head
uv run uvicorn app.main:app --reload
```

## Workers

Training execution is single-sourced (US-032) to one module:

- `app/workers/training_worker.py` — training queue poller backed by a swappable
  `TrainingRunner`. Runs under `python -m app.workers.training_worker`, which is
  what the compose `worker` service (`docker-worker-entrypoint.sh`) executes. It
  processes `PENDING` training runs to completion and is the only caller of
  `model_service.register_model_version`.

For clarity, `app/worker.py` is an unrelated stdlib **inbox/outbox file
poller** (job-dispatch stub, no broker). No service in `docker-compose.yml` runs
it.

## Tests & Quality

```bash
.venv/bin/pytest tests/ -q          # 423 tests, 97% coverage (threshold --cov-fail-under=80)
ruff check .                        # lint
ruff format --check .               # format check
```

With coverage report:

```bash
.venv/bin/pytest tests/ --cov=app --cov-report=term-missing
```

## Migrations

```bash
alembic revision --autogenerate -m "description"
alembic upgrade head
```

## Training concurrency & GPU lock

Training runs are executed exclusively by `app.workers.training_worker`
(`python -m app.workers.training_worker`). To guarantee that only one training
runs at a time across worker processes:

1. **GPU lock** (`app.workers.gpu_lock`) — the worker holds an exclusive
   `flock()` on `data/gpu.lock` (set `GPU_LOCK_FILE`) for the whole
   claim → train → persist block. A `flock` is released by the kernel when the
   owning process exits by *any* path — success, exception, SIGTERM, even
   SIGKILL — so a crashed worker never leaves a permanently stuck lock.
2. **Atomic claim** (`training_service.claim_training_run`) — a compare-and-set
   flips `PENDING → RUNNING` at the SQL level (`UPDATE ... WHERE status='PENDING'`),
   so two workers that both picked the same row can never both win, on SQLite or
   Postgres. Losing the claim returns `None` and the worker skips the poll.
3. **Lock timeout** (set `GPU_LOCK_TIMEOUT`) — if a worker cannot acquire the
   lock within the timeout it skips the poll entirely; the run stays `PENDING`
   and is retried on the next iteration. A busy GPU never fails or loses a run.

Scope caveat (documented in `app/workers/gpu_lock.py`): `flock` only
serializes processes that open the **same lock file on the same host**. In the
Docker deployment every worker mounts `./data:/app/data` so the lock file is
shared. If workers ever run on different hosts (or train on hardware shared
across hosts), coordination must move outside this module.

## Error Codes

| Status | Code | Meaning |
|--------|------|---------|
| `401` | `INVALID_CREDENTIALS` | Wrong username/password |
| `401` | `MISSING_TOKEN` | No `Authorization: Bearer` header |
| `401` | `INVALID_REFRESH_TOKEN` | Expired or invalid refresh token |
| `404` | `MODEL_NOT_FOUND` | Model ID does not exist (or the requested version of it is missing) |
| `404` | `EVAL_SET_NOT_FOUND` | `eval_set_id` (or version) does not exist |
| `404` | `DEPLOYMENT_NOT_FOUND` | Known alias resolves to no deployed version |
| `404` | `UNKNOWN_DEPLOYMENT_ALIAS` | Inference `target` is not a known alias |
| `409` | `USERNAME_TAKEN` | Register with existing username |
| `409` | `EVAL_SET_OVERLAP` | Eval-set record duplicates already-validated training content (H8) |
| `409` | `EVAL_SET_EMPTY` | Validate/train against an eval set that has no versions |
| `409` | `PROMOTION_GATE_BLOCKED` | `PROMOTED` decision failed the eval gate (missing eval-set ref, no majority win, regressions found, or eval-loss worse) |
| `409` | `INFERENCE_NOT_ALLOWED` | Inference `target` version exists but is not the `DEPLOYED` one |
| `409` | `VALIDATION_REQUIRED` | No validation report yet for this dataset version |
| `409` | `VALIDATION_FAILED` | Latest validation gate decision is FAIL or has no valid records |
| `422` | `VALIDATION_RECORDS_REQUIRED` | Validate body missing or `records` empty |
| `422` | validation error | Password too weak, missing fields, unsupported `peft_method`, etc. |
| `429` | `RATE_LIMIT_EXCEEDED` | Too many login/register requests (see headers) |
| `413` | `REQUEST_TOO_LARGE` | Body exceeds `MAX_REQUEST_BODY_SIZE` |
| `502` | `DEPLOY_FAILED` | Serving backend could not load the adapter (vLLM unreachable/rejected the load) |
| `502` | `SMOKE_TEST_FAILED` | Deploy-time smoke test failed; adapter unloaded and alias stayed on the old version |
| `502` | `INFERENCE_FAILED` | Serving backend could not generate during inference |

`training_config.peft_method` menerima `lora`, `qlora`, dan `rslora`. Nilai `dora`,
`qdora`, dan `none` (Full Finetuning) ditolak **permanen** dengan `422` beserta pesan
yang menyebut nilai dan alasannya — ini keputusan governance final (issue #44,
menuntaskan penolakan sementara yang dibuka issue #34), bukan batasan sementara
yang bergantung pada infrastruktur serving saat ini: jalur serving proyek ini
memang hanya untuk vLLM, yang tidak bisa melayani varian weight-decomposed
maupun full finetune. Nilai-nilai itu tidak lagi diterima lalu dipetakan diam-diam
ke LoRA biasa. `qlora` mengaktifkan kuantisasi 4-bit (`load_in_4bit: true`) dan
`rslora` mengaktifkan `use_rslora` pada payload pelatihan, sehingga tidak identik
dengan `lora`.

## Environment Variables

All variables are in [`.env.example`](.env.example) with defaults.

| Variable | Default | Description |
|---|---|---|
| `DATABASE_URL` | `sqlite:///./data/app.db` | SQLAlchemy connection string |
| `DEPLOYMENT_ENVIRONMENT` | `default` | Deployment environment label (deployment records) |
| `UNSLOTH_STUDIO_URL` | `http://localhost:8888` | Unsloth Studio base URL |
| `UNSLOTH_API_KEY` | *(empty)* | Unsloth auth key |
| `UNSLOTH_DEFAULT_MODEL` | `unsloth/Qwen3-0.6B` | Default training model |
| `UNSLOTH_MODELS` | *(comma-separated list)* | Available models offered by the API |
| `SERVING_BACKEND` | `mock` | `mock` (no GPU) or `vllm` (real vLLM serving) |
| `VLLM_URL` | `http://localhost:8001` | vLLM server base URL (host port of the `serving` service) |
| `VLLM_API_KEY` | *(empty)* | Optional bearer token for vLLM |
| `VLLM_TIMEOUT_SECONDS` | `60` | Timeout for vLLM load/unload calls |
| `VLLM_MODEL_NAME` | `unsloth/Qwen3-0.6B` | Base model vLLM serves (`serving` service flag) |
| `VLLM_SERVED_MODEL_NAME` | `defnex-model` | `--served-model-name` for inference requests |
| `VLLM_MAX_LORAS` | `4` | Concurrent adapter slots (`--max-loras`); must be ≥2 since deploy loads the new adapter before unloading the superseded one |
| `VLLM_MAX_LORA_RANK` | `64` | `--max-lora-rank` for runtime LoRA |
| `INFERENCE_SMOKE_ENABLED` | `true` | Run the deploy-time smoke test before the alias moves |
| `INFERENCE_SMOKE_PROMPT` | `Return OK.` | Prompt sent to a just-loaded adapter to prove it generates |
| `INFERENCE_SMOKE_MIN_CHARS` | `1` | Minimum generated-output length for the smoke test to pass |
| `INFERENCE_MAX_TOKENS` | `128` | Max tokens for a generation (smoke test + inference endpoint) |
| `JWT_SECRET` | `dev-secret-change-in-production` | JWT signing secret (**change in prod**) |
| `JWT_ALGORITHM` | `HS256` | JWT algorithm |
| `JWT_EXPIRE_MINUTES` | `1440` (24h) | Access token lifetime |
| `JWT_REFRESH_EXPIRE_MINUTES` | `10080` (7d) | Refresh token lifetime |
| `DB_POOL_SIZE` | `5` | Connection pool size |
| `DB_MAX_OVERFLOW` | `10` | Max overflow connections |
| `DB_POOL_TIMEOUT` | `30` | Seconds to wait for a connection |
| `DB_POOL_RECYCLE` | `1800` | Seconds before a connection is recycled |
| `CORS_ORIGINS` | `http://localhost:3000,...` | Comma-separated allowed browser origins |
| `MAX_REQUEST_BODY_SIZE` | `1048576` (1MB) | Max body in bytes |
| `GPU_LOCK_FILE` | `data/gpu.lock` | Lock file serializing training across workers (must be on a shared filesystem) |
| `GPU_LOCK_TIMEOUT` | `300` | Seconds a worker waits for the GPU lock before skipping the poll |
| `EVAL_GATE_REQUIRE_EVAL_SET_REFERENCE` | `true` | Promotion requires a recorded eval-set reference |
| `EVAL_GATE_REQUIRE_QUALITATIVE_MAJORITY` | `true` | Promotion requires a qualitative majority win on the eval set |
| `EVAL_GATE_REQUIRE_NO_GENERAL_REGRESSION` | `true` | Promotion blocked on general-domain regressions |
| `EVAL_GATE_REQUIRE_EVAL_LOSS_NOT_WORSE` | `true` | Promotion blocked on eval-loss regression |
| `LOG_LEVEL` | `INFO` | Structured log level |
| `DEBUG` | `false` | Debug mode (verbose logging) |
