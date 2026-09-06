# DEFNEX MLOps Backend

![CI](https://github.com/alfazh123/Defnex-MLOps/actions/workflows/ci.yml/badge.svg)

FastAPI backend orchestrating the closed-loop MLOps pipeline:

**Dataset → Validation → Training → Evaluation → Model Registry → Promotion → Deployment**

Integrates with [Unsloth Studio](https://github.com/unslothai/unsloth) as the training/evaluation engine.
Static API spec: [`openapi.yaml`](openapi.yaml) · Live docs: `http://localhost:8000/docs`

## Features

- JWT authentication with admin/user RBAC (first user auto-becomes admin)
- 23 REST endpoints under `/api/v1/` (see [openapi.yaml](openapi.yaml))
- Pagination (`?page=&size=`) on list endpoints
- Query filtering (`?status=&search=&model=`)
- Rate limiting (5/min login, 3/min register) with `X-RateLimit-*` headers
- Request body size limit (1MB default, configurable)
- Structured logging (structlog, JSON)
- Graceful shutdown (SIGTERM → drain in-flight requests → close DB)
- Retry with exponential backoff on Unsloth API failures
- vLLM serving backend (issue #40): runtime LoRA adapter load/unload over the vLLM API,
  GPU profile in docker-compose; a mock backend keeps tests and no-GPU dev green
- DB index optimization on foreign keys + connection pool tuning
- N+1 query prevention via eager loading
- pytest-cov (94%+ coverage, 200 tests)

## Quickstart — Docker

```bash
cp .env.example .env
docker compose up --build
curl http://localhost:8000/api/v1/health
```

Services:
- `backend` — FastAPI on `:8000` (auto-reloads via uvicorn)
- `worker` — stdlib background processor (inbox directory, no broker required)
- `unsloth-studio` — Unsloth GPU image on `:8888`

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

## Tests & Quality

```bash
.venv/bin/pytest tests/ -q          # 200 tests, 94%+ coverage
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

## Error Codes

| Status | Code | Meaning |
|--------|------|---------|
| `401` | `INVALID_CREDENTIALS` | Wrong username/password |
| `401` | `MISSING_TOKEN` | No `Authorization: Bearer` header |
| `401` | `INVALID_REFRESH_TOKEN` | Expired or invalid refresh token |
| `404` | `MODEL_NOT_FOUND` | Model ID does not exist |
| `409` | `USERNAME_TAKEN` | Register with existing username |
| `409` | `VALIDATION_REQUIRED` | No validation report yet for this dataset version |
| `409` | `VALIDATION_FAILED` | Latest validation gate decision is FAIL or has no valid records |
| `422` | `VALIDATION_RECORDS_REQUIRED` | Validate body missing or `records` empty |
| `422` | validation error | Password too weak, missing fields, unsupported `peft_method`, etc. |
| `429` | `RATE_LIMIT_EXCEEDED` | Too many login/register requests (see headers) |
| `413` | `REQUEST_TOO_LARGE` | Body exceeds `MAX_REQUEST_BODY_SIZE` |
| `502` | `DEPLOY_FAILED` | Serving backend could not load the adapter (vLLM unreachable/rejected the load) |

`training_config.peft_method` menerima `lora`, `qlora`, dan `rslora`. Nilai `dora`,
`qdora`, dan `none` (Full Finetuning) ditolak dengan `422` beserta pesan yang
menyebut nilai dan alasannya, karena jalur serving vLLM saat ini tidak dapat
melayani varian weight-decomposed dan full finetune — nilai tersebut tidak lagi
diterima lalu dipetakan diam-diam ke LoRA biasa. `qlora` mengaktifkan kuantisasi
4-bit (`load_in_4bit: true`) dan `rslora` mengaktifkan `use_rslora` pada payload
pelatihan, sehingga tidak identik dengan `lora`.

## Environment Variables

All variables are in [`.env.example`](.env.example) with defaults.

| Variable | Default | Description |
|---|---|---|
| `DATABASE_URL` | `sqlite:///./data/app.db` | SQLAlchemy connection string |
| `UNSLOTH_STUDIO_URL` | `http://localhost:8888` | Unsloth Studio base URL |
| `UNSLOTH_API_KEY` | *(empty)* | Unsloth auth key |
| `UNSLOTH_DEFAULT_MODEL` | `unsloth/Qwen3-0.6B` | Default training model |
| `SERVING_BACKEND` | `mock` | `mock` (no GPU) or `vllm` (real vLLM serving) |
| `VLLM_URL` | `http://localhost:8001` | vLLM server base URL (host port of the `serving` service) |
| `VLLM_API_KEY` | *(empty)* | Optional bearer token for vLLM |
| `VLLM_TIMEOUT_SECONDS` | `60` | Timeout for vLLM load/unload calls |
| `VLLM_MODEL_NAME` | `unsloth/Qwen3-0.6B` | Base model vLLM serves (`serving` service flag) |
| `VLLM_SERVED_MODEL_NAME` | `defnex-model` | `--served-model-name` for inference requests |
| `VLLM_MAX_LORA_RANK` | `64` | `--max-lora-rank` for runtime LoRA |
| `JWT_SECRET` | `dev-secret-change-in-production` | JWT signing secret (**change in prod**) |
| `JWT_ALGORITHM` | `HS256` | JWT algorithm |
| `JWT_EXPIRE_MINUTES` | `1440` (24h) | Access token lifetime |
| `DB_POOL_SIZE` | `5` | Connection pool size |
| `DB_MAX_OVERFLOW` | `10` | Max overflow connections |
| `MAX_REQUEST_BODY_SIZE` | `1048576` (1MB) | Max body in bytes |
| `LOG_LEVEL` | `INFO` | Structured log level |
| `DEBUG` | `false` | Debug mode (verbose logging) |
