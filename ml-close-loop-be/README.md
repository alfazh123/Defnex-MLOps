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
| `404` | `MODEL_NOT_FOUND` | Model ID does not exist |
| `409` | `USERNAME_TAKEN` | Register with existing username |
| `409` | `VALIDATION_REQUIRED` | No validation report yet for this dataset version |
| `409` | `VALIDATION_FAILED` | Latest validation gate decision is FAIL or has no valid records |
| `422` | `VALIDATION_RECORDS_REQUIRED` | Validate body missing or `records` empty |
| `422` | validation error | Password too weak, missing fields, unsupported `peft_method`, etc. |
| `429` | `RATE_LIMIT_EXCEEDED` | Too many login/register requests (see headers) |
| `413` | `REQUEST_TOO_LARGE` | Body exceeds `MAX_REQUEST_BODY_SIZE` |

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
| `JWT_SECRET` | `dev-secret-change-in-production` | JWT signing secret (**change in prod**) |
| `JWT_ALGORITHM` | `HS256` | JWT algorithm |
| `JWT_EXPIRE_MINUTES` | `1440` (24h) | Access token lifetime |
| `DB_POOL_SIZE` | `5` | Connection pool size |
| `DB_MAX_OVERFLOW` | `10` | Max overflow connections |
| `MAX_REQUEST_BODY_SIZE` | `1048576` (1MB) | Max body in bytes |
| `GPU_LOCK_FILE` | `data/gpu.lock` | Lock file serializing training across workers (must be on a shared filesystem) |
| `GPU_LOCK_TIMEOUT` | `300` | Seconds a worker waits for the GPU lock before skipping the poll |
| `LOG_LEVEL` | `INFO` | Structured log level |
| `DEBUG` | `false` | Debug mode (verbose logging) |
