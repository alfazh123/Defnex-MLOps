# DEFNEX MLOps — Closed-Loop Platform

[![CI](https://github.com/alfazh123/Defnex-MLOps/actions/workflows/ci.yml/badge.svg)](https://github.com/alfazh123/Defnex-MLOps/actions/workflows/ci.yml)

An orchestration backend and frontend for the DEFNEX closed-loop MLOps pipeline:
**Dataset → Validation → Training → Evaluation → Model Registry → Promotion → Deployment**,
integrating with [Unsloth Studio](https://github.com/unslothai/unsloth) as the training/evaluation engine.

## Architecture

```
┌──────────────────────────────────────────────────────┐
│                    React Frontend                    │
│             React Router 8 · shadcn/ui · Tailwind    │
│                      :3000                           │
└────────────────────┬─────────────────────────────────┘
                     │  REST API (JSON)
┌────────────────────▼─────────────────────────────────┐
│                  FastAPI Backend  (:8000)            │
│  50 routes · JWT auth (admin/user) · /api/v1/       │
│  ┌────────────────────┐  ┌────────────────────────┐  │
│  │  SQLite + Alembic  │  │  structlog · slowapi   │  │
│  └────────────────────┘  └────────────────────────┘  │
└────────────────────┬─────────────────────────────────┘
                     │  HTTP (async, retry + backoff)
┌────────────────────▼─────────────────────────────────┐
│              Unsloth Studio  (:8888)                 │
│           GPU training · evaluation · model mgmt     │
└──────────────────────────────────────────────────────┘

Docker Compose also runs a `worker` service (`./docker-worker-entrypoint.sh`)
that executes `python -m app.workers.training_worker` — the single
training-run executor (US-032); it processes `PENDING` runs and is the only
caller of `register_model_version`. `ml-close-loop-be/app/worker.py` is a
separate stdlib inbox/outbox file-polling stub that no service runs.
`unsloth-studio` is commented out in `docker-compose.yml` and only enabled on
GPU hosts.
```

## Repository Layout

| Directory | Description |
|---|---|
| `ml-close-loop-be/` | FastAPI backend — API, DB, auth, Unsloth integration, tests |
| `ml-close-loop-fe/` | React frontend — React Router 8, shadcn/ui, Tailwind |

## Tech Stack

**Backend:** FastAPI · SQLAlchemy 2.0 · Alembic · Pydantic Settings · JWT (python-jose) · bcrypt · slowapi · structlog · Uvicorn

**Frontend:** React 19 · React Router 8 · TypeScript · shadcn/ui · Tailwind CSS

**Infra:** Docker Compose · Unsloth Studio (GPU) · SQLite (dev) / PostgreSQL (prod-ready)

## Quickstart — Docker

```bash
cd ml-close-loop-be
cp .env.example .env
docker compose up --build
```

Services:
- **Backend** — http://localhost:8000
- **Swagger docs** — http://localhost:8000/docs
- **Unsloth Studio** — http://localhost:8888, **optional** (NVIDIA GPU). Commented out in `docker-compose.yml`; uncomment to enable
- **Worker** — runs the training worker (`python -m app.workers.training_worker` via `docker-worker-entrypoint.sh`), the single training-run executor

## Quickstart — Local Backend (no Docker)

Requires Python 3.11+.

```bash
cd ml-close-loop-be
python3 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
cp .env.example .env
alembic upgrade head
uvicorn app.main:app --reload
# API available at http://localhost:8000/docs
```

## Frontend

```bash
cd ml-close-loop-fe
pnpm install
pnpm run dev        # http://localhost:3000
pnpm run typecheck  # TypeScript check
pnpm run build      # production build
```

## API Documentation

All endpoints live under the **`/api/v1/`** prefix. Legacy paths (`/auth/login`, etc.) return a `301` redirect automatically.

| Resource | Endpoints |
|---|---|
| Health | `GET /api/v1/health` |
| Auth | `POST /api/v1/auth/login`, `POST /api/v1/auth/register`, `POST /api/v1/auth/refresh` |
| Users | `GET /api/v1/users` (paginated, admin), `DELETE /api/v1/users/{id}` (admin) |
| Datasets | `GET /api/v1/datasets` (paginated, filtered), `POST .../versions`, `GET`, `POST .../validate`, validation reports |
| Training | `POST /api/v1/training-runs`, `GET` (paginated, filtered), `GET .../{id}`, `GET .../progress` (SSE) |
| Models | `GET /api/v1/models`, `GET .../available`, version detail, evaluation, decisions, rollback, deploy, alias resolution (`GET .../deployment/{alias}`) |

Interactive docs: http://localhost:8000/docs (Swagger) · `/redoc` (ReDoc)

Static spec: [`ml-close-loop-be/openapi.yaml`](ml-close-loop-be/openapi.yaml)

## Auth & Security

| Feature | Detail |
|---|---|
| Roles | `admin` (full CRUD) · `user` (train + eval only) |
| First user | Automatically becomes `admin` |
| Access token | JWT, 24-hour expiry (`HS256`) |
| Refresh token | 7-day expiry via `POST /auth/refresh` |
| Rate limiting | `/auth/login` 5/min · `/auth/register` 3/min → `429 RATE_LIMIT_EXCEEDED` |
| Body limit | 1MB default (configurable via `MAX_REQUEST_BODY_SIZE`) → `413 REQUEST_TOO_LARGE` |

`Authorization: Bearer <token>` header required for all endpoints except health and login/register.

## Environment Variables

See [`ml-close-loop-be/.env.example`](ml-close-loop-be/.env.example) for all variables with defaults.

Key groups: `DATABASE_URL`, `DEPLOYMENT_ENVIRONMENT`, `UNSLOTH_*`, `JWT_*`, `DB_POOL_*`, `CORS_ORIGINS`, `GPU_LOCK_*`, `EVAL_GATE_*`, `MAX_REQUEST_BODY_SIZE`, `LOG_LEVEL`, `DEBUG`.

## Testing & Quality

From `ml-close-loop-be/`:

```bash
.venv/bin/pytest tests/ -q       # 392 tests, ~97% coverage (CI enforces --cov-fail-under=80)
ruff check .                     # lint
ruff format --check .            # format check
```

## Migrations

```bash
cd ml-close-loop-be
alembic revision --autogenerate -m "description"
alembic upgrade head
```
