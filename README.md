# DEFNEX MLOps — Closed-Loop Platform

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
│  23 routes · JWT auth (admin/user) · /api/v1/        │
│  ┌────────────────────┐  ┌────────────────────────┐  │
│  │  SQLite + Alembic  │  │  structlog · slowapi   │  │
│  └────────────────────┘  └────────────────────────┘  │
└────────────────────┬─────────────────────────────────┘
                     │  HTTP (async, retry + backoff)
┌────────────────────▼─────────────────────────────────┐
│              Unsloth Studio  (:8888)                 │
│           GPU training · evaluation · model mgmt     │
└──────────────────────────────────────────────────────┘

Docker Compose also includes a lightweight worker service (stdlib-only,
polls a shared inbox directory for job dispatch, no external broker).
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
- **Unsloth Studio** — http://localhost:8888 (requires NVIDIA GPU)
- **Worker** — background job processor (inbox directory)

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
| Models | `GET /api/v1/models`, `GET .../available`, version detail, evaluation, decisions, rollback, deploy |

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

Key groups: `DATABASE_URL`, `UNSLOTH_*`, `JWT_*`, `DB_POOL_*`, `MAX_REQUEST_BODY_SIZE`, `LOG_LEVEL`.

## Testing & Quality

From `ml-close-loop-be/`:

```bash
.venv/bin/pytest tests/ -q       # 200 tests, 94%+ coverage
ruff check .                     # lint
ruff format --check .            # format check
```

## Migrations

```bash
cd ml-close-loop-be
alembic revision --autogenerate -m "description"
alembic upgrade head
```
