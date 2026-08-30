# DEFNEX MLOps Backend

FastAPI backend orchestrating the closed-loop MLOps prototype (Dataset → Validation →
Training → Evaluation → Model Registry → Promotion → Deployment). See
`../docs/prd/mlops-backend-mvp.md` for the full PRD and `../docs/api/openapi.yaml` for
the API contract.

## Status

Phase 0 (Foundation) — project skeleton, FastAPI app, SQLite via SQLAlchemy, Alembic
migrations, Docker Compose, health endpoint, testing foundation. Domain endpoints
(dataset/validation/training/model/evaluation/promotion/deployment) are implemented in
later phases per the PRD's Phase 0–8 order.

## Local development (Docker)

```bash
cp .env.example .env
docker compose up --build
curl http://localhost:8000/health
```

## Local development (without Docker)

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
cp .env.example .env
alembic upgrade head
uvicorn app.main:app --reload
```

## Tests

```bash
pytest
```

## Migrations

```bash
alembic revision --autogenerate -m "message"
alembic upgrade head
```
