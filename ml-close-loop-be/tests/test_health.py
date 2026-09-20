from unittest.mock import MagicMock

from fastapi.testclient import TestClient

from app.main import app

client = TestClient(app)


def test_health():
    response = client.get("/api/v1/health")
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "ok"
    assert "checks" in body
    assert body["checks"]["db"] == "ok"


def test_health_reports_degraded_status_and_503_when_db_check_fails():
    """Issue #159: GET /health used to hardcode status="ok" and HTTP 200 even
    when a dependency check (e.g. db) had already failed internally."""
    from app.db.session import get_db

    def broken_db():
        session = MagicMock()
        session.execute.side_effect = RuntimeError("db unreachable")
        yield session

    app.dependency_overrides[get_db] = broken_db
    try:
        response = client.get("/api/v1/health")
    finally:
        app.dependency_overrides.pop(get_db, None)

    assert response.status_code == 503
    body = response.json()
    assert body["status"] == "degraded"
    assert body["checks"]["db"] == "error"
