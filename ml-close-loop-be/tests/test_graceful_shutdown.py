import re
from unittest.mock import patch

from fastapi.testclient import TestClient


def _strip_ansi(text: str) -> str:
    return re.sub(r"\x1b\[[0-9;]*m", "", text)


def test_engine_dispose_called_on_shutdown():
    """Verify engine.dispose() is called during lifespan shutdown."""
    with patch("app.main.engine") as mock_engine:
        from app.main import app

        with TestClient(app):
            pass
        mock_engine.dispose.assert_called_once()


def test_lifespan_startup_shutdown_logs(capfd):
    """Verify lifespan emits startup and shutdown log messages."""
    from app.main import app

    with TestClient(app) as client:
        resp = client.get("/api/v1/health")
        assert resp.status_code == 200

    captured = capfd.readouterr()
    assert "application_starting" in captured.out
    assert "application_shutting_down" in captured.out


def test_app_handles_requests_during_lifespan():
    """Verify the app processes requests normally while lifespan is active."""
    from app.main import app

    with TestClient(app) as client:
        resp = client.get("/api/v1/health")
        assert resp.status_code == 200
        resp = client.get("/api/v1/health")
        assert resp.status_code == 200
