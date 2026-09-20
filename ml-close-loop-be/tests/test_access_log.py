from fastapi.testclient import TestClient

from app.main import app

client = TestClient(app)


def test_successful_request_is_logged_with_duration(monkeypatch):
    """Issue #160: successful requests used to leave no structured log trace at all -
    only exception handlers logged anything. This asserts the access-log middleware
    actually fires for a plain 200 response."""
    events = []

    class _CapturingLogger:
        def info(self, event, **kwargs):
            events.append((event, kwargs))

    monkeypatch.setattr("app.middleware.access_log.logger", _CapturingLogger())

    response = client.get("/api/v1/health")
    assert response.status_code == 200

    assert len(events) == 1
    event_name, fields = events[0]
    assert event_name == "http_request"
    assert fields["method"] == "GET"
    assert fields["path"] == "/api/v1/health"
    assert fields["status_code"] == 200
    assert isinstance(fields["duration_ms"], float)
