"""Tests for API versioning (US-007): /api/v1/ prefix and 301 redirects."""

from fastapi.testclient import TestClient

from app.main import app

client_no_redirect = TestClient(app, follow_redirects=False)


def test_health_under_v1():
    resp = client_no_redirect.get("/api/v1/health")
    assert resp.status_code == 200
    assert resp.json() == {"status": "ok"}


def test_old_health_redirects_to_v1():
    resp = client_no_redirect.get("/health")
    assert resp.status_code == 301
    assert resp.headers["location"] == "/api/v1/health"


def test_old_auth_register_redirects_to_v1():
    resp = client_no_redirect.post(
        "/auth/register",
        json={"username": "test", "password": "Test1234"},
    )
    assert resp.status_code == 301
    assert resp.headers["location"] == "/api/v1/auth/register"


def test_old_auth_login_redirects_to_v1():
    resp = client_no_redirect.post(
        "/auth/login",
        json={"username": "test", "password": "Test1234"},
    )
    assert resp.status_code == 301
    assert resp.headers["location"] == "/api/v1/auth/login"


def test_old_datasets_redirects_to_v1():
    resp = client_no_redirect.get("/datasets")
    assert resp.status_code == 301
    assert resp.headers["location"] == "/api/v1/datasets"


def test_old_training_runs_redirects_to_v1():
    resp = client_no_redirect.get("/training-runs")
    assert resp.status_code == 301
    assert resp.headers["location"] == "/api/v1/training-runs"


def test_old_models_redirects_to_v1():
    resp = client_no_redirect.get("/models")
    assert resp.status_code == 301
    assert resp.headers["location"] == "/api/v1/models"


def test_v1_path_not_redirected():
    resp = client_no_redirect.get("/api/v1/health")
    assert resp.status_code == 200
    assert "location" not in resp.headers


def test_unknown_path_not_redirected():
    resp = client_no_redirect.get("/unknown")
    assert resp.status_code == 404
    assert "location" not in resp.headers
