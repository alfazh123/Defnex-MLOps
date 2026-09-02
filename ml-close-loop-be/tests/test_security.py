"""Phase 3 — security tests (OWASP Top 10 patterns)."""

import base64
import json
from datetime import datetime, timedelta, timezone

from fastapi.testclient import TestClient
from jose import jwt

from app.config import settings
from app.limiter import limiter
from app.main import app
from tests.conftest import auth_header


# ----------------------------------------------------------------
# Authentication security
# ----------------------------------------------------------------


def _register_and_login(
    client, username: str = "admin", password: str = "Admin1234"
) -> str:
    client.post(
        "/api/v1/auth/register",
        json={"username": username, "password": password, "role": "admin"},
    )
    resp = client.post(
        "/api/v1/auth/login", json={"username": username, "password": password}
    )
    return resp.json()["access_token"]


def test_tampered_token_returns_401(client):
    valid = _register_and_login(client)
    tampered = valid[:-5] + "XXXXX"
    resp = client.get("/api/v1/datasets", headers=auth_header(tampered))
    assert resp.status_code == 401
    assert resp.json()["error"]["code"] == "INVALID_TOKEN"


def test_token_with_wrong_secret_returns_401(client):
    forged = jwt.encode(
        {
            "sub": "1",
            "exp": datetime.now(timezone.utc) + timedelta(hours=1),
            "type": "access",
        },
        "wrong-secret",
        algorithm="HS256",
    )
    resp = client.get("/api/v1/datasets", headers=auth_header(forged))
    assert resp.status_code == 401


def test_token_alg_none_attack(client):
    header = (
        base64.urlsafe_b64encode(json.dumps({"alg": "none", "typ": "JWT"}).encode())
        .rstrip(b"=")
        .decode()
    )
    payload = (
        base64.urlsafe_b64encode(json.dumps({"sub": "1", "exp": 9999999999}).encode())
        .rstrip(b"=")
        .decode()
    )
    none_token = f"{header}.{payload}."
    resp = client.get("/api/v1/datasets", headers=auth_header(none_token))
    assert resp.status_code == 401


def test_missing_bearer_prefix_returns_401(client):
    valid = _register_and_login(client)
    resp = client.get("/api/v1/datasets", headers={"Authorization": valid})
    assert resp.status_code == 401
    assert resp.json()["error"]["code"] == "MISSING_TOKEN"


def test_empty_token_returns_401(client):
    resp = client.get("/api/v1/datasets", headers={"Authorization": "Bearer "})
    assert resp.status_code == 401


def test_token_with_invalid_sub_returns_401(client):
    forged = jwt.encode(
        {
            "sub": "not_a_number",
            "exp": datetime.now(timezone.utc) + timedelta(hours=1),
            "type": "access",
        },
        settings.jwt_secret,
        algorithm=settings.jwt_algorithm,
    )
    resp = client.get("/api/v1/datasets", headers=auth_header(forged))
    assert resp.status_code == 401


def test_token_with_expired_exp_returns_401(client):
    expired = jwt.encode(
        {
            "sub": "1",
            "exp": datetime.now(timezone.utc) - timedelta(hours=1),
            "type": "access",
        },
        settings.jwt_secret,
        algorithm=settings.jwt_algorithm,
    )
    resp = client.get("/api/v1/datasets", headers=auth_header(expired))
    assert resp.status_code == 401


def test_access_token_rejected_as_refresh_token(client):
    access = _register_and_login(client)
    resp = client.post("/api/v1/auth/refresh", json={"refresh_token": access})
    assert resp.status_code == 401
    assert resp.json()["error"]["code"] == "INVALID_REFRESH_TOKEN"


# ----------------------------------------------------------------
# Authorization security
# ----------------------------------------------------------------


def _regular_user_token(client) -> str:
    client.post(
        "/api/v1/auth/register",
        json={"username": "boss", "password": "Boss1234", "role": "admin"},
    )
    client.post(
        "/api/v1/auth/register",
        json={"username": "alice", "password": "Alice1234", "role": "user"},
    )
    resp = client.post(
        "/api/v1/auth/login", json={"username": "alice", "password": "Alice1234"}
    )
    return resp.json()["access_token"]


def test_user_cannot_access_admin_endpoints(client):
    token = _regular_user_token(client)
    resp = client.get("/api/v1/users", headers=auth_header(token))
    assert resp.status_code == 403
    assert resp.json()["error"]["code"] == "FORBIDDEN"


def test_user_cannot_create_dataset(client):
    token = _regular_user_token(client)
    resp = client.post(
        "/api/v1/datasets/ds1/versions",
        json={"rule_set_version": None},
        headers=auth_header(token),
    )
    assert resp.status_code == 403


def test_user_cannot_validate_dataset(client):
    token = _regular_user_token(client)
    resp = client.post(
        "/api/v1/datasets/ds1/versions/1/validate",
        json={},
        headers=auth_header(token),
    )
    assert resp.status_code == 403


def test_user_cannot_deploy_model(client):
    token = _regular_user_token(client)
    resp = client.post(
        "/api/v1/models/m1/versions/1/deploy",
        headers=auth_header(token),
    )
    assert resp.status_code == 403


def test_user_cannot_create_decision(client):
    token = _regular_user_token(client)
    resp = client.post(
        "/api/v1/models/m1/versions/1/decisions",
        json={"decision": "promote"},
        headers=auth_header(token),
    )
    assert resp.status_code == 403


def test_user_cannot_delete_other_users(client):
    token = _regular_user_token(client)
    resp = client.delete("/api/v1/users/1", headers=auth_header(token))
    assert resp.status_code == 403


# ----------------------------------------------------------------
# Input validation security
# ----------------------------------------------------------------


def test_sql_injection_in_search_param(client, user_token):
    resp = client.get(
        "/api/v1/datasets",
        params={"search": "' OR 1=1 --"},
        headers=auth_header(user_token),
    )
    assert resp.status_code == 200
    assert "items" in resp.json()


def test_xss_in_dataset_name(client, user_token):
    resp = client.get(
        "/api/v1/datasets/<script>alert(1)</script>/versions",
        headers=auth_header(user_token),
    )
    assert resp.status_code == 404
    assert "<script>" not in resp.text


def test_oversized_payload_returns_413(client):
    resp = client.post(
        "/api/v1/auth/register",
        content="x" * (settings.max_request_body_size + 1),
        headers={"Content-Type": "text/plain"},
    )
    assert resp.status_code == 413
    assert resp.json()["error"]["code"] == "REQUEST_TOO_LARGE"


def test_malformed_json_returns_422(client):
    resp = client.post(
        "/api/v1/auth/register",
        content="{not valid json",
        headers={"Content-Type": "application/json"},
    )
    assert resp.status_code == 422


def test_unicode_in_password(client):
    resp = client.post(
        "/api/v1/auth/register",
        json={"username": "uni", "password": "Pässwörd123"},
    )
    assert resp.status_code in (201, 422)


def test_empty_body_on_required_endpoint(client):
    resp = client.post(
        "/api/v1/auth/register",
        content=b"{}",
        headers={"Content-Type": "application/json"},
    )
    assert resp.status_code == 422


# ----------------------------------------------------------------
# Sensitive data exposure
# ----------------------------------------------------------------


def test_password_hash_not_in_register_response(client):
    resp = client.post(
        "/api/v1/auth/register",
        json={"username": "bob", "password": "Bob12345"},
    )
    assert resp.status_code == 201
    assert "hashed_password" not in resp.text


def test_password_hash_not_in_login_response(client):
    client.post(
        "/api/v1/auth/register", json={"username": "bob", "password": "Bob12345"}
    )
    resp = client.post(
        "/api/v1/auth/login", json={"username": "bob", "password": "Bob12345"}
    )
    assert resp.status_code == 200
    assert "hashed_password" not in resp.text


def test_error_messages_dont_leak_internal_details(client, admin_token, monkeypatch):
    import app.services.auth_service as auth_service

    def boom(db, user_id):
        raise RuntimeError("SECRET_INTERNAL_DETAIL")

    monkeypatch.setattr(auth_service, "get_user_by_id", boom)
    raw_client = TestClient(app, raise_server_exceptions=False)
    resp = raw_client.get("/api/v1/datasets", headers=auth_header(admin_token))
    assert resp.status_code == 500
    assert "SECRET_INTERNAL_DETAIL" not in resp.text
    assert "Traceback" not in resp.text


def test_jwt_secret_not_in_response_headers(client):
    client.post(
        "/api/v1/auth/register", json={"username": "bob", "password": "Bob12345"}
    )
    resp = client.post(
        "/api/v1/auth/login", json={"username": "bob", "password": "Bob12345"}
    )
    assert resp.status_code == 200
    assert settings.jwt_secret not in resp.text
    for value in resp.headers.values():
        assert settings.jwt_secret not in value


# ----------------------------------------------------------------
# CORS security
# ----------------------------------------------------------------


def test_cors_preflight_returns_correct_origin(client):
    resp = client.options(
        "/api/v1/auth/login",
        headers={
            "Origin": "http://localhost:3000",
            "Access-Control-Request-Method": "POST",
        },
    )
    assert resp.status_code == 200
    assert resp.headers.get("access-control-allow-origin") == "http://localhost:3000"


def test_cors_rejects_unknown_origin(client):
    resp = client.options(
        "/api/v1/auth/login",
        headers={"Origin": "http://evil.com", "Access-Control-Request-Method": "POST"},
    )
    assert resp.status_code == 400


def test_cors_allows_credentials(client):
    resp = client.options(
        "/api/v1/auth/login",
        headers={
            "Origin": "http://localhost:3000",
            "Access-Control-Request-Method": "POST",
        },
    )
    assert resp.status_code == 200
    assert resp.headers.get("access-control-allow-credentials") == "true"


# ----------------------------------------------------------------
# Rate limiting accuracy
# ----------------------------------------------------------------


def test_rate_limit_headers_present(client):
    for i in range(3):
        client.post(
            "/api/v1/auth/register", json={"username": f"h{i}", "password": "Hash1234"}
        )
    resp = client.post(
        "/api/v1/auth/register", json={"username": "h3", "password": "Hash1234"}
    )
    assert resp.status_code == 429
    assert "X-RateLimit-Limit" in resp.headers
    assert "X-RateLimit-Remaining" in resp.headers
    assert "X-RateLimit-Reset" in resp.headers


def test_rate_limit_per_endpoint_independent(client):
    for i in range(3):
        client.post(
            "/api/v1/auth/register", json={"username": f"u{i}", "password": "User1234"}
        )
    blocked = client.post(
        "/api/v1/auth/register", json={"username": "u3", "password": "User1234"}
    )
    assert blocked.status_code == 429
    login = client.post(
        "/api/v1/auth/login", json={"username": "u0", "password": "User1234"}
    )
    assert login.status_code == 200


def test_rate_limit_login_independent_of_register(client):
    for i in range(3):
        client.post(
            "/api/v1/auth/register", json={"username": f"u{i}", "password": "User5678"}
        )
    for i in range(5):
        login = client.post(
            "/api/v1/auth/login", json={"username": f"u{i % 3}", "password": "User5678"}
        )
        assert login.status_code == 200, f"login {i + 1} should succeed"


def test_rate_limit_x_forwarded_for_ignored(client):
    spoofed = {"X-Forwarded-For": "10.0.0.1"}
    for i in range(3):
        resp = client.post(
            "/api/v1/auth/register",
            json={"username": f"sp{i}", "password": "Spoof1234"},
            headers=spoofed,
        )
        assert resp.status_code == 201
    resp = client.post(
        "/api/v1/auth/register",
        json={"username": "sp3", "password": "Spoof1234"},
        headers=spoofed,
    )
    assert resp.status_code == 429


def test_rate_limit_window_resets(client):
    for i in range(3):
        client.post(
            "/api/v1/auth/register", json={"username": f"w{i}", "password": "Win12345"}
        )
    assert (
        client.post(
            "/api/v1/auth/register", json={"username": "w3", "password": "Win12345"}
        ).status_code
        == 429
    )
    limiter.reset()
    assert (
        client.post(
            "/api/v1/auth/register", json={"username": "w4", "password": "Win12345"}
        ).status_code
        == 201
    )
