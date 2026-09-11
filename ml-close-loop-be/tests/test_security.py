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


def test_lowercase_bearer_scheme_accepted(client):
    """HTTPBearer's scheme check is case-insensitive (RFC 7235) -- Scalar/Swagger's
    "Authorize" button relies on this working the same as the exact-case "Bearer"."""
    token = _register_and_login(client)
    resp = client.get("/api/v1/datasets", headers={"Authorization": f"bearer {token}"})
    assert resp.status_code == 200


def test_openapi_declares_bearer_security_scheme():
    """Swagger/Scalar need a real securitySchemes entry to show an "Authorize" button
    that applies the token to every endpoint automatically, instead of requiring a
    manually-typed "Bearer <token>" header per request."""
    spec = app.openapi()
    schemes = spec["components"]["securitySchemes"]
    assert any(s.get("scheme") == "bearer" for s in schemes.values())
    assert spec["paths"]["/api/v1/datasets"]["get"]["security"]


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


# ----------------------------------------------------------------
# Issue #86 — Security Hardening
# ----------------------------------------------------------------


def test_jwt_secret_default_fails_fast_in_production(monkeypatch):
    """jwt_secret == dev default AND deployment_environment=production must raise SystemExit."""
    import app.config as config_mod

    monkeypatch.setenv("JWT_SECRET", "dev-secret-change-in-production")
    monkeypatch.setenv("DEPLOYMENT_ENVIRONMENT", "production")
    try:
        config_mod.Settings()
    except SystemExit:
        pass  # expected
    else:
        raise AssertionError(
            "expected SystemExit for insecure jwt_secret in production"
        )


def test_jwt_secret_default_allowed_in_non_prod(monkeypatch):
    """jwt_secret == dev default is fine when not in production environment."""
    import app.config as config_mod

    monkeypatch.setenv("JWT_SECRET", "dev-secret-change-in-production")
    monkeypatch.setenv("DEPLOYMENT_ENVIRONMENT", "default")
    s = config_mod.Settings()
    assert s.jwt_secret == "dev-secret-change-in-production"


def test_secret_scrubbing_redacts_password_in_logs(capfd):
    """Structlog events with 'password' key are redacted."""
    import logging

    from app.logging import _scrub_secrets

    bound = logging.Logger("test")
    event = {"password": "hunter2", "username": "admin"}
    result = _scrub_secrets(bound, "info", event)
    assert result["password"] == "***REDACTED***"
    assert result["username"] == "admin"


def test_secret_scrubbing_redacts_token_in_logs():
    from app.logging import _scrub_secrets

    event = {"token": "secret-value", "user_id": 1}
    result = _scrub_secrets(None, None, event)
    assert result["token"] == "***REDACTED***"
    assert result["user_id"] == 1


def test_access_token_revocation_via_family_does_not_block_current_token(client):
    """Access tokens are short-lived and NEVER checked against a revocation
    store (issue #125) - natural expiry is the only guard. Revoking the
    refresh-token family (what logout does) must not affect an already-issued
    access token; it stays valid until it expires on its own."""
    from tests.conftest import auth_header

    import app.services.auth_service as svc
    from sqlalchemy.orm import Session

    token = _register_and_login(client)
    resp = client.get("/api/v1/datasets", headers=auth_header(token))
    assert resp.status_code == 200

    payload = svc.decode_token(token)
    with Session(client.engine) as db:
        svc.revoke_refresh_family(db, payload["family_id"])
        db.commit()

    resp = client.get("/api/v1/datasets", headers=auth_header(token))
    assert resp.status_code == 200


def test_logout_revokes_refresh_token_not_access_token(client):
    """POST /auth/logout revokes the refresh-token family (blocks future
    refresh) but leaves the just-issued access token usable until it expires
    naturally (issue #125 - access tokens are never revocation-checked)."""
    from tests.conftest import auth_header

    client.post(
        "/api/v1/auth/register",
        json={"username": "admin", "password": "Admin1234", "role": "admin"},
    )
    login_resp = client.post(
        "/api/v1/auth/login", json={"username": "admin", "password": "Admin1234"}
    )
    token = login_resp.json()["access_token"]
    refresh_token = login_resp.json()["refresh_token"]

    # Access token works before logout.
    resp = client.get("/api/v1/datasets", headers=auth_header(token))
    assert resp.status_code == 200

    resp = client.post("/api/v1/auth/logout", headers=auth_header(token))
    assert resp.status_code == 204

    # Access token still works (not revocation-checked) ...
    resp = client.get("/api/v1/datasets", headers=auth_header(token))
    assert resp.status_code == 200

    # ... but the refresh token can no longer mint new tokens.
    resp = client.post("/api/v1/auth/refresh", json={"refresh_token": refresh_token})
    assert resp.status_code == 401
    assert resp.json()["error"]["code"] == "INVALID_REFRESH_TOKEN"
