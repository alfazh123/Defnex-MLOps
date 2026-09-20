"""Tests for JWT auth endpoints (Phase 9)."""

from tests.conftest import auth_header


def test_register_first_user_becomes_admin(client):
    resp = client.post(
        "/api/v1/auth/register",
        json={"username": "admin", "password": "Pass1234", "role": "user"},
    )
    assert resp.status_code == 201
    data = resp.json()
    assert data["role"] == "admin"  # first user auto-promoted
    assert data["username"] == "admin"


def test_register_second_user_gets_user_role(client):
    client.post(
        "/api/v1/auth/register", json={"username": "admin", "password": "Pass1234"}
    )
    resp = client.post(
        "/api/v1/auth/register",
        json={"username": "bob", "password": "Pass1234", "role": "user"},
    )
    assert resp.status_code == 201
    assert resp.json()["role"] == "user"


def test_register_second_user_cannot_request_admin_role(client):
    """After bootstrap, requesting admin role must not grant admin (P1-4)."""
    client.post(
        "/api/v1/auth/register", json={"username": "admin", "password": "Pass1234"}
    )
    resp = client.post(
        "/api/v1/auth/register",
        json={"username": "attacker", "password": "Pass1234", "role": "admin"},
    )
    assert resp.status_code == 201
    assert resp.json()["role"] == "user"


def test_register_duplicate_username_returns_409(client):
    client.post(
        "/api/v1/auth/register", json={"username": "admin", "password": "Pass1234"}
    )
    resp = client.post(
        "/api/v1/auth/register", json={"username": "admin", "password": "Other1Pass"}
    )
    assert resp.status_code == 409
    assert resp.json()["error"]["code"] == "USERNAME_TAKEN"


def test_register_invalid_role_returns_400(client):
    resp = client.post(
        "/api/v1/auth/register",
        json={"username": "bob", "password": "Pass1234", "role": "superuser"},
    )
    assert resp.status_code == 400
    assert resp.json()["error"]["code"] == "INVALID_ROLE"


def test_login_success(client):
    client.post(
        "/api/v1/auth/register", json={"username": "admin", "password": "Pass1234"}
    )
    resp = client.post(
        "/api/v1/auth/login", json={"username": "admin", "password": "Pass1234"}
    )
    assert resp.status_code == 200
    data = resp.json()
    assert "access_token" in data
    assert "refresh_token" in data
    assert data["user"]["username"] == "admin"
    assert data["user"]["role"] == "admin"


def test_login_wrong_password_returns_401(client):
    client.post(
        "/api/v1/auth/register", json={"username": "admin", "password": "Pass1234"}
    )
    resp = client.post(
        "/api/v1/auth/login", json={"username": "admin", "password": "wrong"}
    )
    assert resp.status_code == 401
    assert resp.json()["error"]["code"] == "INVALID_CREDENTIALS"


def test_login_nonexistent_user_returns_401(client):
    resp = client.post(
        "/api/v1/auth/login", json={"username": "nobody", "password": "Pass1234"}
    )
    assert resp.status_code == 401


def test_protected_endpoint_without_token_returns_401(client):
    resp = client.get("/api/v1/datasets")
    assert resp.status_code == 401
    assert resp.json()["error"]["code"] == "MISSING_TOKEN"


def test_protected_endpoint_with_invalid_token_returns_401(client):
    resp = client.get(
        "/api/v1/datasets", headers={"Authorization": "Bearer invalid.token.here"}
    )
    assert resp.status_code == 401
    assert resp.json()["error"]["code"] == "INVALID_TOKEN"


def test_protected_endpoint_with_valid_token_works(client, admin_token):
    resp = client.get("/api/v1/datasets", headers=auth_header(admin_token))
    assert resp.status_code == 200


def test_list_users_requires_admin(client, admin_token, user_token):
    # admin can list
    resp = client.get("/api/v1/users", headers=auth_header(admin_token))
    assert resp.status_code == 200
    data = resp.json()
    assert "items" in data
    assert data["total"] == 2  # admin + user

    # regular user cannot
    resp = client.get("/api/v1/users", headers=auth_header(user_token))
    assert resp.status_code == 403
    assert resp.json()["error"]["code"] == "FORBIDDEN"


def test_delete_user_requires_admin(client, admin_token, user_token):
    # create a user to delete
    client.post(
        "/api/v1/auth/register", json={"username": "target", "password": "Pass1234"}
    )
    target_id = client.get("/api/v1/users", headers=auth_header(admin_token)).json()[
        "items"
    ][-1]["id"]

    # regular user cannot delete
    resp = client.delete(f"/api/v1/users/{target_id}", headers=auth_header(user_token))
    assert resp.status_code == 403

    # admin can delete
    resp = client.delete(f"/api/v1/users/{target_id}", headers=auth_header(admin_token))
    assert resp.status_code == 204


def test_delete_nonexistent_user_returns_404(client, admin_token):
    resp = client.delete("/api/v1/users/99999", headers=auth_header(admin_token))
    assert resp.status_code == 404


def test_register_short_password_rejected(client):
    resp = client.post(
        "/api/v1/auth/register", json={"username": "bob", "password": "Ab1"}
    )
    assert resp.status_code == 422


def test_register_no_uppercase_password_rejected(client):
    resp = client.post(
        "/api/v1/auth/register", json={"username": "bob", "password": "alllower1"}
    )
    assert resp.status_code == 422


def test_register_no_digit_password_rejected(client):
    resp = client.post(
        "/api/v1/auth/register", json={"username": "bob", "password": "NoDigitHere"}
    )
    assert resp.status_code == 422


def test_register_strong_password_accepted(client):
    resp = client.post(
        "/api/v1/auth/register", json={"username": "bob", "password": "Strong1Pass"}
    )
    assert resp.status_code == 201


def test_refresh_token_flow(client):
    client.post(
        "/api/v1/auth/register", json={"username": "admin", "password": "Pass1234"}
    )
    login_resp = client.post(
        "/api/v1/auth/login", json={"username": "admin", "password": "Pass1234"}
    )
    refresh_token = login_resp.json()["refresh_token"]

    resp = client.post("/api/v1/auth/refresh", json={"refresh_token": refresh_token})
    assert resp.status_code == 200
    data = resp.json()
    assert "access_token" in data
    assert "refresh_token" in data
    assert data["user"]["username"] == "admin"


def test_refresh_with_invalid_token_returns_401(client):
    resp = client.post(
        "/api/v1/auth/refresh", json={"refresh_token": "invalid.token.here"}
    )
    assert resp.status_code == 401
    assert resp.json()["error"]["code"] == "INVALID_REFRESH_TOKEN"


def test_refresh_with_access_token_returns_401(client):
    client.post(
        "/api/v1/auth/register", json={"username": "admin", "password": "Pass1234"}
    )
    login_resp = client.post(
        "/api/v1/auth/login", json={"username": "admin", "password": "Pass1234"}
    )
    access_token = login_resp.json()["access_token"]

    resp = client.post("/api/v1/auth/refresh", json={"refresh_token": access_token})
    assert resp.status_code == 401


def test_refresh_with_expired_token_returns_401(client):
    from datetime import timedelta, timezone
    from datetime import datetime
    from jose import jwt
    from app.config import settings

    client.post(
        "/api/v1/auth/register", json={"username": "admin", "password": "Pass1234"}
    )
    login_resp = client.post(
        "/api/v1/auth/login", json={"username": "admin", "password": "Pass1234"}
    )
    user_id = login_resp.json()["user"]["id"]

    expired = jwt.encode(
        {
            "sub": str(user_id),
            "role": "admin",
            "exp": datetime.now(timezone.utc) - timedelta(hours=1),
            "type": "refresh",
        },
        settings.jwt_secret,
        algorithm=settings.jwt_algorithm,
    )

    resp = client.post("/api/v1/auth/refresh", json={"refresh_token": expired})
    assert resp.status_code == 401


def test_refresh_with_nonexistent_user_returns_401(client):
    from datetime import timedelta, timezone
    from datetime import datetime
    from jose import jwt
    from app.config import settings

    token = jwt.encode(
        {
            "sub": "99999",
            "role": "user",
            "exp": datetime.now(timezone.utc) + timedelta(days=1),
            "type": "refresh",
        },
        settings.jwt_secret,
        algorithm=settings.jwt_algorithm,
    )

    resp = client.post("/api/v1/auth/refresh", json={"refresh_token": token})
    assert resp.status_code == 401


def test_register_with_special_characters_in_password(client):
    resp = client.post(
        "/api/v1/auth/register",
        json={"username": "spec", "password": "P@ss!#$%^&*1"},
    )
    assert resp.status_code == 201


def test_login_returns_token_with_correct_user_fields(client):
    from jose import jwt

    from app.config import settings

    client.post(
        "/api/v1/auth/register", json={"username": "admin", "password": "Pass1234"}
    )
    login_resp = client.post(
        "/api/v1/auth/login", json={"username": "admin", "password": "Pass1234"}
    )
    payload = jwt.decode(
        login_resp.json()["access_token"],
        settings.jwt_secret,
        algorithms=[settings.jwt_algorithm],
    )
    assert payload["sub"] == str(login_resp.json()["user"]["id"])
    assert payload["role"] == "admin"


def test_access_token_expired_returns_401(client):
    from datetime import datetime, timedelta, timezone

    from jose import jwt

    from app.config import settings

    expired = jwt.encode(
        {
            "sub": "1",
            "role": "user",
            "exp": datetime.now(timezone.utc) - timedelta(hours=1),
            "type": "access",
        },
        settings.jwt_secret,
        algorithm=settings.jwt_algorithm,
    )
    resp = client.get(
        "/api/v1/datasets", headers={"Authorization": f"Bearer {expired}"}
    )
    assert resp.status_code == 401
    assert resp.json()["error"]["code"] == "INVALID_TOKEN"


def test_register_empty_username_accepted(client):
    # Gap: UserCreate validates passwords but not usernames, so "" registers fine.
    resp = client.post(
        "/api/v1/auth/register", json={"username": "", "password": "Pass1234"}
    )
    assert resp.status_code == 201
    assert resp.json()["username"] == ""


# ------------------------------------------------------------------
# Refresh-token rotation + reuse detection (issue #125)
# ------------------------------------------------------------------


def test_access_token_ttl_is_short(client):
    """Access tokens must be short-lived (~15 min) since they're never
    revocation-checked (issue #125) - natural expiry is the only guard."""
    from app.config import settings

    assert settings.jwt_expire_minutes <= 15


def test_refresh_rotates_and_invalidates_old_refresh_token(client):
    client.post(
        "/api/v1/auth/register", json={"username": "admin", "password": "Pass1234"}
    )
    login_resp = client.post(
        "/api/v1/auth/login", json={"username": "admin", "password": "Pass1234"}
    )
    old_refresh = login_resp.json()["refresh_token"]

    resp = client.post("/api/v1/auth/refresh", json={"refresh_token": old_refresh})
    assert resp.status_code == 200
    new_refresh = resp.json()["refresh_token"]
    assert new_refresh != old_refresh

    # A legitimate second rotation, using the *new* token, keeps working.
    resp = client.post("/api/v1/auth/refresh", json={"refresh_token": new_refresh})
    assert resp.status_code == 200
    assert resp.json()["refresh_token"] != new_refresh

    # But replaying the original (now stale) refresh token is rejected.
    resp = client.post("/api/v1/auth/refresh", json={"refresh_token": old_refresh})
    assert resp.status_code == 401
    assert resp.json()["error"]["code"] == "INVALID_REFRESH_TOKEN"


def test_refresh_reuse_revokes_entire_token_family(client):
    """Reusing an already-rotated refresh token is a theft indicator: the
    whole family must die, including the legitimate token nobody has used
    yet (issue #125 reuse detection)."""
    client.post(
        "/api/v1/auth/register", json={"username": "admin", "password": "Pass1234"}
    )
    login_resp = client.post(
        "/api/v1/auth/login", json={"username": "admin", "password": "Pass1234"}
    )
    old_refresh = login_resp.json()["refresh_token"]

    resp = client.post("/api/v1/auth/refresh", json={"refresh_token": old_refresh})
    assert resp.status_code == 200
    new_refresh = resp.json()["refresh_token"]

    # Attacker (or a client with a stale token) replays the already-used token.
    resp = client.post("/api/v1/auth/refresh", json={"refresh_token": old_refresh})
    assert resp.status_code == 401

    # The legitimate, never-yet-used new_refresh must also be dead now, since
    # the whole family was revoked - not just the specific reused jti.
    resp = client.post("/api/v1/auth/refresh", json={"refresh_token": new_refresh})
    assert resp.status_code == 401
    assert resp.json()["error"]["code"] == "INVALID_REFRESH_TOKEN"


def test_refresh_revocation_persists_across_fresh_db_session(client):
    """Simulates an API process restart: revocation must be readable from a
    brand-new DB session that shares no Python state with the request that
    wrote it - proving it's durable in the DB, not an in-process cache."""
    from jose import jwt
    from sqlalchemy.orm import Session

    from app.config import settings
    import app.services.auth_service as svc

    client.post(
        "/api/v1/auth/register", json={"username": "admin", "password": "Pass1234"}
    )
    login_resp = client.post(
        "/api/v1/auth/login", json={"username": "admin", "password": "Pass1234"}
    )
    old_refresh = login_resp.json()["refresh_token"]

    resp = client.post("/api/v1/auth/refresh", json={"refresh_token": old_refresh})
    assert resp.status_code == 200

    # Fresh Session over the same persistent engine, no shared Python objects
    # with the request above (equivalent to a different worker process).
    jti = jwt.decode(
        old_refresh, settings.jwt_secret, algorithms=[settings.jwt_algorithm]
    )["jti"]
    with Session(client.engine) as fresh_db:
        assert svc._is_refresh_jti_used(fresh_db, jti) is True

    # And a fresh API request (itself a brand-new Session, see conftest.client)
    # still rejects the rotated-out token.
    resp = client.post("/api/v1/auth/refresh", json={"refresh_token": old_refresh})
    assert resp.status_code == 401


def _issue_reset_code(client, username="bob"):
    from sqlalchemy.orm import Session

    from app.services import auth_service

    with Session(client.engine) as db:
        user = auth_service.get_user_by_username(db, username)
        reset_code = auth_service.create_password_reset_token(db, user)
        db.commit()
    return reset_code


def test_forgot_password_returns_same_response_for_existing_and_nonexistent_username(
    client,
):
    """Issue #177: must not be usable to enumerate registered usernames."""
    client.post(
        "/api/v1/auth/register", json={"username": "bob", "password": "Pass1234"}
    )

    existing = client.post("/api/v1/auth/forgot-password", json={"username": "bob"})
    missing = client.post("/api/v1/auth/forgot-password", json={"username": "nope"})

    assert existing.status_code == 200
    assert missing.status_code == 200
    assert existing.json() == missing.json()


def test_forgot_password_logs_reset_code(client, monkeypatch):
    """Issue #177: no email/SMS provider is wired -- the reset code must be logged (the
    placeholder delivery channel), not silently discarded."""
    client.post(
        "/api/v1/auth/register", json={"username": "bob", "password": "Pass1234"}
    )
    events = []

    class _CapturingLogger:
        def info(self, event, **kwargs):
            events.append((event, kwargs))

    monkeypatch.setattr("app.api.auth.logger", _CapturingLogger())

    client.post("/api/v1/auth/forgot-password", json={"username": "bob"})

    reset_events = [e for e in events if e[0] == "password_reset_requested"]
    assert len(reset_events) == 1
    assert reset_events[0][1]["reset_code"]


def test_reset_password_with_valid_code_allows_login_with_new_password(client):
    client.post(
        "/api/v1/auth/register", json={"username": "bob", "password": "OldPass1"}
    )
    reset_code = _issue_reset_code(client)

    resp = client.post(
        "/api/v1/auth/reset-password",
        json={"token": reset_code, "new_password": "NewPass1"},
    )
    assert resp.status_code == 200

    old_login = client.post(
        "/api/v1/auth/login", json={"username": "bob", "password": "OldPass1"}
    )
    new_login = client.post(
        "/api/v1/auth/login", json={"username": "bob", "password": "NewPass1"}
    )
    assert old_login.status_code == 401
    assert new_login.status_code == 200


def test_reset_password_rejects_reused_code(client):
    client.post(
        "/api/v1/auth/register", json={"username": "bob", "password": "OldPass1"}
    )
    reset_code = _issue_reset_code(client)

    first = client.post(
        "/api/v1/auth/reset-password",
        json={"token": reset_code, "new_password": "NewPass1"},
    )
    second = client.post(
        "/api/v1/auth/reset-password",
        json={"token": reset_code, "new_password": "AnotherPass1"},
    )

    assert first.status_code == 200
    assert second.status_code == 400
    assert second.json()["error"]["code"] == "INVALID_RESET_TOKEN"


def test_reset_password_rejects_unknown_token(client):
    resp = client.post(
        "/api/v1/auth/reset-password",
        json={"token": "not-a-real-token", "new_password": "NewPass1"},
    )
    assert resp.status_code == 400
    assert resp.json()["error"]["code"] == "INVALID_RESET_TOKEN"


def test_reset_password_rejects_expired_code(client):
    from datetime import timedelta

    from sqlalchemy.orm import Session

    from app.models.password_reset import PasswordResetToken
    from app.services import auth_service

    client.post(
        "/api/v1/auth/register", json={"username": "bob", "password": "OldPass1"}
    )
    reset_code = _issue_reset_code(client)
    with Session(client.engine) as db:
        row = db.get(PasswordResetToken, reset_code)
        row.expires_at = auth_service._utc_now_naive() - timedelta(minutes=1)
        db.commit()

    resp = client.post(
        "/api/v1/auth/reset-password",
        json={"token": reset_code, "new_password": "NewPass1"},
    )
    assert resp.status_code == 400
    assert resp.json()["error"]["code"] == "INVALID_RESET_TOKEN"


def test_reset_password_rejects_weak_new_password(client):
    client.post(
        "/api/v1/auth/register", json={"username": "bob", "password": "OldPass1"}
    )
    reset_code = _issue_reset_code(client)

    resp = client.post(
        "/api/v1/auth/reset-password",
        json={"token": reset_code, "new_password": "weak"},
    )
    assert resp.status_code == 422
