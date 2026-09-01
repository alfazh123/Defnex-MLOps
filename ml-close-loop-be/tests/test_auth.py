"""Tests for JWT auth endpoints (Phase 9)."""

from tests.conftest import auth_header


def test_register_first_user_becomes_admin(client):
    resp = client.post(
        "/auth/register",
        json={"username": "admin", "password": "Pass1234", "role": "user"},
    )
    assert resp.status_code == 201
    data = resp.json()
    assert data["role"] == "admin"  # first user auto-promoted
    assert data["username"] == "admin"


def test_register_second_user_gets_requested_role(client):
    client.post("/auth/register", json={"username": "admin", "password": "Pass1234"})
    resp = client.post(
        "/auth/register",
        json={"username": "bob", "password": "Pass1234", "role": "user"},
    )
    assert resp.status_code == 201
    assert resp.json()["role"] == "user"


def test_register_duplicate_username_returns_409(client):
    client.post("/auth/register", json={"username": "admin", "password": "Pass1234"})
    resp = client.post(
        "/auth/register", json={"username": "admin", "password": "Other1Pass"}
    )
    assert resp.status_code == 409
    assert resp.json()["error"]["code"] == "USERNAME_TAKEN"


def test_register_invalid_role_returns_400(client):
    resp = client.post(
        "/auth/register",
        json={"username": "bob", "password": "Pass1234", "role": "superuser"},
    )
    assert resp.status_code == 400
    assert resp.json()["error"]["code"] == "INVALID_ROLE"


def test_login_success(client):
    client.post("/auth/register", json={"username": "admin", "password": "Pass1234"})
    resp = client.post(
        "/auth/login", json={"username": "admin", "password": "Pass1234"}
    )
    assert resp.status_code == 200
    data = resp.json()
    assert "access_token" in data
    assert "refresh_token" in data
    assert data["user"]["username"] == "admin"
    assert data["user"]["role"] == "admin"


def test_login_wrong_password_returns_401(client):
    client.post("/auth/register", json={"username": "admin", "password": "Pass1234"})
    resp = client.post("/auth/login", json={"username": "admin", "password": "wrong"})
    assert resp.status_code == 401
    assert resp.json()["error"]["code"] == "INVALID_CREDENTIALS"


def test_login_nonexistent_user_returns_401(client):
    resp = client.post(
        "/auth/login", json={"username": "nobody", "password": "Pass1234"}
    )
    assert resp.status_code == 401


def test_protected_endpoint_without_token_returns_401(client):
    resp = client.get("/datasets")
    assert resp.status_code == 401
    assert resp.json()["error"]["code"] == "MISSING_TOKEN"


def test_protected_endpoint_with_invalid_token_returns_401(client):
    resp = client.get(
        "/datasets", headers={"Authorization": "Bearer invalid.token.here"}
    )
    assert resp.status_code == 401
    assert resp.json()["error"]["code"] == "INVALID_TOKEN"


def test_protected_endpoint_with_valid_token_works(client, admin_token):
    resp = client.get("/datasets", headers=auth_header(admin_token))
    assert resp.status_code == 200


def test_list_users_requires_admin(client, admin_token, user_token):
    # admin can list
    resp = client.get("/users", headers=auth_header(admin_token))
    assert resp.status_code == 200
    data = resp.json()
    assert "items" in data
    assert data["total"] == 2  # admin + user

    # regular user cannot
    resp = client.get("/users", headers=auth_header(user_token))
    assert resp.status_code == 403
    assert resp.json()["error"]["code"] == "FORBIDDEN"


def test_delete_user_requires_admin(client, admin_token, user_token):
    # create a user to delete
    client.post("/auth/register", json={"username": "target", "password": "Pass1234"})
    target_id = client.get("/users", headers=auth_header(admin_token)).json()["items"][
        -1
    ]["id"]

    # regular user cannot delete
    resp = client.delete(f"/users/{target_id}", headers=auth_header(user_token))
    assert resp.status_code == 403

    # admin can delete
    resp = client.delete(f"/users/{target_id}", headers=auth_header(admin_token))
    assert resp.status_code == 204


def test_delete_nonexistent_user_returns_404(client, admin_token):
    resp = client.delete("/users/99999", headers=auth_header(admin_token))
    assert resp.status_code == 404


def test_register_short_password_rejected(client):
    resp = client.post("/auth/register", json={"username": "bob", "password": "Ab1"})
    assert resp.status_code == 422


def test_register_no_uppercase_password_rejected(client):
    resp = client.post(
        "/auth/register", json={"username": "bob", "password": "alllower1"}
    )
    assert resp.status_code == 422


def test_register_no_digit_password_rejected(client):
    resp = client.post(
        "/auth/register", json={"username": "bob", "password": "NoDigitHere"}
    )
    assert resp.status_code == 422


def test_register_strong_password_accepted(client):
    resp = client.post(
        "/auth/register", json={"username": "bob", "password": "Strong1Pass"}
    )
    assert resp.status_code == 201


def test_refresh_token_flow(client):
    client.post("/auth/register", json={"username": "admin", "password": "Pass1234"})
    login_resp = client.post(
        "/auth/login", json={"username": "admin", "password": "Pass1234"}
    )
    refresh_token = login_resp.json()["refresh_token"]

    resp = client.post("/auth/refresh", json={"refresh_token": refresh_token})
    assert resp.status_code == 200
    data = resp.json()
    assert "access_token" in data
    assert "refresh_token" in data
    assert data["user"]["username"] == "admin"


def test_refresh_with_invalid_token_returns_401(client):
    resp = client.post("/auth/refresh", json={"refresh_token": "invalid.token.here"})
    assert resp.status_code == 401
    assert resp.json()["error"]["code"] == "INVALID_REFRESH_TOKEN"


def test_refresh_with_access_token_returns_401(client):
    client.post("/auth/register", json={"username": "admin", "password": "Pass1234"})
    login_resp = client.post(
        "/auth/login", json={"username": "admin", "password": "Pass1234"}
    )
    access_token = login_resp.json()["access_token"]

    resp = client.post("/auth/refresh", json={"refresh_token": access_token})
    assert resp.status_code == 401


def test_refresh_with_expired_token_returns_401(client):
    from datetime import timedelta, timezone
    from datetime import datetime
    from jose import jwt
    from app.config import settings

    client.post("/auth/register", json={"username": "admin", "password": "Pass1234"})
    login_resp = client.post(
        "/auth/login", json={"username": "admin", "password": "Pass1234"}
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

    resp = client.post("/auth/refresh", json={"refresh_token": expired})
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

    resp = client.post("/auth/refresh", json={"refresh_token": token})
    assert resp.status_code == 401
