"""Tests for JWT auth endpoints (Phase 9)."""

from tests.conftest import auth_header


def test_register_first_user_becomes_admin(client):
    resp = client.post("/auth/register", json={"username": "admin", "password": "pass123", "role": "user"})
    assert resp.status_code == 201
    data = resp.json()
    assert data["role"] == "admin"  # first user auto-promoted
    assert data["username"] == "admin"


def test_register_second_user_gets_requested_role(client):
    client.post("/auth/register", json={"username": "admin", "password": "pass123"})
    resp = client.post("/auth/register", json={"username": "bob", "password": "pass123", "role": "user"})
    assert resp.status_code == 201
    assert resp.json()["role"] == "user"


def test_register_duplicate_username_returns_409(client):
    client.post("/auth/register", json={"username": "admin", "password": "pass123"})
    resp = client.post("/auth/register", json={"username": "admin", "password": "other"})
    assert resp.status_code == 409
    assert resp.json()["error"]["code"] == "USERNAME_TAKEN"


def test_register_invalid_role_returns_400(client):
    resp = client.post("/auth/register", json={"username": "bob", "password": "pass123", "role": "superuser"})
    assert resp.status_code == 400
    assert resp.json()["error"]["code"] == "INVALID_ROLE"


def test_login_success(client):
    client.post("/auth/register", json={"username": "admin", "password": "pass123"})
    resp = client.post("/auth/login", json={"username": "admin", "password": "pass123"})
    assert resp.status_code == 200
    data = resp.json()
    assert "access_token" in data
    assert data["user"]["username"] == "admin"
    assert data["user"]["role"] == "admin"


def test_login_wrong_password_returns_401(client):
    client.post("/auth/register", json={"username": "admin", "password": "pass123"})
    resp = client.post("/auth/login", json={"username": "admin", "password": "wrong"})
    assert resp.status_code == 401
    assert resp.json()["error"]["code"] == "INVALID_CREDENTIALS"


def test_login_nonexistent_user_returns_401(client):
    resp = client.post("/auth/login", json={"username": "nobody", "password": "pass123"})
    assert resp.status_code == 401


def test_protected_endpoint_without_token_returns_401(client):
    resp = client.get("/datasets")
    assert resp.status_code == 401
    assert resp.json()["error"]["code"] == "MISSING_TOKEN"


def test_protected_endpoint_with_invalid_token_returns_401(client):
    resp = client.get("/datasets", headers={"Authorization": "Bearer invalid.token.here"})
    assert resp.status_code == 401
    assert resp.json()["error"]["code"] == "INVALID_TOKEN"


def test_protected_endpoint_with_valid_token_works(client, admin_token):
    resp = client.get("/datasets", headers=auth_header(admin_token))
    assert resp.status_code == 200


def test_list_users_requires_admin(client, admin_token, user_token):
    # admin can list
    resp = client.get("/users", headers=auth_header(admin_token))
    assert resp.status_code == 200
    assert len(resp.json()) == 2  # admin + user

    # regular user cannot
    resp = client.get("/users", headers=auth_header(user_token))
    assert resp.status_code == 403
    assert resp.json()["error"]["code"] == "FORBIDDEN"


def test_delete_user_requires_admin(client, admin_token, user_token):
    # create a user to delete
    client.post("/auth/register", json={"username": "target", "password": "pass123"})
    target_id = client.get("/users", headers=auth_header(admin_token)).json()[-1]["id"]

    # regular user cannot delete
    resp = client.delete(f"/users/{target_id}", headers=auth_header(user_token))
    assert resp.status_code == 403

    # admin can delete
    resp = client.delete(f"/users/{target_id}", headers=auth_header(admin_token))
    assert resp.status_code == 204


def test_delete_nonexistent_user_returns_404(client, admin_token):
    resp = client.delete("/users/99999", headers=auth_header(admin_token))
    assert resp.status_code == 404
