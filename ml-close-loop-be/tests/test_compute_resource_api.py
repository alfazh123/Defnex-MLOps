"""Compute resource Admin CRUD API tests (issue #78, PRD §8.6, §33).

Tests all endpoints: create, list, get, update, delete, health.
Covers RBAC (admin vs user), validation, pagination, filters, and edge cases.
"""

from tests.conftest import auth_header


# --- Helpers ---


def _register_user_direct(client, username="dummy", password="Dummy1234", role="user"):
    """Register a user directly."""
    client.post(
        "/api/v1/auth/register",
        json={"username": username, "password": password, "role": role},
    )


def _get_user_token(client, username="alice", password="Alice1234", role="user"):
    """Register a user and return their JWT. Call AFTER a dummy admin exists."""
    client.post(
        "/api/v1/auth/register",
        json={"username": username, "password": password, "role": role},
    )
    resp = client.post(
        "/api/v1/auth/login", json={"username": username, "password": password}
    )
    return resp.json()["access_token"]


def _resource_payload(**overrides) -> dict:
    payload = {
        "name": "server-2",
        "role": "training",
        "environment": "staging",
        "provider_type": "gpu_vps",
        "host": "10.0.0.2",
        "gpu_info": {"model": "H100", "memory_gb": 80},
        "ssh_host": "10.0.0.2",
        "ssh_port": 22,
        "ssh_username": "root",
        "credential_ref": "secret://gpu-server-key",
    }
    payload.update(overrides)
    return payload


def _create_resource(client, token, **overrides) -> dict:
    resp = client.post(
        "/api/v1/compute-resources",
        json=_resource_payload(**overrides),
        headers=auth_header(token),
    )
    assert resp.status_code == 201
    return resp.json()


# --- Create ---


def test_create_compute_resource_admin(client, admin_token):
    """AC: Admin bisa register server/compute resource."""
    resp = client.post(
        "/api/v1/compute-resources",
        json=_resource_payload(),
        headers=auth_header(admin_token),
    )
    assert resp.status_code == 201
    data = resp.json()
    assert data["name"] == "server-2"
    assert data["role"] == "training"
    assert data["is_healthy"] is True
    assert data["id"] is not None


def test_create_compute_resource_non_admin_403(client, admin_token):
    """AC: RBAC — operasi infra admin-only (PRD §21.5)."""
    # Register a dummy admin first so alice gets "user" role
    _register_user_direct(client, username="admin_dummy", role="admin")
    user_tok = _get_user_token(client, username="alice")
    resp = client.post(
        "/api/v1/compute-resources",
        json=_resource_payload(),
        headers=auth_header(user_tok),
    )
    assert resp.status_code == 403
    assert resp.json()["error"]["code"] == "FORBIDDEN"


def test_create_compute_resource_no_auth_401(client):
    resp = client.post("/api/v1/compute-resources", json=_resource_payload())
    assert resp.status_code in (401, 422)


def test_create_compute_resource_duplicate_name_409(client, admin_token):
    _create_resource(client, admin_token, name="dup-server")
    resp = client.post(
        "/api/v1/compute-resources",
        json=_resource_payload(name="dup-server"),
        headers=auth_header(admin_token),
    )
    assert resp.status_code == 409
    assert resp.json()["error"]["code"] == "RESOURCE_EXISTS"


def test_create_compute_resource_minimal_fields(client, admin_token):
    """Only name is required; defaults handle the rest."""
    resp = client.post(
        "/api/v1/compute-resources",
        json={"name": "minimal-server"},
        headers=auth_header(admin_token),
    )
    assert resp.status_code == 201
    data = resp.json()
    assert data["role"] == "training"
    assert data["environment"] == "staging"
    assert data["provider_type"] == "local"


# --- List ---


def test_list_compute_resources(client, admin_token):
    _create_resource(client, admin_token, name="list-a")
    _create_resource(client, admin_token, name="list-b")

    resp = client.get(
        "/api/v1/compute-resources",
        headers=auth_header(admin_token),
    )
    assert resp.status_code == 200
    data = resp.json()
    assert data["total"] >= 2
    names = [r["name"] for r in data["items"]]
    assert "list-a" in names
    assert "list-b" in names


def test_list_compute_resources_user_can_read(client, admin_token):
    """AC: Health/status read-only untuk user biasa."""
    _register_user_direct(client, username="admin_dummy", role="admin")
    user_tok = _get_user_token(client, username="alice")
    _create_resource(client, admin_token, name="user-readable")
    resp = client.get(
        "/api/v1/compute-resources",
        headers=auth_header(user_tok),
    )
    assert resp.status_code == 200


def test_list_filter_by_role(client, admin_token):
    _create_resource(client, admin_token, name="role-train", role="training")
    _create_resource(client, admin_token, name="role-infer", role="inference")

    resp = client.get(
        "/api/v1/compute-resources?role=inference",
        headers=auth_header(admin_token),
    )
    data = resp.json()
    assert all(r["role"] == "inference" for r in data["items"])


def test_list_filter_by_environment(client, admin_token):
    _create_resource(client, admin_token, name="env-staging", environment="staging")
    _create_resource(client, admin_token, name="env-prod", environment="production")

    resp = client.get(
        "/api/v1/compute-resources?environment=production",
        headers=auth_header(admin_token),
    )
    data = resp.json()
    assert all(r["environment"] == "production" for r in data["items"])


def test_list_search_by_name(client, admin_token):
    _create_resource(client, admin_token, name="search-alpha-1")
    _create_resource(client, admin_token, name="search-beta-2")

    resp = client.get(
        "/api/v1/compute-resources?search=alpha",
        headers=auth_header(admin_token),
    )
    data = resp.json()
    assert data["total"] == 1
    assert data["items"][0]["name"] == "search-alpha-1"


def test_list_pagination(client, admin_token):
    for i in range(5):
        _create_resource(client, admin_token, name=f"page-{i}")

    resp = client.get(
        "/api/v1/compute-resources?page=1&size=2",
        headers=auth_header(admin_token),
    )
    data = resp.json()
    assert data["size"] == 2
    assert data["page"] == 1
    assert len(data["items"]) == 2


# --- Get ---


def test_get_compute_resource(client, admin_token):
    created = _create_resource(client, admin_token)
    resp = client.get(
        f"/api/v1/compute-resources/{created['id']}",
        headers=auth_header(admin_token),
    )
    assert resp.status_code == 200
    assert resp.json()["name"] == "server-2"


def test_get_compute_resource_404(client, admin_token):
    resp = client.get(
        "/api/v1/compute-resources/99999",
        headers=auth_header(admin_token),
    )
    assert resp.status_code == 404
    assert resp.json()["error"]["code"] == "RESOURCE_NOT_FOUND"


# --- Update (PATCH) ---


def test_update_compute_resource(client, admin_token):
    created = _create_resource(client, admin_token)
    resp = client.patch(
        f"/api/v1/compute-resources/{created['id']}",
        json={"host": "10.0.0.99", "role": "inference"},
        headers=auth_header(admin_token),
    )
    assert resp.status_code == 200
    data = resp.json()
    assert data["host"] == "10.0.0.99"
    assert data["role"] == "inference"
    # unchanged fields preserved
    assert data["name"] == "server-2"


def test_update_compute_resource_non_admin_403(client, admin_token):
    _register_user_direct(client, username="admin_dummy", role="admin")
    user_tok = _get_user_token(client, username="alice")
    created = _create_resource(client, admin_token)
    resp = client.patch(
        f"/api/v1/compute-resources/{created['id']}",
        json={"host": "10.0.0.99"},
        headers=auth_header(user_tok),
    )
    assert resp.status_code == 403


def test_update_compute_resource_404(client, admin_token):
    resp = client.patch(
        "/api/v1/compute-resources/99999",
        json={"host": "10.0.0.99"},
        headers=auth_header(admin_token),
    )
    assert resp.status_code == 404


def test_update_compute_resource_duplicate_name_409(client, admin_token):
    _create_resource(client, admin_token, name="orig-name")
    other = _create_resource(client, admin_token, name="other-name")
    resp = client.patch(
        f"/api/v1/compute-resources/{other['id']}",
        json={"name": "orig-name"},
        headers=auth_header(admin_token),
    )
    assert resp.status_code == 409


def test_update_partial_only_changed_fields(client, admin_token):
    """PATCH only updates provided fields."""
    created = _create_resource(client, admin_token)
    resp = client.patch(
        f"/api/v1/compute-resources/{created['id']}",
        json={"ssh_port": 2222},
        headers=auth_header(admin_token),
    )
    assert resp.status_code == 200
    assert resp.json()["ssh_port"] == 2222
    assert resp.json()["host"] == "10.0.0.2"  # unchanged


# --- Delete ---


def test_delete_compute_resource(client, admin_token):
    created = _create_resource(client, admin_token)
    resp = client.delete(
        f"/api/v1/compute-resources/{created['id']}",
        headers=auth_header(admin_token),
    )
    assert resp.status_code == 204

    # Verify it's gone
    resp = client.get(
        f"/api/v1/compute-resources/{created['id']}",
        headers=auth_header(admin_token),
    )
    assert resp.status_code == 404


def test_delete_compute_resource_non_admin_403(client, admin_token):
    _register_user_direct(client, username="admin_dummy", role="admin")
    user_tok = _get_user_token(client, username="alice")
    created = _create_resource(client, admin_token)
    resp = client.delete(
        f"/api/v1/compute-resources/{created['id']}",
        headers=auth_header(user_tok),
    )
    assert resp.status_code == 403


def test_delete_compute_resource_404(client, admin_token):
    resp = client.delete(
        "/api/v1/compute-resources/99999",
        headers=auth_header(admin_token),
    )
    assert resp.status_code == 404


# --- Health ---


def test_health_endpoint(client, admin_token):
    created = _create_resource(client, admin_token)
    resp = client.get(
        f"/api/v1/compute-resources/{created['id']}/health",
        headers=auth_header(admin_token),
    )
    assert resp.status_code == 200
    data = resp.json()
    assert data["is_healthy"] is True
    assert data["message"] == "OK"
    assert data["provider_type"] == "gpu_vps"


def test_health_user_can_read(client, admin_token):
    """AC: Health/status terlihat (read-only untuk user biasa)."""
    _register_user_direct(client, username="admin_dummy", role="admin")
    user_tok = _get_user_token(client, username="alice")
    created = _create_resource(client, admin_token)
    resp = client.get(
        f"/api/v1/compute-resources/{created['id']}/health",
        headers=auth_header(user_tok),
    )
    assert resp.status_code == 200


def test_health_404(client, admin_token):
    resp = client.get(
        "/api/v1/compute-resources/99999/health",
        headers=auth_header(admin_token),
    )
    assert resp.status_code == 404


def test_health_unhealthy_resource(client, admin_token):
    """When is_healthy=False, the message reflects it."""
    created = _create_resource(client, admin_token, name="unhealthy-srv")
    # Set unhealthy via PATCH
    client.patch(
        f"/api/v1/compute-resources/{created['id']}",
        json={"is_healthy": False},
        headers=auth_header(admin_token),
    )
    resp = client.get(
        f"/api/v1/compute-resources/{created['id']}/health",
        headers=auth_header(admin_token),
    )
    data = resp.json()
    assert data["is_healthy"] is False
    assert "unhealthy" in data["message"].lower()


# --- Credential ref ---


def test_credential_ref_stored_as_secret_ref(client, admin_token):
    """AC: Admin bisa konfigurasi credential reference (secret://...), bukan nilai plaintext."""
    resp = client.post(
        "/api/v1/compute-resources",
        json=_resource_payload(
            name="cred-test",
            credential_ref="secret://my-ssh-key",
        ),
        headers=auth_header(admin_token),
    )
    assert resp.status_code == 201
    assert resp.json()["credential_ref"] == "secret://my-ssh-key"


# --- GPU info ---


def test_gpu_info_stored_as_json(client, admin_token):
    resp = client.post(
        "/api/v1/compute-resources",
        json=_resource_payload(
            name="gpu-test",
            gpu_info={"model": "H100", "memory_gb": 80, "count": 2},
        ),
        headers=auth_header(admin_token),
    )
    assert resp.status_code == 201
    gpu = resp.json()["gpu_info"]
    assert gpu["model"] == "H100"
    assert gpu["count"] == 2
