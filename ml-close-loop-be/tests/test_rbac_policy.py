"""RBAC permission policy tests (issue #123).

Covers:
- The `ROLE_PERMISSIONS` map itself (app/rbac.py) matches the intended 5-role matrix.
- Every critical endpoint identified in issue #123 (promote, deploy, rollback, infra
  credential write) is gated by `require_permission`, not the binary `require_admin`:
  each of the 5 roles x each critical endpoint is exercised and asserted allow/deny
  per the matrix, including the failing/403 path - not just the admin happy path
  already covered by test_promotion_api.py / test_deployment_api.py / test_compute_resource_api.py.
"""

import pytest
from sqlalchemy import select, update
from sqlalchemy.orm import Session

from app.models.deployment import Deployment
from app.models.user import User
from app.rbac import (
    DEPLOY,
    INFRA_CREDENTIAL_WRITE,
    PROMOTE,
    ROLLBACK,
    ROLE_PERMISSIONS,
    VALIDATE_STAGING,
    Role,
)

from tests.conftest import auth_header
from tests.test_compute_resource_api import _resource_payload
from tests.test_promotion_api import _evaluated_model_version, _promoted_model_version

ALL_ROLES = [Role.ADMIN, Role.ML_ENGINEER, Role.DATA_ENGINEER, Role.REVIEWER, Role.USER]

# The matrix this issue asks for, spelled out explicitly (not read back from app.rbac) so a
# regression that silently changes ROLE_PERMISSIONS is caught rather than rubber-stamped.
EXPECTED_ROLE_PERMISSIONS: dict[str, set[str]] = {
    Role.ADMIN: {PROMOTE, VALIDATE_STAGING, DEPLOY, ROLLBACK, INFRA_CREDENTIAL_WRITE},
    Role.ML_ENGINEER: {PROMOTE, DEPLOY, ROLLBACK},
    Role.DATA_ENGINEER: set(),
    Role.REVIEWER: {VALIDATE_STAGING},
    Role.USER: set(),
}


def _allowed(permission: str, role: str) -> bool:
    return permission in EXPECTED_ROLE_PERMISSIONS[role]


def _role_token(client, role: str, username: str, password: str = "Passw0rd1") -> str:
    """Register `username` and force their role to `role` directly in the DB.

    The public /auth/register endpoint only ever grants "admin" (first user in an empty
    table) or "user" (app/api/auth.py:102-116) - issue #123 is about authorizing roles once
    assigned, not about self-service role escalation via registration, so that endpoint is
    out of scope here. `get_current_user` (app/api/deps.py) re-reads the user row from the DB
    on every request, so mutating the row after registration is enough for a token issued
    afterwards to carry the new role.
    """
    # A first registration in an empty users table auto-becomes admin; seed one first so
    # `username` reliably starts as "user" before being promoted directly in the DB.
    client.post(
        "/api/v1/auth/register",
        json={
            "username": "_rbac_seed_admin",
            "password": "SeedAdmin1",
            "role": "admin",
        },
    )
    client.post(
        "/api/v1/auth/register",
        json={"username": username, "password": password, "role": "user"},
    )
    with Session(client.engine) as session:
        session.execute(update(User).where(User.username == username).values(role=role))
        session.commit()
    resp = client.post(
        "/api/v1/auth/login", json={"username": username, "password": password}
    )
    return resp.json()["access_token"]


def _assert_matrix(resp, permission: str, role: str, success_status: int) -> None:
    if _allowed(permission, role):
        assert resp.status_code == success_status, resp.text
    else:
        assert resp.status_code == 403
        assert resp.json()["error"]["code"] == "FORBIDDEN"


def test_role_permissions_matrix_matches_spec():
    """AC: `ROLE_PERMISSIONS` map + 5-role enum (ADMIN/ML_ENGINEER/DATA_ENGINEER/REVIEWER/USER)."""
    assert set(ROLE_PERMISSIONS) == set(Role.ALL) == set(EXPECTED_ROLE_PERMISSIONS)
    assert ROLE_PERMISSIONS == EXPECTED_ROLE_PERMISSIONS


# --- POST /models/{model_id}/versions/{version}/decisions -> model:promote ---


@pytest.mark.parametrize("role", ALL_ROLES)
def test_create_decision_permission_matrix(client, admin_token, role):
    model_id, version = _evaluated_model_version(client, admin_token)
    token = _role_token(client, role, "rbac_decision_user")
    resp = client.post(
        f"/api/v1/models/{model_id}/versions/{version}/decisions",
        json={"decision": "PROMOTED", "rationale": "rbac matrix"},
        headers=auth_header(token),
    )
    _assert_matrix(resp, PROMOTE, role, 201)


# --- POST /models/{model_id}/rollback -> deployment:rollback ---


@pytest.mark.parametrize("role", ALL_ROLES)
def test_rollback_model_permission_matrix(client, admin_token, role):
    model_id, version = _promoted_model_version(client, admin_token)
    token = _role_token(client, role, "rbac_rollback_model_user")
    resp = client.post(
        f"/api/v1/models/{model_id}/rollback",
        json={"rollback_of_version": version, "rationale": "rbac matrix"},
        headers=auth_header(token),
    )
    _assert_matrix(resp, ROLLBACK, role, 201)


# --- POST /models/{model_id}/versions/{version}/deploy-staging -> deployment:deploy ---


@pytest.mark.parametrize("role", ALL_ROLES)
def test_deploy_staging_permission_matrix(client, admin_token, role):
    model_id, version = _evaluated_model_version(client, admin_token)
    token = _role_token(client, role, "rbac_deploy_staging_user")
    resp = client.post(
        f"/api/v1/models/{model_id}/versions/{version}/deploy-staging",
        json={"rationale": "rbac matrix"},
        headers=auth_header(token),
    )
    _assert_matrix(resp, DEPLOY, role, 201)


# --- POST /models/{model_id}/versions/{version}/validate-staging -> model:validate ---


@pytest.mark.parametrize("role", ALL_ROLES)
def test_validate_staging_permission_matrix(client, admin_token, role):
    model_id, version = _evaluated_model_version(client, admin_token)
    h = auth_header(admin_token)
    client.post(
        f"/api/v1/models/{model_id}/versions/{version}/deploy-staging",
        json={"rationale": "stage"},
        headers=h,
    )
    token = _role_token(client, role, "rbac_validate_user")
    resp = client.post(
        f"/api/v1/models/{model_id}/versions/{version}/validate-staging",
        json={"rationale": "rbac matrix"},
        headers=auth_header(token),
    )
    _assert_matrix(resp, VALIDATE_STAGING, role, 201)


# --- POST /models/{model_id}/versions/{version}/promote-production -> model:promote ---


@pytest.mark.parametrize("role", ALL_ROLES)
def test_promote_production_permission_matrix(client, admin_token, role):
    model_id, version = _evaluated_model_version(client, admin_token)
    h = auth_header(admin_token)
    url = f"/api/v1/models/{model_id}/versions/{version}"
    client.post(f"{url}/deploy-staging", json={"rationale": "stage"}, headers=h)
    client.post(f"{url}/validate-staging", json={"rationale": "validate"}, headers=h)
    token = _role_token(client, role, "rbac_promote_prod_user")
    resp = client.post(
        f"{url}/promote-production",
        json={"rationale": "rbac matrix"},
        headers=auth_header(token),
    )
    _assert_matrix(resp, PROMOTE, role, 201)


# --- POST /deployments/{deployment_id}/rollback -> deployment:rollback ---


@pytest.mark.parametrize("role", ALL_ROLES)
def test_rollback_deployment_permission_matrix(client, admin_token, role):
    h = auth_header(admin_token)
    model_id, v1 = _promoted_model_version(client, admin_token)
    client.post(f"/api/v1/models/{model_id}/versions/{v1}/deploy", headers=h)
    # A second promoted+deployed version supersedes v1 (-> RETIRED), which is a legal rollback
    # target; v1 fresh off `deploy()` is DEPLOYED, which `promotion_service.rollback` rejects.
    model_id2, v2 = _evaluated_model_version(client, admin_token)
    assert model_id2 == model_id and v2 != v1
    client.post(
        f"/api/v1/models/{model_id}/versions/{v2}/decisions",
        json={"decision": "PROMOTED", "rationale": "ship v2"},
        headers=h,
    )
    client.post(f"/api/v1/models/{model_id}/versions/{v2}/deploy", headers=h)

    with Session(client.engine) as session:
        v1_row = session.scalar(
            select(Deployment)
            .where(Deployment.model_id == model_id, Deployment.model_version == v1)
            .order_by(Deployment.deployed_at.asc())
        )
    token = _role_token(client, role, "rbac_rollback_dep_user")
    resp = client.post(
        f"/api/v1/deployments/{v1_row.deployment_id}/rollback",
        json={"rationale": "rbac matrix"},
        headers=auth_header(token),
    )
    _assert_matrix(resp, ROLLBACK, role, 201)


# --- POST /models/{model_id}/versions/{version}/deploy -> deployment:deploy ---


@pytest.mark.parametrize("role", ALL_ROLES)
def test_deploy_permission_matrix(client, admin_token, role):
    model_id, version = _promoted_model_version(client, admin_token)
    token = _role_token(client, role, "rbac_deploy_user")
    resp = client.post(
        f"/api/v1/models/{model_id}/versions/{version}/deploy",
        headers=auth_header(token),
    )
    _assert_matrix(resp, DEPLOY, role, 200)


# --- POST /compute-resources -> infra:credential:write ---


@pytest.mark.parametrize("role", ALL_ROLES)
def test_create_compute_resource_permission_matrix(client, admin_token, role):
    token = _role_token(client, role, "rbac_infra_create_user")
    resp = client.post(
        "/api/v1/compute-resources",
        json=_resource_payload(name=f"rbac-create-{role}"),
        headers=auth_header(token),
    )
    _assert_matrix(resp, INFRA_CREDENTIAL_WRITE, role, 201)


# --- PATCH /compute-resources/{id} -> infra:credential:write ---


@pytest.mark.parametrize("role", ALL_ROLES)
def test_update_compute_resource_permission_matrix(client, admin_token, role):
    created = client.post(
        "/api/v1/compute-resources",
        json=_resource_payload(name=f"rbac-update-{role}"),
        headers=auth_header(admin_token),
    ).json()
    token = _role_token(client, role, "rbac_infra_update_user")
    resp = client.patch(
        f"/api/v1/compute-resources/{created['id']}",
        json={"host": "10.0.0.99"},
        headers=auth_header(token),
    )
    _assert_matrix(resp, INFRA_CREDENTIAL_WRITE, role, 200)


# --- DELETE /compute-resources/{id} -> infra:credential:write ---


@pytest.mark.parametrize("role", ALL_ROLES)
def test_delete_compute_resource_permission_matrix(client, admin_token, role):
    created = client.post(
        "/api/v1/compute-resources",
        json=_resource_payload(name=f"rbac-delete-{role}"),
        headers=auth_header(admin_token),
    ).json()
    token = _role_token(client, role, "rbac_infra_delete_user")
    resp = client.delete(
        f"/api/v1/compute-resources/{created['id']}",
        headers=auth_header(token),
    )
    _assert_matrix(resp, INFRA_CREDENTIAL_WRITE, role, 204)
