"""Staging/production promotion ladder tests (issue #69) and production rollback (issue #70).

Reuses the importable helpers from `tests/test_promotion_api.py` (eval-gate-compliant version
factory) and `tests/test_deployment_api.py`; gate criteria themselves are exercised by
`tests/test_promotion_gate.py`, this module turns them off like the deployment-mechanics tests do.
"""

from sqlalchemy import select
from sqlalchemy.orm import Session

import pytest

from app.models.deployment import Deployment

from tests.conftest import auth_header
from tests.test_deployment_api import _registered_model_version
from tests.test_promotion_api import _evaluated_model_version, _promoted_model_version


@pytest.fixture(autouse=True)
def _disable_promotion_gates(monkeypatch):
    """These tests exercise the ladder/rollback mechanics, not the #43 eval gate."""
    from app.config import settings

    for name in (
        "eval_gate_require_eval_set_reference",
        "eval_gate_require_qualitative_majority",
        "eval_gate_require_no_general_regression",
        "eval_gate_require_eval_loss_not_worse",
    ):
        monkeypatch.setattr(settings, name, False)


def _production_deployment_row(client, model_id):
    """The deployment-pointer history row for the ladder "production" environment deploy."""
    with Session(client.engine) as session:
        return session.scalar(
            select(Deployment)
            .where(
                Deployment.model_id == model_id,
                Deployment.environment == "production",
            )
            .order_by(Deployment.deployed_at.desc())
        )


def _latest_deployment_row(client, model_id):
    with Session(client.engine) as session:
        return session.scalar(
            select(Deployment)
            .where(Deployment.model_id == model_id)
            .order_by(Deployment.deployed_at.desc())
        )


def _first_deployment_row(client, model_id):
    with Session(client.engine) as session:
        return session.scalar(
            select(Deployment)
            .where(Deployment.model_id == model_id)
            .order_by(Deployment.deployed_at.asc())
        )


def _lineage(client, admin_token, model_id, version):
    return client.get(
        f"/api/v1/models/{model_id}/versions/{version}",
        headers=auth_header(admin_token),
    ).json()


def _pointer(client, admin_token, model_id):
    return client.get(
        f"/api/v1/models/{model_id}/deployment", headers=auth_header(admin_token)
    ).json()


def test_deploy_staging_returns_409_when_not_evaluated(client, admin_token):
    model_id, version = _registered_model_version(client, admin_token)
    response = client.post(
        f"/api/v1/models/{model_id}/versions/{version}/deploy-staging",
        json={"decided_by": "reviewer-1", "rationale": "stage v1"},
        headers=auth_header(admin_token),
    )
    assert response.status_code == 409
    assert response.json()["error"]["code"] == "STAGING_DEPLOY_NOT_ALLOWED"


def test_deploy_staging_moves_candidate_to_staging_without_production_pointer(
    client, admin_token
):
    """Issue #69 AC 1: a deployable candidate can be staged; staging must NOT move the
    production pointer (nothing has ever been deployed -> no DEPLOYED version)."""
    model_id, version = _evaluated_model_version(client, admin_token)
    response = client.post(
        f"/api/v1/models/{model_id}/versions/{version}/deploy-staging",
        json={"decided_by": "reviewer-1", "rationale": "stage v1"},
        headers=auth_header(admin_token),
    )
    assert response.status_code == 201
    body = response.json()
    assert body["decision"] == "STAGING"
    assert body["model_id"] == model_id
    assert body["version"] == version

    assert _lineage(client, admin_token, model_id, version)["status"] == "STAGING"
    assert _pointer(client, admin_token, model_id)["current_deployed_version"] is None


def test_ladder_blocks_production_promotion_until_validation(client, admin_token):
    """Issue #69 AC 2/4/5: production promotion is structurally blocked until the staging
    validation step; the full ladder ends with the registry's DEPLOYED (single source of truth)
    and a recorded PRODUCTION decision."""
    model_id, version = _evaluated_model_version(client, admin_token)
    h = auth_header(admin_token)
    url = f"/api/v1/models/{model_id}/versions/{version}"

    skipped = client.post(
        f"{url}/promote-production", json={"rationale": "jump the gate"}, headers=h
    )
    assert skipped.status_code == 409
    assert skipped.json()["error"]["code"] == "PRODUCTION_PROMOTION_NOT_ALLOWED"

    staged = client.post(
        f"{url}/deploy-staging", json={"rationale": "stage v1"}, headers=h
    )
    assert staged.status_code == 201

    skipped_again = client.post(
        f"{url}/promote-production",
        json={"rationale": "still not validated"},
        headers=h,
    )
    assert skipped_again.status_code == 409
    assert skipped_again.json()["error"]["code"] == "PRODUCTION_PROMOTION_NOT_ALLOWED"

    validated = client.post(
        f"{url}/validate-staging", json={"rationale": "integration passed"}, headers=h
    )
    assert validated.status_code == 201
    assert validated.json()["decision"] == "VALIDATED"
    assert validated.json()["evidence_snapshot"] is not None
    assert _lineage(client, admin_token, model_id, version)["status"] == "VALIDATED"

    promoted = client.post(
        f"{url}/promote-production", json={"rationale": "approve ship"}, headers=h
    )
    assert promoted.status_code == 201
    assert promoted.json()["decision"] == "PRODUCTION"
    assert _lineage(client, admin_token, model_id, version)["status"] == "DEPLOYED"
    assert (
        _pointer(client, admin_token, model_id)["current_deployed_version"] == version
    )


def test_staging_a_new_candidate_keeps_existing_production_pointer(client, admin_token):
    """Issue #69 AC 3 + issue #70 foundation: staging a second candidate must never clobber the
    live production version (the DEPLOYED status stays single-source, issue #36)."""
    model_id, v1 = _promoted_model_version(client, admin_token)
    deployed = client.post(
        f"/api/v1/models/{model_id}/versions/{v1}/deploy",
        headers=auth_header(admin_token),
    )
    assert deployed.status_code == 200

    model_id2, v2 = _evaluated_model_version(client, admin_token)
    assert model_id2 == model_id and v2 != v1

    staged = client.post(
        f"/api/v1/models/{model_id}/versions/{v2}/deploy-staging",
        json={"rationale": "stage v2"},
        headers=auth_header(admin_token),
    )
    assert staged.status_code == 201
    assert _lineage(client, admin_token, model_id, v2)["status"] == "STAGING"
    assert _lineage(client, admin_token, model_id, v1)["status"] == "DEPLOYED"
    assert _pointer(client, admin_token, model_id)["current_deployed_version"] == v1


def test_ladder_steps_require_admin(client, admin_token, user_token):
    """Issue #69 AC authorization: deploy/validate/promote are controlled human actions."""
    model_id, version = _evaluated_model_version(client, admin_token)
    url = f"/api/v1/models/{model_id}/versions/{version}"
    for step in ("deploy-staging", "validate-staging", "promote-production"):
        response = client.post(
            f"{url}/{step}",
            json={"decided_by": "alice", "rationale": "no admin rights"},
            headers=auth_header(user_token),
        )
        assert response.status_code == 403, f"{step} did not require admin"


def test_rollback_deployment_responds_404_when_missing(client, admin_token):
    response = client.post(
        "/api/v1/deployments/no-such-deployment/rollback",
        json={"decided_by": "reviewer-1", "rationale": "n/a"},
        headers=auth_header(admin_token),
    )
    assert response.status_code == 404
    assert response.json()["error"]["code"] == "DEPLOYMENT_NOT_FOUND"


def test_rollback_deployment_restores_previous_production_version(client, admin_token):
    """Issue #70 + PRD §14.4: production v1 -> production v2, v2 fails -> roll back to the
    immutable previous version via its deployment row. v1 is RETIRED (superseded) -> a legal
    rollback target, and the row's recorded environment is carried into the ROLLBACK decision."""
    model_id, v1 = _promoted_model_version(client, admin_token)
    h = auth_header(admin_token)
    client.post(f"/api/v1/models/{model_id}/versions/{v1}/deploy", headers=h)
    model_id2, v2 = _evaluated_model_version(client, admin_token)
    assert model_id2 == model_id and v2 != v1
    client.post(
        f"/api/v1/models/{model_id}/versions/{v2}/decisions",
        json={"decision": "PROMOTED", "rationale": "ship v2"},
        headers=h,
    )
    client.post(f"/api/v1/models/{model_id}/versions/{v2}/deploy", headers=h)
    assert _pointer(client, admin_token, model_id)["current_deployed_version"] == v2

    v1_row = _first_deployment_row(client, model_id)
    assert v1_row.model_version == v1
    response = client.post(
        f"/api/v1/deployments/{v1_row.deployment_id}/rollback",
        json={"decided_by": "reviewer-1", "rationale": "v2 degraded accuracy"},
        headers=h,
    )
    assert response.status_code == 201
    body = response.json()
    assert body["decision"] == "ROLLBACK"
    assert body["rollback_of_version"] == v1
    assert body["evidence_snapshot"] is None

    assert _lineage(client, admin_token, model_id, v1)["status"] == "DEPLOYED"
    assert _lineage(client, admin_token, model_id, v2)["status"] == "RETIRED"
    assert _pointer(client, admin_token, model_id)["current_deployed_version"] == v1

    row = _latest_deployment_row(client, model_id)
    assert row.model_version == v1
    assert row.environment == v1_row.environment


def test_rollback_of_ladder_production_deploy_restores_superseded_version(
    client, admin_token
):
    """Issue #70 against the new ladder: v1 (legacy prod) -> v2 (ladder prod). v2 fails -> the
    superseded v1 deployment row restores production to v1."""
    model_id, v1 = _promoted_model_version(client, admin_token)
    h = auth_header(admin_token)
    client.post(f"/api/v1/models/{model_id}/versions/{v1}/deploy", headers=h)

    model_id2, v2 = _evaluated_model_version(client, admin_token)
    assert model_id2 == model_id and v2 != v1
    url = f"/api/v1/models/{model_id}/versions/{v2}"
    client.post(f"{url}/deploy-staging", json={"rationale": "stage v2"}, headers=h)
    client.post(f"{url}/validate-staging", json={"rationale": "ok"}, headers=h)
    client.post(f"{url}/promote-production", json={"rationale": "ship v2"}, headers=h)
    assert _pointer(client, admin_token, model_id)["current_deployed_version"] == v2
    assert _production_deployment_row(client, model_id).model_version == v2

    v1_row = _first_deployment_row(client, model_id)
    response = client.post(
        f"/api/v1/deployments/{v1_row.deployment_id}/rollback",
        json={"rationale": "v2 degraded"},
        headers=h,
    )
    assert response.status_code == 201
    assert response.json()["decision"] == "ROLLBACK"
    assert _pointer(client, admin_token, model_id)["current_deployed_version"] == v1


def test_rollback_deployment_409_when_target_row_version_is_current_live(
    client, admin_token
):
    """The version currently DEPLOYED is not a rollback target (rolling back to the live version
    is a no-op self-deploy) - mirrors the existing /models/{model_id}/rollback contract."""
    model_id, version = _promoted_model_version(client, admin_token)
    h = auth_header(admin_token)
    client.post(f"/api/v1/models/{model_id}/versions/{version}/deploy", headers=h)
    row = _first_deployment_row(client, model_id)

    response = client.post(
        f"/api/v1/deployments/{row.deployment_id}/rollback",
        json={"rationale": "n/a"},
        headers=h,
    )
    assert response.status_code == 409
    assert response.json()["error"]["code"] == "ROLLBACK_NOT_ALLOWED"
