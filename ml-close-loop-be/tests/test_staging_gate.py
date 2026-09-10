"""Staging gate for production promotion (issue #80): reject production promotion
if the model version hasn't passed STAGING→VALIDATED (PRD §26.3)."""

from sqlalchemy import select
from sqlalchemy.orm import Session

import pytest

from app.models.promotion import PromotionDecision

from tests.conftest import auth_header
from tests.test_promotion_api import _evaluated_model_version


@pytest.fixture(autouse=True)
def _disable_promotion_gates(monkeypatch):
    from app.config import settings

    for name in (
        "eval_gate_require_eval_set_reference",
        "eval_gate_require_qualitative_majority",
        "eval_gate_require_no_general_regression",
        "eval_gate_require_eval_loss_not_worse",
    ):
        monkeypatch.setattr(settings, name, False)


def _decision_count(client, admin_token, model_id, version):
    with Session(client.engine) as session:
        return session.scalar(
            select(PromotionDecision)
            .join(PromotionDecision.model_version)
            .where(
                PromotionDecision.decision == "PRODUCTION",
            )
            .count()
        )


def test_promote_production_requires_validated_staging(client, admin_token):
    """AC1: production promotion requires STAGING→VALIDATED."""
    model_id, version = _evaluated_model_version(client, admin_token)
    h = auth_header(admin_token)

    # EVALUATED status should be blocked
    response = client.post(
        f"/api/v1/models/{model_id}/versions/{version}/promote-production",
        json={"decided_by": "admin", "rationale": "jump gate"},
        headers=h,
    )
    assert response.status_code == 409
    assert response.json()["error"]["code"] in (
        "GATE_NOT_MET",
        "PRODUCTION_PROMOTION_NOT_ALLOWED",
    )


def test_promote_production_blocks_if_not_validated(client, admin_token):
    """AC2: environment gates block promotion when staging not validated."""
    model_id, version = _evaluated_model_version(client, admin_token)
    h = auth_header(admin_token)

    # Deploy to staging first (but don't validate)
    staged = client.post(
        f"/api/v1/models/{model_id}/versions/{version}/deploy-staging",
        json={"decided_by": "admin", "rationale": "stage"},
        headers=h,
    )
    assert staged.status_code == 201

    # Attempt production promotion — should be blocked (STAGING not VALIDATED)
    response = client.post(
        f"/api/v1/models/{model_id}/versions/{version}/promote-production",
        json={"decided_by": "admin", "rationale": "promote anyway"},
        headers=h,
    )
    assert response.status_code == 409
    assert response.json()["error"]["code"] == "GATE_NOT_MET"


def test_promote_production_succeeds_after_validation(client, admin_token):
    """AC2+: after STAGING→VALIDATED, production promotion succeeds."""
    model_id, version = _evaluated_model_version(client, admin_token)
    h = auth_header(admin_token)
    url = f"/api/v1/models/{model_id}/versions/{version}"

    client.post(f"{url}/deploy-staging", json={"rationale": "stage"}, headers=h)
    client.post(f"{url}/validate-staging", json={"rationale": "passed"}, headers=h)

    response = client.post(
        f"{url}/promote-production",
        json={"decided_by": "admin", "rationale": "ship"},
        headers=h,
    )
    assert response.status_code == 201
    assert response.json()["decision"] == "PRODUCTION"


def test_legacy_flow_still_works(client, admin_token):
    """AC: legacy PROMOTED status still allowed through production (backward compat)."""
    from tests.test_promotion_api import _promoted_model_version

    model_id, version = _promoted_model_version(client, admin_token)
    h = auth_header(admin_token)

    response = client.post(
        f"/api/v1/models/{model_id}/versions/{version}/promote-production",
        json={"decided_by": "admin", "rationale": "legacy promote"},
        headers=h,
    )
    assert response.status_code == 201
    assert response.json()["decision"] == "PRODUCTION"


def test_audit_trail_records_approver(client, admin_token):
    """AC3: audit trail records who approved (PromotionDecision.decided_by)."""
    model_id, version = _evaluated_model_version(client, admin_token)
    h = auth_header(admin_token)
    url = f"/api/v1/models/{model_id}/versions/{version}"

    client.post(f"{url}/deploy-staging", json={"rationale": "stage"}, headers=h)
    client.post(f"{url}/validate-staging", json={"rationale": "passed"}, headers=h)

    response = client.post(
        f"{url}/promote-production",
        json={"decided_by": "admin-approval", "rationale": "approved by admin"},
        headers=h,
    )
    assert response.status_code == 201
    body = response.json()
    assert body["decided_by"] == "admin-approval"

    # Verify PromotionDecision row exists with correct decided_by
    with Session(client.engine) as session:
        decision = session.scalar(
            select(PromotionDecision).where(
                PromotionDecision.decision_id == body["decision_id"]
            )
        )
        assert decision is not None
        assert decision.decided_by == "admin-approval"


def test_failed_candidate_cannot_be_promoted(client, admin_token):
    """AC4: rejected model versions cannot be promoted to production."""
    model_id, version = _evaluated_model_version(client, admin_token)
    h = auth_header(admin_token)

    # Reject the model
    client.post(
        f"/api/v1/models/{model_id}/versions/{version}/decisions",
        json={"decision": "REJECTED", "decided_by": "reviewer", "rationale": "no"},
        headers=h,
    )

    # Attempt production promotion — should be blocked
    response = client.post(
        f"/api/v1/models/{model_id}/versions/{version}/promote-production",
        json={"decided_by": "admin", "rationale": "promote rejected"},
        headers=h,
    )
    assert response.status_code == 409
    assert response.json()["error"]["code"] in (
        "GATE_NOT_MET",
        "PRODUCTION_PROMOTION_NOT_ALLOWED",
    )
