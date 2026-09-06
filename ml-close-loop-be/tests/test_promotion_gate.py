"""Eval gate for PROMOTED decisions (issue #43): documented, non-invented checks that a promotion
must clear before it can be recorded (model-promotion-approval-workflow.md §6/§11/§13)."""

from tests.conftest import auth_header
from tests.test_promotion_api import _evaluated_model_version


def _decision(client, admin_token, model_id, version, decision="PROMOTED"):
    return client.post(
        f"/api/v1/models/{model_id}/versions/{version}/decisions",
        json={
            "decision": decision,
            "decided_by": "reviewer-1",
            "rationale": "n/a",
        },
        headers=auth_header(admin_token),
    )


def test_promotion_blocked_without_eval_set_reference(client, admin_token):
    h = auth_header(admin_token)
    model_id, version = _evaluated_model_version(client, admin_token, eval_set_id=None)

    response = _decision(client, admin_token, model_id, version)

    assert response.status_code == 409
    assert response.json()["error"]["code"] == "PROMOTION_GATE_BLOCKED"
    assert "eval set reference" in response.json()["error"]["message"]

    lineage = client.get(
        f"/api/v1/models/{model_id}/versions/{version}", headers=h
    ).json()
    assert lineage["status"] == "EVALUATED"


def test_promotion_blocked_without_majority_win(client, admin_token):
    model_id, version = _evaluated_model_version(client, admin_token, wins=5, losses=13)

    response = _decision(client, admin_token, model_id, version)

    assert response.status_code == 409
    assert response.json()["error"]["code"] == "PROMOTION_GATE_BLOCKED"
    assert "majority win" in response.json()["error"]["message"]


def test_promotion_blocked_on_general_domain_regressions(client, admin_token):
    model_id, version = _evaluated_model_version(
        client, admin_token, regressions_found=[{"category": "reasoning"}]
    )

    response = _decision(client, admin_token, model_id, version)

    assert response.status_code == 409
    assert response.json()["error"]["code"] == "PROMOTION_GATE_BLOCKED"
    assert "regression" in response.json()["error"]["message"]


def test_promotion_blocked_when_eval_loss_worse_than_previous(client, admin_token):
    model_id, version = _evaluated_model_version(
        client, admin_token, this_version_eval_loss=0.9, previous_version_eval_loss=0.5
    )

    response = _decision(client, admin_token, model_id, version)

    assert response.status_code == 409
    assert response.json()["error"]["code"] == "PROMOTION_GATE_BLOCKED"
    assert "eval loss is worse" in response.json()["error"]["message"]


def test_promotion_passes_gate_with_clean_signals(client, admin_token):
    model_id, version = _evaluated_model_version(client, admin_token)

    response = _decision(client, admin_token, model_id, version)

    assert response.status_code == 201
    body = response.json()
    assert body["decision"] == "PROMOTED"
    assert body["eval_set_id"] == "domain-benchmark"
    assert body["eval_set_version"] == 1


def test_rejection_is_never_blocked_by_the_gate(client, admin_token):
    model_id, version = _evaluated_model_version(client, admin_token, eval_set_id=None)

    response = _decision(client, admin_token, model_id, version, decision="REJECTED")

    assert response.status_code == 201
    assert response.json()["decision"] == "REJECTED"
