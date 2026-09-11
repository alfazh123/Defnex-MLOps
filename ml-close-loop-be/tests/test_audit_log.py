"""Audit log tests (issue #129, follow-up to closed #86 whose audit AC was never met).

Each critical action listed in the issue - login, promote/reject, deploy/rollback, retry
training, infra config / credential_ref change - must produce exactly one `audit_logs` row with
the right actor/action/resource/before/after/result/reason. Assertions filter by `resource_id`
(and sometimes `action`) rather than asserting a global row count, since fixtures like
`admin_token`/`user_token` themselves log in and would otherwise add unrelated rows.
"""

from sqlalchemy import select
from sqlalchemy.orm import Session

import pytest

from app.models.audit_log import AuditLog
from tests.conftest import auth_header
from tests.test_compute_resource_api import _resource_payload
from tests.test_deployment_api import (
    DATASET_CREATE_REQUEST,
    TRAINING_RUN_CREATE_REQUEST,
    _promoted_model_version,
)
from tests.test_training_api import _failed_run_via_api


@pytest.fixture(autouse=True)
def _disable_promotion_gates(monkeypatch):
    """Mirrors test_deployment_api.py's fixture - these tests exercise the audit trail, not the
    #43 eval gate."""
    from app.config import settings

    for name in (
        "eval_gate_require_eval_set_reference",
        "eval_gate_require_qualitative_majority",
        "eval_gate_require_no_general_regression",
        "eval_gate_require_eval_loss_not_worse",
    ):
        monkeypatch.setattr(settings, name, False)


def _audit_rows(client, *, action=None, resource_id=None) -> list[AuditLog]:
    with Session(client.engine) as db:
        query = select(AuditLog)
        if action is not None:
            query = query.where(AuditLog.action == action)
        if resource_id is not None:
            query = query.where(AuditLog.resource_id == resource_id)
        return list(db.scalars(query).all())


def test_login_success_records_one_audit_row(client, admin_token):
    rows = _audit_rows(client, action="LOGIN", resource_id="admin")
    assert len(rows) == 1
    row = rows[0]
    assert row.result == "SUCCESS"
    assert row.actor_id is not None
    assert row.resource_type == "user"


def test_login_failure_records_one_audit_row(client):
    client.post(
        "/api/v1/auth/register",
        json={"username": "bob", "password": "Bob12345", "role": "user"},
    )
    resp = client.post(
        "/api/v1/auth/login", json={"username": "bob", "password": "wrong-password"}
    )
    assert resp.status_code == 401

    rows = _audit_rows(client, action="LOGIN", resource_id="bob")
    assert len(rows) == 1
    assert rows[0].result == "FAILURE"
    assert rows[0].reason == "invalid credentials"


def _pass_validation(client, admin_token):
    from app.models.dataset import DatasetVersion

    h = auth_header(admin_token)
    with Session(client.engine) as session:
        row = session.scalar(
            select(DatasetVersion).where(
                DatasetVersion.dataset_id == "no_robots",
                DatasetVersion.version == 1,
            )
        )
        row.status = "PROCESSED"
        session.commit()
    client.post(
        "/api/v1/datasets/no_robots/versions/1/validate",
        json={
            "records": [
                {
                    "id": "r1",
                    "messages": [
                        {"role": "user", "content": "hi"},
                        {
                            "role": "assistant",
                            "content": " ".join(f"w{i}" for i in range(25)),
                        },
                    ],
                    "metadata": {"source_dataset": "no_robots", "source_id": "sq-1"},
                }
            ]
        },
        headers=h,
    )


def _evaluated_model_version(client, admin_token):
    from app.schemas.model import (
        EvalLossTrend,
        EvaluationUpdateRequest,
        GeneralDomainRegressionCheck,
        QualitativeComparison,
    )
    from app.services import model_service, training_service
    from app.workers.mock_runner import MockTrainingRunner

    h = auth_header(admin_token)
    client.post(
        "/api/v1/datasets/no_robots/versions", json=DATASET_CREATE_REQUEST, headers=h
    )
    _pass_validation(client, admin_token)
    created = client.post(
        "/api/v1/training-runs", json=TRAINING_RUN_CREATE_REQUEST, headers=h
    ).json()
    with Session(client.engine) as db:
        training_run = training_service.get_training_run(db, created["training_run_id"])
        training_service.start_training_run(db, training_run)
        artifact_uri = MockTrainingRunner().run(db, training_run)
        training_service.complete_training_run(
            db, training_run, artifact_uri=artifact_uri
        )
        model_version = model_service.register_model_version(db, training_run)
        model_service.submit_evaluation(
            db,
            model_version,
            EvaluationUpdateRequest(
                eval_loss_trend=EvalLossTrend(this_version_eval_loss=0.84),
                qualitative_comparison=QualitativeComparison(
                    question_table_version=1, wins=13, losses=5, ties=2, total=20
                ),
                general_domain_regression_check=GeneralDomainRegressionCheck(
                    checked=True, regressions_found=[]
                ),
            ),
        )
        db.commit()
        return model_version.model_id, model_version.version


def test_promote_decision_records_one_audit_row(client, admin_token):
    model_id, version = _evaluated_model_version(client, admin_token)
    resource_id = f"{model_id}:v{version}"
    h = auth_header(admin_token)

    resp = client.post(
        f"/api/v1/models/{model_id}/versions/{version}/decisions",
        json={
            "decision": "PROMOTED",
            "decided_by": "reviewer-1",
            "rationale": "signals aligned",
        },
        headers=h,
    )
    assert resp.status_code == 201

    rows = _audit_rows(client, action="PROMOTE", resource_id=resource_id)
    assert len(rows) == 1
    row = rows[0]
    assert row.result == "SUCCESS"
    assert row.before_json == {"status": "EVALUATED"}
    assert row.after_json["status"] == "PROMOTED"
    assert row.reason == "signals aligned"
    assert row.actor_id is not None


def test_reject_decision_records_one_audit_row(client, admin_token):
    model_id, version = _evaluated_model_version(client, admin_token)
    resource_id = f"{model_id}:v{version}"
    h = auth_header(admin_token)

    resp = client.post(
        f"/api/v1/models/{model_id}/versions/{version}/decisions",
        json={
            "decision": "REJECTED",
            "decided_by": "reviewer-1",
            "rationale": "not good enough",
        },
        headers=h,
    )
    assert resp.status_code == 201

    rows = _audit_rows(client, action="REJECT", resource_id=resource_id)
    assert len(rows) == 1
    assert rows[0].after_json["status"] == "REJECTED"


def test_deploy_records_one_audit_row(client, admin_token):
    model_id, version = _promoted_model_version(client, admin_token)
    resource_id = f"{model_id}:v{version}"
    h = auth_header(admin_token)

    resp = client.post(
        f"/api/v1/models/{model_id}/versions/{version}/deploy", headers=h
    )
    assert resp.status_code == 200

    rows = _audit_rows(client, action="DEPLOY", resource_id=resource_id)
    assert len(rows) == 1
    row = rows[0]
    assert row.result == "SUCCESS"
    assert row.after_json["model_id"] == model_id
    assert row.after_json["version"] == version
    assert row.actor_id is not None


def test_rollback_records_one_audit_row_distinct_from_deploy(client, admin_token):
    h = auth_header(admin_token)
    model_id, v1 = _promoted_model_version(client, admin_token)
    client.post(f"/api/v1/models/{model_id}/versions/{v1}/deploy", headers=h)
    _, v2 = _promoted_model_version(client, admin_token)
    client.post(f"/api/v1/models/{model_id}/versions/{v2}/deploy", headers=h)

    resp = client.post(
        f"/api/v1/models/{model_id}/rollback",
        json={
            "rollback_of_version": v1,
            "decided_by": "reviewer-1",
            "rationale": "v2 regressed in prod",
        },
        headers=h,
    )
    assert resp.status_code == 201

    resource_id = f"{model_id}:v{v1}"
    rollback_rows = _audit_rows(client, action="ROLLBACK", resource_id=resource_id)
    assert len(rollback_rows) == 1
    assert rollback_rows[0].reason == "v2 regressed in prod"

    # Rollback goes through deployment_service.deploy internally, but that call must be recorded
    # once, as ROLLBACK - not as a second, redundant DEPLOY row. The single DEPLOY row that does
    # exist for v1 is from the earlier, separate direct-deploy call above.
    deploy_rows_for_v1 = _audit_rows(client, action="DEPLOY", resource_id=resource_id)
    assert len(deploy_rows_for_v1) == 1


def test_retry_training_records_one_audit_row(client, admin_token):
    failed_id = _failed_run_via_api(client, admin_token)
    h = auth_header(admin_token)

    resp = client.post(f"/api/v1/training-runs/{failed_id}/retry", headers=h)
    assert resp.status_code == 201

    rows = _audit_rows(client, action="RETRY_TRAINING", resource_id=failed_id)
    assert len(rows) == 1
    row = rows[0]
    assert row.before_json == {"status": "FAILED"}
    assert row.after_json["new_training_run_id"] == resp.json()["training_run_id"]
    assert row.actor_id is not None


def test_create_compute_resource_records_one_audit_row(client, admin_token):
    h = auth_header(admin_token)
    resp = client.post("/api/v1/compute-resources", json=_resource_payload(), headers=h)
    assert resp.status_code == 201
    resource_id = str(resp.json()["id"])

    rows = _audit_rows(client, action="INFRA_CONFIG_CREATE", resource_id=resource_id)
    assert len(rows) == 1
    assert rows[0].after_json["credential_ref"] == "secret://gpu-server-key"


def test_update_compute_resource_credential_ref_records_one_audit_row(
    client, admin_token
):
    h = auth_header(admin_token)
    created = client.post(
        "/api/v1/compute-resources", json=_resource_payload(), headers=h
    ).json()
    resource_id = str(created["id"])

    resp = client.patch(
        f"/api/v1/compute-resources/{created['id']}",
        json={"credential_ref": "secret://rotated-key"},
        headers=h,
    )
    assert resp.status_code == 200

    rows = _audit_rows(client, action="INFRA_CONFIG_UPDATE", resource_id=resource_id)
    assert len(rows) == 1
    row = rows[0]
    assert row.before_json == {"credential_ref": "secret://gpu-server-key"}
    assert row.after_json == {"credential_ref": "secret://rotated-key"}


def test_delete_compute_resource_records_one_audit_row(client, admin_token):
    h = auth_header(admin_token)
    created = client.post(
        "/api/v1/compute-resources", json=_resource_payload(), headers=h
    ).json()
    resource_id = str(created["id"])

    resp = client.delete(f"/api/v1/compute-resources/{created['id']}", headers=h)
    assert resp.status_code == 204

    rows = _audit_rows(client, action="INFRA_CONFIG_DELETE", resource_id=resource_id)
    assert len(rows) == 1
    assert rows[0].before_json["credential_ref"] == "secret://gpu-server-key"


def test_audit_logs_endpoint_requires_admin(client, admin_token, user_token):
    resp = client.get("/api/v1/audit-logs", headers=auth_header(user_token))
    assert resp.status_code == 403


def test_audit_logs_endpoint_filters_by_action_and_paginates(client, admin_token):
    h = auth_header(admin_token)
    client.post("/api/v1/compute-resources", json=_resource_payload(), headers=h)
    client.post(
        "/api/v1/compute-resources", json=_resource_payload(name="server-3"), headers=h
    )

    resp = client.get("/api/v1/audit-logs?action=INFRA_CONFIG_CREATE&size=1", headers=h)
    assert resp.status_code == 200
    body = resp.json()
    assert body["total"] == 2
    assert len(body["items"]) == 1
    assert body["items"][0]["action"] == "INFRA_CONFIG_CREATE"


def test_audit_logs_endpoint_filters_by_actor(client, admin_token):
    h = auth_header(admin_token)
    # Resolve the real admin id from a LOGIN row rather than guessing (fixtures may vary).
    login_rows = _audit_rows(client, action="LOGIN", resource_id="admin")
    admin_id = login_rows[0].actor_id

    client.post("/api/v1/compute-resources", json=_resource_payload(), headers=h)

    resp = client.get(f"/api/v1/audit-logs?actor_id={admin_id}", headers=h)
    assert resp.status_code == 200
    body = resp.json()
    assert body["total"] >= 1
    assert all(item["actor_id"] == admin_id for item in body["items"])
