from sqlalchemy.orm import Session

import pytest

from tests.conftest import auth_header


@pytest.fixture(autouse=True)
def _disable_promotion_gates(monkeypatch):
    """These tests exercise deployment mechanics, not the #43 eval gate
    (gate criteria are covered by tests/test_promotion_gate.py)."""
    from app.config import settings

    for name in (
        "eval_gate_require_eval_set_reference",
        "eval_gate_require_qualitative_majority",
        "eval_gate_require_no_general_regression",
        "eval_gate_require_eval_loss_not_worse",
    ):
        monkeypatch.setattr(settings, name, False)


DATASET_CREATE_REQUEST = {
    "source_type": "huggingface",
    "source_dataset": "HuggingFaceH4/no_robots",
    "source_commit_or_snapshot_date": "2026-08-01",
    "source_format": "chatml",
}

TRAINING_RUN_CREATE_REQUEST = {
    "dataset_id": "no_robots",
    "dataset_version": 1,
    "model_id": "qwen-sft-domain-x",
    "base_model": "Qwen/Qwen3.8-27B",
    "training_config": {
        "peft_method": "lora",
        "load_in_4bit": False,
        "lora_r": 16,
        "lora_alpha": 16,
        "learning_rate": None,
        "epochs": 2,
        "max_seq_length": 4096,
    },
    "triggered_by": "user-1",
}


VALID_RECORD = {
    "id": "r1",
    "messages": [
        {"role": "user", "content": "What is the capital of France?"},
        {"role": "assistant", "content": " ".join(f"word{i}" for i in range(25))},
    ],
    "metadata": {"source_dataset": "no_robots", "source_id": "sq-1"},
}


def _pass_validation(client, admin_token):
    from sqlalchemy import select

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
    report = client.post(
        "/api/v1/datasets/no_robots/versions/1/validate",
        json={"records": [VALID_RECORD]},
        headers=h,
    )
    assert report.status_code == 201, report.text
    assert report.json()["gate_decision"] == "PASS"


def _registered_model_version(client, admin_token):
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
        db.commit()
        return model_version.model_id, model_version.version


def _promoted_model_version(client, admin_token):
    model_id, version = _registered_model_version(client, admin_token)
    h = auth_header(admin_token)
    url = f"/api/v1/models/{model_id}/versions/{version}/evaluation"
    client.post(
        url, json={"eval_loss_trend": {"this_version_eval_loss": 0.84}}, headers=h
    )
    client.post(
        url,
        json={
            "qualitative_comparison": {
                "question_table_version": 1,
                "wins": 13,
                "losses": 5,
                "ties": 2,
                "total": 20,
            }
        },
        headers=h,
    )
    client.post(
        url,
        json={
            "general_domain_regression_check": {
                "checked": True,
                "regressions_found": [],
            }
        },
        headers=h,
    )
    client.post(
        f"/api/v1/models/{model_id}/versions/{version}/decisions",
        json={
            "decision": "PROMOTED",
            "decided_by": "reviewer-1",
            "rationale": "Signals aligned.",
        },
        headers=h,
    )
    return model_id, version


def test_deploy_returns_404_when_version_missing(client, admin_token):
    response = client.post(
        "/api/v1/models/no-such-model/versions/1/deploy",
        headers=auth_header(admin_token),
    )

    assert response.status_code == 404
    assert response.json()["error"]["code"] == "MODEL_NOT_FOUND"


def test_deploy_returns_409_when_not_promoted(client, admin_token):
    model_id, version = _registered_model_version(client, admin_token)

    response = client.post(
        f"/api/v1/models/{model_id}/versions/{version}/deploy",
        headers=auth_header(admin_token),
    )

    assert response.status_code == 409
    assert response.json()["error"]["code"] == "DEPLOY_NOT_ALLOWED"


def test_deploy_promoted_version_updates_pointer_and_registry(client, admin_token):
    model_id, version = _promoted_model_version(client, admin_token)
    h = auth_header(admin_token)

    response = client.post(
        f"/api/v1/models/{model_id}/versions/{version}/deploy", headers=h
    )

    assert response.status_code == 200
    assert response.json() == {
        "model_id": model_id,
        "current_deployed_version": version,
        "previous_deployed_version": None,
    }
    assert (
        client.get(f"/api/v1/models/{model_id}/versions/{version}", headers=h).json()[
            "status"
        ]
        == "DEPLOYED"
    )


def test_deploy_to_named_environment_records_target(client, admin_token):
    """Issue #67: deploy targets a named environment; the deployment row records it."""
    model_id, version = _promoted_model_version(client, admin_token)
    h = auth_header(admin_token)

    response = client.post(
        f"/api/v1/models/{model_id}/versions/{version}/deploy",
        json={"environment": "production"},
        headers=h,
    )

    assert response.status_code == 200
    with Session(client.engine) as session:
        from sqlalchemy import select

        from app.models.deployment import Deployment

        row = session.scalars(select(Deployment)).first()
        assert row.environment == "production"


def test_deploy_unknown_environment_still_succeeds(client, admin_token):
    """Issue #67: unknown environment names are tolerated (environments are advisory metadata
    on deployment history, not a hard FK), preserving the pre-#67 permissive behavior."""
    model_id, version = _promoted_model_version(client, admin_token)
    h = auth_header(admin_token)

    response = client.post(
        f"/api/v1/models/{model_id}/versions/{version}/deploy",
        json={"environment": "canary"},
        headers=h,
    )

    assert response.status_code == 200
    assert response.json()["current_deployed_version"] == version


def test_list_environments_requires_auth(client):
    response = client.get("/api/v1/environments")

    assert response.status_code == 401
    assert response.json()["error"]["code"] == "MISSING_TOKEN"


def test_list_environments_returns_seeded_targets(client, admin_token):
    """Issue #67: GET /api/v1/environments reflects the environments table (seeded
    default/staging/production by the migration; the test DB seeds rows directly)."""
    from app.models.environment import Environment

    with Session(client.engine) as session:
        session.add_all(
            [
                Environment(name="default", description=None),
                Environment(name="staging", description="Integration validation."),
                Environment(name="production", description="Live serving."),
            ]
        )
        session.commit()

    response = client.get("/api/v1/environments", headers=auth_header(admin_token))

    assert response.status_code == 200
    assert response.json() == [
        {"name": "default", "description": None},
        {"name": "production", "description": "Live serving."},
        {"name": "staging", "description": "Integration validation."},
    ]


def test_deploy_reports_and_retires_the_superseded_version(client, admin_token):
    h = auth_header(admin_token)
    model_id, v1 = _promoted_model_version(client, admin_token)
    client.post(f"/api/v1/models/{model_id}/versions/{v1}/deploy", headers=h)
    _, v2 = _promoted_model_version(client, admin_token)

    response = client.post(f"/api/v1/models/{model_id}/versions/{v2}/deploy", headers=h)

    assert response.status_code == 200
    assert response.json()["current_deployed_version"] == v2
    assert response.json()["previous_deployed_version"] == v1
    assert (
        client.get(f"/api/v1/models/{model_id}/versions/{v1}", headers=h).json()[
            "status"
        ]
        == "RETIRED"
    )


def test_get_deployment_returns_404_when_model_missing(client, admin_token):
    response = client.get(
        "/api/v1/models/no-such-model/deployment", headers=auth_header(admin_token)
    )

    assert response.status_code == 404
    assert response.json()["error"]["code"] == "MODEL_NOT_FOUND"


def test_get_deployment_returns_null_pointer_before_any_deploy(client, admin_token):
    model_id, _ = _registered_model_version(client, admin_token)

    response = client.get(
        f"/api/v1/models/{model_id}/deployment", headers=auth_header(admin_token)
    )

    assert response.status_code == 200
    assert response.json() == {
        "model_id": model_id,
        "current_deployed_version": None,
        "deployed_at": None,
        "status": None,
    }


def test_get_deployment_returns_current_pointer_after_deploy(client, admin_token):
    model_id, version = _promoted_model_version(client, admin_token)
    h = auth_header(admin_token)
    client.post(f"/api/v1/models/{model_id}/versions/{version}/deploy", headers=h)

    response = client.get(f"/api/v1/models/{model_id}/deployment", headers=h)

    assert response.status_code == 200
    body = response.json()
    assert body["model_id"] == model_id
    assert body["current_deployed_version"] == version
    assert body["status"] == "DEPLOYED"
    assert body["deployed_at"] is not None


def test_rollback_moves_the_deployment_pointer_back(client, admin_token):
    h = auth_header(admin_token)
    model_id, v1 = _promoted_model_version(client, admin_token)
    client.post(f"/api/v1/models/{model_id}/versions/{v1}/deploy", headers=h)
    _, v2 = _promoted_model_version(client, admin_token)
    client.post(f"/api/v1/models/{model_id}/versions/{v2}/deploy", headers=h)

    response = client.post(
        f"/api/v1/models/{model_id}/rollback",
        json={
            "rollback_of_version": v1,
            "decided_by": "reviewer-1",
            "rationale": "Prod regression.",
        },
        headers=h,
    )

    assert response.status_code == 201
    # DecisionRecord.version is "the version being rolled back from".
    assert response.json()["version"] == v2
    assert (
        client.get(f"/api/v1/models/{model_id}/deployment", headers=h).json()[
            "current_deployed_version"
        ]
        == v1
    )


def test_deploy_requires_admin_role(client, admin_token, user_token):
    response = client.post(
        "/api/v1/models/no-such-model/versions/1/deploy",
        headers=auth_header(user_token),
    )

    assert response.status_code == 403
    assert response.json()["error"]["code"] == "FORBIDDEN"


def test_deploy_replays_cached_result_for_same_idempotency_key(client, admin_token):
    """Issue #124: the deploy endpoint's `X-Idempotency-Key` is now backed by the durable
    `idempotency_keys` table (app.services.idempotency_service) instead of an in-process dict.
    Sending the same key twice must return the exact same body and must not create a second
    `Deployment` row / re-run the deploy (a real request replay, not just "a 200 both times")."""
    from sqlalchemy import select

    from app.models.deployment import Deployment

    model_id, version = _promoted_model_version(client, admin_token)
    h = {**auth_header(admin_token), "X-Idempotency-Key": "deploy-replay-key"}

    resp1 = client.post(
        f"/api/v1/models/{model_id}/versions/{version}/deploy", headers=h
    )
    assert resp1.status_code == 200, resp1.text

    resp2 = client.post(
        f"/api/v1/models/{model_id}/versions/{version}/deploy", headers=h
    )
    assert resp2.status_code == 200, resp2.text
    assert resp2.json() == resp1.json()

    with Session(client.engine) as session:
        deployments = session.scalars(
            select(Deployment).where(Deployment.model_id == model_id)
        ).all()
        assert len(deployments) == 1, (
            "a replayed idempotent request must not create a second Deployment row"
        )


def test_deploy_idempotency_cache_survives_a_fresh_session(client, admin_token):
    """Same guarantee as above, but explicitly checked against a brand-new `Session` (no
    connection/object reused from the request that wrote it) - the point of #124 is that the
    cache is a DB row lookup, not a variable that would reset on a worker restart."""
    from sqlalchemy import select as sa_select

    from app.models.idempotency import IdempotencyKey

    model_id, version = _promoted_model_version(client, admin_token)
    h = {**auth_header(admin_token), "X-Idempotency-Key": "deploy-restart-key"}

    resp1 = client.post(
        f"/api/v1/models/{model_id}/versions/{version}/deploy", headers=h
    )
    assert resp1.status_code == 200, resp1.text

    with Session(client.engine) as fresh:
        rows = fresh.scalars(
            sa_select(IdempotencyKey).where(
                IdempotencyKey.endpoint == "deploy_model_version"
            )
        ).all()
        assert len(rows) == 1

    resp2 = client.post(
        f"/api/v1/models/{model_id}/versions/{version}/deploy", headers=h
    )
    assert resp2.status_code == 200, resp2.text
    assert resp2.json() == resp1.json()
