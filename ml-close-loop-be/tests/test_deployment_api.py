from sqlalchemy.orm import Session

from tests.conftest import auth_header

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
        "peft_method": "dora",
        "load_in_4bit": False,
        "lora_r": 16,
        "lora_alpha": 16,
        "learning_rate": None,
        "epochs": 2,
        "max_seq_length": 4096,
    },
    "triggered_by": "user-1",
}


def _registered_model_version(client, admin_token):
    from app.services import model_service, training_service
    from app.workers.mock_runner import MockTrainingRunner

    h = auth_header(admin_token)
    client.post(
        "/api/v1/datasets/no_robots/versions", json=DATASET_CREATE_REQUEST, headers=h
    )
    created = client.post(
        "/api/v1/training-runs", json=TRAINING_RUN_CREATE_REQUEST, headers=h
    ).json()
    with Session(client.engine) as db:
        training_run = training_service.get_training_run(db, created["training_run_id"])
        training_service.start_training_run(db, training_run)
        artifact_uri = MockTrainingRunner().run(training_run)
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
