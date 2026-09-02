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


def _evaluated_model_version(client, admin_token):
    """Drive a training run to COMPLETED, register it, and submit all 3 evaluation signals."""
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
        model_id, version = model_version.model_id, model_version.version

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
    return model_id, version


def test_create_decision_returns_404_when_missing(client, admin_token):
    response = client.post(
        "/api/v1/models/no-such-model/versions/1/decisions",
        json={"decision": "PROMOTED", "decided_by": "reviewer-1", "rationale": "n/a"},
        headers=auth_header(admin_token),
    )

    assert response.status_code == 404
    assert response.json()["error"]["code"] == "MODEL_NOT_FOUND"


def test_create_decision_promotes_and_returns_evidence_snapshot(client, admin_token):
    model_id, version = _evaluated_model_version(client, admin_token)
    h = auth_header(admin_token)

    response = client.post(
        f"/api/v1/models/{model_id}/versions/{version}/decisions",
        json={
            "decision": "PROMOTED",
            "decided_by": "reviewer-1",
            "rationale": "All three signals aligned.",
        },
        headers=h,
    )

    assert response.status_code == 201
    body = response.json()
    assert body["model_id"] == model_id
    assert body["version"] == version
    assert body["decision"] == "PROMOTED"
    assert body["decided_by"] == "reviewer-1"
    assert body["evidence_snapshot"]["qualitative_comparison"]["wins"] == 13
    assert body["rollback_of_version"] is None

    lineage = client.get(
        f"/api/v1/models/{model_id}/versions/{version}", headers=h
    ).json()
    assert lineage["status"] == "PROMOTED"
    assert lineage["promotion_decision_ref"] == body["decision_id"]


def test_create_decision_returns_409_when_not_evaluated(client, admin_token):
    from app.services import training_service
    from app.workers.mock_runner import MockTrainingRunner
    from app.services import model_service

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
        model_id, version = model_version.model_id, model_version.version

    response = client.post(
        f"/api/v1/models/{model_id}/versions/{version}/decisions",
        json={"decision": "PROMOTED", "decided_by": "reviewer-1", "rationale": "n/a"},
        headers=h,
    )

    assert response.status_code == 409
    assert response.json()["error"]["code"] == "DECISION_NOT_ALLOWED"


def test_create_decision_returns_409_when_already_decided(client, admin_token):
    model_id, version = _evaluated_model_version(client, admin_token)
    h = auth_header(admin_token)
    url = f"/api/v1/models/{model_id}/versions/{version}/decisions"
    client.post(
        url,
        json={"decision": "PROMOTED", "decided_by": "reviewer-1", "rationale": "first"},
        headers=h,
    )

    response = client.post(
        url,
        json={
            "decision": "REJECTED",
            "decided_by": "reviewer-1",
            "rationale": "second",
        },
        headers=h,
    )

    assert response.status_code == 409
    assert response.json()["error"]["code"] == "DECISION_NOT_ALLOWED"


def _promoted_model_version(client, admin_token):
    model_id, version = _evaluated_model_version(client, admin_token)
    client.post(
        f"/api/v1/models/{model_id}/versions/{version}/decisions",
        json={
            "decision": "PROMOTED",
            "decided_by": "reviewer-1",
            "rationale": "All three signals aligned.",
        },
        headers=auth_header(admin_token),
    )
    return model_id, version


def test_rollback_returns_404_when_missing(client, admin_token):
    response = client.post(
        "/api/v1/models/no-such-model/rollback",
        json={
            "rollback_of_version": 1,
            "decided_by": "reviewer-1",
            "rationale": "Prod regression.",
        },
        headers=auth_header(admin_token),
    )

    assert response.status_code == 404
    assert response.json()["error"]["code"] == "MODEL_NOT_FOUND"


def test_rollback_deploys_target_and_returns_decision_record(client, admin_token):
    model_id, version = _promoted_model_version(client, admin_token)
    h = auth_header(admin_token)

    response = client.post(
        f"/api/v1/models/{model_id}/rollback",
        json={
            "rollback_of_version": version,
            "decided_by": "reviewer-1",
            "rationale": "Prod regression.",
        },
        headers=h,
    )

    assert response.status_code == 201
    body = response.json()
    assert body["model_id"] == model_id
    assert body["decision"] == "ROLLBACK"
    assert body["rollback_of_version"] == version
    assert body["evidence_snapshot"] is None

    lineage = client.get(
        f"/api/v1/models/{model_id}/versions/{version}", headers=h
    ).json()
    assert lineage["status"] == "DEPLOYED"
    assert lineage["promotion_decision_ref"] == body["decision_id"]


def test_rollback_returns_409_when_target_not_promoted(client, admin_token):
    model_id, version = _evaluated_model_version(client, admin_token)

    response = client.post(
        f"/api/v1/models/{model_id}/rollback",
        json={
            "rollback_of_version": version,
            "decided_by": "reviewer-1",
            "rationale": "Prod regression.",
        },
        headers=auth_header(admin_token),
    )

    assert response.status_code == 409
    assert response.json()["error"]["code"] == "ROLLBACK_NOT_ALLOWED"
