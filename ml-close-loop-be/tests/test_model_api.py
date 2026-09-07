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
    """Drive a training run through to COMPLETED via the mock worker, then register it."""
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


def test_get_model_version_returns_404_when_missing(client, admin_token):
    response = client.get(
        "/api/v1/models/no-such-model/versions/1", headers=auth_header(admin_token)
    )

    assert response.status_code == 404
    assert response.json()["error"]["code"] == "MODEL_NOT_FOUND"


def test_get_model_version_returns_full_lineage(client, admin_token):
    model_id, version = _registered_model_version(client, admin_token)

    response = client.get(
        f"/api/v1/models/{model_id}/versions/{version}",
        headers=auth_header(admin_token),
    )

    assert response.status_code == 200
    body = response.json()
    assert body["model_id"] == model_id
    assert body["version"] == version
    assert body["status"] == "REGISTERED"
    assert body["training_run_id"]
    assert body["base_model"] == "Qwen/Qwen3.8-27B"
    assert body["dataset_id"] == "no_robots"
    assert body["dataset_version"] == 1
    assert body["artifacts"][0]["type"] == "adapter"
    assert body["promotion_decision_ref"] is None


def test_list_models_returns_latest_version_and_status(client, admin_token):
    model_id, version = _registered_model_version(client, admin_token)

    response = client.get("/api/v1/models", headers=auth_header(admin_token))

    assert response.status_code == 200
    body = response.json()
    assert {
        "model_id": model_id,
        "latest_version": version,
        "status": "REGISTERED",
    } in body


def test_list_models_filters_by_status(client, admin_token):
    _registered_model_version(client, admin_token)

    response = client.get(
        "/api/v1/models",
        params={"status": "PROMOTED"},
        headers=auth_header(admin_token),
    )

    assert response.status_code == 200
    assert response.json() == []


def test_get_evaluation_returns_404_when_missing(client, admin_token):
    response = client.get(
        "/api/v1/models/no-such-model/versions/1/evaluation",
        headers=auth_header(admin_token),
    )

    assert response.status_code == 404
    assert response.json()["error"]["code"] == "MODEL_NOT_FOUND"


def test_get_evaluation_is_all_null_before_any_submission(client, admin_token):
    model_id, version = _registered_model_version(client, admin_token)

    response = client.get(
        f"/api/v1/models/{model_id}/versions/{version}/evaluation",
        headers=auth_header(admin_token),
    )

    assert response.status_code == 200
    assert response.json() == {
        "eval_loss_trend": None,
        "qualitative_comparison": None,
        "general_domain_regression_check": None,
    }


def test_submit_evaluation_returns_404_when_missing(client, admin_token):
    response = client.post(
        "/api/v1/models/no-such-model/versions/1/evaluation",
        json={"eval_loss_trend": {"this_version_eval_loss": 0.84}},
        headers=auth_header(admin_token),
    )

    assert response.status_code == 404
    assert response.json()["error"]["code"] == "MODEL_NOT_FOUND"


def test_submit_evaluation_partial_payload_stays_registered(client, admin_token):
    model_id, version = _registered_model_version(client, admin_token)

    response = client.post(
        f"/api/v1/models/{model_id}/versions/{version}/evaluation",
        json={"eval_loss_trend": {"this_version_eval_loss": 0.84}},
        headers=auth_header(admin_token),
    )

    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "REGISTERED"
    assert body["evaluation"]["eval_loss_trend"]["this_version_eval_loss"] == 0.84
    assert body["evaluation"]["qualitative_comparison"] is None


def test_submit_evaluation_all_three_signals_transitions_to_evaluated(
    client, admin_token
):
    model_id, version = _registered_model_version(client, admin_token)
    url = f"/api/v1/models/{model_id}/versions/{version}/evaluation"
    h = auth_header(admin_token)

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
    response = client.post(
        url,
        json={
            "general_domain_regression_check": {
                "checked": True,
                "regressions_found": [],
            }
        },
        headers=h,
    )

    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "EVALUATED"
    assert body["evaluation"]["qualitative_comparison"]["wins"] == 13

    lineage = client.get(
        f"/api/v1/models/{model_id}/versions/{version}", headers=h
    ).json()
    assert lineage["status"] == "EVALUATED"
    assert lineage["evaluation"]["general_domain_regression_check"]["checked"] is True


def test_submit_evaluation_returns_409_once_promoted(client, admin_token):
    model_id, version = _registered_model_version(client, admin_token)

    with Session(client.engine) as db:
        from app.services import model_service

        model_version = model_service.get_model_version(db, model_id, version)
        model_version.status = "PROMOTED"
        db.commit()

    response = client.post(
        f"/api/v1/models/{model_id}/versions/{version}/evaluation",
        json={"eval_loss_trend": {"this_version_eval_loss": 0.84}},
        headers=auth_header(admin_token),
    )

    assert response.status_code == 409
    assert response.json()["error"]["code"] == "EVALUATION_NOT_EDITABLE"


def test_list_models_empty(client, admin_token):
    response = client.get("/api/v1/models", headers=auth_header(admin_token))

    assert response.status_code == 200
    assert response.json() == []
