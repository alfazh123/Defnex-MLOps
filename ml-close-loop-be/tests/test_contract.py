"""API contract validation (openapi.yaml / mlops-api-contract.md): assert each endpoint's
response shape and status code, driven purely through the HTTP client — no service calls."""

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.models.dataset import DatasetVersion
from app.workers.mock_runner import MockTrainingRunner
from app.workers.training_worker import process_next_job
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


VALID_RECORD = {
    "id": "r1",
    "messages": [
        {"role": "user", "content": "What is the capital of France?"},
        {"role": "assistant", "content": " ".join(f"word{i}" for i in range(25))},
    ],
    "metadata": {"source_dataset": "no_robots", "source_id": "sq-1"},
}


def _mark_processed(client, dataset_id="no_robots", version=1):
    with Session(client.engine) as db:
        row = db.scalar(
            select(DatasetVersion).where(
                DatasetVersion.dataset_id == dataset_id,
                DatasetVersion.version == version,
            )
        )
        row.status = "PROCESSED"
        db.commit()


def _registered_model_version(client, admin_token):
    h = auth_header(admin_token)
    client.post(
        "/api/v1/datasets/no_robots/versions", json=DATASET_CREATE_REQUEST, headers=h
    )
    _mark_processed(client)
    report = client.post(
        "/api/v1/datasets/no_robots/versions/1/validate",
        json={"records": [VALID_RECORD]},
        headers=h,
    )
    assert report.status_code == 201
    assert report.json()["gate_decision"] == "PASS"
    client.post("/api/v1/training-runs", json=TRAINING_RUN_CREATE_REQUEST, headers=h)
    with Session(client.engine) as db:
        process_next_job(db, MockTrainingRunner())
        db.commit()
    return "qwen-sft-domain-x", 1


def _evaluated_model_version(client, admin_token):
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
    return model_id, version


def _promoted_model_version(client, admin_token):
    model_id, version = _evaluated_model_version(client, admin_token)
    client.post(
        f"/api/v1/models/{model_id}/versions/{version}/decisions",
        json={
            "decision": "PROMOTED",
            "decided_by": "reviewer-1",
            "rationale": "Signals aligned.",
        },
        headers=auth_header(admin_token),
    )
    return model_id, version


def test_health_returns_200_with_status_ok(client):
    resp = client.get("/api/v1/health")

    assert resp.status_code == 200
    assert resp.json() == {"status": "ok"}


def test_register_returns_201_with_user_response_shape(client):
    client.post(
        "/api/v1/auth/register",
        json={"username": "baseadmin", "password": "Admin1234", "role": "user"},
    )
    resp = client.post(
        "/api/v1/auth/register",
        json={"username": "fresh", "password": "Fresh1234", "role": "user"},
    )

    assert resp.status_code == 201
    data = resp.json()
    assert set(data) == {"id", "username", "role", "created_at"}
    assert isinstance(data["id"], int)
    assert data["username"] == "fresh"
    assert data["role"] == "user"
    assert isinstance(data["created_at"], str)


def test_login_returns_200_with_token_response_shape(client):
    client.post(
        "/api/v1/auth/register",
        json={"username": "user1", "password": "User1234", "role": "user"},
    )
    resp = client.post(
        "/api/v1/auth/login", json={"username": "user1", "password": "User1234"}
    )

    assert resp.status_code == 200
    data = resp.json()
    assert set(data) == {"access_token", "refresh_token", "token_type", "user"}
    assert isinstance(data["access_token"], str)
    assert isinstance(data["refresh_token"], str)
    assert data["token_type"] == "bearer"
    assert data["user"]["username"] == "user1"


def test_list_datasets_returns_paginated_shape(client, admin_token):
    resp = client.get("/api/v1/datasets", headers=auth_header(admin_token))

    assert resp.status_code == 200
    data = resp.json()
    assert set(data) == {"items", "total", "page", "size", "pages"}
    assert isinstance(data["items"], list)
    assert isinstance(data["total"], int)
    assert data["page"] == 1
    assert data["size"] == 20
    assert data["pages"] == 0


def test_create_dataset_returns_201_with_dataset_version_shape(client, admin_token):
    resp = client.post(
        "/api/v1/datasets/no_robots/versions",
        json=DATASET_CREATE_REQUEST,
        headers=auth_header(admin_token),
    )

    assert resp.status_code == 201
    data = resp.json()
    assert set(data) == {"dataset_id", "version", "status", "manifest"}
    assert data["dataset_id"] == "no_robots"
    assert isinstance(data["version"], int)
    assert data["status"] == "PROCESSED"
    assert isinstance(data["manifest"]["created_at"], str)
    assert data["manifest"]["source_format"] == "chatml"


def test_create_training_run_returns_201_with_training_run_shape(client, admin_token):
    h = auth_header(admin_token)
    client.post(
        "/api/v1/datasets/no_robots/versions", json=DATASET_CREATE_REQUEST, headers=h
    )
    _mark_processed(client)
    report = client.post(
        "/api/v1/datasets/no_robots/versions/1/validate",
        json={"records": [VALID_RECORD]},
        headers=h,
    )
    assert report.status_code == 201
    assert report.json()["gate_decision"] == "PASS"

    resp = client.post(
        "/api/v1/training-runs", json=TRAINING_RUN_CREATE_REQUEST, headers=h
    )

    assert resp.status_code == 201
    data = resp.json()
    assert isinstance(data["training_run_id"], str)
    assert data["status"] == "PENDING"
    assert data["dataset_id"] == "no_robots"
    assert data["dataset_version"] == 1
    assert isinstance(data["training_config"], dict)
    assert data["model_version"] is None


def test_list_models_returns_list_shape(client, admin_token):
    h = auth_header(admin_token)
    _registered_model_version(client, admin_token)

    resp = client.get("/api/v1/models", headers=h)

    assert resp.status_code == 200
    data = resp.json()
    assert isinstance(data, list)
    assert set(data[0]) == {"model_id", "latest_version", "status"}
    assert data[0]["model_id"] == "qwen-sft-domain-x"


def test_get_model_version_returns_full_lineage_shape(client, admin_token):
    h = auth_header(admin_token)
    model_id, version = _registered_model_version(client, admin_token)

    resp = client.get(f"/api/v1/models/{model_id}/versions/{version}", headers=h)

    assert resp.status_code == 200
    data = resp.json()
    assert set(data) == {
        "model_id",
        "version",
        "status",
        "training_run_id",
        "base_model",
        "dataset_id",
        "dataset_version",
        "dataset_validation_report_ref",
        "training_config",
        "created_at",
        "created_by",
        "evaluation",
        "artifacts",
        "promotion_decision_ref",
        "previous_model_id",
    }
    assert data["status"] == "REGISTERED"
    assert isinstance(data["training_config"], dict)
    assert isinstance(data["artifacts"], list)


def test_validation_report_returns_report_shape(client, admin_token):
    h = auth_header(admin_token)
    client.post(
        "/api/v1/datasets/no_robots/versions", json=DATASET_CREATE_REQUEST, headers=h
    )
    _mark_processed(client)

    resp = client.post(
        "/api/v1/datasets/no_robots/versions/1/validate",
        json={"records": [VALID_RECORD]},
        headers=h,
    )

    assert resp.status_code == 201
    data = resp.json()
    assert set(data) == {
        "dataset_id",
        "dataset_version",
        "rule_set_version",
        "run_at",
        "record_count",
        "status_counts",
        "warnings_summary",
        "dataset_statistics",
        "per_record_errors",
        "content_hash",
        "gate_decision",
        "gate_reason",
    }
    assert data["dataset_id"] == "no_robots"
    assert data["status_counts"] == {"VALID": 1, "INVALID": 0, "NEEDS_REVIEW": 0}
    assert data["record_count"] == 1
    assert isinstance(data["content_hash"], str)
    assert data["per_record_errors"] == [[]]
    assert data["gate_decision"] == "PASS"
    assert isinstance(data["run_at"], str)


def test_promotion_decision_returns_decision_shape(client, admin_token):
    h = auth_header(admin_token)
    model_id, version = _evaluated_model_version(client, admin_token)

    resp = client.post(
        f"/api/v1/models/{model_id}/versions/{version}/decisions",
        json={
            "decision": "PROMOTED",
            "decided_by": "reviewer-1",
            "rationale": "Signals aligned.",
        },
        headers=h,
    )

    assert resp.status_code == 201
    data = resp.json()
    assert set(data) == {
        "decision_id",
        "model_id",
        "version",
        "decision",
        "decided_by",
        "decided_at",
        "evidence_snapshot",
        "rationale",
        "rollback_of_version",
    }
    assert data["decision"] == "PROMOTED"
    assert isinstance(data["evidence_snapshot"], dict)
    assert data["rollback_of_version"] is None
    assert isinstance(data["decided_at"], str)


def test_deployment_result_returns_deploy_result_shape(client, admin_token):
    h = auth_header(admin_token)
    model_id, version = _promoted_model_version(client, admin_token)

    resp = client.post(
        f"/api/v1/models/{model_id}/versions/{version}/deploy", headers=h
    )

    assert resp.status_code == 200
    data = resp.json()
    assert set(data) == {
        "model_id",
        "current_deployed_version",
        "previous_deployed_version",
    }
    assert data["model_id"] == model_id
    assert data["current_deployed_version"] == version
    assert data["previous_deployed_version"] is None


def test_error_response_always_has_error_envelope(client, admin_token):
    h = auth_header(admin_token)
    cases = [
        ("GET", "/api/v1/datasets/missing/versions", None),
        ("POST", "/api/v1/datasets/missing/versions/1/validate", None),
        ("GET", "/api/v1/training-runs/missing", None),
        ("GET", "/api/v1/models/missing/versions/1", None),
        (
            "POST",
            "/api/v1/models/missing/versions/1/decisions",
            {"decision": "PROMOTED", "rationale": "n/a"},
        ),
        ("GET", "/api/v1/models/missing/deployment", None),
    ]

    for method, path, body in cases:
        resp = client.request(method, path, json=body, headers=h)
        assert resp.status_code == 404, path
        data = resp.json()
        assert set(data) == {"error"}, path
        assert set(data["error"]) == {"code", "message"}, path
        assert isinstance(data["error"]["code"], str), path
        assert isinstance(data["error"]["message"], str), path


def test_wrong_content_type_is_rejected(client):
    resp = client.post(
        "/api/v1/auth/register",
        content='{"username": "bob", "password": "Bob1234", "role": "admin"}',
        headers={"Content-Type": "text/plain"},
    )

    assert resp.status_code == 422


def test_pagination_max_page_size_rejected(client, admin_token):
    resp = client.get("/api/v1/datasets?size=101", headers=auth_header(admin_token))

    assert resp.status_code == 422


def test_combined_filters_work(client, admin_token):
    h = auth_header(admin_token)
    client.post(
        "/api/v1/datasets/no_robots/versions", json=DATASET_CREATE_REQUEST, headers=h
    )
    _mark_processed(client)

    resp = client.get("/api/v1/datasets?status=PROCESSED&search=robots", headers=h)

    assert resp.status_code == 200
    data = resp.json()
    assert data["total"] == 1
    assert data["items"][0]["dataset_id"] == "no_robots"
    assert data["items"][0]["status"] == "PROCESSED"
