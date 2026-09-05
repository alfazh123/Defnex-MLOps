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


def test_create_training_run_returns_404_when_dataset_version_missing(
    client, admin_token
):
    response = client.post(
        "/api/v1/training-runs",
        json=TRAINING_RUN_CREATE_REQUEST,
        headers=auth_header(admin_token),
    )

    assert response.status_code == 404
    assert response.json()["error"]["code"] == "DATASET_NOT_FOUND"


def test_create_training_run_returns_201_queued(client, admin_token):
    h = auth_header(admin_token)
    client.post(
        "/api/v1/datasets/no_robots/versions", json=DATASET_CREATE_REQUEST, headers=h
    )

    response = client.post(
        "/api/v1/training-runs", json=TRAINING_RUN_CREATE_REQUEST, headers=h
    )

    assert response.status_code == 201
    body = response.json()
    assert body["training_run_id"].startswith("run-")
    assert body["status"] == "PENDING"
    assert body["dataset_id"] == "no_robots"
    assert body["dataset_version"] == 1
    assert body["model_id"] == "qwen-sft-domain-x"
    assert body["base_model"] == "Qwen/Qwen3.8-27B"
    assert body["training_config"]["peft_method"] == "dora"
    assert body["model_version"] is None


def test_create_training_run_rejects_invalid_body(client, admin_token):
    response = client.post(
        "/api/v1/training-runs",
        json={"dataset_id": "no_robots"},
        headers=auth_header(admin_token),
    )

    assert response.status_code == 422


def test_get_training_run_returns_404_when_missing(client, admin_token):
    response = client.get(
        "/api/v1/training-runs/run-doesnotexist", headers=auth_header(admin_token)
    )

    assert response.status_code == 404
    assert response.json()["error"]["code"] == "TRAINING_RUN_NOT_FOUND"


def test_get_training_run_returns_created_run(client, admin_token):
    h = auth_header(admin_token)
    client.post(
        "/api/v1/datasets/no_robots/versions", json=DATASET_CREATE_REQUEST, headers=h
    )
    created = client.post(
        "/api/v1/training-runs", json=TRAINING_RUN_CREATE_REQUEST, headers=h
    ).json()

    response = client.get(
        f"/api/v1/training-runs/{created['training_run_id']}", headers=h
    )

    assert response.status_code == 200
    assert response.json() == created


def test_create_training_run_does_not_start_training_inline(client, admin_token):
    """Create must only queue the run (PENDING); the worker owns execution."""
    from unittest.mock import AsyncMock, patch

    h = auth_header(admin_token)
    client.post(
        "/api/v1/datasets/no_robots/versions", json=DATASET_CREATE_REQUEST, headers=h
    )

    start_training = AsyncMock(return_value={"job_id": "job-123"})
    with patch(
        "app.api.training.unsloth_client.start_training", new=start_training
    ) as mocked:
        created = client.post(
            "/api/v1/training-runs", json=TRAINING_RUN_CREATE_REQUEST, headers=h
        ).json()

    assert created["status"] == "PENDING"
    mocked.assert_not_called()


def test_create_training_run_does_not_set_artifact_uri(client, admin_token):
    """artifact_uri stays NULL on create; only the worker populates it on COMPLETED."""
    h = auth_header(admin_token)
    client.post(
        "/api/v1/datasets/no_robots/versions", json=DATASET_CREATE_REQUEST, headers=h
    )
    created = client.post(
        "/api/v1/training-runs", json=TRAINING_RUN_CREATE_REQUEST, headers=h
    ).json()

    from sqlalchemy import select
    from sqlalchemy.orm import Session
    from app.models.training import TrainingRun

    with Session(client.engine) as db:
        row = db.scalar(
            select(TrainingRun).where(
                TrainingRun.training_run_id == created["training_run_id"]
            )
        )
        assert row.status == "PENDING"
        assert row.artifact_uri is None


def _create_runs(client, admin_token, count):
    h = auth_header(admin_token)
    client.post(
        "/api/v1/datasets/no_robots/versions", json=DATASET_CREATE_REQUEST, headers=h
    )
    for _ in range(count):
        client.post(
            "/api/v1/training-runs", json=TRAINING_RUN_CREATE_REQUEST, headers=h
        )


def test_create_training_run_permissive_model_id(client, admin_token):
    # Gap: TrainingRunCreateRequest.model_id is an unvalidated str, so any format registers.
    h = auth_header(admin_token)
    client.post(
        "/api/v1/datasets/no_robots/versions", json=DATASET_CREATE_REQUEST, headers=h
    )
    request = dict(TRAINING_RUN_CREATE_REQUEST)
    request["model_id"] = "bad model!@#/with spaces"

    response = client.post("/api/v1/training-runs", json=request, headers=h)

    assert response.status_code == 201
    assert response.json()["model_id"] == "bad model!@#/with spaces"


def test_create_training_run_returns_correct_status_field(client, admin_token):
    h = auth_header(admin_token)
    client.post(
        "/api/v1/datasets/no_robots/versions", json=DATASET_CREATE_REQUEST, headers=h
    )
    created = client.post(
        "/api/v1/training-runs", json=TRAINING_RUN_CREATE_REQUEST, headers=h
    ).json()

    fetched = client.get(
        f"/api/v1/training-runs/{created['training_run_id']}", headers=h
    )

    assert fetched.status_code == 200
    assert fetched.json()["status"] == "PENDING"
    assert fetched.json()["current_epoch"] is None


def test_list_training_runs_empty(client, admin_token):
    response = client.get("/api/v1/training-runs", headers=auth_header(admin_token))

    assert response.status_code == 200
    body = response.json()
    assert body["items"] == []
    assert body["total"] == 0


def test_list_training_runs_paginated_response_shape(client, admin_token):
    _create_runs(client, admin_token, count=2)

    response = client.get(
        "/api/v1/training-runs",
        params={"page": 1, "size": 1},
        headers=auth_header(admin_token),
    )

    assert response.status_code == 200
    body = response.json()
    assert set(body.keys()) == {"items", "total", "page", "size", "pages"}
    assert len(body["items"]) == 1
    assert body["total"] == 2
    assert body["page"] == 1
    assert body["size"] == 1
    assert body["pages"] == 2
