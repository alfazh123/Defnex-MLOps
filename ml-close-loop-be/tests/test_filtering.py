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


def _create_dataset(client, headers, dataset_id="no_robots"):
    client.post(
        f"/api/v1/datasets/{dataset_id}/versions",
        json=DATASET_CREATE_REQUEST,
        headers=headers,
    )


VALID_RECORD = {
    "id": "r1",
    "messages": [
        {"role": "user", "content": "What is the capital of France?"},
        {"role": "assistant", "content": " ".join(f"word{i}" for i in range(25))},
    ],
    "metadata": {"source_dataset": "no_robots", "source_id": "sq-1"},
}


def _mark_processed(client, dataset_id="no_robots", version=1):
    from sqlalchemy import select

    from app.models.dataset import DatasetVersion

    with Session(client.engine) as session:
        row = session.scalar(
            select(DatasetVersion).where(
                DatasetVersion.dataset_id == dataset_id,
                DatasetVersion.version == version,
            )
        )
        row.status = "PROCESSED"
        session.commit()


def _pass_validation(client, headers):
    _mark_processed(client)
    response = client.post(
        "/api/v1/datasets/no_robots/versions/1/validate",
        json={"records": [VALID_RECORD]},
        headers=headers,
    )
    assert response.status_code == 201, response.text
    assert response.json()["gate_decision"] == "PASS"


def _create_training_run(client, headers, model_id="qwen-sft-domain-x"):
    _pass_validation(client, headers)
    req = {**TRAINING_RUN_CREATE_REQUEST, "model_id": model_id}
    client.post("/api/v1/training-runs", json=req, headers=headers)


# --- Dataset filtering ---


def test_list_datasets_filter_by_status(client, admin_token):
    h = auth_header(admin_token)
    _create_dataset(client, h)

    response = client.get("/api/v1/datasets", params={"status": "PROCESSED"}, headers=h)
    assert response.status_code == 200
    data = response.json()
    assert data["total"] == 1
    assert data["items"][0]["status"] == "PROCESSED"


def test_list_datasets_filter_by_status_no_match(client, admin_token):
    h = auth_header(admin_token)
    _create_dataset(client, h)

    response = client.get("/api/v1/datasets", params={"status": "FAILED"}, headers=h)
    assert response.status_code == 200
    assert response.json()["total"] == 0
    assert response.json()["items"] == []


def test_list_datasets_search_by_id(client, admin_token):
    h = auth_header(admin_token)
    _create_dataset(client, h, "no_robots")
    _create_dataset(client, h, "alpaca_cleaned")

    response = client.get("/api/v1/datasets", params={"search": "no_robots"}, headers=h)
    assert response.status_code == 200
    data = response.json()
    assert data["total"] == 1
    assert data["items"][0]["dataset_id"] == "no_robots"


def test_list_datasets_search_case_insensitive(client, admin_token):
    h = auth_header(admin_token)
    _create_dataset(client, h, "MyDataset")

    response = client.get("/api/v1/datasets", params={"search": "mydataset"}, headers=h)
    assert response.status_code == 200
    assert response.json()["total"] == 1


# --- Training run filtering ---


def test_list_training_runs_filter_by_status(client, admin_token):
    h = auth_header(admin_token)
    _create_dataset(client, h)
    _create_training_run(client, h)

    response = client.get(
        "/api/v1/training-runs", params={"status": "PENDING"}, headers=h
    )
    assert response.status_code == 200
    data = response.json()
    assert data["total"] == 1
    assert data["items"][0]["status"] == "PENDING"


def test_list_training_runs_filter_by_status_no_match(client, admin_token):
    h = auth_header(admin_token)
    _create_dataset(client, h)
    _create_training_run(client, h)

    response = client.get(
        "/api/v1/training-runs", params={"status": "COMPLETED"}, headers=h
    )
    assert response.status_code == 200
    assert response.json()["total"] == 0


def test_list_training_runs_filter_by_model(client, admin_token):
    h = auth_header(admin_token)
    _create_dataset(client, h)
    _create_training_run(client, h, model_id="qwen-sft-domain-x")
    _create_training_run(client, h, model_id="llama-finetune-y")

    response = client.get("/api/v1/training-runs", params={"model": "qwen"}, headers=h)
    assert response.status_code == 200
    data = response.json()
    assert data["total"] == 1
    assert data["items"][0]["model_id"] == "qwen-sft-domain-x"


# --- Model filtering ---


def _registered_model_version(client, admin_token):
    from app.services import model_service, training_service
    from app.workers.mock_runner import MockTrainingRunner

    h = auth_header(admin_token)
    client.post(
        "/api/v1/datasets/no_robots/versions", json=DATASET_CREATE_REQUEST, headers=h
    )
    _pass_validation(client, h)
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


def test_list_models_search_by_id(client, admin_token):
    _registered_model_version(client, admin_token)

    response = client.get(
        "/api/v1/models", params={"search": "qwen"}, headers=auth_header(admin_token)
    )
    assert response.status_code == 200
    data = response.json()
    assert len(data) == 1
    assert "qwen" in data[0]["model_id"].lower()


def test_list_models_search_no_match(client, admin_token):
    _registered_model_version(client, admin_token)

    response = client.get(
        "/api/v1/models", params={"search": "gpt4"}, headers=auth_header(admin_token)
    )
    assert response.status_code == 200
    assert response.json() == []


def test_list_models_filter_status_and_search_combined(client, admin_token):
    _registered_model_version(client, admin_token)

    response = client.get(
        "/api/v1/models",
        params={"status": "REGISTERED", "search": "qwen"},
        headers=auth_header(admin_token),
    )
    assert response.status_code == 200
    assert len(response.json()) == 1

    response = client.get(
        "/api/v1/models",
        params={"status": "PROMOTED", "search": "qwen"},
        headers=auth_header(admin_token),
    )
    assert response.status_code == 200
    assert response.json() == []
