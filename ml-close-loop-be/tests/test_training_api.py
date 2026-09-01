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
        "/training-runs",
        json=TRAINING_RUN_CREATE_REQUEST,
        headers=auth_header(admin_token),
    )

    assert response.status_code == 404
    assert response.json()["error"]["code"] == "DATASET_NOT_FOUND"


def test_create_training_run_returns_201_queued(client, admin_token):
    h = auth_header(admin_token)
    client.post("/datasets/no_robots/versions", json=DATASET_CREATE_REQUEST, headers=h)

    response = client.post(
        "/training-runs", json=TRAINING_RUN_CREATE_REQUEST, headers=h
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
        "/training-runs",
        json={"dataset_id": "no_robots"},
        headers=auth_header(admin_token),
    )

    assert response.status_code == 422


def test_get_training_run_returns_404_when_missing(client, admin_token):
    response = client.get(
        "/training-runs/run-doesnotexist", headers=auth_header(admin_token)
    )

    assert response.status_code == 404
    assert response.json()["error"]["code"] == "TRAINING_RUN_NOT_FOUND"


def test_get_training_run_returns_created_run(client, admin_token):
    h = auth_header(admin_token)
    client.post("/datasets/no_robots/versions", json=DATASET_CREATE_REQUEST, headers=h)
    created = client.post(
        "/training-runs", json=TRAINING_RUN_CREATE_REQUEST, headers=h
    ).json()

    response = client.get(f"/training-runs/{created['training_run_id']}", headers=h)

    assert response.status_code == 200
    assert response.json() == created
