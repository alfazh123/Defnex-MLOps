from unittest.mock import patch

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


def _create_run(client, admin_token):
    from sqlalchemy import select

    from app.models.dataset import DatasetVersion

    h = auth_header(admin_token)
    client.post(
        "/api/v1/datasets/no_robots/versions", json=DATASET_CREATE_REQUEST, headers=h
    )
    from sqlalchemy.orm import Session

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
        json={
            "records": [
                {
                    "id": "r1",
                    "messages": [
                        {"role": "user", "content": "What is the capital of France?"},
                        {
                            "role": "assistant",
                            "content": " ".join(f"word{i}" for i in range(25)),
                        },
                    ],
                    "metadata": {"source_dataset": "no_robots", "source_id": "sq-1"},
                }
            ]
        },
        headers=h,
    )
    assert report.status_code == 201, report.text
    created = client.post(
        "/api/v1/training-runs", json=TRAINING_RUN_CREATE_REQUEST, headers=h
    ).json()
    return created["training_run_id"], h


def test_progress_returns_404_when_run_missing(client, admin_token):
    response = client.get(
        "/api/v1/training-runs/run-doesnotexist/progress",
        headers=auth_header(admin_token),
    )
    assert response.status_code == 404
    assert response.json()["error"]["code"] == "TRAINING_RUN_NOT_FOUND"


def test_progress_requires_auth(client, admin_token):
    response = client.get("/api/v1/training-runs/run-x/progress")
    assert response.status_code == 401
    assert response.json()["error"]["code"] == "MISSING_TOKEN"


def test_progress_streams_sse_events(client, admin_token):
    run_id, h = _create_run(client, admin_token)

    async def fake_progress(run_id):
        yield '{"epoch": 1, "step": 10}'

    with patch("app.api.training.unsloth_client.stream_progress", fake_progress):
        response = client.get(f"/api/v1/training-runs/{run_id}/progress", headers=h)

    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/event-stream")
    assert response.text.startswith("data:")
    assert '{"epoch": 1, "step": 10}' in response.text


def test_progress_streams_multiple_events(client, admin_token):
    run_id, h = _create_run(client, admin_token)

    async def fake_progress(run_id):
        yield '{"epoch": 1, "step": 10}'
        yield '{"epoch": 1, "step": 20}'

    with patch("app.api.training.unsloth_client.stream_progress", fake_progress):
        response = client.get(f"/api/v1/training-runs/{run_id}/progress", headers=h)

    assert response.status_code == 200
    assert response.text.count("data:") == 2
