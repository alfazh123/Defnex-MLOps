import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import Session
from sqlalchemy.pool import StaticPool

from app.db.base import Base
from app.db.session import get_db
from app.main import app

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


@pytest.fixture
def client():
    engine = create_engine("sqlite://", connect_args={"check_same_thread": False}, poolclass=StaticPool)
    Base.metadata.create_all(engine)

    def override_get_db():
        with Session(engine) as session:
            yield session

    app.dependency_overrides[get_db] = override_get_db
    test_client = TestClient(app)
    test_client.engine = engine
    yield test_client
    app.dependency_overrides.clear()
    engine.dispose()


def test_create_training_run_returns_404_when_dataset_version_missing(client):
    response = client.post("/training-runs", json=TRAINING_RUN_CREATE_REQUEST)

    assert response.status_code == 404
    assert response.json()["error"]["code"] == "DATASET_NOT_FOUND"


def test_create_training_run_returns_201_queued(client):
    client.post("/datasets/no_robots/versions", json=DATASET_CREATE_REQUEST)

    response = client.post("/training-runs", json=TRAINING_RUN_CREATE_REQUEST)

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


def test_create_training_run_rejects_invalid_body(client):
    response = client.post("/training-runs", json={"dataset_id": "no_robots"})

    assert response.status_code == 422


def test_get_training_run_returns_404_when_missing(client):
    response = client.get("/training-runs/run-doesnotexist")

    assert response.status_code == 404
    assert response.json()["error"]["code"] == "TRAINING_RUN_NOT_FOUND"


def test_get_training_run_returns_created_run(client):
    client.post("/datasets/no_robots/versions", json=DATASET_CREATE_REQUEST)
    created = client.post("/training-runs", json=TRAINING_RUN_CREATE_REQUEST).json()

    response = client.get(f"/training-runs/{created['training_run_id']}")

    assert response.status_code == 200
    assert response.json() == created
