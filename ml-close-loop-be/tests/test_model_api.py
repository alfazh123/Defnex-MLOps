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


def _registered_model_version(client):
    """Drive a training run through to COMPLETED via the mock worker, then register it."""
    from app.services import model_service, training_service
    from app.workers.mock_runner import MockTrainingRunner

    client.post("/datasets/no_robots/versions", json=DATASET_CREATE_REQUEST)
    created = client.post("/training-runs", json=TRAINING_RUN_CREATE_REQUEST).json()

    with Session(client.engine) as db:
        training_run = training_service.get_training_run(db, created["training_run_id"])
        training_service.start_training_run(db, training_run)
        artifact_uri = MockTrainingRunner().run(training_run)
        training_service.complete_training_run(db, training_run, artifact_uri=artifact_uri)
        model_version = model_service.register_model_version(db, training_run)
        db.commit()
        return model_version.model_id, model_version.version


def test_get_model_version_returns_404_when_missing(client):
    response = client.get("/models/no-such-model/versions/1")

    assert response.status_code == 404
    assert response.json()["error"]["code"] == "MODEL_NOT_FOUND"


def test_get_model_version_returns_full_lineage(client):
    model_id, version = _registered_model_version(client)

    response = client.get(f"/models/{model_id}/versions/{version}")

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


def test_list_models_returns_latest_version_and_status(client):
    model_id, version = _registered_model_version(client)

    response = client.get("/models")

    assert response.status_code == 200
    body = response.json()
    assert {"model_id": model_id, "latest_version": version, "status": "REGISTERED"} in body


def test_list_models_filters_by_status(client):
    _registered_model_version(client)

    response = client.get("/models", params={"status": "PROMOTED"})

    assert response.status_code == 200
    assert response.json() == []
