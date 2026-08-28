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


def _evaluated_model_version(client):
    """Drive a training run to COMPLETED, register it, and submit all 3 evaluation signals."""
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
        model_id, version = model_version.model_id, model_version.version

    url = f"/models/{model_id}/versions/{version}/evaluation"
    client.post(url, json={"eval_loss_trend": {"this_version_eval_loss": 0.84}})
    client.post(
        url,
        json={"qualitative_comparison": {"question_table_version": 1, "wins": 13, "losses": 5, "ties": 2, "total": 20}},
    )
    client.post(url, json={"general_domain_regression_check": {"checked": True, "regressions_found": []}})
    return model_id, version


def test_create_decision_returns_404_when_missing(client):
    response = client.post(
        "/models/no-such-model/versions/1/decisions",
        json={"decision": "PROMOTED", "decided_by": "reviewer-1", "rationale": "n/a"},
    )

    assert response.status_code == 404
    assert response.json()["error"]["code"] == "MODEL_NOT_FOUND"


def test_create_decision_promotes_and_returns_evidence_snapshot(client):
    model_id, version = _evaluated_model_version(client)

    response = client.post(
        f"/models/{model_id}/versions/{version}/decisions",
        json={
            "decision": "PROMOTED",
            "decided_by": "reviewer-1",
            "rationale": "All three signals aligned.",
        },
    )

    assert response.status_code == 201
    body = response.json()
    assert body["model_id"] == model_id
    assert body["version"] == version
    assert body["decision"] == "PROMOTED"
    assert body["decided_by"] == "reviewer-1"
    assert body["evidence_snapshot"]["qualitative_comparison"]["wins"] == 13
    assert body["rollback_of_version"] is None

    lineage = client.get(f"/models/{model_id}/versions/{version}").json()
    assert lineage["status"] == "PROMOTED"
    assert lineage["promotion_decision_ref"] == body["decision_id"]


def test_create_decision_returns_409_when_not_evaluated(client):
    from app.services import model_service

    client.post("/datasets/no_robots/versions", json=DATASET_CREATE_REQUEST)
    created = client.post("/training-runs", json=TRAINING_RUN_CREATE_REQUEST).json()

    from app.services import training_service
    from app.workers.mock_runner import MockTrainingRunner

    with Session(client.engine) as db:
        training_run = training_service.get_training_run(db, created["training_run_id"])
        training_service.start_training_run(db, training_run)
        artifact_uri = MockTrainingRunner().run(training_run)
        training_service.complete_training_run(db, training_run, artifact_uri=artifact_uri)
        model_version = model_service.register_model_version(db, training_run)
        db.commit()
        model_id, version = model_version.model_id, model_version.version

    response = client.post(
        f"/models/{model_id}/versions/{version}/decisions",
        json={"decision": "PROMOTED", "decided_by": "reviewer-1", "rationale": "n/a"},
    )

    assert response.status_code == 409
    assert response.json()["error"]["code"] == "DECISION_NOT_ALLOWED"


def test_create_decision_returns_409_when_already_decided(client):
    model_id, version = _evaluated_model_version(client)
    url = f"/models/{model_id}/versions/{version}/decisions"
    client.post(url, json={"decision": "PROMOTED", "decided_by": "reviewer-1", "rationale": "first"})

    response = client.post(url, json={"decision": "REJECTED", "decided_by": "reviewer-1", "rationale": "second"})

    assert response.status_code == 409
    assert response.json()["error"]["code"] == "DECISION_NOT_ALLOWED"
