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


def _promoted_model_version(client):
    model_id, version = _registered_model_version(client)
    url = f"/models/{model_id}/versions/{version}/evaluation"
    client.post(url, json={"eval_loss_trend": {"this_version_eval_loss": 0.84}})
    client.post(
        url,
        json={"qualitative_comparison": {"question_table_version": 1, "wins": 13, "losses": 5, "ties": 2, "total": 20}},
    )
    client.post(url, json={"general_domain_regression_check": {"checked": True, "regressions_found": []}})
    client.post(
        f"/models/{model_id}/versions/{version}/decisions",
        json={"decision": "PROMOTED", "decided_by": "reviewer-1", "rationale": "Signals aligned."},
    )
    return model_id, version


def test_deploy_returns_404_when_version_missing(client):
    response = client.post("/models/no-such-model/versions/1/deploy")

    assert response.status_code == 404
    assert response.json()["error"]["code"] == "MODEL_NOT_FOUND"


def test_deploy_returns_409_when_not_promoted(client):
    model_id, version = _registered_model_version(client)

    response = client.post(f"/models/{model_id}/versions/{version}/deploy")

    assert response.status_code == 409
    assert response.json()["error"]["code"] == "DEPLOY_NOT_ALLOWED"


def test_deploy_promoted_version_updates_pointer_and_registry(client):
    model_id, version = _promoted_model_version(client)

    response = client.post(f"/models/{model_id}/versions/{version}/deploy")

    assert response.status_code == 200
    assert response.json() == {
        "model_id": model_id,
        "current_deployed_version": version,
        "previous_deployed_version": None,
    }
    assert client.get(f"/models/{model_id}/versions/{version}").json()["status"] == "DEPLOYED"


def test_deploy_reports_and_retires_the_superseded_version(client):
    model_id, v1 = _promoted_model_version(client)
    client.post(f"/models/{model_id}/versions/{v1}/deploy")
    _, v2 = _promoted_model_version(client)

    response = client.post(f"/models/{model_id}/versions/{v2}/deploy")

    assert response.status_code == 200
    assert response.json()["current_deployed_version"] == v2
    assert response.json()["previous_deployed_version"] == v1
    assert client.get(f"/models/{model_id}/versions/{v1}").json()["status"] == "RETIRED"


def test_get_deployment_returns_404_when_model_missing(client):
    response = client.get("/models/no-such-model/deployment")

    assert response.status_code == 404
    assert response.json()["error"]["code"] == "MODEL_NOT_FOUND"


def test_get_deployment_returns_null_pointer_before_any_deploy(client):
    model_id, _ = _registered_model_version(client)

    response = client.get(f"/models/{model_id}/deployment")

    assert response.status_code == 200
    assert response.json() == {
        "model_id": model_id,
        "current_deployed_version": None,
        "deployed_at": None,
        "status": None,
    }


def test_get_deployment_returns_current_pointer_after_deploy(client):
    model_id, version = _promoted_model_version(client)
    client.post(f"/models/{model_id}/versions/{version}/deploy")

    response = client.get(f"/models/{model_id}/deployment")

    assert response.status_code == 200
    body = response.json()
    assert body["model_id"] == model_id
    assert body["current_deployed_version"] == version
    assert body["status"] == "DEPLOYED"
    assert body["deployed_at"] is not None


def test_rollback_moves_the_deployment_pointer_back(client):
    model_id, v1 = _promoted_model_version(client)
    client.post(f"/models/{model_id}/versions/{v1}/deploy")
    _, v2 = _promoted_model_version(client)
    client.post(f"/models/{model_id}/versions/{v2}/deploy")

    response = client.post(
        f"/models/{model_id}/rollback",
        json={"rollback_of_version": v1, "decided_by": "reviewer-1", "rationale": "Prod regression."},
    )

    assert response.status_code == 201
    # DecisionRecord.version is "the version being rolled back from".
    assert response.json()["version"] == v2
    assert client.get(f"/models/{model_id}/deployment").json()["current_deployed_version"] == v1
