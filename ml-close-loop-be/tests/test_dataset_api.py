import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import Session
from sqlalchemy.pool import StaticPool

from app.db.base import Base
from app.db.session import get_db
from app.main import app


@pytest.fixture
def client():
    engine = create_engine("sqlite://", connect_args={"check_same_thread": False}, poolclass=StaticPool)
    Base.metadata.create_all(engine)

    def override_get_db():
        with Session(engine) as session:
            yield session

    app.dependency_overrides[get_db] = override_get_db
    yield TestClient(app)
    app.dependency_overrides.clear()
    engine.dispose()


CREATE_REQUEST = {
    "source_type": "huggingface",
    "source_dataset": "HuggingFaceH4/no_robots",
    "source_commit_or_snapshot_date": "2026-08-01",
    "source_format": "chatml",
}


def test_list_datasets_empty(client):
    response = client.get("/datasets")
    assert response.status_code == 200
    assert response.json() == []


def test_create_dataset_version_returns_201_with_body(client):
    response = client.post("/datasets/no_robots/versions", json=CREATE_REQUEST)

    assert response.status_code == 201
    body = response.json()
    assert body["dataset_id"] == "no_robots"
    assert body["version"] == 1
    assert body["status"] == "PROCESSING"
    assert body["manifest"]["source_format"] == "chatml"


def test_create_dataset_version_rejects_invalid_body(client):
    response = client.post("/datasets/no_robots/versions", json={"source_type": "bogus"})
    assert response.status_code == 422


def test_list_datasets_reflects_latest_version_and_status(client):
    client.post("/datasets/no_robots/versions", json=CREATE_REQUEST)
    client.post("/datasets/no_robots/versions", json=CREATE_REQUEST)

    response = client.get("/datasets")

    assert response.status_code == 200
    assert response.json() == [{"dataset_id": "no_robots", "latest_version": 2, "status": "PROCESSING"}]


def test_list_dataset_versions_returns_404_error_envelope_when_missing(client):
    response = client.get("/datasets/missing/versions")

    assert response.status_code == 404
    assert response.json() == {"error": {"code": "DATASET_NOT_FOUND", "message": 'dataset_id "missing" not found'}}


def test_list_dataset_versions_returns_all_versions(client):
    client.post("/datasets/no_robots/versions", json=CREATE_REQUEST)
    client.post("/datasets/no_robots/versions", json=CREATE_REQUEST)

    response = client.get("/datasets/no_robots/versions")

    assert response.status_code == 200
    assert [v["version"] for v in response.json()] == [2, 1]


def test_get_dataset_version_returns_404_error_envelope_when_missing(client):
    response = client.get("/datasets/no_robots/versions/1")

    assert response.status_code == 404
    assert response.json()["error"]["code"] == "DATASET_NOT_FOUND"


def test_get_dataset_version_returns_manifest(client):
    client.post("/datasets/no_robots/versions", json=CREATE_REQUEST)

    response = client.get("/datasets/no_robots/versions/1")

    assert response.status_code == 200
    body = response.json()
    assert body["version"] == 1
    assert body["manifest"]["source_url_or_hf_id"] == "HuggingFaceH4/no_robots"
