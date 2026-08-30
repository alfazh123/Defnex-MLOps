import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine, select
from sqlalchemy.orm import Session
from sqlalchemy.pool import StaticPool

from app.db.base import Base
from app.db.session import get_db
from app.main import app
from app.models.dataset import DatasetVersion

CREATE_REQUEST = {
    "source_type": "huggingface",
    "source_dataset": "HuggingFaceH4/no_robots",
    "source_commit_or_snapshot_date": "2026-08-01",
    "source_format": "chatml",
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


def _mark_processed(client, dataset_id="no_robots", version=1):
    with Session(client.engine) as session:
        row = session.scalar(
            select(DatasetVersion).where(
                DatasetVersion.dataset_id == dataset_id, DatasetVersion.version == version
            )
        )
        row.status = "PROCESSED"
        session.commit()


def test_validate_returns_404_when_dataset_version_missing(client):
    response = client.post("/datasets/no_robots/versions/1/validate")

    assert response.status_code == 404
    assert response.json()["error"]["code"] == "DATASET_NOT_FOUND"


def test_validate_returns_409_when_not_processed(client):
    client.post("/datasets/no_robots/versions", json=CREATE_REQUEST)

    response = client.post("/datasets/no_robots/versions/1/validate")

    assert response.status_code == 409
    assert response.json() == {
        "error": {
            "code": "VALIDATION_INCOMPLETE",
            "message": "Dataset version has not completed processing yet.",
        }
    }


def test_validate_returns_201_report_when_processed(client):
    client.post("/datasets/no_robots/versions", json=CREATE_REQUEST)
    _mark_processed(client)

    response = client.post("/datasets/no_robots/versions/1/validate")

    assert response.status_code == 201
    body = response.json()
    assert body["dataset_id"] == "no_robots"
    assert body["dataset_version"] == 1
    assert body["status_counts"] == {"VALID": 0, "INVALID": 0, "NEEDS_REVIEW": 0}
    assert body["gate_decision"] == "PASS"


def test_validate_accepts_rule_set_version_override(client):
    client.post("/datasets/no_robots/versions", json=CREATE_REQUEST)
    _mark_processed(client)

    response = client.post(
        "/datasets/no_robots/versions/1/validate", json={"rule_set_version": "9.9.9"}
    )

    assert response.status_code == 201
    assert response.json()["rule_set_version"] == "9.9.9"


def test_validate_rejects_invalid_body(client):
    client.post("/datasets/no_robots/versions", json=CREATE_REQUEST)
    _mark_processed(client)

    response = client.post(
        "/datasets/no_robots/versions/1/validate", json={"rule_set_version": 123}
    )

    assert response.status_code == 422


def test_list_validation_reports_returns_404_when_missing(client):
    response = client.get("/datasets/no_robots/versions/1/validation-reports")

    assert response.status_code == 404
    assert response.json()["error"]["code"] == "DATASET_NOT_FOUND"


def test_list_validation_reports_returns_empty_list_before_any_run(client):
    client.post("/datasets/no_robots/versions", json=CREATE_REQUEST)

    response = client.get("/datasets/no_robots/versions/1/validation-reports")

    assert response.status_code == 200
    assert response.json() == []


def test_list_validation_reports_returns_all_runs_most_recent_first(client):
    client.post("/datasets/no_robots/versions", json=CREATE_REQUEST)
    _mark_processed(client)
    client.post("/datasets/no_robots/versions/1/validate")
    client.post("/datasets/no_robots/versions/1/validate")

    response = client.get("/datasets/no_robots/versions/1/validation-reports")

    assert response.status_code == 200
    assert len(response.json()) == 2


def test_get_latest_validation_report_returns_404_when_dataset_version_missing(client):
    response = client.get("/datasets/no_robots/versions/1/validation-reports/latest")

    assert response.status_code == 404
    assert response.json()["error"]["code"] == "DATASET_NOT_FOUND"


def test_get_latest_validation_report_returns_404_when_none_run_yet(client):
    client.post("/datasets/no_robots/versions", json=CREATE_REQUEST)

    response = client.get("/datasets/no_robots/versions/1/validation-reports/latest")

    assert response.status_code == 404
    assert response.json()["error"]["code"] == "VALIDATION_REPORT_NOT_FOUND"


def test_get_latest_validation_report_returns_most_recent(client):
    client.post("/datasets/no_robots/versions", json=CREATE_REQUEST)
    _mark_processed(client)
    client.post("/datasets/no_robots/versions/1/validate", json={"rule_set_version": "1.0.0"})
    client.post("/datasets/no_robots/versions/1/validate", json={"rule_set_version": "2.0.0"})

    response = client.get("/datasets/no_robots/versions/1/validation-reports/latest")

    assert response.status_code == 200
    assert response.json()["rule_set_version"] == "2.0.0"
