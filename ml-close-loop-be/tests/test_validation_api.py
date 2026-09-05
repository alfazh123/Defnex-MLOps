from sqlalchemy import select

from app.models.dataset import DatasetVersion
from tests.conftest import auth_header

CREATE_REQUEST = {
    "source_type": "huggingface",
    "source_dataset": "HuggingFaceH4/no_robots",
    "source_commit_or_snapshot_date": "2026-08-01",
    "source_format": "chatml",
}


def _mark_processed(client, dataset_id="no_robots", version=1):
    from sqlalchemy.orm import Session

    with Session(client.engine) as session:
        row = session.scalar(
            select(DatasetVersion).where(
                DatasetVersion.dataset_id == dataset_id,
                DatasetVersion.version == version,
            )
        )
        row.status = "PROCESSED"
        session.commit()


def test_validate_returns_404_when_dataset_version_missing(client, admin_token):
    response = client.post(
        "/api/v1/datasets/no_robots/versions/1/validate",
        headers=auth_header(admin_token),
    )

    assert response.status_code == 404
    assert response.json()["error"]["code"] == "DATASET_NOT_FOUND"


def test_validate_returns_409_when_not_processed(client, admin_token):
    h = auth_header(admin_token)
    client.post("/api/v1/datasets/no_robots/versions", json=CREATE_REQUEST, headers=h)

    from sqlalchemy.orm import Session

    # Lifecycle decisions (which statuses exist and when versions reach PROCESSED)
    # belong to issue #35; forcing PROCESSING here only re-creates the not-yet-ready
    # state the 409 guard is meant for.
    with Session(client.engine) as session:
        row = session.scalar(
            select(DatasetVersion).where(
                DatasetVersion.dataset_id == "no_robots",
                DatasetVersion.version == 1,
            )
        )
        row.status = "PROCESSING"
        session.commit()

    response = client.post("/api/v1/datasets/no_robots/versions/1/validate", headers=h)

    assert response.status_code == 409
    assert response.json() == {
        "error": {
            "code": "VALIDATION_INCOMPLETE",
            "message": "Dataset version has not completed processing yet.",
        }
    }


def test_validate_returns_201_report_when_processed(client, admin_token):
    h = auth_header(admin_token)
    client.post("/api/v1/datasets/no_robots/versions", json=CREATE_REQUEST, headers=h)
    _mark_processed(client)

    response = client.post("/api/v1/datasets/no_robots/versions/1/validate", headers=h)

    assert response.status_code == 201
    body = response.json()
    assert body["dataset_id"] == "no_robots"
    assert body["dataset_version"] == 1
    assert body["status_counts"] == {"VALID": 0, "INVALID": 0, "NEEDS_REVIEW": 0}
    assert body["gate_decision"] == "PASS"


def test_validate_accepts_rule_set_version_override(client, admin_token):
    h = auth_header(admin_token)
    client.post("/api/v1/datasets/no_robots/versions", json=CREATE_REQUEST, headers=h)
    _mark_processed(client)

    response = client.post(
        "/api/v1/datasets/no_robots/versions/1/validate",
        json={"rule_set_version": "9.9.9"},
        headers=h,
    )

    assert response.status_code == 201
    assert response.json()["rule_set_version"] == "9.9.9"


def test_validate_rejects_invalid_body(client, admin_token):
    h = auth_header(admin_token)
    client.post("/api/v1/datasets/no_robots/versions", json=CREATE_REQUEST, headers=h)
    _mark_processed(client)

    response = client.post(
        "/api/v1/datasets/no_robots/versions/1/validate",
        json={"rule_set_version": 123},
        headers=h,
    )

    assert response.status_code == 422


def test_list_validation_reports_returns_404_when_missing(client, admin_token):
    response = client.get(
        "/api/v1/datasets/no_robots/versions/1/validation-reports",
        headers=auth_header(admin_token),
    )

    assert response.status_code == 404
    assert response.json()["error"]["code"] == "DATASET_NOT_FOUND"


def test_list_validation_reports_returns_empty_list_before_any_run(client, admin_token):
    h = auth_header(admin_token)
    client.post("/api/v1/datasets/no_robots/versions", json=CREATE_REQUEST, headers=h)

    response = client.get(
        "/api/v1/datasets/no_robots/versions/1/validation-reports", headers=h
    )

    assert response.status_code == 200
    assert response.json() == []


def test_list_validation_reports_returns_all_runs_most_recent_first(
    client, admin_token
):
    h = auth_header(admin_token)
    client.post("/api/v1/datasets/no_robots/versions", json=CREATE_REQUEST, headers=h)
    _mark_processed(client)
    client.post("/api/v1/datasets/no_robots/versions/1/validate", headers=h)
    client.post("/api/v1/datasets/no_robots/versions/1/validate", headers=h)

    response = client.get(
        "/api/v1/datasets/no_robots/versions/1/validation-reports", headers=h
    )

    assert response.status_code == 200
    assert len(response.json()) == 2


def test_get_latest_validation_report_returns_404_when_dataset_version_missing(
    client, admin_token
):
    response = client.get(
        "/api/v1/datasets/no_robots/versions/1/validation-reports/latest",
        headers=auth_header(admin_token),
    )

    assert response.status_code == 404
    assert response.json()["error"]["code"] == "DATASET_NOT_FOUND"


def test_get_latest_validation_report_returns_404_when_none_run_yet(
    client, admin_token
):
    h = auth_header(admin_token)
    client.post("/api/v1/datasets/no_robots/versions", json=CREATE_REQUEST, headers=h)

    response = client.get(
        "/api/v1/datasets/no_robots/versions/1/validation-reports/latest", headers=h
    )

    assert response.status_code == 404
    assert response.json()["error"]["code"] == "VALIDATION_REPORT_NOT_FOUND"


def test_get_latest_validation_report_returns_most_recent(client, admin_token):
    h = auth_header(admin_token)
    client.post("/api/v1/datasets/no_robots/versions", json=CREATE_REQUEST, headers=h)
    _mark_processed(client)
    client.post(
        "/api/v1/datasets/no_robots/versions/1/validate",
        json={"rule_set_version": "1.0.0"},
        headers=h,
    )
    client.post(
        "/api/v1/datasets/no_robots/versions/1/validate",
        json={"rule_set_version": "2.0.0"},
        headers=h,
    )

    response = client.get(
        "/api/v1/datasets/no_robots/versions/1/validation-reports/latest", headers=h
    )

    assert response.status_code == 200
    assert response.json()["rule_set_version"] == "2.0.0"
