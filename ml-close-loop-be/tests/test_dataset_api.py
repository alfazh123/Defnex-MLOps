from tests.conftest import auth_header

CREATE_REQUEST = {
    "source_type": "huggingface",
    "source_dataset": "HuggingFaceH4/no_robots",
    "source_commit_or_snapshot_date": "2026-08-01",
    "source_format": "chatml",
}


def test_list_datasets_empty(client, admin_token):
    response = client.get("/api/v1/datasets", headers=auth_header(admin_token))
    assert response.status_code == 200
    data = response.json()
    assert data["items"] == []
    assert data["total"] == 0


def test_create_dataset_version_returns_201_with_body(client, admin_token):
    response = client.post(
        "/api/v1/datasets/no_robots/versions",
        json=CREATE_REQUEST,
        headers=auth_header(admin_token),
    )

    assert response.status_code == 201
    body = response.json()
    assert body["dataset_id"] == "no_robots"
    assert body["version"] == 1
    assert body["status"] == "PROCESSED"
    assert body["manifest"]["source_format"] == "chatml"


def test_create_dataset_version_rejects_invalid_body(client, admin_token):
    response = client.post(
        "/api/v1/datasets/no_robots/versions",
        json={"source_type": "bogus"},
        headers=auth_header(admin_token),
    )
    assert response.status_code == 422


def test_list_datasets_reflects_latest_version_and_status(client, admin_token):
    h = auth_header(admin_token)
    client.post("/api/v1/datasets/no_robots/versions", json=CREATE_REQUEST, headers=h)
    client.post("/api/v1/datasets/no_robots/versions", json=CREATE_REQUEST, headers=h)

    response = client.get("/api/v1/datasets", headers=h)

    assert response.status_code == 200
    data = response.json()
    assert data["items"] == [
        {"dataset_id": "no_robots", "latest_version": 2, "status": "PROCESSED"}
    ]
    assert data["total"] == 1


def test_list_dataset_versions_returns_404_error_envelope_when_missing(
    client, admin_token
):
    response = client.get(
        "/api/v1/datasets/missing/versions", headers=auth_header(admin_token)
    )

    assert response.status_code == 404
    body = response.json()
    assert body["error"]["code"] == "DATASET_NOT_FOUND"
    assert body["error"]["message"] == 'dataset_id "missing" not found'
    assert isinstance(body["error"]["request_id"], str)


def test_list_dataset_versions_returns_all_versions(client, admin_token):
    h = auth_header(admin_token)
    client.post("/api/v1/datasets/no_robots/versions", json=CREATE_REQUEST, headers=h)
    client.post("/api/v1/datasets/no_robots/versions", json=CREATE_REQUEST, headers=h)

    response = client.get("/api/v1/datasets/no_robots/versions", headers=h)

    assert response.status_code == 200
    assert [v["version"] for v in response.json()] == [2, 1]


def test_get_dataset_version_returns_404_error_envelope_when_missing(
    client, admin_token
):
    response = client.get(
        "/api/v1/datasets/no_robots/versions/1", headers=auth_header(admin_token)
    )

    assert response.status_code == 404
    assert response.json()["error"]["code"] == "DATASET_NOT_FOUND"


def test_get_dataset_version_returns_manifest(client, admin_token):
    h = auth_header(admin_token)
    client.post("/api/v1/datasets/no_robots/versions", json=CREATE_REQUEST, headers=h)

    response = client.get("/api/v1/datasets/no_robots/versions/1", headers=h)

    assert response.status_code == 200
    body = response.json()
    assert body["version"] == 1
    assert body["manifest"]["source_url_or_hf_id"] == "HuggingFaceH4/no_robots"


def test_create_dataset_version_invalid_source_type_returns_422(client, admin_token):
    response = client.post(
        "/api/v1/datasets/no_robots/versions",
        json={
            "source_type": "s3",
            "source_dataset": "bucket/data",
            "source_format": "chatml",
        },
        headers=auth_header(admin_token),
    )

    assert response.status_code == 422


def test_create_dataset_version_missing_required_field_returns_422(client, admin_token):
    response = client.post(
        "/api/v1/datasets/no_robots/versions",
        json={
            "source_type": "huggingface",
            "source_dataset": "HuggingFaceH4/no_robots",
        },
        headers=auth_header(admin_token),
    )

    assert response.status_code == 422


def test_get_dataset_version_returns_correct_manifest(client, admin_token):
    h = auth_header(admin_token)
    client.post("/api/v1/datasets/no_robots/versions", json=CREATE_REQUEST, headers=h)

    response = client.get("/api/v1/datasets/no_robots/versions/1", headers=h)

    assert response.status_code == 200
    manifest = response.json()["manifest"]
    assert manifest["source_url_or_hf_id"] == "HuggingFaceH4/no_robots"
    assert manifest["source_commit_or_snapshot_date"] == "2026-08-01"
    assert manifest["source_format"] == "chatml"
    assert manifest["cleaning_steps_applied"] == []
    assert manifest["created_at"] is not None


def test_list_dataset_versions_empty_after_delete(client, admin_token):
    from sqlalchemy import select
    from sqlalchemy.orm import Session

    from app.models.dataset import DatasetVersion

    h = auth_header(admin_token)
    client.post("/api/v1/datasets/no_robots/versions", json=CREATE_REQUEST, headers=h)
    with Session(client.engine) as session:
        row = session.scalar(
            select(DatasetVersion).where(DatasetVersion.dataset_id == "no_robots")
        )
        session.delete(row)
        session.commit()

    response = client.get("/api/v1/datasets/no_robots/versions", headers=h)

    assert response.status_code == 404
    assert response.json()["error"]["code"] == "DATASET_NOT_FOUND"
