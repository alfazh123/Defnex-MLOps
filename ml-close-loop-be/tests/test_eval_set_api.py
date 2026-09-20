"""Eval-set storage API (issue #43): create/list/get versions, and the reverse-direction H8
leakage guard that keeps a new eval set disjoint from already-validated training data."""

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.models.dataset import DatasetVersion
from app.models.eval_set import EvalSetVersion
from app.models.validation import ValidationReport
from tests.conftest import auth_header

DATASET_CREATE_REQUEST = {
    "source_type": "huggingface",
    "source_dataset": "HuggingFaceH4/no_robots",
    "source_commit_or_snapshot_date": "2026-08-01",
    "source_format": "chatml",
}

GOOD_ANSWER = " ".join(f"word{i}" for i in range(25))


def _valid_record(record_id="r1", user="What is the capital of France?"):
    return {
        "id": record_id,
        "messages": [
            {"role": "user", "content": user},
            {"role": "assistant", "content": GOOD_ANSWER},
        ],
        "metadata": {"source_dataset": "no_robots", "source_id": record_id},
    }


def _mark_processed(client, dataset_id="no_robots", version=1):
    with Session(client.engine) as session:
        row = session.scalar(
            select(DatasetVersion).where(
                DatasetVersion.dataset_id == dataset_id,
                DatasetVersion.version == version,
            )
        )
        row.status = "PROCESSED"
        session.commit()


def _create_dataset_and_validate(client, admin_token, records):
    h = auth_header(admin_token)
    client.post(
        "/api/v1/datasets/no_robots/versions", json=DATASET_CREATE_REQUEST, headers=h
    )
    _mark_processed(client)
    response = client.post(
        "/api/v1/datasets/no_robots/versions/1/validate",
        json={"records": records},
        headers=h,
    )
    assert response.status_code == 201, response.text
    return response.json()


def test_create_eval_set_version_returns_201_with_echoed_content(client, admin_token):
    h = auth_header(admin_token)
    record = _valid_record()

    response = client.post(
        "/api/v1/eval-sets/domain-benchmark/versions",
        json={"records": [record]},
        headers=h,
    )

    assert response.status_code == 201
    body = response.json()
    assert body["eval_set_id"] == "domain-benchmark"
    assert body["version"] == 1
    assert body["record_count"] == 1
    assert body["records"] == [record]


def test_create_eval_set_version_increments_version(client, admin_token):
    h = auth_header(admin_token)
    client.post(
        "/api/v1/eval-sets/domain-benchmark/versions",
        json={"records": [_valid_record()]},
        headers=h,
    )

    response = client.post(
        "/api/v1/eval-sets/domain-benchmark/versions",
        json={"records": [_valid_record("r2")]},
        headers=h,
    )

    assert response.status_code == 201
    assert response.json()["version"] == 2


def test_create_eval_set_requires_admin(client, admin_token, user_token):
    response = client.post(
        "/api/v1/eval-sets/domain-benchmark/versions",
        json={"records": [_valid_record()]},
        headers=auth_header(user_token),
    )

    assert response.status_code == 403


def test_list_and_get_eval_set_versions(client, admin_token):
    h = auth_header(admin_token)
    client.post(
        "/api/v1/eval-sets/domain-benchmark/versions",
        json={"records": [_valid_record()]},
        headers=h,
    )
    client.post(
        "/api/v1/eval-sets/domain-benchmark/versions",
        json={"records": [_valid_record("r2")]},
        headers=h,
    )

    summary = client.get("/api/v1/eval-sets", headers=h)
    assert summary.status_code == 200
    assert summary.json()["items"] == [
        {"eval_set_id": "domain-benchmark", "latest_version": 2, "version_count": 2}
    ]

    versions = client.get("/api/v1/eval-sets/domain-benchmark/versions", headers=h)
    assert versions.status_code == 200
    assert [v["version"] for v in versions.json()] == [2, 1]

    single = client.get("/api/v1/eval-sets/domain-benchmark/versions/1", headers=h)
    assert single.status_code == 200
    assert single.json()["version"] == 1
    assert single.json()["record_count"] == 1


def test_list_eval_sets_empty(client, admin_token):
    response = client.get("/api/v1/eval-sets", headers=auth_header(admin_token))

    assert response.status_code == 200
    assert response.json()["items"] == []


def test_list_eval_sets_respects_pagination_params(client, admin_token):
    """Issue #176: GET /eval-sets is now paginated like every other list endpoint."""
    h = auth_header(admin_token)
    client.post(
        "/api/v1/eval-sets/domain-benchmark/versions",
        json={"records": [_valid_record()]},
        headers=h,
    )

    page2 = client.get("/api/v1/eval-sets", params={"page": 2, "size": 1}, headers=h)

    assert page2.status_code == 200
    body = page2.json()
    assert body["items"] == []
    assert body["total"] == 1
    assert body["page"] == 2
    assert body["size"] == 1
    assert body["pages"] == 1


def test_get_missing_eval_set_returns_404(client, admin_token):
    response = client.get(
        "/api/v1/eval-sets/nope/versions", headers=auth_header(admin_token)
    )

    assert response.status_code == 404
    assert response.json()["error"]["code"] == "EVAL_SET_NOT_FOUND"


def test_create_eval_set_version_rejects_overlap_with_validated_training(
    client, admin_token
):
    """Reverse H8: eval-set content must not duplicate already-validated training data."""
    h = auth_header(admin_token)
    report = _create_dataset_and_validate(client, admin_token, [_valid_record("t1")])
    assert report["gate_decision"] == "PASS"
    with Session(client.engine) as db:
        assert db.scalar(select(ValidationReport)).records == [_valid_record("t1")]

    overlap = _valid_record("r2", user="What is the capital of France?")
    response = client.post(
        "/api/v1/eval-sets/domain-benchmark/versions",
        json={"records": [overlap]},
        headers=h,
    )

    assert response.status_code == 409
    assert response.json()["error"]["code"] == "EVAL_SET_OVERLAP"
    with Session(client.engine) as db:
        assert db.scalar(select(EvalSetVersion)) is None


def test_create_eval_set_version_allows_records_not_in_validated_data(
    client, admin_token
):
    h = auth_header(admin_token)
    report = _create_dataset_and_validate(client, admin_token, [_valid_record("t1")])
    assert report["gate_decision"] == "PASS"

    response = client.post(
        "/api/v1/eval-sets/domain-benchmark/versions",
        json={"records": [_valid_record("e1", user="pertanyaan fresh")]},
        headers=h,
    )

    assert response.status_code == 201
    assert response.json()["version"] == 1
