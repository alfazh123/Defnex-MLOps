from sqlalchemy import select

from app.models.dataset import DatasetVersion
from tests.conftest import auth_header

CREATE_REQUEST = {
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


def _create_eval_set(client, admin_token, eval_set_id="domain-benchmark", records=None):
    """Create eval set version 1 and return its id/version."""
    resp = client.post(
        f"/api/v1/eval-sets/{eval_set_id}/versions",
        json={"records": records or []},
        headers=auth_header(admin_token),
    )
    assert resp.status_code == 201, resp.text
    return resp.json()["version"]


def test_validate_returns_404_when_dataset_version_missing(client, admin_token):
    response = client.post(
        "/api/v1/datasets/no_robots/versions/1/validate",
        headers=auth_header(admin_token),
    )

    assert response.status_code == 404
    assert response.json()["error"]["code"] == "DATASET_NOT_FOUND"


def test_validate_runs_on_newly_created_version(client, admin_token):
    """Dataset versions are born PROCESSED (no separate intake step), so validation runs directly."""
    h = auth_header(admin_token)
    client.post("/api/v1/datasets/no_robots/versions", json=CREATE_REQUEST, headers=h)

    response = client.post(
        "/api/v1/datasets/no_robots/versions/1/validate",
        json={"records": [_valid_record()]},
        headers=h,
    )

    assert response.status_code == 201
    assert response.json()["gate_decision"] == "PASS"


def test_validate_returns_201_report_when_processed(client, admin_token):
    h = auth_header(admin_token)
    client.post("/api/v1/datasets/no_robots/versions", json=CREATE_REQUEST, headers=h)
    _mark_processed(client)

    response = client.post(
        "/api/v1/datasets/no_robots/versions/1/validate",
        json={"records": [_valid_record()]},
        headers=h,
    )

    assert response.status_code == 201
    body = response.json()
    assert body["dataset_id"] == "no_robots"
    assert body["dataset_version"] == 1
    assert body["record_count"] == 1
    assert body["content_hash"]
    assert body["status_counts"] == {"VALID": 1, "INVALID": 0, "NEEDS_REVIEW": 0}
    assert body["gate_decision"] == "PASS"


def test_validate_accepts_rule_set_version_override(client, admin_token):
    h = auth_header(admin_token)
    client.post("/api/v1/datasets/no_robots/versions", json=CREATE_REQUEST, headers=h)
    _mark_processed(client)

    response = client.post(
        "/api/v1/datasets/no_robots/versions/1/validate",
        json={"rule_set_version": "9.9.9", "records": [_valid_record()]},
        headers=h,
    )

    assert response.status_code == 201
    assert response.json()["rule_set_version"] == "9.9.9"


def test_validate_with_valid_records_reports_real_counts(client, admin_token):
    h = auth_header(admin_token)
    client.post("/api/v1/datasets/no_robots/versions", json=CREATE_REQUEST, headers=h)
    _mark_processed(client)
    records = [
        _valid_record("r1", user="What is the capital of France?"),
        _valid_record("r2", user="What is 2+2?"),
        _valid_record("r3", user="Apa itu quantum computing?"),
    ]

    response = client.post(
        "/api/v1/datasets/no_robots/versions/1/validate",
        json={"records": records},
        headers=h,
    )

    assert response.status_code == 201
    body = response.json()
    assert body["record_count"] == 3
    assert body["content_hash"]
    assert body["status_counts"] == {"VALID": 3, "INVALID": 0, "NEEDS_REVIEW": 0}
    assert body["per_record_errors"] == [[], [], []]
    assert body["gate_decision"] == "PASS"


def test_validate_without_records_returns_422(client, admin_token):
    h = auth_header(admin_token)
    client.post("/api/v1/datasets/no_robots/versions", json=CREATE_REQUEST, headers=h)
    _mark_processed(client)

    response = client.post("/api/v1/datasets/no_robots/versions/1/validate", headers=h)

    assert response.status_code == 422
    assert response.json()["error"]["code"] == "VALIDATION_RECORDS_REQUIRED"
    listing = client.get(
        "/api/v1/datasets/no_robots/versions/1/validation-reports", headers=h
    )
    assert listing.json() == []


def test_validate_with_empty_records_returns_422(client, admin_token):
    h = auth_header(admin_token)
    client.post("/api/v1/datasets/no_robots/versions", json=CREATE_REQUEST, headers=h)
    _mark_processed(client)

    response = client.post(
        "/api/v1/datasets/no_robots/versions/1/validate",
        json={"records": []},
        headers=h,
    )

    assert response.status_code == 422
    assert response.json()["error"]["code"] == "VALIDATION_RECORDS_REQUIRED"
    listing = client.get(
        "/api/v1/datasets/no_robots/versions/1/validation-reports", headers=h
    )
    assert listing.json() == []


def test_content_hash_is_stable_and_sensitive_to_content(client, admin_token):
    h = auth_header(admin_token)
    client.post("/api/v1/datasets/no_robots/versions", json=CREATE_REQUEST, headers=h)
    _mark_processed(client)

    first = client.post(
        "/api/v1/datasets/no_robots/versions/1/validate",
        json={"records": [_valid_record("r1")]},
        headers=h,
    ).json()
    second = client.post(
        "/api/v1/datasets/no_robots/versions/1/validate",
        json={"records": [_valid_record("r1")]},
        headers=h,
    ).json()
    changed = client.post(
        "/api/v1/datasets/no_robots/versions/1/validate",
        json={"records": [_valid_record("r1"), _valid_record("r2")]},
        headers=h,
    ).json()

    assert first["content_hash"] == second["content_hash"]
    assert first["content_hash"] != changed["content_hash"]


def test_validate_with_hard_errors_reports_invalid_and_per_record_codes(
    client, admin_token
):
    h = auth_header(admin_token)
    client.post("/api/v1/datasets/no_robots/versions", json=CREATE_REQUEST, headers=h)
    _mark_processed(client)
    bad = {
        "id": "bad1",
        "messages": [
            {"role": "user", "content": "   "},
            {"role": "assistant", "content": "short"},
        ],
        # no metadata.source_dataset / source_id -> H1
    }

    response = client.post(
        "/api/v1/datasets/no_robots/versions/1/validate",
        json={"records": [bad]},
        headers=h,
    )

    assert response.status_code == 201
    body = response.json()
    assert body["record_count"] == 1
    assert body["status_counts"] == {"VALID": 0, "INVALID": 1, "NEEDS_REVIEW": 0}
    # H1 (missing metadata) + H3 (empty/whitespace content) + H4 (assistant too short)
    assert all(
        code in body["per_record_errors"][0]
        for code in (
            "H1_missing_required_field",
            "H3_empty_content",
            "H4_below_min_length",
        )
    )


def test_validate_marks_exact_duplicate_pair(client, admin_token):
    h = auth_header(admin_token)
    client.post("/api/v1/datasets/no_robots/versions", json=CREATE_REQUEST, headers=h)
    _mark_processed(client)
    first = _valid_record("r1")
    # Identical (user, assistant) content pair => second occurrence is marked duplicate.
    records = [first, _valid_record("r2", user="What is 2+2?"), dict(first)]

    response = client.post(
        "/api/v1/datasets/no_robots/versions/1/validate",
        json={"records": records},
        headers=h,
    )

    assert response.status_code == 201
    body = response.json()
    assert body["status_counts"] == {"VALID": 2, "INVALID": 1, "NEEDS_REVIEW": 0}
    assert body["per_record_errors"] == [[], [], ["H7_duplicate"]]
    assert body["dataset_statistics"]["duplicate_count"] == 1


def test_validate_with_eval_overlap_fails_gate(client, admin_token):
    h = auth_header(admin_token)
    client.post("/api/v1/datasets/no_robots/versions", json=CREATE_REQUEST, headers=h)
    _mark_processed(client)
    leaked = _valid_record("r1", user="pertanyaan rahasia")
    eval_record = {"messages": [{"role": "user", "content": "pertanyaan rahasia"}]}

    response = client.post(
        "/api/v1/datasets/no_robots/versions/1/validate",
        json={"records": [leaked], "eval_records": [eval_record]},
        headers=h,
    )

    assert response.status_code == 201
    body = response.json()
    assert body["status_counts"]["INVALID"] == 1
    assert "H8_leakage" in body["per_record_errors"][0]
    assert body["gate_decision"] == "FAIL"
    assert body["dataset_statistics"]["leakage_check"]["overlaps_found"] == 1


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


def test_validate_references_stored_eval_set_and_reports_leakage(client, admin_token):
    h = auth_header(admin_token)
    client.post("/api/v1/datasets/no_robots/versions", json=CREATE_REQUEST, headers=h)
    _mark_processed(client)
    eval_version = _create_eval_set(
        client,
        admin_token,
        records=[{"messages": [{"role": "user", "content": "pertanyaan rahasia"}]}],
    )

    leaked = _valid_record("r1", user="pertanyaan rahasia")
    response = client.post(
        "/api/v1/datasets/no_robots/versions/1/validate",
        json={
            "records": [leaked],
            "eval_set_id": "domain-benchmark",
            "eval_set_version": eval_version,
        },
        headers=h,
    )

    assert response.status_code == 201
    body = response.json()
    assert "H8_leakage" in body["per_record_errors"][0]
    assert body["gate_decision"] == "FAIL"
    assert body["dataset_statistics"]["leakage_check"]["checked_against"] == [
        "domain-benchmark@1"
    ]
    assert body["dataset_statistics"]["leakage_check"]["overlaps_found"] == 1


def test_validate_returns_404_for_missing_eval_set(client, admin_token):
    h = auth_header(admin_token)
    client.post("/api/v1/datasets/no_robots/versions", json=CREATE_REQUEST, headers=h)
    _mark_processed(client)

    response = client.post(
        "/api/v1/datasets/no_robots/versions/1/validate",
        json={"records": [_valid_record()], "eval_set_id": "nope"},
        headers=h,
    )

    assert response.status_code == 404
    assert response.json()["error"]["code"] == "EVAL_SET_NOT_FOUND"


def test_validate_uses_latest_eval_set_version_by_default(client, admin_token):
    h = auth_header(admin_token)
    client.post("/api/v1/datasets/no_robots/versions", json=CREATE_REQUEST, headers=h)
    _mark_processed(client)
    _create_eval_set(client, admin_token, records=[])
    second = client.post(
        "/api/v1/eval-sets/domain-benchmark/versions",
        json={"records": [{"messages": [{"role": "user", "content": "rahasia-2"}]}]},
        headers=h,
    )
    assert second.status_code == 201
    assert second.json()["version"] == 2

    leaked = _valid_record("r1", user="rahasia-2")
    response = client.post(
        "/api/v1/datasets/no_robots/versions/1/validate",
        json={"records": [leaked], "eval_set_id": "domain-benchmark"},
        headers=h,
    )

    assert response.status_code == 201
    body = response.json()
    assert body["gate_decision"] == "FAIL"
    assert body["dataset_statistics"]["leakage_check"]["checked_against"] == [
        "domain-benchmark@2"
    ]


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
    client.post(
        "/api/v1/datasets/no_robots/versions/1/validate",
        json={"records": [_valid_record()]},
        headers=h,
    )
    client.post(
        "/api/v1/datasets/no_robots/versions/1/validate",
        json={"records": [_valid_record()]},
        headers=h,
    )

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
        json={"rule_set_version": "1.0.0", "records": [_valid_record()]},
        headers=h,
    )
    client.post(
        "/api/v1/datasets/no_robots/versions/1/validate",
        json={"rule_set_version": "2.0.0", "records": [_valid_record()]},
        headers=h,
    )

    response = client.get(
        "/api/v1/datasets/no_robots/versions/1/validation-reports/latest", headers=h
    )

    assert response.status_code == 200
    assert response.json()["rule_set_version"] == "2.0.0"
