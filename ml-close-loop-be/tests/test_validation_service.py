from app.schemas.dataset import DatasetVersionCreateRequest
from app.services import dataset_service, validation_service

GOOD_ANSWER = " ".join(["kata"] * 25)


def _make_version(db_session):
    request = DatasetVersionCreateRequest(
        source_type="huggingface",
        source_dataset="HuggingFaceH4/no_robots",
        source_commit_or_snapshot_date="2026-08-01",
        source_format="chatml",
    )
    return dataset_service.create_dataset_version(db_session, "no_robots", request)


def _record(record_id="r1", user="What is the capital of France?", assistant=GOOD_ANSWER, **overrides):
    record = {
        "id": record_id,
        "messages": [
            {"role": "user", "content": user},
            {"role": "assistant", "content": assistant},
        ],
        "metadata": {"source_dataset": "no_robots", "source_id": record_id, "language": "en"},
    }
    record.update(overrides)
    return record


def test_all_valid_records_pass_gate(db_session):
    version = _make_version(db_session)
    records = [_record("r1"), _record("r2", user="What is 2+2?")]

    report = validation_service.validate_dataset_version(db_session, version, records)

    assert report.record_count == 2
    assert report.status_counts == {"VALID": 2, "INVALID": 0, "NEEDS_REVIEW": 0}
    assert report.gate_decision == "PASS"


def test_h1_missing_required_field_is_invalid(db_session):
    version = _make_version(db_session)
    record = _record("r1")
    record["metadata"] = {}

    report = validation_service.validate_dataset_version(db_session, version, [record])

    assert report.status_counts == {"VALID": 0, "INVALID": 1, "NEEDS_REVIEW": 0}


def test_h2_invalid_role_is_invalid(db_session):
    version = _make_version(db_session)
    record = _record("r1")
    record["messages"][0]["role"] = "narrator"

    report = validation_service.validate_dataset_version(db_session, version, [record])

    assert report.status_counts["INVALID"] == 1


def test_h3_empty_content_is_invalid(db_session):
    version = _make_version(db_session)
    record = _record("r1", assistant="   ")

    report = validation_service.validate_dataset_version(db_session, version, [record])

    assert report.status_counts["INVALID"] == 1


def test_h4_below_min_length_is_invalid(db_session):
    version = _make_version(db_session)
    record = _record("r1", assistant="too short")

    report = validation_service.validate_dataset_version(db_session, version, [record])

    assert report.status_counts["INVALID"] == 1


def test_h6_boilerplate_marker_is_invalid(db_session):
    version = _make_version(db_session)
    record = _record("r1", assistant=f"{GOOD_ANSWER} baca juga artikel terkait lainnya di situs kami")

    report = validation_service.validate_dataset_version(db_session, version, [record])

    assert report.status_counts["INVALID"] == 1


def test_h7_duplicate_marks_second_occurrence_invalid(db_session):
    version = _make_version(db_session)
    records = [_record("r1"), _record("r2")]  # identical user+assistant content

    report = validation_service.validate_dataset_version(db_session, version, records)

    assert report.status_counts == {"VALID": 1, "INVALID": 1, "NEEDS_REVIEW": 0}
    assert report.dataset_statistics["duplicate_count"] == 1


def test_h8_leakage_fails_gate_and_flags_record(db_session):
    version = _make_version(db_session)
    records = [_record("r1", user="What is the capital of France?")]
    eval_records = [_record("bench1", user="What is the capital of France?")]

    report = validation_service.validate_dataset_version(db_session, version, records, eval_records=eval_records)

    assert report.status_counts["INVALID"] == 1
    assert report.gate_decision == "FAIL"
    assert report.dataset_statistics["leakage_check"]["overlaps_found"] == 1


def test_no_leakage_source_defaults_to_zero_overlaps(db_session):
    version = _make_version(db_session)
    records = [_record("r1")]

    report = validation_service.validate_dataset_version(db_session, version, records)

    assert report.dataset_statistics["leakage_check"] == {"checked_against": [], "overlaps_found": 0}


def test_list_and_get_latest_validation_reports(db_session):
    version = _make_version(db_session)
    validation_service.validate_dataset_version(db_session, version, [_record("r1")])
    second = validation_service.validate_dataset_version(db_session, version, [_record("r1"), _record("r2")])

    reports = validation_service.list_validation_reports(db_session, version)
    latest = validation_service.get_latest_validation_report(db_session, version)

    assert len(reports) == 2
    assert latest.id == second.id


def test_to_schema_derives_dataset_id_and_version(db_session):
    version = _make_version(db_session)
    report = validation_service.validate_dataset_version(db_session, version, [_record("r1")])

    schema = validation_service.to_schema(report)

    assert schema.dataset_id == "no_robots"
    assert schema.dataset_version == 1
    assert schema.status_counts.VALID == 1
