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


def _record(
    record_id="r1",
    user="What is the capital of France?",
    assistant=GOOD_ANSWER,
    **overrides,
):
    record = {
        "id": record_id,
        "messages": [
            {"role": "user", "content": user},
            {"role": "assistant", "content": assistant},
        ],
        "metadata": {
            "source_dataset": "no_robots",
            "source_id": record_id,
            "language": "en",
        },
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
    record = _record(
        "r1", assistant=f"{GOOD_ANSWER} baca juga artikel terkait lainnya di situs kami"
    )

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

    report = validation_service.validate_dataset_version(
        db_session, version, records, eval_records=eval_records
    )

    assert report.status_counts["INVALID"] == 1
    assert report.gate_decision == "FAIL"
    assert report.dataset_statistics["leakage_check"]["overlaps_found"] == 1


def test_no_leakage_source_defaults_to_zero_overlaps(db_session):
    version = _make_version(db_session)
    records = [_record("r1")]

    report = validation_service.validate_dataset_version(db_session, version, records)

    assert report.dataset_statistics["leakage_check"] == {
        "checked_against": [],
        "overlaps_found": 0,
    }


def test_list_and_get_latest_validation_reports(db_session):
    version = _make_version(db_session)
    validation_service.validate_dataset_version(db_session, version, [_record("r1")])
    second = validation_service.validate_dataset_version(
        db_session, version, [_record("r1"), _record("r2")]
    )

    reports = validation_service.list_validation_reports(db_session, version)
    latest = validation_service.get_latest_validation_report(db_session, version)

    assert len(reports) == 2
    assert latest.id == second.id


def test_to_schema_derives_dataset_id_and_version(db_session):
    version = _make_version(db_session)
    report = validation_service.validate_dataset_version(
        db_session, version, [_record("r1")]
    )

    schema = validation_service.to_schema(report)

    assert schema.dataset_id == "no_robots"
    assert schema.dataset_version == 1
    assert schema.status_counts.VALID == 1


def test_empty_records_pass_gate(db_session):
    version = _make_version(db_session)

    report = validation_service.validate_dataset_version(db_session, version, [])

    assert report.record_count == 0
    assert report.status_counts == {"VALID": 0, "INVALID": 0, "NEEDS_REVIEW": 0}
    assert report.gate_decision == "PASS"


def test_validate_with_custom_rule_set_version(db_session):
    version = _make_version(db_session)

    report = validation_service.validate_dataset_version(
        db_session, version, [_record("r1")], rule_set_version="9.9.9"
    )

    assert report.rule_set_version == "9.9.9"


def test_multiple_validations_produce_separate_reports(db_session):
    version = _make_version(db_session)
    first = validation_service.validate_dataset_version(
        db_session, version, [_record("r1")]
    )
    second = validation_service.validate_dataset_version(
        db_session, version, [_record("r2")]
    )

    reports = validation_service.list_validation_reports(db_session, version)

    assert len(reports) == 2
    assert {r.id for r in reports} == {first.id, second.id}


# --- PII screening (issue #132): warning-only, must never reject a record ---------------


def test_pii_email_produces_warning_not_rejection(db_session):
    version = _make_version(db_session)
    record = _record(
        "r1",
        user=f"{GOOD_ANSWER} hubungi saya di budi.santoso@example.com untuk detail",
    )

    report = validation_service.validate_dataset_version(db_session, version, [record])

    # Still VALID/PASS - a PII match is a warning, not a hard-error.
    assert report.status_counts == {"VALID": 1, "INVALID": 0, "NEEDS_REVIEW": 0}
    assert report.gate_decision == "PASS"
    assert report.per_record_errors == [[]]
    assert report.warnings_summary["PII_EMAIL"] == 1
    assert report.warnings_summary["total_warnings"] == 1
    assert report.dataset_statistics["pii_screening"]["records_flagged"] == 1


def test_pii_id_number_produces_warning_not_rejection(db_session):
    version = _make_version(db_session)
    # Synthetic 16-digit ID number pattern - not a real NIK.
    record = _record(
        "r1", user=f"{GOOD_ANSWER} NIK saya 3271010101990001 untuk verifikasi"
    )

    report = validation_service.validate_dataset_version(db_session, version, [record])

    assert report.status_counts == {"VALID": 1, "INVALID": 0, "NEEDS_REVIEW": 0}
    assert report.gate_decision == "PASS"
    assert report.warnings_summary["PII_ID_NUMBER"] == 1


def test_pii_phone_number_produces_warning_not_rejection(db_session):
    version = _make_version(db_session)
    record = _record(
        "r1", user=f"{GOOD_ANSWER} hubungi 081234567890 kalau ada pertanyaan"
    )

    report = validation_service.validate_dataset_version(db_session, version, [record])

    assert report.status_counts == {"VALID": 1, "INVALID": 0, "NEEDS_REVIEW": 0}
    assert report.gate_decision == "PASS"
    assert report.warnings_summary["PII_PHONE_NUMBER"] == 1


def test_no_pii_produces_no_pii_warning(db_session):
    version = _make_version(db_session)
    record = _record("r1")

    report = validation_service.validate_dataset_version(db_session, version, [record])

    assert "PII_EMAIL" not in report.warnings_summary
    assert "PII_ID_NUMBER" not in report.warnings_summary
    assert "PII_PHONE_NUMBER" not in report.warnings_summary
    assert report.warnings_summary["total_warnings"] == 0
    assert report.dataset_statistics["pii_screening"] == {
        "records_flagged": 0,
        "pattern_counts": {"PII_EMAIL": 0, "PII_ID_NUMBER": 0, "PII_PHONE_NUMBER": 0},
    }


def test_multiple_pii_patterns_in_one_record_counted_once_each(db_session):
    version = _make_version(db_session)
    record = _record(
        "r1",
        user=(
            f"{GOOD_ANSWER} email saya budi@example.com dan telepon 081234567890, "
            "email cadangan budi.kedua@example.com juga"
        ),
    )

    report = validation_service.validate_dataset_version(db_session, version, [record])

    # Two emails in the same record still count as one PII_EMAIL hit for that record.
    assert report.warnings_summary["PII_EMAIL"] == 1
    assert report.warnings_summary["PII_PHONE_NUMBER"] == 1
    assert report.warnings_summary["total_warnings"] == 2
    assert report.dataset_statistics["pii_screening"]["records_flagged"] == 1


def test_pii_warning_does_not_affect_h1_hard_error_records(db_session):
    version = _make_version(db_session)
    bad_record = _record("r1")
    bad_record["metadata"] = {}  # triggers H1_missing_required_field
    pii_record = _record("r2", user=f"{GOOD_ANSWER} email saya budi@example.com")

    report = validation_service.validate_dataset_version(
        db_session, version, [bad_record, pii_record]
    )

    assert report.status_counts == {"VALID": 1, "INVALID": 1, "NEEDS_REVIEW": 0}
    assert report.per_record_errors[0] == ["H1_missing_required_field"]
    assert report.per_record_errors[1] == []
    assert report.warnings_summary["PII_EMAIL"] == 1
