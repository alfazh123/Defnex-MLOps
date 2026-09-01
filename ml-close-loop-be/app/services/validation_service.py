import re
import statistics
from datetime import datetime, timezone

from sqlalchemy.orm import Session

from app.models.dataset import DatasetVersion as DatasetVersionModel
from app.models.validation import ValidationReport as ValidationReportModel
from app.schemas.validation import ValidationReport, ValidationStatusCounts

DEFAULT_RULE_SET_VERSION = "2.2.0"

VALID_ROLES = {"system", "user", "assistant"}

# H5: literal empty-enumeration artifact left over from bad link-stripping,
# e.g. "yaitu: , , dan ." (validation-rules.md H5).
_EMPTY_ENUMERATION_RE = re.compile(
    r"(yaitu|antara lain)\s*:?\s*(,\s*)+(dan\s*)?\.?", re.IGNORECASE
)

# H6: known source-boilerplate phrase variants named in validation-rules.md H6
# ("site footers, 'baca/klik/simak ulasan' variants").
_BOILERPLATE_PHRASES = (
    "baca juga",
    "baca selengkapnya",
    "klik di sini",
    "simak selengkapnya",
    "simak ulasan",
)

# H9: control characters other than tab/newline/carriage-return.
_CONTROL_CHAR_RE = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")


def _hard_errors_for_record(record: dict) -> list[str]:
    """H1-H6, H9: per-record hard-error checks that don't need the full dataset."""

    errors = []
    messages = record.get("messages")
    metadata = record.get("metadata") or {}

    # H1: required fields present.
    if (
        not record.get("id")
        or not messages
        or not metadata.get("source_dataset")
        or not metadata.get("source_id")
    ):
        errors.append("H1_missing_required_field")

    for message in messages or []:
        role = message.get("role")
        content = message.get("content")

        # H2: valid role values.
        if role not in VALID_ROLES:
            errors.append("H2_invalid_role")

        # H3: content is non-empty text.
        if not isinstance(content, str) or not content.strip():
            errors.append("H3_empty_content")
            continue

        # H4: assistant content minimum length (>20 words).
        if role == "assistant" and len(content.split()) <= 20:
            errors.append("H4_below_min_length")

        # H5: residual empty-enumeration artifact.
        if _EMPTY_ENUMERATION_RE.search(content):
            errors.append("H5_empty_enumeration_artifact")

        # H6: known boilerplate markers.
        lowered = content.lower()
        if any(phrase in lowered for phrase in _BOILERPLATE_PHRASES):
            errors.append("H6_boilerplate_marker")

        # H9: valid encoding (no control characters).
        if _CONTROL_CHAR_RE.search(content):
            errors.append("H9_invalid_encoding")

    return sorted(set(errors))


def _normalized_pair(record: dict) -> tuple[str, str] | None:
    """Normalized (user, assistant) content pair used for H7 duplicate detection."""

    user_content = None
    assistant_content = None
    for message in record.get("messages") or []:
        content = message.get("content")
        if not isinstance(content, str):
            return None
        if message.get("role") == "user" and user_content is None:
            user_content = content.strip()
        elif message.get("role") == "assistant" and assistant_content is None:
            assistant_content = content.strip()
    if user_content is None or assistant_content is None:
        return None
    return (user_content, assistant_content)


def _user_content(record: dict) -> str | None:
    for message in record.get("messages") or []:
        if message.get("role") == "user" and isinstance(message.get("content"), str):
            return message["content"].strip()
    return None


def validate_dataset_version(
    db: Session,
    dataset_version: DatasetVersionModel,
    records: list[dict],
    eval_records: list[dict] | None = None,
    rule_set_version: str = DEFAULT_RULE_SET_VERSION,
) -> ValidationReportModel:
    """Run the WBS 2.2 hard-error rule set (H1-H9) against `records` and persist a report.

    Scoped to the rules named in this story's acceptance criteria (schema/required-field/
    role-message/malformed-data/duplicate/leakage detection) plus the gate decision - all of
    which have a concrete check defined in validation-rules.md. Warning rules (W1-W5) and
    quality-review rules (Q1-Q4) are deliberately not implemented yet: several of them (W2, W5,
    Q2, Q4) have no concrete threshold or algorithm defined in validation-rules.md, and the doc
    itself (§9) recommends deferring exactly this kind of undefined threshold rather than
    inventing one. NEEDS_REVIEW and warnings_summary therefore stay at zero/empty until a future
    story adds those rules with real, non-invented thresholds.

    `eval_records` represents already-known eval-set content (e.g. the curated benchmark set) to
    check leakage (H8) against. There is no eval-set storage in this system yet (no story creates
    one), so it defaults to None and leakage detection reports zero overlaps - the check itself is
    fully implemented, it simply has nothing to compare against until that storage exists.
    """

    run_at = datetime.now(timezone.utc)

    per_record_errors = [_hard_errors_for_record(record) for record in records]

    # H7: exact-duplicate detection - first occurrence of a normalized pair stays as-is,
    # every later occurrence is a duplicate.
    seen_pairs: set[tuple[str, str]] = set()
    duplicate_count = 0
    for index, record in enumerate(records):
        pair = _normalized_pair(record)
        if pair is None:
            continue
        if pair in seen_pairs:
            per_record_errors[index].append("H7_duplicate")
            duplicate_count += 1
        else:
            seen_pairs.add(pair)

    # H8: train/eval leakage detection - exact match of normalized user content against
    # known eval-set records.
    eval_user_contents = {
        content
        for record in (eval_records or [])
        if (content := _user_content(record)) is not None
    }
    leakage_overlaps = 0
    for index, record in enumerate(records):
        user_content = _user_content(record)
        if user_content is not None and user_content in eval_user_contents:
            per_record_errors[index].append("H8_leakage")
            leakage_overlaps += 1

    valid_count = sum(1 for errors in per_record_errors if not errors)
    invalid_count = len(records) - valid_count

    word_counts = [
        len(message["content"].split())
        for record in records
        for message in record.get("messages") or []
        if message.get("role") == "assistant"
        and isinstance(message.get("content"), str)
    ]
    length_distribution_words = (
        {
            "min": min(word_counts),
            "median": statistics.median(word_counts),
            "mean": round(statistics.mean(word_counts), 1),
            "max": max(word_counts),
        }
        if word_counts
        else {}
    )

    leakage_found = leakage_overlaps > 0
    gate_decision = "FAIL" if leakage_found else "PASS"
    gate_reason = (
        "Leakage found between dataset records and the eval set - blocked regardless of other rates."
        if leakage_found
        else "No leakage found. Hard-error/NEEDS_REVIEW rate thresholds are not yet defined "
        "(validation-rules.md §9), so the gate stays permissive on those counts for now."
    )

    report = ValidationReportModel(
        dataset_version_id=dataset_version.id,
        rule_set_version=rule_set_version,
        run_at=run_at,
        record_count=len(records),
        status_counts={
            "VALID": valid_count,
            "INVALID": invalid_count,
            "NEEDS_REVIEW": 0,
        },
        warnings_summary={},
        dataset_statistics={
            "length_distribution_words": length_distribution_words,
            "duplicate_count": duplicate_count,
            "leakage_check": {
                "checked_against": ["eval/benchmark"] if eval_records else [],
                "overlaps_found": leakage_overlaps,
            },
        },
        gate_decision=gate_decision,
        gate_reason=gate_reason,
    )
    db.add(report)
    db.commit()
    db.refresh(report)
    return report


def list_validation_reports(
    db: Session, dataset_version: DatasetVersionModel
) -> list[ValidationReportModel]:
    return sorted(
        dataset_version.validation_reports, key=lambda r: r.run_at, reverse=True
    )


def get_latest_validation_report(
    db: Session, dataset_version: DatasetVersionModel
) -> ValidationReportModel | None:
    reports = list_validation_reports(db, dataset_version)
    return reports[0] if reports else None


def to_schema(report: ValidationReportModel) -> ValidationReport:
    """Compose the flat ORM row + its dataset_version relationship into the flat response schema."""

    return ValidationReport(
        dataset_id=report.dataset_version.dataset_id,
        dataset_version=report.dataset_version.version,
        rule_set_version=report.rule_set_version,
        run_at=report.run_at,
        record_count=report.record_count,
        status_counts=ValidationStatusCounts(**report.status_counts),
        warnings_summary=report.warnings_summary,
        dataset_statistics=report.dataset_statistics,
        gate_decision=report.gate_decision,
        gate_reason=report.gate_reason,
    )
