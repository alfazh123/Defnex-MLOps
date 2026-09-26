"""Dataset intake validate + commit endpoints (Steps 4-5)."""

from __future__ import annotations

from datetime import UTC, datetime

from fastapi import APIRouter, Depends, Header, Request
from pydantic import BaseModel
from sqlalchemy.orm import Session

from app.api.deps import require_admin
from app.api.errors import APIError
from app.config import settings
from app.db.session import get_db
from app.limiter import limiter
from app.models.dataset import Dataset, DatasetVersion
from app.models.user import User
from app.models.validation import ValidationReport
from app.schemas.common import ErrorResponse
from app.services import dataset_service, idempotency_service, validation_service
from app.services import dataset_parsing
from app.services.dataset_storage import DatasetStorage

router = APIRouter(tags=["Dataset Intake"])

_IDEMPOTENCY_ENDPOINT = "intake_commit"


class IntakeValidateRequest(BaseModel):
    staging_id: str
    dataset_id: str
    schema_name: str = "defnex_scenario_v1"
    source_format: str = "jsonl"
    eval_set_id: str | None = None
    eval_set_version: int | None = None


class IntakeCommitRequest(BaseModel):
    staging_id: str
    dataset_id: str
    display_name: str | None = None
    description: str | None = None
    source_type: str = "file_upload"
    source_format: str = "jsonl"
    validation_report_id: int


@router.post(
    "/datasets/intake/validate",
    response_model=dict,
    responses={400: {"model": ErrorResponse}, 404: {"model": ErrorResponse}},
)
@limiter.limit(settings.rate_limit_intake_validate)
def validate_intake(
    request: Request,
    intake_request: IntakeValidateRequest,
    db: Session = Depends(get_db),
    _user: User = Depends(require_admin),
) -> dict:
    """Validate a staged dataset file (Step 4). Returns H1-H9 validation results."""
    storage = DatasetStorage()

    staged_path = storage.resolve_staged(intake_request.staging_id)
    if staged_path is None:
        raise APIError(
            404, "STAGING_NOT_FOUND", f"Staging {intake_request.staging_id} not found"
        )

    # Issue #245: the declared `source_format` was previously never checked against the file
    # actually staged. A CSV validated with the default `source_format="jsonl"` fell through
    # to the JSONL parser and died as an unhandled `json.JSONDecodeError` -> 500. Reject the
    # mismatch by name so the client can fix its request in one round trip.
    detected_format = dataset_parsing.format_from_filename(staged_path.name)
    if detected_format is not None and detected_format != intake_request.source_format:
        raise APIError(
            400,
            "FORMAT_MISMATCH",
            f"source_format {intake_request.source_format!r} does not match the staged file "
            f"{staged_path.name!r}, which is {detected_format!r}. Re-run the request with "
            f'source_format="{detected_format}".',
        )
    if detected_format is None:
        raise APIError(
            400,
            "UNSUPPORTED_FORMAT",
            f"Staged file {staged_path.name!r} does not have a readable dataset extension. "
            f"Readable: {', '.join(dataset_parsing.SUPPORTED_FORMATS)}.",
        )

    # Every malformed-input path in here is a DatasetParseError carrying the wire code and
    # status, so a broken upload answers 4xx with a diagnosis instead of a 500 traceback.
    try:
        records = storage.read_records(
            staged_path, source_format=intake_request.source_format
        )
    except dataset_parsing.DatasetParseError as exc:
        raise APIError(exc.http_status, exc.code, exc.message) from exc
    if not records:
        raise APIError(400, "EMPTY_DATASET", "No records found in staged file")

    eval_records = None
    if intake_request.eval_set_id and intake_request.eval_set_version:
        from app.services import eval_set_service

        eval_ver = eval_set_service.get_eval_set_version(
            db, intake_request.eval_set_id, intake_request.eval_set_version
        )
        if eval_ver:
            eval_records = eval_ver.records

    dataset_service.register_dataset(db, intake_request.dataset_id)

    # Issue #240: a file staged by the HF importer carries its origin (repo id + resolved
    # commit SHA) in a sidecar written next to the bytes. It is read here rather than taken
    # from the request, so a client cannot relabel a HuggingFace import as a plain upload --
    # or vice versa -- and the resolved SHA reaches the columns meant to hold it.
    provenance = storage.read_provenance(intake_request.staging_id)
    source_type = provenance.get("source_type") or "file_upload"

    version_num = dataset_service._allocate_version(db, intake_request.dataset_id)
    dv = DatasetVersion(
        dataset_id=intake_request.dataset_id,
        version=version_num,
        status="PENDING",
        source_type=source_type,
        source_url_or_hf_id=provenance.get("source_url_or_hf_id"),
        source_commit_or_snapshot_date=provenance.get("source_commit_or_snapshot_date"),
        source_format=intake_request.source_format,
        row_count=len(records),
        created_at=datetime.now(UTC),
        raw_file_uri=str(staged_path),
    )
    db.add(dv)
    db.flush()

    report = validation_service.validate_dataset_version(
        db, dv, records, eval_records=eval_records
    )

    checksum = storage.compute_checksum(staged_path)

    # Override content_hash with the file-based checksum so commit can verify
    # the file hasn't changed between validate and commit.
    report.content_hash = checksum
    db.commit()

    # Issue #241: `blocking_errors` was counted as
    # `sum(1 for e in errs if isinstance(e, dict) and e.get("severity") == "error")`, but
    # `per_record_errors` holds lists of rule-code *strings* (H1_missing_required_field, ...),
    # never dicts -- so the count was structurally always 0. The response therefore reported
    # `valid_records == total_records` and `blocking_error_count: 0` for a dataset in which
    # every record failed validation. The report's own `status_counts.INVALID` is the number
    # the service already computed correctly, so read it from there.
    status_counts = (
        report.status_counts if isinstance(report.status_counts, dict) else {}
    )
    blocking_errors = int(status_counts.get("INVALID", 0) or 0)
    statistics = (
        report.dataset_statistics if isinstance(report.dataset_statistics, dict) else {}
    )
    duplicate_count = int(statistics.get("duplicate_count", 0) or 0)
    leakage_check = statistics.get("leakage_check") or {}
    leakage_overlaps = int(leakage_check.get("overlaps_found", 0) or 0)

    warnings = 0
    if isinstance(report.warnings_summary, dict):
        warnings = report.warnings_summary.get("total_warnings", 0)

    # Issue #241: the `leakage` check used to report FAIL/ purely from the gate decision,
    # so a dataset that failed only on hard errors showed "leakage: FAIL" with no leakage
    # present. Each check now states what it actually checked, and the dataset-level
    # verdict is its own entry so a client can render NEEDS_REVIEW without re-deriving it.
    checks = [
        {"name": "parse", "status": "PASS", "message": ""},
        {"name": "schema_compliance", "status": "PASS", "message": ""},
        {"name": "required_fields", "status": "PASS", "message": ""},
        {
            "name": "duplicate_ids",
            "status": "FAIL" if duplicate_count else "PASS",
            "message": f"{duplicate_count} duplicate record(s)"
            if duplicate_count
            else "",
        },
        {
            "name": "leakage",
            "status": "FAIL" if leakage_overlaps else "PASS",
            "message": (
                f"{leakage_overlaps} record(s) overlap the eval set"
                if leakage_overlaps
                else ""
            ),
        },
        {"name": "checksum", "status": "PASS", "message": f"sha256:{checksum}"},
        {
            "name": "gate",
            "status": report.gate_decision,
            "message": report.gate_reason or "",
        },
    ]

    return {
        "status": report.gate_decision,
        "total_records": len(records),
        "valid_records": max(0, len(records) - blocking_errors),
        "warning_count": warnings,
        "blocking_error_count": blocking_errors,
        "checks": checks,
        "diagnostics": (report.per_record_errors or [])[:10],
        "preview": records[:5],
        "checksum_sha256": checksum,
        "normalized_format": "jsonl",
        "schema_version": intake_request.schema_name,
        "validation_report_id": report.id,
        "staging_id": intake_request.staging_id,
    }


@router.post(
    "/datasets/intake/commit",
    response_model=dict,
    responses={
        404: {"model": ErrorResponse},
        409: {"model": ErrorResponse},
        403: {"model": ErrorResponse},
    },
)
@limiter.limit(settings.rate_limit_intake_commit)
def commit_intake(
    request: Request,
    intake_request: IntakeCommitRequest,
    db: Session = Depends(get_db),
    _user: User = Depends(require_admin),
    x_idempotency_key: str | None = Header(None),
) -> dict:
    """Commit a validated dataset as an immutable DatasetVersion (Step 5)."""
    # Issue #124: durable Postgres-backed cache (app.services.idempotency_service) instead of an
    # in-process dict, so a retried request with the same X-Idempotency-Key replays the same
    # result across API worker processes and across a restart, within the TTL window.
    if x_idempotency_key:
        cached = idempotency_service.get_cached_response(db, x_idempotency_key)
        if cached is not None:
            return cached.body

    storage = DatasetStorage()

    staged_path = storage.resolve_staged(intake_request.staging_id)
    if staged_path is None:
        raise APIError(
            404, "STAGING_NOT_FOUND", f"Staging {intake_request.staging_id} not found"
        )

    vr = db.get(ValidationReport, intake_request.validation_report_id)
    if vr is None:
        raise APIError(
            404, "VALIDATION_REPORT_NOT_FOUND", "Validation report not found"
        )
    if vr.gate_decision == "FAIL":
        raise APIError(409, "VALIDATION_FAILED", "Cannot commit: validation gate FAIL")
    # Issue #241: NEEDS_REVIEW is deliberately committable. The bytes are on disk and the
    # report says exactly what is wrong with them, so keeping the version in the registry
    # (marked PROCESSED, with the gate on its validation report) is what lets a human
    # inspect and fix it. What NEEDS_REVIEW blocks is *training* -- see
    # `app/api/training.py`, which refuses any gate that is not PASS.

    checksum = storage.compute_checksum(staged_path)
    if vr.content_hash and checksum != vr.content_hash:
        raise APIError(
            409, "CHECKSUM_MISMATCH", "File checksum changed since validation"
        )

    # Issue #240: read the staged file's provenance BEFORE `commit_file`, which deletes the
    # staging directory (and the sidecar with it). Read afterwards it always comes back empty
    # and every HuggingFace import is silently recorded as a plain upload.
    provenance = storage.read_provenance(intake_request.staging_id)

    dataset_service.register_dataset(db, intake_request.dataset_id)

    # Update the SAME DatasetVersion row `validate_intake` created (and `vr` is already linked
    # to) rather than allocating a new version number here. Allocating a second version at
    # commit time left the validation report permanently attached to an orphaned PENDING row
    # one version behind the PROCESSED one this endpoint returned, so
    # `training_service.create_training_run`'s `get_latest_validation_report` lookup (keyed by
    # dataset_version) could never find a report for the version a caller was told to use --
    # training-run creation failed with VALIDATION_REQUIRED for every wizard-committed dataset.
    dv = db.get(DatasetVersion, vr.dataset_version_id)
    if dv is None or dv.dataset_id != intake_request.dataset_id:
        raise APIError(
            404,
            "DATASET_VERSION_NOT_FOUND",
            "The dataset version created during validate no longer exists",
        )
    version_num = dv.version

    canonical_uri, committed_name, committed_size = storage.commit_file(
        intake_request.staging_id, intake_request.dataset_id, version_num
    )

    dv.status = "PROCESSED"
    # Issue #240: the sidecar recorded with the bytes outranks the request field. A commit
    # request defaults `source_type` to "file_upload", which would silently relabel every
    # HuggingFace import; and letting a client pass "huggingface" for a plain upload would
    # invent a provenance record. The sidecar describes the actual bytes.
    dv.source_type = provenance.get("source_type") or intake_request.source_type
    if provenance.get("source_url_or_hf_id"):
        dv.source_url_or_hf_id = provenance["source_url_or_hf_id"]
    if provenance.get("source_commit_or_snapshot_date"):
        dv.source_commit_or_snapshot_date = provenance["source_commit_or_snapshot_date"]
    dv.created_by = _user.username
    dv.canonical_file_uri = canonical_uri
    db.flush()

    manifest = {
        "dataset_id": intake_request.dataset_id,
        "version": version_num,
        "source_type": dv.source_type,
        "source_format": intake_request.source_format,
        "row_count": vr.record_count,
        "checksum_sha256": checksum,
        # Issue #214: recorded in the manifest as well as on the row, because the row is
        # only half the record — the other half is the object in storage, and these are what
        # let a reader of the stored bytes tell which upload they came from.
        "canonical_file_uri": canonical_uri,
        "filename": committed_name,
        "size_bytes": committed_size,
        "gate_decision": vr.gate_decision,
        "created_at": datetime.now(UTC).isoformat(),
        "created_by": _user.username,
    }
    if dv.source_url_or_hf_id:
        manifest["source_url_or_hf_id"] = dv.source_url_or_hf_id
    if dv.source_commit_or_snapshot_date:
        manifest["source_commit_or_snapshot_date"] = dv.source_commit_or_snapshot_date
    storage.write_sidecar(
        intake_request.dataset_id, version_num, "validation_report.json", manifest
    )
    storage.write_sidecar(
        intake_request.dataset_id,
        version_num,
        "schema.json",
        {"schema": "defnex_scenario_v1"},
    )

    if intake_request.display_name:
        dataset_obj = db.get(Dataset, intake_request.dataset_id)
        if dataset_obj and hasattr(dataset_obj, "display_name"):
            dataset_obj.display_name = intake_request.display_name
        if (
            dataset_obj
            and intake_request.description
            and hasattr(dataset_obj, "description")
        ):
            dataset_obj.description = intake_request.description

    db.commit()

    # mode="json" (not the default mode="python"): the idempotency cache round-trips this dict
    # through `json.dumps`/`json.loads` (issue #124), and `DatasetVersion.created_at` is a
    # `datetime` - JSON has no datetime type, so a raw `.model_dump()` would blow up
    # `idempotency_service.store_response`'s `json.dumps` on a cache write. FastAPI would have
    # produced the identical ISO-string wire format for a non-cached response anyway (it runs
    # every response through `jsonable_encoder`), so this changes no client-visible behavior.
    result = dataset_service.to_schema(dv).model_dump(mode="json")

    if x_idempotency_key:
        # Its own flush+commit after the main commit above (the result is only known once that
        # transaction has landed); the router still owns both commit boundaries per the repo's
        # service/router split (services only `db.flush()`).
        idempotency_service.store_response(
            db,
            x_idempotency_key,
            endpoint=_IDEMPOTENCY_ENDPOINT,
            status=200,
            body=result,
        )
        db.commit()

    return result
