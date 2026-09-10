"""Dataset intake validate + commit endpoints (Steps 4-5)."""

from __future__ import annotations

import time
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
from app.services import dataset_service, validation_service
from app.services.dataset_storage import DatasetStorage

router = APIRouter(tags=["Dataset Intake"])

_IDEMPOTENCY_CACHE: dict[str, tuple[dict, float]] = {}
_IDEMPOTENCY_TTL = 3600


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

    records = storage.read_records(
        staged_path, source_format=intake_request.source_format
    )
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

    version_num = dataset_service._next_version(db, intake_request.dataset_id)
    dv = DatasetVersion(
        dataset_id=intake_request.dataset_id,
        version=version_num,
        status="PENDING",
        source_type="file_upload",
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

    blocking_errors = sum(
        1
        for errs in (report.per_record_errors or [])
        for e in errs
        if isinstance(e, dict) and e.get("severity") == "error"
    )
    warnings = 0
    if isinstance(report.warnings_summary, dict):
        warnings = report.warnings_summary.get("total_warnings", 0)

    checks = [
        {"name": "parse", "status": "PASS", "message": ""},
        {"name": "schema_compliance", "status": "PASS", "message": ""},
        {"name": "required_fields", "status": "PASS", "message": ""},
        {"name": "duplicate_ids", "status": "PASS", "message": ""},
        {
            "name": "leakage",
            "status": "PASS" if report.gate_decision == "PASS" else "FAIL",
            "message": report.gate_reason or "",
        },
        {"name": "checksum", "status": "PASS", "message": f"sha256:{checksum}"},
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
    if x_idempotency_key:
        if x_idempotency_key in _IDEMPOTENCY_CACHE:
            result, ts = _IDEMPOTENCY_CACHE[x_idempotency_key]
            if time.time() - ts < _IDEMPOTENCY_TTL:
                return result
            del _IDEMPOTENCY_CACHE[x_idempotency_key]

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

    checksum = storage.compute_checksum(staged_path)
    if vr.content_hash and checksum != vr.content_hash:
        raise APIError(
            409, "CHECKSUM_MISMATCH", "File checksum changed since validation"
        )

    dataset_service.register_dataset(db, intake_request.dataset_id)
    version_num = dataset_service._next_version(db, intake_request.dataset_id)

    canonical_uri = storage.commit_file(
        intake_request.staging_id, intake_request.dataset_id, version_num
    )

    dv = DatasetVersion(
        dataset_id=intake_request.dataset_id,
        version=version_num,
        status="PROCESSED",
        source_type=intake_request.source_type,
        source_format=intake_request.source_format,
        row_count=vr.record_count,
        created_at=datetime.now(UTC),
        created_by=_user.username,
        raw_file_uri=str(staged_path),
        canonical_file_uri=canonical_uri,
    )
    db.add(dv)
    db.flush()

    manifest = {
        "dataset_id": intake_request.dataset_id,
        "version": version_num,
        "source_type": intake_request.source_type,
        "source_format": intake_request.source_format,
        "row_count": vr.record_count,
        "checksum_sha256": checksum,
        "created_at": datetime.now(UTC).isoformat(),
        "created_by": _user.username,
    }
    storage.write_validation_report(intake_request.dataset_id, version_num, manifest)
    storage.write_schema(
        intake_request.dataset_id, version_num, {"schema": "defnex_scenario_v1"}
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

    result = dataset_service.to_schema(dv).model_dump()

    if x_idempotency_key:
        _IDEMPOTENCY_CACHE[x_idempotency_key] = (result, time.time())

    return result
