"""Dataset intake inspect endpoint — Step 1 of intake wizard."""

from __future__ import annotations

import csv
import hashlib
import io
import json

from fastapi import APIRouter, Depends, File, UploadFile
from pydantic import BaseModel

from app.api.deps import require_admin
from app.api.errors import APIError
from app.models.user import User
from app.schemas.common import ErrorResponse
from app.services.dataset_storage import DatasetStorage

router = APIRouter(tags=["Dataset Intake"])

ALLOWED_EXTENSIONS = {".jsonl", ".json", ".csv", ".xlsx"}
MAX_FILE_SIZE = 100 * 1024 * 1024  # 100MB


class DatasetInspectResponse(BaseModel):
    staging_id: str
    filename: str
    file_size: int
    detected_format: str
    normalized_format: str
    detected_sample_count: int
    checksum_sha256: str
    parse_status: str
    error: str | None = None


def _detect_format(filename: str, content: bytes) -> str:
    """Detect file format from extension and content sniffing."""
    lower = filename.lower()
    if lower.endswith(".jsonl"):
        return "jsonl"
    if lower.endswith(".json"):
        try:
            data = json.loads(content)
            if isinstance(data, list):
                return "json"
        except (json.JSONDecodeError, UnicodeDecodeError):
            pass
        return "unknown"
    if lower.endswith(".csv"):
        return "csv"
    if lower.endswith(".xlsx"):
        return "xlsx"
    return "unknown"


def _count_records(content: bytes, fmt: str) -> int:
    """Count records in the file — authoritative backend count."""
    if fmt == "jsonl":
        return sum(1 for line in content.decode("utf-8").splitlines() if line.strip())
    if fmt == "json":
        data = json.loads(content)
        return len(data) if isinstance(data, list) else 0
    if fmt == "csv":
        reader = csv.reader(io.StringIO(content.decode("utf-8")))
        next(reader, None)  # skip header
        return sum(1 for _ in reader)
    if fmt == "xlsx":
        try:
            import openpyxl
        except ImportError:
            raise APIError(400, "MISSING_DEP", "openpyxl not installed")
        wb = openpyxl.load_workbook(io.BytesIO(content), read_only=True)
        ws = wb.active
        rows = ws.max_row or 0
        wb.close()
        return max(0, rows - 1)
    return 0


@router.post(
    "/datasets/intake/inspect",
    response_model=DatasetInspectResponse,
    responses={
        400: {"model": ErrorResponse},
        403: {"model": ErrorResponse},
        413: {"model": ErrorResponse},
    },
)
async def inspect_dataset_source(
    file: UploadFile = File(...),
    _user: User = Depends(require_admin),
) -> DatasetInspectResponse:
    """Inspect an uploaded dataset file (Step 1 of intake wizard).

    Returns detected format, sample count, checksum, and staging_id
    for subsequent validate/commit calls. Admin only.
    """
    filename = file.filename or "unknown"
    ext = ("." + filename.rsplit(".", 1)[-1].lower()) if "." in filename else ""

    if ext not in ALLOWED_EXTENSIONS:
        raise APIError(
            400,
            "UNSUPPORTED_FORMAT",
            f"Unsupported file format: {ext}. Allowed: {', '.join(sorted(ALLOWED_EXTENSIONS))}",
        )

    content = await file.read()
    if len(content) == 0:
        raise APIError(400, "EMPTY_FILE", "Uploaded file is empty")
    if len(content) > MAX_FILE_SIZE:
        raise APIError(
            413,
            "FILE_TOO_LARGE",
            f"File exceeds maximum size of {MAX_FILE_SIZE // (1024 * 1024)}MB",
        )

    fmt = _detect_format(filename, content)
    if fmt == "unknown":
        raise APIError(400, "UNPARSEABLE_FILE", f"Could not parse file: {filename}")

    try:
        sample_count = _count_records(content, fmt)
    except APIError:
        raise
    except Exception as exc:
        raise APIError(400, "PARSE_ERROR", f"Failed to parse file: {exc}")

    if sample_count == 0:
        raise APIError(400, "EMPTY_DATASET", "File contains no records")

    storage = DatasetStorage()
    staged = storage.stage_upload(filename, content)
    checksum = hashlib.sha256(content).hexdigest()

    return DatasetInspectResponse(
        staging_id=staged["staging_id"],
        filename=filename,
        file_size=staged["size_bytes"],
        detected_format=fmt,
        normalized_format="jsonl",
        detected_sample_count=sample_count,
        checksum_sha256=checksum,
        parse_status="ok",
        error=None,
    )
