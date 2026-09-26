"""Dataset intake inspect endpoint — Step 1 of intake wizard."""

from __future__ import annotations

import hashlib

from fastapi import APIRouter, Depends, File, UploadFile
from pydantic import BaseModel

from app.api.deps import require_admin
from app.api.errors import APIError
from app.config import settings
from app.models.user import User
from app.schemas.common import ErrorResponse
from app.services import dataset_parsing
from app.services.dataset_storage import (
    DatasetStorage,
    UnsafeFilenameError,
    sanitize_filename,
)

router = APIRouter(tags=["Dataset Intake"])

# The 100 MB cap is `settings.max_upload_file_size` (issue #242). It used to be a local
# constant here while the global `RequestSizeLimitMiddleware` rejected anything over 1 MB,
# so this check was unreachable and the documented limit was a lie. The single source of
# truth is now the setting, and the middleware raises the body cap for these routes to
# match it — see `app/middleware/request_size.py`.
ALLOWED_EXTENSIONS = frozenset(dataset_parsing.EXTENSION_FORMATS)


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
    # Sanitize first so the extension check, the staged file and the echoed `filename` all
    # agree on one name. A traversal attempt is rejected here rather than silently
    # rewritten, so a client that sends `../../etc/x.jsonl` learns about it instead of
    # getting a success response for a file it did not upload (issue #246).
    try:
        filename = sanitize_filename(file.filename or "")
    except UnsafeFilenameError as exc:
        raise APIError(400, "INVALID_FILENAME", str(exc)) from exc

    ext = ("." + filename.rsplit(".", 1)[-1].lower()) if "." in filename else ""
    if ext not in ALLOWED_EXTENSIONS:
        raise APIError(
            400,
            "UNSUPPORTED_FORMAT",
            f"Unsupported file format: {ext or filename!r}. "
            f"Allowed: {', '.join(sorted(ALLOWED_EXTENSIONS))}",
        )

    content = await file.read()
    if len(content) == 0:
        raise APIError(400, "EMPTY_FILE", "Uploaded file is empty")
    max_file_size = settings.max_upload_file_size
    if len(content) > max_file_size:
        raise APIError(
            413,
            "FILE_TOO_LARGE",
            f"File exceeds maximum size of {max_file_size // (1024 * 1024)}MB",
        )

    fmt = dataset_parsing.detect_format(filename, content)
    if fmt == "unknown":
        raise APIError(
            400,
            "UNPARSEABLE_FILE",
            f"Could not determine a readable dataset format from {filename!r}. "
            f"Allowed extensions: {', '.join(sorted(ALLOWED_EXTENSIONS))}",
        )

    # One parse, one answer (issue #245): `detected_sample_count` is the length of the very
    # record list `validate` will build, so inspect and validate can no longer disagree
    # about how many records a file holds. Parse failures are coded, never 500.
    try:
        records = dataset_parsing.parse_bytes(content, fmt)
    except dataset_parsing.DatasetParseError as exc:
        raise APIError(exc.http_status, exc.code, exc.message) from exc

    if not records:
        raise APIError(400, "EMPTY_DATASET", "File contains no records")

    storage = DatasetStorage()
    staged = storage.stage_upload(filename, content)
    checksum = hashlib.sha256(content).hexdigest()

    return DatasetInspectResponse(
        staging_id=staged["staging_id"],
        filename=staged["filename"],
        file_size=staged["size_bytes"],
        detected_format=fmt,
        normalized_format="jsonl",
        detected_sample_count=len(records),
        checksum_sha256=checksum,
        parse_status="ok",
        error=None,
    )
