"""Dataset intake inspect endpoint — Step 1 of intake wizard."""

from __future__ import annotations

import hashlib

from fastapi import APIRouter, Depends, File, UploadFile
from pydantic import BaseModel, Field
from sqlalchemy.orm import Session

from app.api.deps import require_admin
from app.api.errors import APIError
from app.config import settings
from app.db.session import get_db
from app.models.dataset import Dataset
from app.models.user import User
from app.schemas.common import ErrorResponse
from app.services import dataset_parsing, dataset_service, hf_dataset_import
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


class HfImportRequest(BaseModel):
    """Body for POST /api/v1/datasets/intake/import-hf (issue #240)."""

    dataset_id: str
    repo_id: str = Field(
        description="Hub dataset id, 'owner/name'",
        examples=["HuggingFaceH4/no_robots"],
    )
    # Required on purpose: see `hf_dataset_import.import_hf_dataset`. The endpoint has no
    # default, so a client that omits it gets a 422 rather than silently importing whatever
    # `main` points at today.
    revision: str = Field(
        description="Exact Hub revision to pin: a commit SHA or a tag",
        examples=["refs/convert/parquet"],
    )
    split: str | None = Field(
        default=None,
        description="Split to ingest, e.g. 'train'. Omit to let the importer pick the first "
        "readable data file.",
    )
    display_name: str | None = None
    description: str | None = None


class HfImportResponse(BaseModel):
    """The imported file is staged, not yet committed.

    The wizard continues through the existing `validate` → `commit` steps rather than
    committing here, so an HF dataset is judged by exactly the same H0-H9 gate as an
    uploaded one instead of a second, divergent code path.
    """

    staging_id: str
    dataset_id: str
    repo_id: str
    # Both are returned because they are usually different: `revision` is what the client
    # asked for (a tag, a branch, a short SHA), `resolved_revision` is the commit the bytes
    # actually came from. The latter is what gets recorded on the dataset version.
    revision: str
    resolved_revision: str
    filename: str
    file_size: int
    detected_format: str
    checksum_sha256: str
    split: str | None = None
    next_step: str = "/api/v1/datasets/intake/validate"


@router.post(
    "/datasets/intake/import-hf",
    response_model=HfImportResponse,
    responses={
        400: {"model": ErrorResponse},
        403: {"model": ErrorResponse},
        409: {"model": ErrorResponse},
        502: {"model": ErrorResponse},
    },
)
def import_huggingface_dataset(
    body: HfImportRequest,
    db: Session = Depends(get_db),
    _user: User = Depends(require_admin),
) -> HfImportResponse:
    """Import a pinned HuggingFace dataset revision into the intake pipeline (issue #240).

    Downloads the snapshot, stages its data file, and returns a `staging_id` for the existing
    `intake/validate` step. Admin only.

    The dataset row is registered here (not at commit) so the wizard's later steps have a
    dataset to attach the version to, matching how `validate` already calls
    `dataset_service.register_dataset`.
    """

    try:
        result = hf_dataset_import.import_hf_dataset(
            repo_id=body.repo_id,
            revision=body.revision,
            dataset_id=body.dataset_id,
            split=body.split,
            # Resolved here rather than inside the importer so this endpoint picks the store
            # in the same place the upload path does -- and so the later validate/commit
            # calls, which are handed nothing but a staging_id, are guaranteed to be looking
            # at the same storage.
            storage=DatasetStorage(),
        )
    except hf_dataset_import.HuggingFaceImportError as exc:
        # 502 when the Hub was reached and refused/failed; 400 when the request itself was
        # wrong (bad repo id, missing revision, nothing readable in the snapshot).
        raise APIError(exc.http_status, exc.code, exc.message) from exc

    dataset_service.register_dataset(db, body.dataset_id)
    if body.display_name or body.description:
        dataset_obj = db.get(Dataset, body.dataset_id)
        if dataset_obj:
            if body.display_name:
                dataset_obj.display_name = body.display_name
            if body.description:
                dataset_obj.description = body.description
    db.commit()

    return HfImportResponse(
        staging_id=result.staging_id,
        dataset_id=body.dataset_id,
        repo_id=result.repo_id,
        revision=body.revision,
        resolved_revision=result.resolved_revision,
        filename=result.filename,
        file_size=result.size_bytes,
        detected_format=result.detected_format,
        checksum_sha256=result.checksum_sha256,
        split=result.split,
    )
