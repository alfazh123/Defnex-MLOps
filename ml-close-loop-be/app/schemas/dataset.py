from datetime import datetime
from typing import Literal

from pydantic import BaseModel

DatasetSourceFormat = Literal["alpaca", "sharegpt", "chatml", "other"]
DatasetVersionStatus = Literal["PENDING", "PROCESSING", "PROCESSED", "FAILED"]


class DatasetManifest(BaseModel):
    """Per-version manifest fields (dataset-lifecycle-and-schema.md §2)."""

    source_url_or_hf_id: str | None = None
    source_commit_or_snapshot_date: str | None = None
    source_format: DatasetSourceFormat
    seed: int | None = None
    row_count: int | None = None
    cleaning_steps_applied: list[str] = []
    created_at: datetime
    created_by: str | None = None


class DatasetVersionCreateRequest(BaseModel):
    """Request to intake a new dataset version (openapi.yaml DatasetVersionCreateRequest)."""

    source_type: Literal["huggingface", "file_upload"]
    source_dataset: str | None = None
    source_commit_or_snapshot_date: str | None = None
    source_format: DatasetSourceFormat


class DatasetVersion(BaseModel):
    """A dataset version's manifest and current processing status (openapi.yaml DatasetVersion)."""

    dataset_id: str
    version: int
    status: DatasetVersionStatus
    manifest: DatasetManifest


class DatasetSummary(BaseModel):
    """Lightweight list-view entry for GET /datasets."""

    dataset_id: str
    latest_version: int
    status: DatasetVersionStatus
