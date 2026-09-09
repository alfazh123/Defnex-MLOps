from datetime import datetime
from typing import Literal

from pydantic import BaseModel


class TransferInitiateRequest(BaseModel):
    """Request body for POST /api/v1/transfers."""

    artifact_uri: str
    source_host: str
    target_host: str


class TransferResponse(BaseModel):
    """Response for transfer operations (openapi.yaml TransferRecord)."""

    transfer_id: str
    artifact_uri: str
    source_host: str
    target_host: str
    checksum_before: str
    checksum_after: str | None = None
    status: Literal["PENDING", "TRANSFERRING", "COMPLETED", "FAILED"]
    created_at: datetime
    completed_at: datetime | None = None
    error_message: str | None = None
