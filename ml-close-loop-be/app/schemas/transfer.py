"""Schemas for artifact transfer API (issue #72, PRD §20.3)."""

from datetime import datetime

from pydantic import BaseModel, ConfigDict


class TransferInitiateRequest(BaseModel):
    """Request to initiate an artifact transfer."""

    artifact_uri: str
    source_host: str
    target_host: str


class Transfer(BaseModel):
    """A transfer record."""

    model_config = ConfigDict(from_attributes=True)

    transfer_id: str
    artifact_uri: str
    source_host: str
    target_host: str
    checksum_before: str
    checksum_after: str | None = None
    status: str
    created_at: datetime
    completed_at: datetime | None = None
    error_message: str | None = None
