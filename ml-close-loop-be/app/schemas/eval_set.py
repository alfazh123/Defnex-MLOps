from datetime import datetime

from pydantic import BaseModel


class EvalSetVersionCreateRequest(BaseModel):
    """Request body for POST /eval-sets/{eval_set_id}/versions (openapi.yaml)."""

    records: list[dict]


class EvalSetVersion(BaseModel):
    """A versioned eval set's record content (openapi.yaml EvalSetVersion)."""

    eval_set_id: str
    version: int
    record_count: int
    records: list[dict]
    created_at: datetime
    created_by: str | None = None


class EvalSetSummary(BaseModel):
    """Lightweight list-view entry for GET /eval-sets (openapi.yaml EvalSetSummary)."""

    eval_set_id: str
    latest_version: int | None = None
    version_count: int = 0
