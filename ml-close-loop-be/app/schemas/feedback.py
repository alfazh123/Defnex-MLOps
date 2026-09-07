from datetime import datetime
from typing import Literal

from pydantic import BaseModel, Field

from app.schemas.dataset import DatasetSourceFormat

FeedbackCurationStatus = Literal["PENDING", "APPROVED", "REJECTED"]


class FeedbackCreateRequest(BaseModel):
    """Submit feedback on an inference response (issue #42).

    `model_id`/`version` identify the concrete `ModelVersion` that produced `response`
    (`InferenceResponse.model_id`/`version`, issue #41) - not an alias, so the reference is
    traceable to an exact version regardless of what is DEPLOYED later. `rating` is a plain
    1-5 input-validation bound (a common feedback scale), not an invented evaluation metric.
    `correction` is optional - a human-authored better answer, when the reviewer has one.
    """

    model_id: str
    version: int
    prompt: str = Field(min_length=1)
    response: str = Field(min_length=1)
    rating: int = Field(ge=1, le=5)
    correction: str | None = None


class FeedbackCurateRequest(BaseModel):
    """Optional body for the approve/reject endpoints - who curated it, for the audit trail
    (same pattern as `DecisionCreateRequest.decided_by`)."""

    curated_by: str | None = None


class FeedbackRecord(BaseModel):
    """A feedback row (openapi.yaml Feedback)."""

    feedback_id: str
    model_id: str
    version: int
    prompt: str
    response: str
    rating: int
    correction: str | None = None
    curation_status: FeedbackCurationStatus
    submitted_by: str | None = None
    submitted_at: datetime
    curated_by: str | None = None
    curated_at: datetime | None = None


class FeedbackDatasetVersionRequest(BaseModel):
    """Create a dataset version from approved feedback candidates (issue #42).

    `feedback_ids` is an explicit, manual selection - there is no automatic threshold-based
    trigger (issue #42's own "out of scope"). `source_format` still has to be chosen the same
    way it does for `DatasetVersionCreateRequest`; feedback rows don't carry one themselves.
    """

    feedback_ids: list[str] = Field(min_length=1)
    source_format: DatasetSourceFormat
