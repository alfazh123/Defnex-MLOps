from datetime import datetime
from typing import Literal

from pydantic import BaseModel

from app.schemas.model import EvaluationObject


class DecisionCreateRequest(BaseModel):
    """Request body for POST .../decisions (openapi.yaml DecisionCreateRequest). Restricted to
    PROMOTED|REJECTED - ROLLBACK is only ever produced via POST /models/{model_id}/rollback (US-017)."""

    decision: Literal["PROMOTED", "REJECTED"]
    decided_by: str | None = None
    rationale: str


class LadderActionRequest(BaseModel):
    """Request body for the staging/production ladder steps and the deployment-level rollback
    (openapi.yaml LadderActionRequest). The version being acted on is in the URL; the body carries
    only the audit fields (who approved, why), matching `DecisionCreateRequest.decided_by`."""

    decided_by: str | None = None
    rationale: str


class RollbackRequest(BaseModel):
    """Request body for POST /models/{model_id}/rollback (openapi.yaml RollbackRequest)."""

    rollback_of_version: int
    decided_by: str | None = None
    rationale: str


class DecisionRecord(BaseModel):
    """WBS 3.3 §8 decision record (openapi.yaml DecisionRecord) - covers promotion, rejection,
    rollback, and the staging/production ladder steps with one schema."""

    decision_id: str
    model_id: str
    version: int
    decision: Literal[
        "PROMOTED", "REJECTED", "ROLLBACK", "STAGING", "VALIDATED", "PRODUCTION"
    ]
    decided_by: str | None = None
    decided_at: datetime
    evidence_snapshot: EvaluationObject | None = None
    eval_set_id: str | None = None
    eval_set_version: int | None = None
    rationale: str
    rollback_of_version: int | None = None
