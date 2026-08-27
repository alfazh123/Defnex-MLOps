from datetime import datetime
from typing import Literal

from pydantic import BaseModel

ValidationGateDecision = Literal["PASS", "FAIL"]


class ValidationStatusCounts(BaseModel):
    """Per-record status tally (validation-rules.md §2)."""

    VALID: int = 0
    INVALID: int = 0
    NEEDS_REVIEW: int = 0


class ValidationReport(BaseModel):
    """Dataset-level validation report (validation-rules.md §5/§7, openapi.yaml ValidationReport)."""

    dataset_id: str
    dataset_version: int
    rule_set_version: str
    run_at: datetime
    record_count: int
    status_counts: ValidationStatusCounts
    warnings_summary: dict[str, int] = {}
    dataset_statistics: dict = {}
    gate_decision: ValidationGateDecision
    gate_reason: str
