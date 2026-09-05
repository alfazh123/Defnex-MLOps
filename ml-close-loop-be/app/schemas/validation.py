from datetime import datetime
from typing import Literal

from pydantic import BaseModel

ValidationGateDecision = Literal["PASS", "FAIL"]


class ValidationStatusCounts(BaseModel):
    """Per-record status tally (validation-rules.md §2)."""

    VALID: int = 0
    INVALID: int = 0
    NEEDS_REVIEW: int = 0


class ValidateDatasetVersionRequest(BaseModel):
    """Optional body for POST .../validate (openapi.yaml ValidateDatasetVersionRequest).

    `records` is the actual record content to run the hard-error rules against — the
    client supplies it because no intake/normalization pipeline persists record content
    in this system yet (see progress.txt US-005/US-006). It is required and must be
    non-empty so a validation report can never claim PASS over content that was never
    examined; the report records a SHA-256 fingerprint of exactly what was checked.
    `eval_records` is the known eval-set content used for the H8 leakage check.
    """

    rule_set_version: str | None = None
    records: list[dict]
    eval_records: list[dict] = []


class ValidationReport(BaseModel):
    """Dataset-level validation report (validation-rules.md §5/§7, openapi.yaml ValidationReport)."""

    dataset_id: str
    dataset_version: int
    rule_set_version: str
    run_at: datetime
    record_count: int
    content_hash: str
    status_counts: ValidationStatusCounts
    warnings_summary: dict[str, int] = {}
    dataset_statistics: dict = {}
    per_record_errors: list[list[str]] = []
    gate_decision: ValidationGateDecision
    gate_reason: str
