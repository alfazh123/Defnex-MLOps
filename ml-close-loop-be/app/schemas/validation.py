from datetime import datetime
from typing import Any, Literal

from pydantic import BaseModel

# Issue #241: NEEDS_REVIEW is now reachable. Previously the gate only ever produced PASS
# or FAIL, and the model refused to describe the middle state the policy now defines.
ValidationGateDecision = Literal["PASS", "NEEDS_REVIEW", "FAIL"]


class ValidationStatusCounts(BaseModel):
    """Per-record status tally (validation-rules.md §2).

    NEEDS_REVIEW is always 0: a per-record NEEDS_REVIEW status would need a warning-class
    threshold (W2/W5), which validation-rules.md does not define. Do not confuse it with the
    dataset-level `gate_decision == "NEEDS_REVIEW"`, which is driven by the hard-error ratio
    (issue #241) and is reachable.
    """

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
    `eval_records` is the legacy inline eval-set content for the H8 leakage check;
    `eval_set_id`/`eval_set_version` reference a stored golden/eval set (issue #43) and
    take precedence when both are supplied -- the stored eval set is the real comparison
    surface for the H8 check.
    """

    rule_set_version: str | None = None
    # Loosely typed on purpose (issue #245): a non-object record is a rule-set finding
    # (`H0_record_not_object`), not a request-shape error, so it is reported in
    # `per_record_errors` instead of rejected as a 422 before validation ever runs. This
    # keeps both validate entry points (inline body and the intake pipeline) governed by
    # one rule set.
    records: list[Any]
    eval_records: list[dict] = []
    eval_set_id: str | None = None
    eval_set_version: int | None = None


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
    # Echoes exactly what was examined, so it is not narrowed to `dict`: intake can surface a
    # parsed value that is not a JSON object (issue #245), and the H0 shape rules report that
    # per record rather than this field silently dropping or crashing on it.
    records: list[Any] = []
    gate_decision: ValidationGateDecision
    gate_reason: str
