"""Periodic post-promotion quality re-check against the golden set (issue #136, PRD §41
Principle 4 / §28.1: the quality gate today only runs once, at promotion time, with no
continuous check afterward).

`run_drift_check_for_version` is the entry point the scheduled worker
(`app/workers/drift_check_worker.py`) calls for each `DEPLOYED` model version. It reuses
`evaluation_engine.compute_evaluation_update` unchanged - same real inference through the
existing `ServingBackend` against the model's stored golden/eval set, same `ValueError`-raises-
means-skip contract `evaluation_worker.process_next_evaluation` already relies on - but it never
calls `model_service.submit_evaluation`. That function auto-transitions
`REGISTERED -> EVALUATED` (model_service.py) once all three signals are present; a `DEPLOYED`
model version must never re-enter that state machine. Instead each run is appended to
`model_drift_checks` (`app/models/drift_check.py`), tagged `trigger=TRIGGER_SCHEDULED_DRIFT_CHECK`,
leaving `ModelVersion`'s own evaluation columns - the promotion-time snapshot - untouched. Those
untouched columns double as the "baseline saat promosi" this module diffs each periodic run
against.

"Significant" regression: CLAUDE.md forbids inventing evaluation metrics/thresholds not already
grounded in the code, and no numeric drop threshold exists anywhere in this codebase for these
signals. So rather than pick an arbitrary percentage, `has_new_regression` uses the binary check
the issue's own acceptance criteria fall back to: a golden-set record
(`general_domain_regression_check.regressions_found`, keyed by `record_index` - the eval set
itself, `model_version.eval_set_id`/`eval_set_version`, does not change between checks) that
fails in the new run but did not fail in the baseline is a *new* regression, which is what
triggers `notify_admins`. A record that was already failing at promotion time is not "new" drift
and does not re-notify on every subsequent run.
"""

from __future__ import annotations

from datetime import datetime, timezone

import structlog
from sqlalchemy.orm import Session

from app.models.drift_check import ModelDriftCheck
from app.models.model import ModelVersion
from app.services import notification_service
from app.services.evaluation_engine import compute_evaluation_update
from app.services.serving import ServingBackend

logger = structlog.get_logger(__name__)

TRIGGER_SCHEDULED_DRIFT_CHECK = "scheduled_drift_check"

# Notification type for issue #130's notify_admins (Notification.type is free text, no
# registered enum - see app/models/notification.py).
MODEL_QUALITY_DRIFT_DETECTED = "MODEL_QUALITY_DRIFT_DETECTED"


def has_new_regression(baseline_check: dict | None, new_check: dict | None) -> bool:
    """True when `new_check.regressions_found` contains a `record_index` that
    `baseline_check.regressions_found` does not - i.e. a golden-set record that used to pass and
    now fails. Both are `GeneralDomainRegressionCheck.model_dump()` dicts (or None).

    A missing/never-evaluated baseline can't prove a failure is old, so any regression in the
    new run is treated as new (fail safe toward notifying rather than staying silent).
    """
    new_indices = {
        r.get("record_index") for r in (new_check or {}).get("regressions_found", [])
    }
    if not new_indices:
        return False
    baseline_indices = {
        r.get("record_index")
        for r in (baseline_check or {}).get("regressions_found", [])
    }
    return bool(new_indices - baseline_indices)


def run_drift_check_for_version(
    db: Session, model_version: ModelVersion, backend: ServingBackend
) -> ModelDriftCheck | None:
    """Re-run the golden-set evaluation for one `DEPLOYED` model version and record the result.

    Returns the created `ModelDriftCheck` row, or `None` when the version has no eval set to
    check against (or the eval set/adapter can't be loaded) - `compute_evaluation_update`'s
    `ValueError` is caught and logged exactly like `evaluation_worker.process_next_evaluation`
    does, so one model version missing an eval set never stops the cycle for the rest.
    """
    try:
        update = compute_evaluation_update(db, model_version, backend)
    except ValueError as exc:
        logger.warning(
            "drift_check_skipped",
            model_id=model_version.model_id,
            version=model_version.version,
            reason=str(exc),
        )
        return None

    new_regression_check = (
        update.general_domain_regression_check.model_dump()
        if update.general_domain_regression_check
        else None
    )
    regressed = has_new_regression(
        model_version.general_domain_regression_check, new_regression_check
    )

    record = ModelDriftCheck(
        model_version_id=model_version.id,
        trigger=TRIGGER_SCHEDULED_DRIFT_CHECK,
        eval_set_id=update.eval_set_id,
        eval_set_version=update.eval_set_version,
        eval_loss_trend=(
            update.eval_loss_trend.model_dump() if update.eval_loss_trend else None
        ),
        qualitative_comparison=(
            update.qualitative_comparison.model_dump()
            if update.qualitative_comparison
            else None
        ),
        general_domain_regression_check=new_regression_check,
        new_regression_detected=regressed,
        created_at=datetime.now(timezone.utc),
    )
    db.add(record)
    db.flush()

    if regressed:
        notification_service.notify_admins(
            db,
            type=MODEL_QUALITY_DRIFT_DETECTED,
            message=(
                f"Model {model_version.model_id!r} version {model_version.version} "
                "(DEPLOYED) failed a golden-set record in scheduled drift check #"
                f"{record.id} that passed at promotion time - possible quality regression."
            ),
            resource_ref=f"{model_version.model_id}:v{model_version.version}",
        )

    return record
