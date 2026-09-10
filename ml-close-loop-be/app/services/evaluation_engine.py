"""Server-side evaluation computation (issue #128, PRD §15/§48 InferenceTarget).

`compute_evaluation_update` is the only thing the evaluation worker
(`app/workers/evaluation_worker.py`) calls: it runs real inference through the existing
`ServingBackend` against the model version's stored golden/eval set (issue #43) and returns an
`EvaluationUpdateRequest` built entirely from that inference, for `model_service.submit_evaluation`
to apply - no caller-supplied numbers involved.

What each signal is grounded in (deliberately not inventing a judge/scoring model - CLAUDE.md
"Don't invent evaluation metrics... that aren't already in the code or openapi.yaml", and
validation_service.py's own precedent of deferring undefined thresholds, e.g. its W2/W5/Q2/Q4
notes at validation_service.py:124-128):

- `qualitative_comparison`: a "win" is a golden-set question the candidate produced a valid,
  non-empty completion for (PRD §15.1 Layer 1 "Structural Validity" - valid output, no malformed
  response); a "loss" is a `ServingBackend.generate` failure (`InferenceError`) or empty output.
  `ties` is always 0 - there is no reference-answer judge in this codebase to produce a tie.
- `general_domain_regression_check`: reuses the same generation pass; a "regression" is a record
  whose generation failed, recorded so a human can inspect it (mirrors the same Layer 1 bar).
- `eval_loss_trend`: sourced from `training_run.eval_loss` (already persisted, self-reported by
  the training subprocess - training_service.py:222, a separate concern from this issue) rather
  than a raw float in the evaluation request body; `previous_version_eval_loss` looks at the
  immediately preceding version of the same model_id with a recorded training loss.
"""

from __future__ import annotations

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.models.model import ModelVersion
from app.schemas.model import (
    EvalLossTrend,
    EvaluationUpdateRequest,
    GeneralDomainRegressionCheck,
    QualitativeComparison,
)
from app.services import eval_set_service
from app.services.serving import InferenceError, ServingBackend, ServingError
from app.services.validation_service import _user_content


def _previous_version_eval_loss(
    db: Session, model_version: ModelVersion
) -> float | None:
    """The most recent earlier version of the same model_id with a recorded training eval_loss,
    for the `eval_loss_trend.previous_version_eval_loss` comparison point."""

    prev = db.scalar(
        select(ModelVersion)
        .where(
            ModelVersion.model_id == model_version.model_id,
            ModelVersion.version < model_version.version,
        )
        .order_by(ModelVersion.version.desc())
    )
    if prev is None or prev.training_run is None:
        return None
    return prev.training_run.eval_loss


def compute_evaluation_update(
    db: Session, model_version: ModelVersion, backend: ServingBackend
) -> EvaluationUpdateRequest:
    """Run real inference against the stored golden/eval set and build the resulting
    `EvaluationUpdateRequest`. Raises `ValueError` (caller/worker skips this cycle, logs, and
    leaves the model version unevaluated for a future retrigger) when there is no eval set
    reference, the referenced eval set/version does not exist, or the adapter cannot be loaded
    for evaluation."""

    if model_version.eval_set_id is None or model_version.eval_set_version is None:
        raise ValueError(
            f"model_id {model_version.model_id!r} version {model_version.version} has no "
            "eval_set_id/eval_set_version to evaluate against"
        )
    eval_set_version = eval_set_service.get_eval_set_version(
        db, model_version.eval_set_id, model_version.eval_set_version
    )
    if eval_set_version is None:
        raise ValueError(
            f'eval set "{model_version.eval_set_id}" version {model_version.eval_set_version} '
            "not found"
        )

    try:
        backend.deploy(model_version)
    except ServingError as exc:
        raise ValueError(
            f"could not load adapter for evaluation "
            f"({model_version.model_id} v{model_version.version}): {exc}"
        ) from exc

    wins = 0
    losses = 0
    regressions: list[dict] = []
    try:
        for index, record in enumerate(eval_set_version.records or []):
            prompt = _user_content(record)
            if prompt is None:
                continue
            try:
                text = backend.generate(
                    prompt, model_version.model_id, model_version.version
                )
                ok = bool(text)
            except InferenceError:
                ok = False
            if ok:
                wins += 1
            else:
                losses += 1
                regressions.append({"record_index": index, "prompt": prompt[:200]})
    finally:
        backend.unload(model_version)

    total = wins + losses
    qualitative_comparison = (
        QualitativeComparison(
            question_table_version=eval_set_version.version,
            wins=wins,
            losses=losses,
            ties=0,
            total=total,
        )
        if total
        else None
    )
    general_domain_regression_check = (
        GeneralDomainRegressionCheck(checked=True, regressions_found=regressions)
        if total
        else None
    )

    eval_loss_trend = None
    this_version_eval_loss = (
        model_version.training_run.eval_loss if model_version.training_run else None
    )
    if this_version_eval_loss is not None:
        eval_loss_trend = EvalLossTrend(
            this_version_eval_loss=this_version_eval_loss,
            previous_version_eval_loss=_previous_version_eval_loss(db, model_version),
        )

    return EvaluationUpdateRequest(
        eval_set_id=model_version.eval_set_id,
        eval_set_version=model_version.eval_set_version,
        eval_loss_trend=eval_loss_trend,
        qualitative_comparison=qualitative_comparison,
        general_domain_regression_check=general_domain_regression_check,
    )
