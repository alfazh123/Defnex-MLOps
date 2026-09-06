import uuid
from datetime import datetime, timezone

from sqlalchemy.orm import Session

from app.config import settings
from app.models.model import ModelVersion
from app.models.promotion import PromotionDecision
from app.schemas.promotion import DecisionCreateRequest, DecisionRecord, RollbackRequest
from app.services import deployment_service
from app.services.model_service import get_evaluation

# EVALUATED -> PROMOTED|REJECTED is the only transition this story records
# (model-promotion-approval-workflow.md §2). Every other state - including a version that
# already has a decision (no longer EVALUATED) - is rejected the same way, matching openapi.yaml's
# combined 409 case ("not EVALUATED, or a decision already exists").
_VALID_TRANSITIONS: dict[str, set[str]] = {
    "EVALUATED": {"PROMOTED", "REJECTED"},
}


class EvalGateBlocked(Exception):
    """Raised when a PROMOTED decision fails the eval gate (issue #43).

    Distinct from `ValueError` (transition not allowed) so the API layer can map it to
    its own 409 code (`PROMOTION_GATE_BLOCKED`).
    """


def _gate_reasons(model_version: ModelVersion) -> list[str]:
    """Documented, non-invented checks that a PROMOTED decision must clear before it can be
    recorded (model-promotion-approval-workflow.md §6: "any signal negative or absent ->
    default to not promoting"; §13 majority-win; §5 regressions; §11 eval-loss comparison).
    Each toggle lives in Settings; gate criteria never turn into automatic promotion."""

    reasons: list[str] = []
    if settings.eval_gate_require_eval_set_reference and (
        model_version.eval_set_id is None or model_version.eval_set_version is None
    ):
        reasons.append(
            "this model version has no eval set reference recorded; promotion requires "
            "evaluation against a stored golden/eval set"
        )

    qualitative = model_version.qualitative_comparison
    if (
        settings.eval_gate_require_qualitative_majority
        and qualitative is not None
        and qualitative.get("wins", 0) <= qualitative.get("losses", 0)
    ):
        reasons.append(
            "qualitative comparison has no majority win "
            f"(wins {qualitative.get('wins')} <= losses {qualitative.get('losses')})"
        )

    regression = model_version.general_domain_regression_check
    if (
        settings.eval_gate_require_no_general_regression
        and regression is not None
        and regression.get("regressions_found")
    ):
        reasons.append(
            "general-domain regression check found "
            f"{len(regression['regressions_found'])} regression(s)"
        )

    trend = model_version.eval_loss_trend
    if (
        settings.eval_gate_require_eval_loss_not_worse
        and trend is not None
        and (previous := trend.get("previous_version_eval_loss")) is not None
        and trend["this_version_eval_loss"] > previous
    ):
        reasons.append(
            "eval loss is worse than the previous version "
            f"({trend['this_version_eval_loss']} > {previous})"
        )

    return reasons


def create_decision(
    db: Session, model_version: ModelVersion, request: DecisionCreateRequest
) -> PromotionDecision:
    """Record a human promotion/rejection decision and transition the model version
    EVALUATED -> PROMOTED|REJECTED (model-promotion-approval-workflow.md §2/§7/§8). Human-triggered
    only - no automatic promotion based on numeric thresholds (§10 Decision 1).

    A PROMOTED decision must also clear the eval gate (`EvalGateBlocked`); REJECTED is always
    allowed - a rejection is a human call that needs no evidence that promotion would accept.
    """

    allowed = _VALID_TRANSITIONS.get(model_version.status, set())
    if request.decision not in allowed:
        raise ValueError(
            f"Cannot record decision {request.decision!r} for model_id {model_version.model_id!r} "
            f"version {model_version.version} in status {model_version.status!r} (must be EVALUATED)"
        )

    if request.decision == "PROMOTED":
        reasons = _gate_reasons(model_version)
        if reasons:
            raise EvalGateBlocked(
                "Promotion blocked by the eval gate: " + "; ".join(reasons)
            )

    decision = PromotionDecision(
        decision_id=f"decision-{uuid.uuid4().hex[:6]}",
        model_version_id=model_version.id,
        decision=request.decision,
        decided_by=request.decided_by,
        decided_at=datetime.now(timezone.utc),
        evidence_snapshot=get_evaluation(model_version).model_dump(),
        eval_set_id=model_version.eval_set_id,
        eval_set_version=model_version.eval_set_version,
        rationale=request.rationale,
        rollback_of_version=None,
    )
    db.add(decision)
    model_version.status = request.decision
    model_version.promotion_decision_ref = decision.decision_id
    db.flush()
    return decision


def rollback(
    db: Session, target: ModelVersion, request: RollbackRequest
) -> PromotionDecision:
    """Roll back a model's deployed version to an earlier `target` (rollback-of-version)
    (model-promotion-approval-workflow.md §9), reusing the PromotionDecision record with
    decision=ROLLBACK and evidence_snapshot=None - a rollback responds to an observed production
    problem, not new offline evaluation data (the problem must be described in `rationale`
    instead). Human-triggered only, same reasoning as `create_decision` (§10 Decision 1)."""

    # openapi.yaml RollbackRequest: target "must already be PROMOTED or have been previously
    # DEPLOYED". RETIRED is exactly "was DEPLOYED, then superseded" (`deployment_service.deploy`
    # is the only thing that sets it), so the pair covers the documented condition.
    if target.status not in ("PROMOTED", "RETIRED"):
        raise ValueError(
            f'model_id "{target.model_id}" version {target.version} is {target.status}; '
            "rollback target must be PROMOTED or have been previously DEPLOYED"
        )

    # openapi.yaml describes rollback as moving the same deployment pointer POST .../deploy moves,
    # so it delegates rather than keeping a second, divergent implementation of the transition.
    # `from_version` is "the version being rolled back from" (DecisionRecord.version's ROLLBACK
    # semantics) - None when nothing was deployed, in which case the decision anchors to `target`
    # itself, since PromotionDecision.model_version_id is NOT NULL.
    _, from_version = deployment_service.deploy(db, target)

    decision = PromotionDecision(
        decision_id=f"rollback-{uuid.uuid4().hex[:6]}",
        model_version_id=(from_version or target).id,
        decision="ROLLBACK",
        decided_by=request.decided_by,
        decided_at=datetime.now(timezone.utc),
        evidence_snapshot=None,
        rationale=request.rationale,
        rollback_of_version=target.version,
    )
    db.add(decision)
    target.promotion_decision_ref = decision.decision_id
    db.flush()
    return decision


def to_schema(decision: PromotionDecision) -> DecisionRecord:
    model_version = decision.model_version
    return DecisionRecord(
        decision_id=decision.decision_id,
        model_id=model_version.model_id,
        version=model_version.version,
        decision=decision.decision,
        decided_by=decision.decided_by,
        decided_at=decision.decided_at,
        evidence_snapshot=decision.evidence_snapshot,
        eval_set_id=decision.eval_set_id,
        eval_set_version=decision.eval_set_version,
        rationale=decision.rationale,
        rollback_of_version=decision.rollback_of_version,
    )
