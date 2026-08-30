import uuid
from datetime import datetime, timezone

from sqlalchemy.orm import Session

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


def create_decision(db: Session, model_version: ModelVersion, request: DecisionCreateRequest) -> PromotionDecision:
    """Record a human promotion/rejection decision and transition the model version
    EVALUATED -> PROMOTED|REJECTED (model-promotion-approval-workflow.md §2/§7/§8). Human-triggered
    only - no automatic promotion based on numeric thresholds (§10 Decision 1)."""

    allowed = _VALID_TRANSITIONS.get(model_version.status, set())
    if request.decision not in allowed:
        raise ValueError(
            f"Cannot record decision {request.decision!r} for model_id {model_version.model_id!r} "
            f"version {model_version.version} in status {model_version.status!r} (must be EVALUATED)"
        )

    decision = PromotionDecision(
        decision_id=f"decision-{uuid.uuid4().hex[:6]}",
        model_version_id=model_version.id,
        decision=request.decision,
        decided_by=request.decided_by,
        decided_at=datetime.now(timezone.utc),
        evidence_snapshot=get_evaluation(model_version).model_dump(),
        rationale=request.rationale,
        rollback_of_version=None,
    )
    db.add(decision)
    model_version.status = request.decision
    model_version.promotion_decision_ref = decision.decision_id
    db.flush()
    return decision


def rollback(db: Session, target: ModelVersion, request: RollbackRequest) -> PromotionDecision:
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


def list_decisions_for_model(db: Session, model_id: str) -> list[PromotionDecision]:
    """Return all promotion decisions for a given model_id, newest first."""
    from sqlalchemy import select
    from app.models.model import ModelVersion

    return list(
        db.scalars(
            select(PromotionDecision)
            .join(ModelVersion, PromotionDecision.model_version_id == ModelVersion.id)
            .where(ModelVersion.model_id == model_id)
            .order_by(PromotionDecision.decided_at.desc())
        )
    )


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
        rationale=decision.rationale,
        rollback_of_version=decision.rollback_of_version,
    )
