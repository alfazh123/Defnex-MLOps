import uuid
from datetime import datetime, timezone

from sqlalchemy.orm import Session

from app.config import settings
from app.models.model import ModelVersion
from app.models.promotion import PromotionDecision
from app.schemas.promotion import (
    DecisionCreateRequest,
    DecisionRecord,
    LadderActionRequest,
    RollbackRequest,
)
from app.services import deployment_service
from app.services.model_service import get_evaluation

# EVALUATED -> PROMOTED|REJECTED is the only transition this story records
# (model-promotion-approval-workflow.md §2). Every other state - including a version that
# already has a decision (no longer EVALUATED) - is rejected the same way, matching openapi.yaml's
# combined 409 case ("not EVALUATED, or a decision already exists").
_VALID_TRANSITIONS: dict[str, set[str]] = {
    "EVALUATED": {"PROMOTED", "REJECTED"},
}

# Staging/production ladder (issues #69/#70, PRD §16.2 "Promotion Ladder", §41 Principle 4):
#   EVALUATED --deploy-staging-->  STAGING --validate-staging--> VALIDATED
#   VALIDATED --promote-production--> [production pointer move -> DEPLOYED]
# The terminal registry status stays DEPLOYED - the codebase's single source of truth for
# "which version is production" (deployment_service.py:27) and what the `prod` alias, inventory
# status, and get_deployment_status resolve. The ladder's PRODUCTION step is recorded in the
# audit history as a "PRODUCTION" PromotionDecision; the registry never holds a second,
# competing "which version is live" flag (PRD §41 Principle 2, issue #36).
#
# Legacy flow (unchanged, backward compatible): EVALUATED -> PROMOTED --deploy--> DEPLOYED.
# A PRODUCTION deploy is therefore offered to legacy PROMOTED versions as well.
_LADDER_STAGING_SOURCE = {"EVALUATED"}
_LADDER_VALIDATION_SOURCE = {"STAGING"}
_LADDER_PROMOTION_SOURCE = {"VALIDATED", "PROMOTED"}

# Statuses that represent a validated production candidate (PRD §26.3).
# Used by _validate_staging_gate to block unvalidated versions from production.
_VALIDATED_PRODUCTION_STATUSES = {"VALIDATED", "PROMOTED", "DEPLOYED", "PRODUCTION"}

# Versions a rollback may restore to (issues #69/#70, PRD §14.4 "Rollback"): an earlier version
# that is still immutable and available. RETIRED = "was DEPLOYED, then superseded" (the only thing
# that sets it is deployment_service.deploy), PROMOTED = approved but never deployed, and the
# ladder statuses STAGING/VALIDATED/PRODUCTION cover candidates and previously active versions.
# The currently-active DEPLOYED version is deliberately NOT a rollback target (rolling back to it
# is a no-op self-deploy, and the pre-#70 test suite locked that behavior).
_ROLLBACK_TARGET_STATUSES = (
    "PROMOTED",
    "RETIRED",
    "STAGING",
    "VALIDATED",
    "PRODUCTION",
)


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


def _record_ladder_step(
    db: Session,
    model_version: ModelVersion,
    *,
    decision: str,
    request: LadderActionRequest,
    evidence: dict | None,
) -> PromotionDecision:
    """Append a PromotionDecision audit row for a ladder step (issues #69/#70) - who approved,
    when, and (for the staging-validation gate) the frozen signal evidence the approval was based
    on. Same schema as the existing PROMOTED/REJECTED/ROLLBACK decisions so one audit trail covers
    every human trigger; the deploy steps (STAGING/PRODUCTION) carry no evidence snapshot because
    they respond to an operator action, not to a fresh evaluation."""
    decision_row = PromotionDecision(
        decision_id=f"{decision.lower()}-{uuid.uuid4().hex[:6]}",
        model_version_id=model_version.id,
        decision=decision,
        decided_by=request.decided_by,
        decided_at=datetime.now(timezone.utc),
        evidence_snapshot=evidence,
        eval_set_id=model_version.eval_set_id,
        eval_set_version=model_version.eval_set_version,
        rationale=request.rationale,
        rollback_of_version=None,
    )
    db.add(decision_row)
    model_version.promotion_decision_ref = decision_row.decision_id
    db.flush()
    return decision_row


def stage_deploy(
    db: Session, model_version: ModelVersion
) -> tuple[deployment_service.Deployment, None]:
    """Move a candidate onto the staging target without disturbing the production pointer
    (issues #69/#70, PRD §16.2/§16.3).

    `deployment_service.deploy` is the only place the adapter is loaded, smoke-tested
    (`inference_smoke_enabled`, PRD §38.4 "Staging smoke test runs automatically") and the
    pointer moved, so it is reused - but it retires the current DEPLOYED (production) version and
    returns the candidate as DEPLOYED. Staging must not steal the production pointer, so after the
    deploy the candidate is demoted to STAGING and (when a production version existed) that version
    is restored as DEPLOYED. The deployment rows still record the staging deploy (history); the
    registry keeps exactly one DEPLOYED production version - and that one stays the pre-staging
    version. Returns the new Deployment row and None (nothing was superseded from the caller's
    point of view; the prod pointer is unchanged)."""
    deployment, previous = deployment_service.deploy(
        db, model_version, environment="staging"
    )
    model_version.status = "STAGING"
    if previous is not None:
        # demote the candidate first (flush) so the partial unique index
        # `uq_model_versions_one_deployed` never sees two DEPLOYED rows, then restore.
        db.flush()
        previous.status = "DEPLOYED"
    db.flush()
    return deployment, None


def deploy_to_staging(
    db: Session, model_version: ModelVersion, request: LadderActionRequest
) -> PromotionDecision:
    """Ladder step 1 (issue #69 AC 1): deploy an EVALUATED candidate to staging and record the
    audit decision. Human/authorized-triggered only (the endpoint requires an admin) - a candidate
    is staged because an operator stages it, never on a numeric threshold (PRD §41)."""
    if model_version.status not in _LADDER_STAGING_SOURCE:
        raise ValueError(
            f"Cannot deploy model_id {model_version.model_id!r} version {model_version.version} "
            f"to staging: status is {model_version.status!r}, requires EVALUATED"
        )
    stage_deploy(db, model_version)
    return _record_ladder_step(
        db, model_version, decision="STAGING", request=request, evidence=None
    )


def validate_staging(
    db: Session, model_version: ModelVersion, request: LadderActionRequest
) -> PromotionDecision:
    """Ladder step 2 (issue #69 AC 2/3): human approval that the staged candidate passed its
    staging checks. The candidate must already be STAGING (only reachable through
    `deploy_to_staging`, so production is structurally blocked until this step runs) and the
    frozen evaluation snapshot is recorded as the gate result the approval was based on."""
    if model_version.status not in _LADDER_VALIDATION_SOURCE:
        raise ValueError(
            f"Cannot validate model_id {model_version.model_id!r} version {model_version.version} "
            f"for production: status is {model_version.status!r}, requires STAGING"
        )
    model_version.status = "VALIDATED"
    return _record_ladder_step(
        db,
        model_version,
        decision="VALIDATED",
        request=request,
        evidence=get_evaluation(model_version).model_dump(),
    )


class StagingGateNotMet(Exception):
    """Raised when production promotion is blocked because the model version
    has not passed through STAGING→VALIDATED (PRD §26.3, issue #80).

    Distinct from `ValueError` so the API layer can map it to its own 409 code.
    """


def _validate_staging_gate(model_version: ModelVersion) -> None:
    """Reject production promotion if staging hasn't been validated (PRD §26.3, issue #80)."""
    if model_version.status not in _VALIDATED_PRODUCTION_STATUSES:
        raise StagingGateNotMet(
            f"Model version {model_version.version} has not been validated in staging; "
            f"must pass STAGING→VALIDATED before production (PRD §26.3)"
        )


def promote_to_production(
    db: Session, model_version: ModelVersion, request: LadderActionRequest
) -> PromotionDecision:
    """Ladder step 3 (issue #69 AC 2/4/5): the authorized promotion of a VALIDATED candidate to the
    production pointer, calling `deployment_service.deploy(environment='production')` so the move
    goes through the same safe path as every deployment (smoke test before the pointer moves,
    issue #41; checksum verification, issue #62). The registry's terminal status is DEPLOYED
    (single source of truth), and the audit decision records the PRODUCTION step.

    A legacy PROMOTED version may also take this path (backward compatibility), where it is
    equivalent to the existing EVALUATED -> PROMOTED --deploy--> DEPLOYED flow plus an audit row.
    """
    _validate_staging_gate(model_version)
    if model_version.status not in _LADDER_PROMOTION_SOURCE:
        raise ValueError(
            f"Cannot promote model_id {model_version.model_id!r} version {model_version.version} "
            f"to production: status is {model_version.status!r}, requires VALIDATED "
            "(ladder) or PROMOTED (legacy)"
        )
    deployment_service.deploy(db, model_version, environment="production")
    return _record_ladder_step(
        db, model_version, decision="PRODUCTION", request=request, evidence=None
    )


def rollback(
    db: Session,
    target: ModelVersion,
    request: RollbackRequest,
    environment: str | None = None,
) -> PromotionDecision:
    """Roll back a model's deployed version to an earlier `target` (rollback-of-version)
    (model-promotion-approval-workflow.md §9), reusing the PromotionDecision record with
    decision=ROLLBACK and evidence_snapshot=None - a rollback responds to an observed production
    problem, not new offline evaluation data (the problem must be described in `rationale`
    instead). Human-triggered only, same reasoning as `create_decision` (§10 Decision 1).

    Issues #69/#70 (PRD §14.4 "Rollback"): the target set is widened to the ladder statuses
    (STAGING/VALIDATED/PRODUCTION) so a ladder-created version can be rolled back exactly like a
    legacy one, and `environment` lets the deployment rows record which environment the pointer
    moved to (POST /models/{model_id}/rollback keeps its legacy no-environment behavior). The
    rollback goes through `deployment_service.deploy` (same pointer move + smoke test +
    checksum path, so a failed production deploy cannot destroy the current healthy version -
    PRD §38.4/§43)."""

    # openapi.yaml RollbackRequest: target "must already be PROMOTED or have been previously
    # DEPLOYED". RETIRED is exactly "was DEPLOYED, then superseded" (`deployment_service.deploy`
    # is the only thing that sets it), so PROMOTED|RETIRED covers the documented condition, and the
    # ladder statuses cover the #69/#70 additions.
    if target.status not in _ROLLBACK_TARGET_STATUSES:
        raise ValueError(
            f'model_id "{target.model_id}" version {target.version} is {target.status}; '
            "rollback target must be PROMOTED, have been previously DEPLOYED, or be on the "
            "staging ladder (STAGING/VALIDATED/PRODUCTION)"
        )

    # openapi.yaml describes rollback as moving the same deployment pointer POST .../deploy moves,
    # so it delegates rather than keeping a second, divergent implementation of the transition.
    # `from_version` is "the version being rolled back from" (DecisionRecord.version's ROLLBACK
    # semantics) - None when nothing was deployed, in which case the decision anchors to `target`
    # itself, since PromotionDecision.model_version_id is NOT NULL.
    _, from_version = deployment_service.deploy(db, target, environment=environment)

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
