from fastapi import APIRouter, Depends, Request
import structlog
from sqlalchemy.orm import Session

from app.api.deps import get_model_version_or_404, require_permission
from app.api.errors import APIError
from app.config import settings
from app.db.session import get_db
from app.limiter import limiter
from app.models.deployment import Deployment
from app.models.user import User
from app.rbac import DEPLOY, PROMOTE, ROLLBACK, VALIDATE_STAGING
from app.schemas.common import ErrorResponse
from app.schemas.promotion import (
    DecisionCreateRequest,
    DecisionRecord,
    LadderActionRequest,
    RollbackRequest,
)
from app.services import deployment_service, promotion_service

logger = structlog.get_logger(__name__)

router = APIRouter(tags=["Decisions"])


@router.post(
    "/models/{model_id}/versions/{version}/decisions",
    response_model=DecisionRecord,
    status_code=201,
    responses={404: {"model": ErrorResponse}, 409: {"model": ErrorResponse}},
)
@limiter.limit(settings.rate_limit_promotion_decision)
def create_decision(
    request: Request,
    model_id: str,
    version: int,
    body: DecisionCreateRequest,
    db: Session = Depends(get_db),
    _admin: User = Depends(require_permission(PROMOTE)),
) -> DecisionRecord:
    model_version = get_model_version_or_404(db, model_id, version)
    try:
        decision = promotion_service.create_decision(db, model_version, body)
    except promotion_service.EvalGateBlocked as exc:
        raise APIError(409, "PROMOTION_GATE_BLOCKED", str(exc)) from exc
    except ValueError as exc:
        raise APIError(409, "DECISION_NOT_ALLOWED", str(exc)) from exc
    db.commit()
    return promotion_service.to_schema(decision)


@router.post(
    "/models/{model_id}/rollback",
    response_model=DecisionRecord,
    status_code=201,
    responses={404: {"model": ErrorResponse}, 409: {"model": ErrorResponse}},
)
def rollback_model(
    model_id: str,
    request: RollbackRequest,
    db: Session = Depends(get_db),
    _admin: User = Depends(require_permission(ROLLBACK)),
) -> DecisionRecord:
    target = get_model_version_or_404(db, model_id, request.rollback_of_version)
    try:
        decision = promotion_service.rollback(db, target, request)
    except deployment_service.DeploymentLockTimeout as exc:
        raise APIError(503, "GPU_LOCK_TIMEOUT", str(exc)) from exc
    except ValueError as exc:
        raise APIError(409, "ROLLBACK_NOT_ALLOWED", str(exc)) from exc
    db.commit()
    return promotion_service.to_schema(decision)


def _ladder_error(exc: ValueError, code: str) -> APIError:
    """Map a ladder/release step's domain error to the endpoint's 409 (mirrors the existing
    DECISION_NOT_ALLOWED pattern for create_decision, so every human-triggered decision returns
    the same shape)."""
    return APIError(409, code, str(exc))


@router.post(
    "/models/{model_id}/versions/{version}/deploy-staging",
    response_model=DecisionRecord,
    status_code=201,
    responses={
        404: {"model": ErrorResponse},
        409: {"model": ErrorResponse},
        503: {"model": ErrorResponse},
    },
)
def deploy_to_staging_endpoint(
    model_id: str,
    version: int,
    request: LadderActionRequest,
    db: Session = Depends(get_db),
    _admin: User = Depends(require_permission(DEPLOY)),
) -> DecisionRecord:
    """Ladder step 1 (issue #69 AC 1), PRD §16.2: deploy an EVALUATED candidate to staging without
    touching the production pointer. Records the STAGING decision; smoke test runs automatically as
    part of the deploy (PRD §38.4)."""
    model_version = get_model_version_or_404(db, model_id, version)
    try:
        decision = promotion_service.deploy_to_staging(db, model_version, request)
    except deployment_service.DeploymentLockTimeout as exc:
        raise APIError(503, "GPU_LOCK_TIMEOUT", str(exc)) from exc
    except ValueError as exc:
        raise _ladder_error(exc, "STAGING_DEPLOY_NOT_ALLOWED") from exc
    db.commit()
    return promotion_service.to_schema(decision)


@router.post(
    "/models/{model_id}/versions/{version}/validate-staging",
    response_model=DecisionRecord,
    status_code=201,
    responses={404: {"model": ErrorResponse}, 409: {"model": ErrorResponse}},
)
def validate_staging_endpoint(
    model_id: str,
    version: int,
    request: LadderActionRequest,
    db: Session = Depends(get_db),
    _admin: User = Depends(require_permission(VALIDATE_STAGING)),
) -> DecisionRecord:
    """Ladder step 2 (issue #69 AC 2/3): the human quality-gate approval that the staged candidate
    passed integration validation. Only a STAGING version may be validated, which structurally
    blocks production promotion until staging happened. Records the VALIDATED decision with the
    frozen evaluation snapshot as the gate result (PRD §8.4)."""
    model_version = get_model_version_or_404(db, model_id, version)
    try:
        decision = promotion_service.validate_staging(db, model_version, request)
    except ValueError as exc:
        raise _ladder_error(exc, "VALIDATION_NOT_ALLOWED") from exc
    db.commit()
    return promotion_service.to_schema(decision)


@router.post(
    "/models/{model_id}/versions/{version}/promote-production",
    response_model=DecisionRecord,
    status_code=201,
    responses={
        404: {"model": ErrorResponse},
        409: {"model": ErrorResponse},
        503: {"model": ErrorResponse},
    },
)
def promote_production_endpoint(
    model_id: str,
    version: int,
    request: LadderActionRequest,
    db: Session = Depends(get_db),
    _admin: User = Depends(require_permission(PROMOTE)),
) -> DecisionRecord:
    """Ladder step 3 (issue #69 AC 2/4/5): the authorized controlled promotion of a VALIDATED
    (or legacy PROMOTED) candidate to the production pointer. Records the PRODUCTION decision;
    the registry terminal status stays DEPLOYED (single source of truth for production)."""
    model_version = get_model_version_or_404(db, model_id, version)
    try:
        decision = promotion_service.promote_to_production(db, model_version, request)
    except deployment_service.DeploymentLockTimeout as exc:
        raise APIError(503, "GPU_LOCK_TIMEOUT", str(exc)) from exc
    except promotion_service.StagingGateNotMet as exc:
        raise APIError(409, "GATE_NOT_MET", str(exc)) from exc
    except ValueError as exc:
        raise _ladder_error(exc, "PRODUCTION_PROMOTION_NOT_ALLOWED") from exc
    db.commit()
    logger.info(
        "production_promotion",
        model_id=model_id,
        version=version,
        decided_by=request.decided_by,
        decision_id=decision.decision_id,
    )
    return promotion_service.to_schema(decision)


@router.post(
    "/deployments/{deployment_id}/rollback",
    response_model=DecisionRecord,
    status_code=201,
    responses={
        404: {"model": ErrorResponse},
        409: {"model": ErrorResponse},
        503: {"model": ErrorResponse},
    },
)
def rollback_deployment(
    deployment_id: str,
    request: LadderActionRequest,
    db: Session = Depends(get_db),
    _admin: User = Depends(require_permission(ROLLBACK)),
) -> DecisionRecord:
    """Issue #70, PRD §14.4 "Rollback": restore the previous immutable version for a deployment
    target, discovered by the deployment row (POST .../deployments/{id}/rollback, PRD §25). The
    version is located via the deployment's recorded (model_id, model_version), the pointer is
    moved to the previous version, and the ROLLBACK decision records which environment was rolled
    back. The previous version is never deleted - artifacts are immutable (artifact_storage)."""
    deployment = db.get(Deployment, deployment_id)
    if deployment is None:
        raise APIError(
            404, "DEPLOYMENT_NOT_FOUND", f'deployment_id "{deployment_id}" not found'
        )
    target = get_model_version_or_404(db, deployment.model_id, deployment.model_version)
    try:
        decision = promotion_service.rollback(
            db,
            target,
            RollbackRequest(
                rollback_of_version=deployment.model_version,
                decided_by=request.decided_by,
                rationale=request.rationale,
            ),
            environment=deployment.environment,
        )
    except deployment_service.DeploymentLockTimeout as exc:
        raise APIError(503, "GPU_LOCK_TIMEOUT", str(exc)) from exc
    except ValueError as exc:
        raise _ladder_error(exc, "ROLLBACK_NOT_ALLOWED") from exc
    db.commit()
    return promotion_service.to_schema(decision)
