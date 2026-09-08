from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session

from app.api.deps import get_model_version_or_404, require_admin
from app.api.errors import APIError
from app.db.session import get_db
from app.models.user import User
from app.schemas.common import ErrorResponse
from app.schemas.promotion import DecisionCreateRequest, DecisionRecord, RollbackRequest
from app.services import deployment_service, promotion_service

router = APIRouter(tags=["Decisions"])


@router.post(
    "/models/{model_id}/versions/{version}/decisions",
    response_model=DecisionRecord,
    status_code=201,
    responses={404: {"model": ErrorResponse}, 409: {"model": ErrorResponse}},
)
def create_decision(
    model_id: str,
    version: int,
    request: DecisionCreateRequest,
    db: Session = Depends(get_db),
    _admin: User = Depends(require_admin),
) -> DecisionRecord:
    model_version = get_model_version_or_404(db, model_id, version)
    try:
        decision = promotion_service.create_decision(db, model_version, request)
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
    _admin: User = Depends(require_admin),
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
