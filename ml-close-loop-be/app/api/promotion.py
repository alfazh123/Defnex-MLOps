from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session

from app.api.errors import APIError
from app.api.models import _get_model_version_or_404
from app.db.session import get_db
from app.schemas.common import ErrorResponse
from app.schemas.promotion import DecisionCreateRequest, DecisionRecord, RollbackRequest
from app.services import promotion_service

router = APIRouter(tags=["Decisions"])


@router.post(
    "/models/{model_id}/versions/{version}/decisions",
    response_model=DecisionRecord,
    status_code=201,
    responses={404: {"model": ErrorResponse}, 409: {"model": ErrorResponse}},
)
def create_decision(
    model_id: str, version: int, request: DecisionCreateRequest, db: Session = Depends(get_db)
) -> DecisionRecord:
    model_version = _get_model_version_or_404(db, model_id, version)
    try:
        decision = promotion_service.create_decision(db, model_version, request)
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
def rollback_model(model_id: str, request: RollbackRequest, db: Session = Depends(get_db)) -> DecisionRecord:
    target = _get_model_version_or_404(db, model_id, request.rollback_of_version)
    try:
        decision = promotion_service.rollback(db, target, request)
    except ValueError as exc:
        raise APIError(409, "ROLLBACK_NOT_ALLOWED", str(exc)) from exc
    db.commit()
    return promotion_service.to_schema(decision)
