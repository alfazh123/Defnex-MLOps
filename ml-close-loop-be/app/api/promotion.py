from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session

from app.api.deps import get_model_version_or_404
from app.api.errors import APIError
from app.db.session import get_db
from app.schemas.common import ErrorResponse
from app.schemas.promotion import DecisionCreateRequest, DecisionRecord, RollbackRequest
from app.services import promotion_service

router = APIRouter(tags=["Decisions"])


@router.get(
    "/models/{model_id}/decisions",
    response_model=list[DecisionRecord],
    responses={404: {"model": ErrorResponse}},
)
def list_model_decisions(model_id: str, db: Session = Depends(get_db)) -> list[DecisionRecord]:
    from app.models.model import Model as ModelORM
    if db.get(ModelORM, model_id) is None:
        raise APIError(404, "MODEL_NOT_FOUND", f'model_id "{model_id}" not found')
    decisions = promotion_service.list_decisions_for_model(db, model_id)
    return [promotion_service.to_schema(d) for d in decisions]


@router.post(
    "/models/{model_id}/versions/{version}/decisions",
    response_model=DecisionRecord,
    status_code=201,
    responses={404: {"model": ErrorResponse}, 409: {"model": ErrorResponse}},
)
def create_decision(
    model_id: str, version: int, request: DecisionCreateRequest, db: Session = Depends(get_db)
) -> DecisionRecord:
    model_version = get_model_version_or_404(db, model_id, version)
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
    target = get_model_version_or_404(db, model_id, request.rollback_of_version)
    try:
        decision = promotion_service.rollback(db, target, request)
    except ValueError as exc:
        raise APIError(409, "ROLLBACK_NOT_ALLOWED", str(exc)) from exc
    db.commit()
    return promotion_service.to_schema(decision)
