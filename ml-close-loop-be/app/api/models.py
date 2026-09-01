from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session

from app.api.deps import get_current_user, get_model_version_or_404
from app.api.errors import APIError
from app.config import settings
from app.db.session import get_db
from app.models.user import User
from app.schemas.common import ErrorResponse
from app.schemas.model import (
    EvaluationObject,
    EvaluationSubmitResponse,
    EvaluationUpdateRequest,
    ModelLifecycleStatus,
    ModelRegistryRecord,
    ModelSummary,
)
from app.services import model_service

router = APIRouter(tags=["Models"])


@router.get("/models/available")
def list_available_models(_user: User = Depends(get_current_user)) -> dict:
    raw = settings.unsloth_models
    models = [m.strip() for m in raw.split(",") if m.strip()]
    default = settings.unsloth_default_model
    return {"models": models, "default": default}


@router.get("/models", response_model=list[ModelSummary])
def list_models(
    status: ModelLifecycleStatus | None = None,
    db: Session = Depends(get_db),
    _user: User = Depends(get_current_user),
) -> list[ModelSummary]:
    return model_service.list_models(db, status)


@router.get(
    "/models/{model_id}/versions/{version}",
    response_model=ModelRegistryRecord,
    responses={404: {"model": ErrorResponse}},
)
def get_model_version(
    model_id: str, version: int, db: Session = Depends(get_db), _user: User = Depends(get_current_user)
) -> ModelRegistryRecord:
    model_version = get_model_version_or_404(db, model_id, version)
    return model_service.to_schema(model_version)


@router.post(
    "/models/{model_id}/versions/{version}/evaluation",
    response_model=EvaluationSubmitResponse,
    responses={404: {"model": ErrorResponse}, 409: {"model": ErrorResponse}},
)
def submit_evaluation(
    model_id: str,
    version: int,
    request: EvaluationUpdateRequest,
    db: Session = Depends(get_db),
    _user: User = Depends(get_current_user),
) -> EvaluationSubmitResponse:
    model_version = get_model_version_or_404(db, model_id, version)
    if model_version.status not in ("REGISTERED", "EVALUATED"):
        raise APIError(
            409,
            "EVALUATION_NOT_EDITABLE",
            f'model_id "{model_id}" version {version} is {model_version.status}; '
            "evaluation data is not editable after a decision has been made against it.",
        )
    model_service.submit_evaluation(db, model_version, request)
    db.commit()
    return EvaluationSubmitResponse(
        evaluation=model_service.get_evaluation(model_version), status=model_version.status
    )


@router.get(
    "/models/{model_id}/versions/{version}/evaluation",
    response_model=EvaluationObject,
    responses={404: {"model": ErrorResponse}},
)
def get_evaluation(
    model_id: str, version: int, db: Session = Depends(get_db), _user: User = Depends(get_current_user)
) -> EvaluationObject:
    model_version = get_model_version_or_404(db, model_id, version)
    return model_service.get_evaluation(model_version)
