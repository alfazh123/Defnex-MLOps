from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session

from app.api.errors import APIError
from app.db.session import get_db
from app.schemas.common import ErrorResponse
from app.schemas.model import ModelLifecycleStatus, ModelRegistryRecord, ModelSummary
from app.services import model_service

router = APIRouter(tags=["Models"])


@router.get("/models", response_model=list[ModelSummary])
def list_models(status: ModelLifecycleStatus | None = None, db: Session = Depends(get_db)) -> list[ModelSummary]:
    return model_service.list_models(db, status)


@router.get(
    "/models/{model_id}/versions/{version}",
    response_model=ModelRegistryRecord,
    responses={404: {"model": ErrorResponse}},
)
def get_model_version(model_id: str, version: int, db: Session = Depends(get_db)) -> ModelRegistryRecord:
    model_version = model_service.get_model_version(db, model_id, version)
    if model_version is None:
        raise APIError(404, "MODEL_NOT_FOUND", f'model_id "{model_id}" version {version} not found')
    return model_service.to_schema(model_version)
