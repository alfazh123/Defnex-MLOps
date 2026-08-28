from sqlalchemy.orm import Session

from app.api.errors import APIError
from app.models.model import ModelVersion
from app.services import model_service


def get_model_version_or_404(db: Session, model_id: str, version: int) -> ModelVersion:
    """Resolve a model version or raise the standard 404 error envelope (openapi.yaml)."""
    model_version = model_service.get_model_version(db, model_id, version)
    if model_version is None:
        raise APIError(404, "MODEL_NOT_FOUND", f'model_id "{model_id}" version {version} not found')
    return model_version
