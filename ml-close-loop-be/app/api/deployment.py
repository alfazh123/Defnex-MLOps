from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session

from app.api.deps import get_model_version_or_404
from app.api.errors import APIError
from app.db.session import get_db
from app.models.model import Model
from app.schemas.common import ErrorResponse
from app.schemas.deployment import DeployResult, DeploymentStatus
from app.services import deployment_service

router = APIRouter(tags=["Deployment"])


@router.post(
    "/models/{model_id}/versions/{version}/deploy",
    response_model=DeployResult,
    responses={404: {"model": ErrorResponse}, 409: {"model": ErrorResponse}},
)
def deploy_model_version(model_id: str, version: int, db: Session = Depends(get_db)) -> DeployResult:
    model_version = get_model_version_or_404(db, model_id, version)
    if model_version.status != "PROMOTED":
        raise APIError(
            409,
            "DEPLOY_NOT_ALLOWED",
            f'model_id "{model_id}" version {version} is {model_version.status}; only a PROMOTED version '
            "can be deployed.",
        )
    deployment, previous = deployment_service.deploy(db, model_version)
    db.commit()
    return deployment_service.to_deploy_result(deployment, previous)


@router.get(
    "/models/{model_id}/deployment",
    response_model=DeploymentStatus,
    responses={404: {"model": ErrorResponse}},
)
def get_deployment_status(model_id: str, db: Session = Depends(get_db)) -> DeploymentStatus:
    if db.get(Model, model_id) is None:
        raise APIError(404, "MODEL_NOT_FOUND", f'model_id "{model_id}" not found')
    return deployment_service.get_deployment_status(db, model_id)
