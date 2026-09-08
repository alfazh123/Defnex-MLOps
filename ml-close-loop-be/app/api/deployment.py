from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session

from app.api.deps import get_current_user, get_model_version_or_404, require_admin
from app.api.errors import APIError
from app.db.session import get_db
from app.models.model import Model
from app.models.user import User
from app.schemas.common import ErrorResponse
from app.schemas.deployment import DeployResult, DeploymentStatus
from app.services import deployment_service

router = APIRouter(tags=["Deployment"])


@router.post(
    "/models/{model_id}/versions/{version}/deploy",
    response_model=DeployResult,
    responses={404: {"model": ErrorResponse}, 409: {"model": ErrorResponse}},
)
def deploy_model_version(
    model_id: str,
    version: int,
    db: Session = Depends(get_db),
    _admin: User = Depends(require_admin),
) -> DeployResult:
    model_version = get_model_version_or_404(db, model_id, version)
    if model_version.status != "PROMOTED":
        raise APIError(
            409,
            "DEPLOY_NOT_ALLOWED",
            f'model_id "{model_id}" version {version} is {model_version.status}; only a PROMOTED version '
            "can be deployed.",
        )
    try:
        deployment, previous = deployment_service.deploy(db, model_version)
        db.commit()
    except deployment_service.DeploymentLockTimeout as exc:
        # The GPU lock shared with training was still held when the timeout passed (issue #59):
        # the deploy must not touch the GPU while training does, and must surface an explicit
        # "GPU busy, retry" state (503) instead of hanging.
        raise APIError(503, "GPU_LOCK_TIMEOUT", str(exc)) from exc
    except ValueError as exc:
        # A concurrent deploy won the race for this model_id (partial unique index
        # uq_model_versions_one_deployed) - a real conflict, not a fake success.
        raise APIError(409, "DEPLOY_CONFLICT", str(exc)) from exc
    return deployment_service.to_deploy_result(deployment, previous)


@router.get(
    "/models/{model_id}/deployment",
    response_model=DeploymentStatus,
    responses={404: {"model": ErrorResponse}},
)
def get_deployment_status(
    model_id: str,
    db: Session = Depends(get_db),
    _user: User = Depends(get_current_user),
) -> DeploymentStatus:
    if db.get(Model, model_id) is None:
        raise APIError(404, "MODEL_NOT_FOUND", f'model_id "{model_id}" not found')
    return deployment_service.get_deployment_status(db, model_id)


@router.get(
    "/models/{model_id}/deployment/{alias}",
    response_model=DeploymentStatus,
    responses={404: {"model": ErrorResponse}, 422: {"model": ErrorResponse}},
)
def get_deployment_by_alias(
    model_id: str,
    alias: str,
    db: Session = Depends(get_db),
    _user: User = Depends(get_current_user),
) -> DeploymentStatus:
    """Resolve a deployment alias (currently only ``prod``) to the production version, via the
    single resolution function `deployment_service.resolve_alias` (openapi.yaml DeploymentStatus
    alias resolution). Unknown alias -> 422, model unknown -> 404 MODEL_NOT_FOUND, model never
    deployed -> 404 DEPLOYMENT_NOT_FOUND (explicit - never a None that propagates)."""
    if db.get(Model, model_id) is None:
        raise APIError(404, "MODEL_NOT_FOUND", f'model_id "{model_id}" not found')
    try:
        deployment_service.resolve_alias(db, model_id, alias)
    except ValueError as exc:
        if alias not in deployment_service.SUPPORTED_ALIASES:
            raise APIError(422, "UNKNOWN_DEPLOYMENT_ALIAS", str(exc)) from exc
        raise APIError(404, "DEPLOYMENT_NOT_FOUND", str(exc)) from exc
    return deployment_service.get_deployment_status(db, model_id)
