from fastapi import APIRouter, Depends, Header, Request
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.api.deps import get_current_user, get_model_version_or_404, require_admin
from app.api.errors import APIError
from app.config import settings
from app.db.session import get_db
from app.limiter import limiter
from app.models.environment import Environment
from app.models.model import Model
from app.models.user import User
from app.schemas.common import ErrorResponse
from app.schemas.deployment import (
    DeployRequest,
    DeployResult,
    DeploymentStatus,
    EnvironmentOut,
)
from app.services import deployment_service, promotion_service

router = APIRouter(tags=["Deployment"])


@router.post(
    "/models/{model_id}/versions/{version}/deploy",
    response_model=DeployResult,
    responses={404: {"model": ErrorResponse}, 409: {"model": ErrorResponse}},
)
@limiter.limit(settings.rate_limit_deploy)
def deploy_model_version(
    request: Request,
    model_id: str,
    version: int,
    body: DeployRequest | None = None,
    db: Session = Depends(get_db),
    _admin: User = Depends(require_admin),
    x_idempotency_key: str | None = Header(None, alias="X-Idempotency-Key"),
) -> DeployResult:
    model_version = get_model_version_or_404(db, model_id, version)
    cached = deployment_service.check_idempotency(x_idempotency_key, model_version.id)
    if cached is not None:
        return DeployResult(**cached)
    # Batch 3 (issues #69/#70): environment-aware deploy gate for the staging/production
    # promotion ladder (PRD §16.2 "Promotion Ladder", §41 Principle 4 "Staging Before Production").
    #   - default/None  : legacy gate, PROMOTED only (unchanged; backward compatible).
    #   - staging       : EVALUATED|STAGING candidates (AC: "must be in the candidate registry").
    #   - production    : VALIDATED|PRODUCTION (ladder) plus PROMOTED (legacy) - a production
    #                     deploy must never bypass the staging/validation gate once on the ladder.
    # env="staging" deploys also route through promotion_service.stage_deploy so that staging a
    # candidate does NOT steal the production pointer (deployment_service.deploy's behavior) - the
    # candidate is demoted to STAGING and the previous production version is restored as DEPLOYED.
    # Agent 1's serving.py changes do not overlap this block (reading only, no serving coupling).
    environment = body.environment if body else None
    if environment == "staging":
        if model_version.status not in ("EVALUATED", "STAGING"):
            raise APIError(
                409,
                "DEPLOY_NOT_ALLOWED",
                f'model_id "{model_id}" version {version} is {model_version.status}; a staging '
                "deploy requires an EVALUATED candidate (or an already-STAGING version).",
            )
    elif environment == "production":
        if model_version.status not in ("VALIDATED", "PRODUCTION", "PROMOTED"):
            raise APIError(
                409,
                "DEPLOY_NOT_ALLOWED",
                f'model_id "{model_id}" version {version} is {model_version.status}; a production '
                "deploy requires a VALIDATED/PRODUCTION (ladder) or PROMOTED (legacy) version.",
            )
    else:
        if model_version.status != "PROMOTED":
            raise APIError(
                409,
                "DEPLOY_NOT_ALLOWED",
                f'model_id "{model_id}" version {version} is {model_version.status}; only a PROMOTED version '
                "can be deployed.",
            )
    try:
        if environment == "staging":
            deployment, previous = promotion_service.stage_deploy(db, model_version)
        else:
            deployment, previous = deployment_service.deploy(
                db, model_version, environment=environment
            )
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
    result = deployment_service.to_deploy_result(deployment, previous)
    deployment_service.store_idempotency(
        x_idempotency_key, model_version.id, result.model_dump()
    )
    return result


@router.get(
    "/environments",
    response_model=list[EnvironmentOut],
    responses={401: {"model": ErrorResponse}},
)
def list_environments(
    db: Session = Depends(get_db),
    _user: User = Depends(get_current_user),
) -> list[EnvironmentOut]:
    """List the registered deployment environments (PRD §16.1), seeded as `default`/`staging`/
    `production` by the environments migration."""
    rows = db.scalars(select(Environment).order_by(Environment.name)).all()
    return [EnvironmentOut(name=r.name, description=r.description) for r in rows]


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
