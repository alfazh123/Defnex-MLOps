from fastapi import APIRouter, Depends
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.api.deps import get_current_user
from app.api.errors import APIError
from app.db.session import get_db
from app.models.model import Model, ModelVersion
from app.models.user import User
from app.schemas.common import ErrorResponse
from app.schemas.inference import InferenceRequest, InferenceResponse
from app.services import deployment_service, inference_service
from app.services.serving import InferenceError, get_serving_backend

router = APIRouter(tags=["Inference"])


@router.post(
    "/models/{model_id}/inference",
    response_model=InferenceResponse,
    responses={
        404: {"model": ErrorResponse},
        409: {"model": ErrorResponse},
        502: {"model": ErrorResponse},
    },
)
def run_inference(
    model_id: str,
    request: InferenceRequest,
    db: Session = Depends(get_db),
    _user: User = Depends(get_current_user),
) -> InferenceResponse:
    """Generate from a model, addressed by deployment alias (`prod`, resolved via the single
    `deployment_service.resolve_alias` function from issue #36) or by explicit version number.

    Unknown alias / missing version / never-deployed model are explicit 404s; a version that
    exists but is not the DEPLOYED one is 409; an upstream serving failure is 502 - never a
    500 traceback (issue #41)."""
    if db.get(Model, model_id) is None:
        raise APIError(404, "MODEL_NOT_FOUND", f'model_id "{model_id}" not found')
    try:
        model_version = inference_service.resolve_target(db, model_id, request.target)
    except ValueError as exc:
        if request.target.isdigit():
            # Version path: the only failures are "version missing" (404) and
            # "version exists but is not the DEPLOYED one" (409). Disambiguate by whether the
            # version exists at all, mirroring how the deployment-alias router disambiguates
            # by input value rather than by parsing exception messages.
            version = int(request.target)
            existing: ModelVersion | None = db.scalar(
                select(ModelVersion).where(
                    ModelVersion.model_id == model_id, ModelVersion.version == version
                )
            )
            if existing is None:
                raise APIError(404, "MODEL_NOT_FOUND", str(exc)) from exc
            raise APIError(409, "INFERENCE_NOT_ALLOWED", str(exc)) from exc
        # Alias path: unknown alias or known-but-never-deployed. Both are explicit 404s
        # (issue #41), with the more specific code when the alias itself is unrecognized.
        if request.target not in deployment_service.SUPPORTED_ALIASES:
            raise APIError(404, "UNKNOWN_DEPLOYMENT_ALIAS", str(exc)) from exc
        raise APIError(404, "DEPLOYMENT_NOT_FOUND", str(exc)) from exc
    try:
        generation = inference_service.generate(
            get_serving_backend(), model_version, request.prompt
        )
    except InferenceError as exc:
        raise APIError(502, "INFERENCE_FAILED", str(exc)) from exc
    return InferenceResponse(
        model_id=model_version.model_id,
        version=model_version.version,
        generation=generation,
    )
