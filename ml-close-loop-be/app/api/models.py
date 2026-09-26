from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session

from app.api.deps import (
    FilterParams,
    PaginationParams,
    get_current_user,
    get_filters,
    get_model_version_or_404,
    get_pagination,
)
from app.api.errors import APIError
from app.config import settings
from app.db.session import get_db
from app.models.model import Model
from app.models.user import User
from app.schemas.common import ErrorResponse, PaginatedResponse
from app.schemas.model import (
    EvaluationObject,
    EvaluationSubmitResponse,
    EvaluationTriggerRequest,
    ModelRegistryRecord,
    ModelSummary,
)
from app.services import lineage_service, model_service

router = APIRouter(tags=["Models"])


@router.get("/models/available")
def list_available_models(_user: User = Depends(get_current_user)) -> dict:
    raw = settings.unsloth_models
    models = [m.strip() for m in raw.split(",") if m.strip()]
    default = settings.unsloth_default_model
    return {"models": models, "default": default}


@router.get("/models", response_model=PaginatedResponse[ModelSummary])
def list_models(
    db: Session = Depends(get_db),
    _user: User = Depends(get_current_user),
    fl: FilterParams = Depends(get_filters),
    pg: PaginationParams = Depends(get_pagination),
) -> PaginatedResponse[ModelSummary]:
    summaries, total = model_service.list_models(
        db, status=fl.status, search=fl.search, limit=pg.limit, offset=pg.offset
    )
    return PaginatedResponse(
        items=summaries,
        total=total,
        page=pg.page,
        size=pg.size,
        pages=PaginationParams.pages_from(total, pg.size),
    )


@router.get(
    "/models/{model_id}/versions",
    response_model=PaginatedResponse[ModelRegistryRecord],
    responses={404: {"model": ErrorResponse}},
)
def list_model_versions(
    model_id: str,
    db: Session = Depends(get_db),
    _user: User = Depends(get_current_user),
    pg: PaginationParams = Depends(get_pagination),
) -> PaginatedResponse[ModelRegistryRecord]:
    """List every version (all statuses, including RETIRED) of a model, most-legible for the
    rollback picker and history browsing (PRD §14.2). Paginated like other list endpoints."""
    if db.get(Model, model_id) is None:
        raise APIError(404, "MODEL_NOT_FOUND", f'model_id "{model_id}" not found')
    versions, total = model_service.list_model_versions(
        db, model_id, limit=pg.limit, offset=pg.offset
    )
    return PaginatedResponse(
        items=[model_service.to_schema(v) for v in versions],
        total=total,
        page=pg.page,
        size=pg.size,
        pages=PaginationParams.pages_from(total, pg.size),
    )


@router.get(
    "/models/{model_id}/versions/{version}/lineage",
    responses={404: {"model": ErrorResponse}},
)
def get_model_version_lineage(
    model_id: str,
    version: int,
    db: Session = Depends(get_db),
    _user: User = Depends(get_current_user),
) -> dict:
    """Everything needed to answer "where did this artifact come from?" in one call.

    Issue #236, from the meeting's fourth agenda item: which training, when, which dataset,
    what metadata, what config. All of it existed, but across five tables plus a sidecar, so a
    caller had to join them by hand — and the dataset's checksum and validation-report
    reference were not reachable at all (issue #237 writes the report ref).

    Unresolvable pieces come back in `gaps[]` with a reason, so "we do not know this" is
    distinguishable from "this is null". Deliberately not a pydantic response model: a fixed
    schema here would have to either drop the gap detail or invent empty fields for answers
    that do not exist, and this record's whole value is being honest about what is missing.
    """
    try:
        return lineage_service.build_lineage(db, model_id, version)
    except lineage_service.LineageError as exc:
        raise APIError(404, "MODEL_VERSION_NOT_FOUND", str(exc)) from exc


@router.get(
    "/models/{model_id}/versions/{version}",
    response_model=ModelRegistryRecord,
    responses={404: {"model": ErrorResponse}},
)
def get_model_version(
    model_id: str,
    version: int,
    db: Session = Depends(get_db),
    _user: User = Depends(get_current_user),
) -> ModelRegistryRecord:
    model_version = get_model_version_or_404(db, model_id, version)
    return model_service.to_schema(model_version)


@router.post(
    "/models/{model_id}/versions/{version}/evaluation",
    response_model=EvaluationSubmitResponse,
    responses={
        400: {"model": ErrorResponse},
        404: {"model": ErrorResponse},
        409: {"model": ErrorResponse},
    },
)
def submit_evaluation(
    model_id: str,
    version: int,
    request: EvaluationTriggerRequest,
    db: Session = Depends(get_db),
    _user: User = Depends(get_current_user),
) -> EvaluationSubmitResponse:
    """Trigger async server-side evaluation (issue #128) - no longer accepts caller-supplied
    signal numbers (`EvaluationTriggerRequest` rejects them with 422, see the schema's
    docstring). The evaluation worker computes the real signals and applies them, so the
    response here reflects whatever evaluation state already exists, not the outcome of this
    trigger (which runs asynchronously)."""

    model_version = get_model_version_or_404(db, model_id, version)
    if model_version.status not in ("REGISTERED", "EVALUATED"):
        raise APIError(
            409,
            "EVALUATION_NOT_EDITABLE",
            f'model_id "{model_id}" version {version} is {model_version.status}; '
            "evaluation data is not editable after a decision has been made against it.",
        )
    eval_set_id = request.eval_set_id or model_version.eval_set_id
    eval_set_version = request.eval_set_version or model_version.eval_set_version
    if eval_set_id is None or eval_set_version is None:
        raise APIError(
            400,
            "EVAL_SET_REQUIRED",
            "an eval_set_id and eval_set_version are required to trigger evaluation "
            "(either on this request or already stored on the model version).",
        )
    model_service.trigger_evaluation(
        db,
        model_version,
        eval_set_id=request.eval_set_id,
        eval_set_version=request.eval_set_version,
    )
    db.commit()
    return EvaluationSubmitResponse(
        evaluation=model_service.get_evaluation(model_version),
        status=model_version.status,
    )


@router.get(
    "/models/{model_id}/versions/{version}/evaluation",
    response_model=EvaluationObject,
    responses={404: {"model": ErrorResponse}},
)
def get_evaluation(
    model_id: str,
    version: int,
    db: Session = Depends(get_db),
    _user: User = Depends(get_current_user),
) -> EvaluationObject:
    model_version = get_model_version_or_404(db, model_id, version)
    return model_service.get_evaluation(model_version)
