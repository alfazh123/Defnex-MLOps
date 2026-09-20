from fastapi import APIRouter, Depends, Header, Request
from fastapi.responses import StreamingResponse
from sqlalchemy.orm import Session

from app.api.deps import (
    FilterParams,
    PaginationParams,
    get_current_user,
    get_filters,
    get_pagination,
)
from app.api.errors import APIError
from app.config import settings
from app.db.session import get_db
from app.limiter import limiter
from app.models.user import User
from app.schemas.common import ErrorResponse, PaginatedResponse
from app.schemas.training import TrainingRun, TrainingRunCreateRequest
from app.services import (
    dataset_service,
    idempotency_service,
    training_service,
    validation_service,
)
from app.services import unsloth_client

_IDEMPOTENCY_ENDPOINT = "create_training_run"

router = APIRouter(tags=["Training"])


@router.post(
    "/training-runs",
    response_model=TrainingRun,
    status_code=201,
    responses={404: {"model": ErrorResponse}, 409: {"model": ErrorResponse}},
)
@limiter.limit(settings.rate_limit_training_create)
async def create_training_run(
    request: Request,
    body: TrainingRunCreateRequest,
    db: Session = Depends(get_db),
    _user: User = Depends(get_current_user),
    x_idempotency_key: str | None = Header(None, alias="X-Idempotency-Key"),
) -> dict:
    # Issue #170: durable Postgres-backed cache (app.services.idempotency_service), same
    # pattern as deployment_service/intake_validate.py -- a retried POST with the same
    # X-Idempotency-Key replays the earlier response instead of creating a second TrainingRun.
    if x_idempotency_key:
        cached = idempotency_service.get_cached_response(db, x_idempotency_key)
        if cached is not None:
            return cached.body

    dataset_version = dataset_service.get_dataset_version(
        db, body.dataset_id, body.dataset_version
    )
    if dataset_version is None:
        raise APIError(
            404,
            "DATASET_NOT_FOUND",
            f'dataset_id "{body.dataset_id}" version {body.dataset_version} not found',
        )

    latest_report = validation_service.get_latest_validation_report(db, dataset_version)
    if latest_report is None:
        raise APIError(
            409,
            "VALIDATION_REQUIRED",
            f'Dataset version {body.dataset_version} of dataset_id "{body.dataset_id}" '
            "must have a validation report with gate_decision PASS before training.",
        )
    if latest_report.gate_decision == "FAIL":
        raise APIError(
            409,
            "VALIDATION_FAILED",
            f'Dataset version {body.dataset_version} of dataset_id "{body.dataset_id}" '
            "failed validation and cannot be trained on.",
        )
    if latest_report.record_count == 0:
        raise APIError(
            409,
            "VALIDATION_FAILED",
            f'Dataset version {body.dataset_version} of dataset_id "{body.dataset_id}" '
            "has a validation report that examined no records and cannot be trained on.",
        )

    training_run = training_service.create_training_run(db, dataset_version, body)
    db.commit()

    result = training_service.to_schema(training_run).model_dump(mode="json")
    if x_idempotency_key:
        idempotency_service.store_response(
            db,
            x_idempotency_key,
            endpoint=_IDEMPOTENCY_ENDPOINT,
            status=201,
            body=result,
        )
        db.commit()
    return result


@router.get(
    "/training-runs",
    response_model=PaginatedResponse[TrainingRun],
)
def list_training_runs(
    db: Session = Depends(get_db),
    _user: User = Depends(get_current_user),
    pg: PaginationParams = Depends(get_pagination),
    fl: FilterParams = Depends(get_filters),
) -> PaginatedResponse[TrainingRun]:
    runs, total = training_service.list_training_runs(
        db,
        limit=pg.limit,
        offset=pg.offset,
        status=fl.status,
        model=fl.model,
    )
    return PaginatedResponse(
        items=[training_service.to_schema(r) for r in runs],
        total=total,
        page=pg.page,
        size=pg.size,
        pages=PaginationParams.pages_from(total, pg.size),
    )


@router.get(
    "/training-runs/{training_run_id}",
    response_model=TrainingRun,
    responses={404: {"model": ErrorResponse}},
)
def get_training_run(
    training_run_id: str,
    db: Session = Depends(get_db),
    _user: User = Depends(get_current_user),
) -> TrainingRun:
    training_run = training_service.get_training_run(db, training_run_id)
    if training_run is None:
        raise APIError(
            404,
            "TRAINING_RUN_NOT_FOUND",
            f'training_run_id "{training_run_id}" not found',
        )
    return training_service.to_schema(training_run)


@router.post(
    "/training-runs/{training_run_id}/retry",
    response_model=TrainingRun,
    status_code=201,
    responses={404: {"model": ErrorResponse}, 409: {"model": ErrorResponse}},
)
def retry_training_run(
    training_run_id: str,
    db: Session = Depends(get_db),
    _user: User = Depends(get_current_user),
) -> TrainingRun:
    """Retry a FAILED run: creates a new PENDING run with `retry_of` back at the original
    (PRD §10.5). The original stays immutable; only a FAILED run is retryable, any other
    status is a deterministic 409."""
    training_run = training_service.get_training_run(db, training_run_id)
    if training_run is None:
        raise APIError(
            404,
            "TRAINING_RUN_NOT_FOUND",
            f'training_run_id "{training_run_id}" not found',
        )
    try:
        new_run = training_service.retry_training_run(
            db, training_run, actor_id=_user.id
        )
    except ValueError as exc:
        raise APIError(409, "TRAINING_RUN_NOT_RETRYABLE", str(exc)) from exc
    db.commit()
    return training_service.to_schema(new_run)


@router.get(
    "/training-runs/{training_run_id}/progress",
    responses={404: {"model": ErrorResponse}},
)
async def get_training_run_progress(
    training_run_id: str,
    db: Session = Depends(get_db),
    _user: User = Depends(get_current_user),
):
    training_run = training_service.get_training_run(db, training_run_id)
    if training_run is None:
        raise APIError(
            404,
            "TRAINING_RUN_NOT_FOUND",
            f'training_run_id "{training_run_id}" not found',
        )

    async def event_generator():
        async for event in unsloth_client.stream_progress(training_run_id):
            yield f"data: {event}\n\n"

    return StreamingResponse(event_generator(), media_type="text/event-stream")
