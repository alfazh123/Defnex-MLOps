import logging

from fastapi import APIRouter, Depends
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
from app.db.session import get_db
from app.models.user import User
from app.schemas.common import ErrorResponse, PaginatedResponse
from app.schemas.training import TrainingRun, TrainingRunCreateRequest
from app.services import dataset_service, training_service
from app.services import unsloth_client

logger = logging.getLogger(__name__)

router = APIRouter(tags=["Training"])


@router.post(
    "/training-runs",
    response_model=TrainingRun,
    status_code=201,
    responses={404: {"model": ErrorResponse}},
)
async def create_training_run(
    request: TrainingRunCreateRequest,
    db: Session = Depends(get_db),
    _user: User = Depends(get_current_user),
) -> TrainingRun:
    dataset_version = dataset_service.get_dataset_version(
        db, request.dataset_id, request.dataset_version
    )
    if dataset_version is None:
        raise APIError(
            404,
            "DATASET_NOT_FOUND",
            f'dataset_id "{request.dataset_id}" version {request.dataset_version} not found',
        )

    training_run = training_service.create_training_run(db, dataset_version, request)
    db.commit()

    try:
        result = await unsloth_client.start_training(
            training_run.training_run_id,
            training_run.base_model,
            training_run.training_config,
        )
        training_run.artifact_uri = result.get("job_id", "")
        training_service.start_training_run(db, training_run)
        db.commit()
    except Exception:
        logger.exception("Failed to start training on Unsloth Studio")

    return training_service.to_schema(training_run)


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
