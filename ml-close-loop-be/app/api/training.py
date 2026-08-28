from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session

from app.api.errors import APIError
from app.db.session import get_db
from app.schemas.common import ErrorResponse
from app.schemas.training import TrainingRun, TrainingRunCreateRequest
from app.services import dataset_service, training_service

router = APIRouter(tags=["Training"])


@router.post(
    "/training-runs",
    response_model=TrainingRun,
    status_code=201,
    responses={404: {"model": ErrorResponse}},
)
def create_training_run(request: TrainingRunCreateRequest, db: Session = Depends(get_db)) -> TrainingRun:
    dataset_version = dataset_service.get_dataset_version(db, request.dataset_id, request.dataset_version)
    if dataset_version is None:
        raise APIError(
            404,
            "DATASET_NOT_FOUND",
            f'dataset_id "{request.dataset_id}" version {request.dataset_version} not found',
        )

    training_run = training_service.create_training_run(db, dataset_version, request)
    db.commit()
    return training_service.to_schema(training_run)


@router.get(
    "/training-runs/{training_run_id}",
    response_model=TrainingRun,
    responses={404: {"model": ErrorResponse}},
)
def get_training_run(training_run_id: str, db: Session = Depends(get_db)) -> TrainingRun:
    training_run = training_service.get_training_run(db, training_run_id)
    if training_run is None:
        raise APIError(404, "TRAINING_RUN_NOT_FOUND", f'training_run_id "{training_run_id}" not found')
    return training_service.to_schema(training_run)
