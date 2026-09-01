from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session

from app.api.deps import (
    PaginationParams,
    get_current_user,
    get_pagination,
    require_admin,
)
from app.api.errors import APIError
from app.db.session import get_db
from app.models.user import User
from app.schemas.common import ErrorResponse, PaginatedResponse
from app.schemas.dataset import (
    DatasetSummary,
    DatasetVersion,
    DatasetVersionCreateRequest,
)
from app.services import dataset_service

router = APIRouter(tags=["Datasets"])


@router.get("/datasets", response_model=PaginatedResponse[DatasetSummary])
def list_datasets(
    db: Session = Depends(get_db),
    _user: User = Depends(get_current_user),
    pg: PaginationParams = Depends(get_pagination),
) -> PaginatedResponse[DatasetSummary]:
    items, total = dataset_service.list_datasets(db, limit=pg.limit, offset=pg.offset)
    return PaginatedResponse(
        items=items,
        total=total,
        page=pg.page,
        size=pg.size,
        pages=PaginationParams.pages_from(total, pg.size),
    )


@router.post(
    "/datasets/{dataset_id}/versions", response_model=DatasetVersion, status_code=201
)
def create_dataset_version(
    dataset_id: str,
    request: DatasetVersionCreateRequest,
    db: Session = Depends(get_db),
    _admin: User = Depends(require_admin),
) -> DatasetVersion:
    version = dataset_service.create_dataset_version(db, dataset_id, request)
    return dataset_service.to_schema(version)


@router.get(
    "/datasets/{dataset_id}/versions",
    response_model=list[DatasetVersion],
    responses={404: {"model": ErrorResponse}},
)
def list_dataset_versions(
    dataset_id: str,
    db: Session = Depends(get_db),
    _user: User = Depends(get_current_user),
) -> list[DatasetVersion]:
    versions = dataset_service.list_dataset_versions(db, dataset_id)
    if not versions:
        raise APIError(404, "DATASET_NOT_FOUND", f'dataset_id "{dataset_id}" not found')
    return [dataset_service.to_schema(v) for v in versions]


@router.get(
    "/datasets/{dataset_id}/versions/{version}",
    response_model=DatasetVersion,
    responses={404: {"model": ErrorResponse}},
)
def get_dataset_version(
    dataset_id: str,
    version: int,
    db: Session = Depends(get_db),
    _user: User = Depends(get_current_user),
) -> DatasetVersion:
    result = dataset_service.get_dataset_version(db, dataset_id, version)
    if result is None:
        raise APIError(
            404,
            "DATASET_NOT_FOUND",
            f'dataset_id "{dataset_id}" version {version} not found',
        )
    return dataset_service.to_schema(result)
