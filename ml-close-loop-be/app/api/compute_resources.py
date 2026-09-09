"""Admin CRUD API for compute resources (issue #78, PRD §8.6, §33).

All write operations require admin role. Health status is read-only for any
authenticated user. Adding/removing/updating a compute resource is a config
change, not a code change (PRD §19.4).
"""

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
from app.schemas.compute_resource import (
    ComputeResource,
    ComputeResourceCreateRequest,
    ComputeResourceUpdateRequest,
    HealthStatus,
)
from app.services import compute_resource_service

router = APIRouter(tags=["Compute Resources"])


@router.post(
    "/compute-resources",
    response_model=ComputeResource,
    status_code=201,
    responses={409: {"model": ErrorResponse}},
)
def create_compute_resource(
    request: ComputeResourceCreateRequest,
    db: Session = Depends(get_db),
    _admin: User = Depends(require_admin),
) -> ComputeResource:
    """Register a new compute resource (PRD §19.4). Admin only."""
    try:
        resource = compute_resource_service.create_compute_resource(db, request)
    except ValueError as exc:
        raise APIError(409, "RESOURCE_EXISTS", str(exc))
    db.commit()
    return resource


@router.get(
    "/compute-resources",
    response_model=PaginatedResponse[ComputeResource],
)
def list_compute_resources(
    db: Session = Depends(get_db),
    _user: User = Depends(get_current_user),
    pg=Depends(get_pagination),
    role: str | None = None,
    environment: str | None = None,
    search: str | None = None,
) -> PaginatedResponse[ComputeResource]:
    """List compute resources with optional filters. Any authenticated user."""
    resources, total = compute_resource_service.list_compute_resources(
        db,
        limit=pg.limit,
        offset=pg.offset,
        role=role,
        environment=environment,
        search=search,
    )
    return PaginatedResponse(
        items=[ComputeResource.model_validate(r) for r in resources],
        total=total,
        page=pg.page,
        size=pg.size,
        pages=PaginationParams.pages_from(total, pg.size),
    )


@router.get(
    "/compute-resources/{resource_id}",
    response_model=ComputeResource,
    responses={404: {"model": ErrorResponse}},
)
def get_compute_resource(
    resource_id: int,
    db: Session = Depends(get_db),
    _user: User = Depends(get_current_user),
) -> ComputeResource:
    """Get a compute resource by ID. Any authenticated user."""
    resource = compute_resource_service.get_compute_resource(db, resource_id)
    if resource is None:
        raise APIError(
            404,
            "RESOURCE_NOT_FOUND",
            f"compute resource {resource_id} not found",
        )
    return resource


@router.patch(
    "/compute-resources/{resource_id}",
    response_model=ComputeResource,
    responses={404: {"model": ErrorResponse}, 409: {"model": ErrorResponse}},
)
def update_compute_resource(
    resource_id: int,
    request: ComputeResourceUpdateRequest,
    db: Session = Depends(get_db),
    _admin: User = Depends(require_admin),
) -> ComputeResource:
    """Partial update of a compute resource. Admin only."""
    resource = compute_resource_service.get_compute_resource(db, resource_id)
    if resource is None:
        raise APIError(
            404,
            "RESOURCE_NOT_FOUND",
            f"compute resource {resource_id} not found",
        )
    try:
        resource = compute_resource_service.update_compute_resource(
            db, resource, request
        )
    except ValueError as exc:
        raise APIError(409, "RESOURCE_EXISTS", str(exc))
    db.commit()
    return resource


@router.delete(
    "/compute-resources/{resource_id}",
    status_code=204,
    responses={404: {"model": ErrorResponse}},
)
def delete_compute_resource(
    resource_id: int,
    db: Session = Depends(get_db),
    _admin: User = Depends(require_admin),
) -> None:
    """Delete a compute resource. Admin only."""
    resource = compute_resource_service.get_compute_resource(db, resource_id)
    if resource is None:
        raise APIError(
            404,
            "RESOURCE_NOT_FOUND",
            f"compute resource {resource_id} not found",
        )
    compute_resource_service.delete_compute_resource(db, resource)
    db.commit()


@router.get(
    "/compute-resources/{resource_id}/health",
    response_model=HealthStatus,
    responses={404: {"model": ErrorResponse}},
)
def get_compute_resource_health(
    resource_id: int,
    db: Session = Depends(get_db),
    _user: User = Depends(get_current_user),
) -> HealthStatus:
    """Read-only health/connectivity status (PRD §8.6). Any authenticated user."""
    resource = compute_resource_service.get_compute_resource(db, resource_id)
    if resource is None:
        raise APIError(
            404,
            "RESOURCE_NOT_FOUND",
            f"compute resource {resource_id} not found",
        )
    result = compute_resource_service.check_health(db, resource)
    return HealthStatus(**result)
