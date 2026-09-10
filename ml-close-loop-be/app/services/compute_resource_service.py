"""Compute resource business logic (issue #78, PRD §8.6, §19.4).

Services handle validation and state; the router owns the transaction boundary.
"""

from datetime import datetime, timezone

from sqlalchemy import func, select
from sqlalchemy.orm import Session

from app.models.compute_resource import ComputeResource
from app.schemas.compute_resource import (
    ComputeResourceCreateRequest,
    ComputeResourceUpdateRequest,
)


def create_compute_resource(
    db: Session, request: ComputeResourceCreateRequest
) -> ComputeResource:
    """Register a new compute resource (PRD §19.4 — config, not code)."""
    existing = db.scalar(
        select(ComputeResource).where(ComputeResource.name == request.name)
    )
    if existing is not None:
        raise ValueError(f'Compute resource "{request.name}" already exists')

    resource = ComputeResource(
        name=request.name,
        role=request.role,
        environment=request.environment,
        provider_type=request.provider_type,
        host=request.host,
        gpu_info=request.gpu_info,
        ssh_host=request.ssh_host,
        ssh_port=request.ssh_port,
        ssh_username=request.ssh_username,
        credential_ref=request.credential_ref,
        notebook_url=request.notebook_url,
        is_healthy=True,
    )
    db.add(resource)
    db.flush()
    return resource


def get_compute_resource(db: Session, resource_id: int) -> ComputeResource | None:
    return db.get(ComputeResource, resource_id)


def get_compute_resource_by_name(db: Session, name: str) -> ComputeResource | None:
    return db.scalar(select(ComputeResource).where(ComputeResource.name == name))


def list_compute_resources(
    db: Session,
    *,
    limit: int = 20,
    offset: int = 0,
    role: str | None = None,
    environment: str | None = None,
    search: str | None = None,
) -> tuple[list[ComputeResource], int]:
    """List compute resources with optional filters (PRD §33)."""
    base = select(ComputeResource)

    if role is not None:
        base = base.where(ComputeResource.role == role)
    if environment is not None:
        base = base.where(ComputeResource.environment == environment)
    if search is not None:
        base = base.where(ComputeResource.name.ilike(f"%{search}%"))

    total = db.scalar(select(func.count()).select_from(base.subquery()))
    resources = list(
        db.scalars(
            base.order_by(ComputeResource.created_at.desc()).limit(limit).offset(offset)
        ).all()
    )
    return resources, total


def update_compute_resource(
    db: Session,
    resource: ComputeResource,
    request: ComputeResourceUpdateRequest,
) -> ComputeResource:
    """Partial update of a compute resource. Only provided fields are changed."""
    update_data = request.model_dump(exclude_unset=True)
    if not update_data:
        return resource

    # Name uniqueness check if renaming
    if "name" in update_data and update_data["name"] != resource.name:
        existing = db.scalar(
            select(ComputeResource).where(
                ComputeResource.name == update_data["name"],
                ComputeResource.id != resource.id,
            )
        )
        if existing is not None:
            raise ValueError(f'Compute resource "{update_data["name"]}" already exists')

    for field, value in update_data.items():
        setattr(resource, field, value)

    resource.updated_at = datetime.now(timezone.utc)
    db.flush()
    return resource


def delete_compute_resource(db: Session, resource: ComputeResource) -> None:
    """Delete a compute resource."""
    db.delete(resource)
    db.flush()


def check_health(db: Session, resource: ComputeResource) -> dict:
    """Read-only health check for a compute resource (PRD §8.6).

    For the MVP, health is a stored boolean — real connectivity checks
    (SSH ping, GPU status) will be added with the VPS provider (issue #76).
    """
    return {
        "resource_id": resource.id,
        "name": resource.name,
        "is_healthy": resource.is_healthy,
        "provider_type": resource.provider_type,
        "message": "OK" if resource.is_healthy else "Marked unhealthy by admin",
    }
