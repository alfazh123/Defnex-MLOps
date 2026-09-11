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
from app.services import audit_service


def create_compute_resource(
    db: Session, request: ComputeResourceCreateRequest, actor_id: int | None = None
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

    # issue #129 (audit AC "ubah infra config" / "ubah credential_ref"): a new resource is an
    # infra config change too, so it gets an audit row from the moment it exists.
    audit_service.record_audit(
        db,
        actor_id=actor_id,
        action=audit_service.INFRA_CONFIG_CREATE,
        resource_type="compute_resource",
        resource_id=str(resource.id),
        after={
            "name": resource.name,
            "provider_type": resource.provider_type,
            "credential_ref": resource.credential_ref,
        },
    )
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
    actor_id: int | None = None,
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

    # issue #129 (audit AC "ubah infra config" / "ubah credential_ref"): snapshot only the fields
    # actually being changed, before they're overwritten, so before/after line up field-for-field
    # (credential_ref included whenever it's one of the changed fields).
    before = {field: getattr(resource, field) for field in update_data}

    for field, value in update_data.items():
        setattr(resource, field, value)

    resource.updated_at = datetime.now(timezone.utc)
    db.flush()

    audit_service.record_audit(
        db,
        actor_id=actor_id,
        action=audit_service.INFRA_CONFIG_UPDATE,
        resource_type="compute_resource",
        resource_id=str(resource.id),
        before=before,
        after=update_data,
    )
    return resource


def delete_compute_resource(
    db: Session, resource: ComputeResource, actor_id: int | None = None
) -> None:
    """Delete a compute resource."""
    resource_id = resource.id
    before = {
        "name": resource.name,
        "provider_type": resource.provider_type,
        "credential_ref": resource.credential_ref,
    }
    db.delete(resource)
    db.flush()

    audit_service.record_audit(
        db,
        actor_id=actor_id,
        action=audit_service.INFRA_CONFIG_DELETE,
        resource_type="compute_resource",
        resource_id=str(resource_id),
        before=before,
    )


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
