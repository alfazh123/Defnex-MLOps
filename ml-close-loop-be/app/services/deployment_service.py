import uuid
from datetime import datetime, timezone

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.config import settings
from app.models.deployment import Deployment
from app.models.model import ModelVersion
from app.schemas.deployment import DeployResult, DeploymentStatus
from app.services.serving import MockServingBackend, ServingBackend


def deploy(
    db: Session, model_version: ModelVersion, backend: ServingBackend | None = None
) -> tuple[Deployment, ModelVersion | None]:
    """Move the deployment pointer to `model_version`, retiring whichever version currently holds
    it (WBS 3.3 §3 release gate + §4 supersession). Returns the new Deployment row and the
    superseded ModelVersion, if any.

    Deliberately has no `status` guard of its own: the PROMOTED-only gate belongs to the deploy
    endpoint, while `promotion_service.rollback` legitimately points at a RETIRED version. This is
    the single place the pointer moves, so both paths stay consistent.
    """

    # Queried rather than read off `model_version.model.versions`: that collection is loaded once
    # per Session and does not pick up versions registered afterwards, so a sibling deployed in the
    # same Session would be missed and never retired.
    previous = db.scalars(
        select(ModelVersion).where(
            ModelVersion.model_id == model_version.model_id,
            ModelVersion.status == "DEPLOYED",
            ModelVersion.id != model_version.id,
        )
    ).first()
    if previous is not None:
        previous.status = "RETIRED"

    (backend or MockServingBackend()).deploy(model_version)

    model_version.status = "DEPLOYED"
    deployment = Deployment(
        deployment_id=f"deployment-{uuid.uuid4().hex[:6]}",
        model_id=model_version.model_id,
        model_version=model_version.version,
        environment=settings.deployment_environment,
        status="DEPLOYED",
        deployed_at=datetime.now(timezone.utc),
    )
    db.add(deployment)
    db.flush()
    return deployment, previous


def get_deployment_status(db: Session, model_id: str) -> DeploymentStatus:
    """The current pointer for `model_id` - all-null except `model_id` when nothing has ever been
    deployed (openapi.yaml DeploymentStatus.current_deployed_version)."""

    deployment = db.scalars(
        select(Deployment).where(Deployment.model_id == model_id).order_by(Deployment.deployed_at.desc())
    ).first()
    if deployment is None:
        return DeploymentStatus(model_id=model_id)
    return DeploymentStatus(
        model_id=model_id,
        current_deployed_version=deployment.model_version,
        deployed_at=deployment.deployed_at,
        status="DEPLOYED",
    )


def to_deploy_result(deployment: Deployment, previous: ModelVersion | None) -> DeployResult:
    return DeployResult(
        model_id=deployment.model_id,
        current_deployed_version=deployment.model_version,
        previous_deployed_version=previous.version if previous is not None else None,
    )
