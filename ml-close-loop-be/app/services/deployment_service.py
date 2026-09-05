import uuid
from datetime import datetime, timezone

from sqlalchemy import select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from app.config import settings
from app.models.deployment import Deployment
from app.models.model import ModelVersion
from app.schemas.deployment import DeployResult, DeploymentStatus
from app.services.serving import MockServingBackend, ServingBackend

# Single source of truth for "which version is production": ModelVersion.status == "DEPLOYED".
# The `deployments` table is append-only pointer *history* - the deployed_at audit trail for the
# currently DEPLOYED version - never a competing claim about it. The partial unique index
# `uq_model_versions_one_deployed` (model_versions.model_id WHERE status='DEPLOYED') makes a
# second DEPLOYED version for one model_id impossible at the DB level, on both SQLite and
# Postgres (issue #36).

# Deployment aliases that can be used to refer to the production version (issue #36).
# `prod` resolves to whatever ModelVersion currently holds the DEPLOYED status.
SUPPORTED_ALIASES = frozenset({"prod"})

_DEPLOYED_CONFLICT_MESSAGE = (
    "another version of this model is already DEPLOYED; only one can hold the production "
    "pointer at a time (a concurrent deploy won the race)."
)


def _deployed_model_version(db: Session, model_id: str) -> ModelVersion | None:
    """The single source of truth query: the DEPLOYED ModelVersion for `model_id`, or None."""
    return db.scalars(
        select(ModelVersion).where(
            ModelVersion.model_id == model_id,
            ModelVersion.status == "DEPLOYED",
        )
    ).first()


def _status_for(
    db: Session, model_id: str, deployed: ModelVersion | None
) -> DeploymentStatus:
    """Shape the DeploymentStatus for the DEPLOYED version, taking `deployed_at` from the
    deployment history rows for that version (the deployments table is history, not truth)."""
    if deployed is None:
        return DeploymentStatus(model_id=model_id)
    history = db.scalars(
        select(Deployment)
        .where(
            Deployment.model_id == model_id,
            Deployment.model_version == deployed.version,
        )
        .order_by(Deployment.deployed_at.desc())
    ).first()
    return DeploymentStatus(
        model_id=model_id,
        current_deployed_version=deployed.version,
        deployed_at=history.deployed_at if history is not None else None,
        status="DEPLOYED",
    )


def deploy(
    db: Session, model_version: ModelVersion, backend: ServingBackend | None = None
) -> tuple[Deployment, ModelVersion | None]:
    """Move the deployment pointer to `model_version`, retiring whichever version currently holds
    it (WBS 3.3 §3 release gate + §4 supersession). Returns the new Deployment row and the
    superseded ModelVersion, if any.

    Deliberately has no `status` guard of its own: the PROMOTED-only gate belongs to the deploy
    endpoint, while `promotion_service.rollback` legitimately points at a RETIRED version. This is
    the single place the pointer moves, so both paths stay consistent.

    The Registry status is the only thing that moves here; the `deployments` row is appended as
    history. Two genuinely concurrent deploys race on the partial unique index
    `uq_model_versions_one_deployed`: exactly one commits, and the loser's IntegrityError is
    re-raised as a clear ValueError (mapped to 409 by the router) instead of silently succeeding.
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
        # Flush the retire on its own before marking `model_version` DEPLOYED: SQLAlchemy would
        # otherwise batch both status UPDATEs into one statement, and SQLite/Postgres evaluate the
        # partial unique index per row mid-statement - the transient "still DEPLOYED (previous),
        # now DEPLOYED (target)" state would trip the index even though the committed end state is
        # legal. Each UPDATE must therefore hit the table when at most one DEPLOYED row exists.
        db.flush()

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
    try:
        db.flush()
    except IntegrityError as exc:
        # A concurrent deploy committed first; the partial unique index rejects a second DEPLOYED
        # version for this model_id. Surface it as a domain conflict, not a raw IntegrityError.
        raise ValueError(_DEPLOYED_CONFLICT_MESSAGE) from exc
    return deployment, previous


def get_deployment_status(db: Session, model_id: str) -> DeploymentStatus:
    """The current pointer for `model_id` - derived from the Registry's DEPLOYED version (the
    single source of truth), all-null except `model_id` when nothing has ever been deployed
    (openapi.yaml DeploymentStatus.current_deployed_version). The deployments table contributes
    only `deployed_at` history."""
    return _status_for(db, model_id, _deployed_model_version(db, model_id))


def resolve_alias(db: Session, model_id: str, alias: str) -> ModelVersion:
    """Resolve a named alias (only ``prod`` today) to the currently DEPLOYED ModelVersion.

    This is the *single* resolution point every alias consumer (the alias endpoint, and later the
    inference path, #40) must use; it raises ValueError rather than returning None so "no deployed
    version yet" can never silently propagate. Resolution follows the same single source of truth
    as `get_deployment_status` (the Registry DEPLOYED status), so both always agree.
    """
    if alias not in SUPPORTED_ALIASES:
        raise ValueError(f"unknown deployment alias {alias!r} (supported: prod)")
    deployed = _deployed_model_version(db, model_id)
    if deployed is None:
        raise ValueError(
            f'model_id "{model_id}" has no deployed version to resolve alias {alias!r} against'
        )
    return deployed


def to_deploy_result(
    deployment: Deployment, previous: ModelVersion | None
) -> DeployResult:
    return DeployResult(
        model_id=deployment.model_id,
        current_deployed_version=deployment.model_version,
        previous_deployed_version=previous.version if previous is not None else None,
    )
