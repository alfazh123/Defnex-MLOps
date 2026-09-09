import uuid
from datetime import datetime, timezone

import structlog
from sqlalchemy import select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from app.config import settings
from app.models.deployment import Deployment
from app.models.model import ModelVersion
from app.schemas.deployment import DeployResult, DeploymentStatus
from app.services.artifact_storage import (
    ArtifactChecksumError,
    ArtifactStorage,
    LocalFilesystemArtifactStorage,
)
from app.services.serving import (
    ServingBackend,
    _validate_base_model,
    get_serving_backend,
)
from app.workers.gpu_lock import gpu_lock

logger = structlog.get_logger(__name__)

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


class SmokeTestError(Exception):
    """The just-loaded adapter failed the deploy-time smoke test (issue #41): vLLM could not
    produce a generation at or above the configured threshold. Raised by `deploy` *after* the
    adapter was loaded but *before* the pointer moved, so the previous version stays DEPLOYED
    (and keeps serving) and the failure is recorded in the logs."""


class DeploymentLockTimeout(Exception):
    """The GPU lock shared with training (#33) could not be acquired within the configured
    timeout, so the GPU-touching deploy must NOT proceed (issue #59).

    Raised by `deploy` in real (vllm) serving mode when training or another GPU-hoisting
    operation still holds the lock. Mapped to a clear 503 `GPU_LOCK_TIMEOUT` by the API so an
    operator sees an explicit "GPU busy, retry" state instead of an indefinite hang."""


def verify_artifact_checksum(
    model_version: ModelVersion, storage: ArtifactStorage | None = None
) -> None:
    """Recompute each artifact's SHA-256 against the checksum recorded at finalize time
    (issue #62). Raises ArtifactChecksumError on any mismatch so the caller (deploy) aborts
    *before* the pointer moves; the artifact is treated as verified when no recorded checksum
    exists (pre-#62 artifacts), so existing deployments keep working."""
    storage = storage or LocalFilesystemArtifactStorage()
    for artifact in model_version.artifacts or []:
        uri = artifact.get("uri")
        if not uri:
            continue
        if not storage.verify_checksum(uri):
            recorded = artifact.get("checksum")
            raise ArtifactChecksumError(
                f"artifact checksum mismatch for model_id {model_version.model_id!r} "
                f"version {model_version.version}: stored checksum {recorded!r} does not "
                f"match the on-disk payload; deployment refused"
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
    db: Session,
    model_version: ModelVersion,
    backend: ServingBackend | None = None,
    *,
    lock_file: str | None = None,
    lock_timeout: float | None = None,
    artifact_storage: ArtifactStorage | None = None,
    environment: str | None = None,
) -> tuple[Deployment, ModelVersion | None]:
    """Move the deployment pointer to `model_version`, retiring whichever version currently holds
    it (WBS 3.3 §3 release gate + §4 supersession). Returns the new Deployment row and the
    superseded ModelVersion, if any.

    The GPU-works portion of the deploy (loading/unloading the LoRA adapter into the serving
    process) serializes with training on the SAME exclusive GPU lock that `training_worker`
    holds (issue #59, PRD §17.5/§18.1): a hot-swap must never load onto VRAM that training is
    actively using, and vice versa. The lock is only taken when the serving backend actually
    touches the GPU (`SERVING_BACKEND=vllm`); the mock backend never touches a GPU, so tests and
    no-GPU local dev keep the exact pre-#59 behavior. When the lock cannot be acquired within the
    timeout the deploy raises `DeploymentLockTimeout` — an explicit state/error, never a hang —
    and the lock is always released via the context manager's `finally` (PRD §18.4).

    Deliberately has no `status` guard of its own: the PROMOTED-only gate belongs to the deploy
    endpoint, while `promotion_service.rollback` legitimately points at a RETIRED version. This is
    the single place the pointer moves, so both paths stay consistent.

    The Registry status is the only thing that moves here; the `deployments` row is appended as
    history. Two genuinely concurrent deploys race on the partial unique index
    `uq_model_versions_one_deployed`: exactly one commits, and the loser's IntegrityError is
    re-raised as a clear ValueError (mapped to 409 by the router) instead of silently succeeding.

    Ordering vs. the serving backend (issue #40): `backend.deploy(model_version)` runs *before*
    any DB mutation, so a failed load leaves the registry untouched - the previous version
    stays DEPLOYED in the DB and keeps serving. Only after the new adapter is loaded is the
    previous version retired via `backend.unload` (best-effort: a failed unload is logged, not
    raised) and then retired in the DB. If the deploy then loses the race on the partial unique
    index, its own freshly-loaded adapter is unloaded again so the winner's adapter is the only
    one left resident in vLLM.
    """

    if settings.serving_backend != "vllm":
        # Mock backend never touches a GPU, so there is nothing to serialize against training;
        # keep the pre-#59 behavior exact for tests and no-GPU local dev. Only a real vllm
        # deploy mutates the shared H100 and must take the same flock training uses.
        return _deploy_locked(
            db,
            model_version,
            backend,
            artifact_storage=artifact_storage,
            environment=environment,
        )

    try:
        with gpu_lock(
            lock_file or settings.gpu_lock_file,
            lock_timeout if lock_timeout is not None else settings.gpu_lock_timeout,
        ):
            return _deploy_locked(
                db,
                model_version,
                backend,
                artifact_storage=artifact_storage,
                environment=environment,
            )
    except TimeoutError as exc:
        logger.warning(
            "deploy_gpu_lock_timeout",
            model_id=model_version.model_id,
            version=model_version.version,
            lock_file=lock_file or settings.gpu_lock_file,
        )
        raise DeploymentLockTimeout(str(exc)) from exc


def _deploy_locked(
    db: Session,
    model_version: ModelVersion,
    backend: ServingBackend | None = None,
    *,
    artifact_storage: ArtifactStorage | None = None,
    environment: str | None = None,
) -> tuple[Deployment, ModelVersion | None]:
    """The locked body of `deploy` (issue #59): the pointer move plus all GPU-touching calls
    (`backend.deploy`, the smoke test, `backend.unload`). Runs inside the GPU lock when the
    serving backend is real; see `deploy` for the lock/timeout/`finally` semantics."""

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

    # Identity snapshot taken before any flush: if the deploy below loses the race and "this
    # call's" flush fails, SQLAlchemy expires the attributes of `model_version`, and re-reading
    # them (e.g. in the cleanup `unload` below) would lazy-load against the dead transaction and
    # raise PendingRollbackError instead of the conflict. The cleanup therefore unloads a detached
    # snapshot built from these captured values instead of the live ORM object.
    model_id = model_version.model_id
    version = model_version.version

    # A real backend (VLLMServingBackend) is created once per process and reused; the default
    # (mock) is what pre-#40 callers got from `backend or MockServingBackend()`. When the caller
    # targets a named environment without passing an explicit backend, resolve the backend for
    # that environment (issue #68, PRD §16.1) so a staging/prod deploy can hit a different host;
    # a None/`default` environment resolves to the process singleton (unchanged).
    if backend is None:
        backend = get_serving_backend(environment)

    # Issue #62: verify artifact integrity against the checksum recorded at finalize time
    # BEFORE loading the adapter and before any pointer moves. On mismatch the deploy aborts
    # here — nothing is loaded, nothing is unloaded, and the previous version stays DEPLOYED;
    # the failure is recorded as a log line.
    verify_artifact_checksum(model_version, artifact_storage)

    # Issue #65: reject a deploy whose artifact base_model doesn't match the served base model
    # (BaseModelMismatchError), before any load/pointer move. Skipped when served_base_model
    # is unset (default), so unconfigured/legacy deploys behave as before.
    _validate_base_model(model_version)

    backend.deploy(model_version)

    # Smoke test before the alias/pointer moves (issue #41): the `prod` alias must never point
    # at an adapter that cannot actually generate. Run a real generation against the just-loaded
    # adapter; on failure the adapter is unloaded again and the deploy aborts, leaving the
    # previous version DEPLOYED (and serving). The pointer only moves below, inside the same
    # transaction, so concurrent inference resolves the alias to the old adapter throughout.
    if settings.inference_smoke_enabled:
        _run_smoke_test(backend, model_version)

    if previous is not None:
        # Best-effort: unload failure must not block the retire - the newly loaded `model_version`
        # is the one that now serves. `ServingBackend.unload` is defined to not raise, but a
        # Protocol implementation could; never let it roll the deploy back.
        try:
            backend.unload(previous)
        except Exception:  # noqa: BLE001 - see comment above
            logger.warning(
                "serving_unload_failed",
                model_id=previous.model_id,
                version=previous.version,
                exc_info=True,
            )
        previous.status = "RETIRED"
        # Flush the retire on its own before marking `model_version` DEPLOYED: SQLAlchemy would
        # otherwise batch both status UPDATEs into one statement, and SQLite/Postgres evaluate the
        # partial unique index per row mid-statement - the transient "still DEPLOYED (previous),
        # now DEPLOYED (target)" state would trip the index even though the committed end state is
        # legal. Each UPDATE must therefore hit the table when at most one DEPLOYED row exists.
        db.flush()

    model_version.status = "DEPLOYED"
    deployment = Deployment(
        deployment_id=f"deployment-{uuid.uuid4().hex[:6]}",
        model_id=model_version.model_id,
        model_version=model_version.version,
        environment=environment or settings.deployment_environment,
        status="DEPLOYED",
        deployed_at=datetime.now(timezone.utc),
    )
    db.add(deployment)
    try:
        db.flush()
    except IntegrityError as exc:
        # A concurrent deploy committed first; the partial unique index rejects a second DEPLOYED
        # version for this model_id. Surface it as a domain conflict, not a raw IntegrityError.
        # We lost the race, so the adapter this call loaded must not stay resident - the winner's
        # adapter is the one the serving backend should keep.
        logger.warning(
            "deploy_conflict_cleanup_unload",
            model_id=model_id,
            version=version,
        )
        # Detached snapshot: `model_version`'s attributes were expired by the failed flush above,
        # so reading them (even just for `backend.unload`'s lora_name) would lazy-load against the
        # dead transaction and raise PendingRollbackError, masking this very conflict.
        try:
            backend.unload(ModelVersion(model_id=model_id, version=version))
        except Exception:  # noqa: BLE001 - cleanup must never mask the real conflict
            logger.warning(
                "deploy_conflict_cleanup_unload_failed",
                model_id=model_id,
                version=version,
                exc_info=True,
            )
        raise ValueError(_DEPLOYED_CONFLICT_MESSAGE) from exc
    return deployment, previous


def _run_smoke_test(backend: ServingBackend, model_version: ModelVersion) -> None:
    """Run the deploy-time smoke test (issue #41) against the just-loaded adapter and raise
    `SmokeTestError` on failure. Prompt and pass threshold are the explicit, configurable
    settings (`inference_smoke_prompt` / `inference_smoke_min_chars`), never a magic value.

    On failure the just-loaded adapter is unloaded again (best-effort, mirroring the deploy
    race-cleanup pattern) so the serving backend is left as it was before this deploy, the
    failure is recorded as a warning log line, and `deploy` aborts with the old version still
    DEPLOYED. Uses a detached `ModelVersion(model_id=..., version=...)` snapshot for the
    cleanup unload so a later failed flush / expired attributes can never mask the smoke
    failure with a lazy-load error.
    """
    model_id = model_version.model_id
    version = model_version.version
    try:
        output = backend.generate(settings.inference_smoke_prompt, model_id, version)
    except Exception as exc:  # noqa: BLE001 - any generation failure fails the smoke test
        logger.warning(
            "smoke_test_failed",
            model_id=model_id,
            version=version,
            reason="generation_error",
            error=str(exc),
        )
        _unload_after_smoke_failure(backend, model_id, version)
        raise SmokeTestError(
            f"smoke test failed for model_id {model_id!r} version {version}: "
            f"generation error: {exc}"
        ) from exc
    if len(output) < settings.inference_smoke_min_chars:
        logger.warning(
            "smoke_test_failed",
            model_id=model_id,
            version=version,
            reason="output_too_short",
            output_chars=len(output),
            min_chars=settings.inference_smoke_min_chars,
        )
        _unload_after_smoke_failure(backend, model_id, version)
        raise SmokeTestError(
            f"smoke test failed for model_id {model_id!r} version {version}: "
            f"generated {len(output)} chars, below the configured minimum "
            f"{settings.inference_smoke_min_chars}"
        )
    logger.info(
        "smoke_test_passed",
        model_id=model_id,
        version=version,
        output_chars=len(output),
    )


def _unload_after_smoke_failure(
    backend: ServingBackend, model_id: str, version: int
) -> None:
    """Best-effort unload of the adapter that just failed its smoke test, so the serving
    backend is left without the half-verified adapter. Must never mask the smoke failure."""
    try:
        backend.unload(ModelVersion(model_id=model_id, version=version))
    except Exception:  # noqa: BLE001 - cleanup must never mask the smoke failure
        logger.warning(
            "smoke_test_cleanup_unload_failed",
            model_id=model_id,
            version=version,
            exc_info=True,
        )


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
