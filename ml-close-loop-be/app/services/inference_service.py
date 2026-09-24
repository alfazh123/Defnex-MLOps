"""Inference service (issue #41): resolve a model reference (alias or explicit version) to the
concrete version that must serve a request, then generate from it through the `ServingBackend`.

Alias resolution delegates to the *single* resolution function from issue #36
(`deployment_service.resolve_alias`) - there is no second alias-resolution logic here. An
explicit version number is a separate, non-alias lookup: it must exist and currently hold
`DEPLOYED` status, because only the DEPLOYED adapter is guaranteed to be loaded (and to have
passed the deploy-time smoke test) in the serving backend.
"""

from __future__ import annotations

from sqlalchemy.orm import Session

from app.models.model import ModelVersion
from app.services import deployment_service, model_service
from app.services.serving import ServingBackend


def resolve_target(db: Session, model_id: str, target: str) -> ModelVersion:
    """Resolve an inference request's `target` (a deployment alias like ``prod``, or an
    explicit version number) to the concrete ModelVersion that will serve the request.

    Raises `ValueError` (translated by the router) for every failure case so an unknown alias,
    a missing version, or a never-deployed model can never silently propagate as a None.

    - alias (``prod``): delegates to `deployment_service.resolve_alias` (issue #36) - the
      single alias-resolution point; unknown alias / no deployed version raise ValueError.
    - explicit version number: must exist (else ValueError) and be `DEPLOYED` (else
      ValueError) - only the deployed adapter is loaded and smoke-tested.
    - anything else: ValueError - a target must be an alias or a version number.
    """
    if target in deployment_service.SUPPORTED_ALIASES:
        return deployment_service.resolve_alias(db, model_id, target)
    if target.isdigit():
        model_version = model_service.get_model_version(db, model_id, int(target))
        if model_version is None:
            raise ValueError(f'model_id "{model_id}" version {target} not found')
        if model_version.status != "DEPLOYED":
            raise ValueError(
                f'model_id "{model_id}" version {target} is {model_version.status}; '
                "only the DEPLOYED version can serve inference"
            )
        return model_version
    raise ValueError(
        f"target {target!r} must be a deployment alias "
        f"(supported: {sorted(deployment_service.SUPPORTED_ALIASES)}) or a version number"
    )


def generate(
    backend: ServingBackend,
    model_version: ModelVersion,
    prompt: str,
    *,
    max_tokens: int | None = None,
    temperature: float | None = None,
) -> str:
    """Generate a completion for `prompt` from `model_version`'s adapter. `max_tokens`/
    `temperature` are optional per-request sampling overrides (issue #179). Raises the
    backend's `InferenceError` on upstream failure (translated to 502 by the router)."""
    return backend.generate(
        prompt,
        model_version.model_id,
        model_version.version,
        max_tokens=max_tokens,
        temperature=temperature,
    )
