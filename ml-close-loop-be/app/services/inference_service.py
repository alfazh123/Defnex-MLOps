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
from app.services.serving import GenerationResult, ServingBackend

# Roles whose content is prompt scaffolding rather than conversation. They are flattened
# into the single text prompt the serving backend takes (issue #227), but kept separate from
# the transcript so a client debugging a request can see what was instruction and what was
# dialogue.
_SYSTEM_ROLES = ("system", "developer")


def flatten_messages(messages: list) -> tuple[str, str]:
    """Flatten OpenAI-style `messages[]` into `(system_prompt, transcript)`.

    The serving backend is a completion endpoint that takes one flat prompt; the chat-shaped
    surface sits on top of it. So the messages are rendered back into text here, which is
    the one place that has to know how: keeping it out of the router and out of the backend
    means a future `/v1/completions` consumer cannot accidentally inherit a second,
    different rendering.

    `developer` and `system` both become the system prompt. A request carrying both is not
    an error -- they are concatenated in order, which is the least surprising reading of
    "two instruction blocks, both apply".
    """

    system_parts: list[str] = []
    turns: list[str] = []
    for message in messages:
        role = message.role
        content = message.content
        if role in _SYSTEM_ROLES:
            system_parts.append(content)
            continue
        # "User:"/"Assistant:" prefixes rather than a chat template: the vLLM adapter path
        # has no template of its own, and a stable readable rendering beats a guessed one.
        turns.append(f"{role.capitalize()}: {content}")

    if not turns and not system_parts:
        return "", ""
    system_prompt = "\n\n".join(system_parts)
    transcript = "\n".join(turns)
    if system_prompt and transcript:
        return system_prompt, f"{system_prompt}\n\n{transcript}"
    return system_prompt, transcript


def parse_model_ref(ref: str) -> tuple[str, int | None]:
    """Split a `model` reference into `(model_id, version)`.

    `"name"` -> version None (serve whatever is DEPLOYED); `"name:3"` -> version 3.

    Only the last colon is a separator, so a `model_id` that somehow contains a colon keeps
    working as a bare name rather than being silently truncated at the wrong place.
    """

    head, sep, tail = ref.rpartition(":")
    if sep and tail.isdigit():
        return head, int(tail)
    return ref, None


def resolve_model_ref(db: Session, ref: str) -> ModelVersion:
    """Resolve an OpenAI-style `model` field to the concrete version that will serve it.

    A bare `model_id` resolves through the deployment alias (`prod`), so a client can be
    pointed at "the current model" the way it would be against any OpenAI-compatible server.
    An explicit `model_id:version` must exist and be DEPLOYED — only the DEPLOYED adapter is
    loaded and smoke-tested, so serving any other status would fail upstream anyway.

    Raises `ValueError` for every failure; the router turns those into 404/409.
    """

    model_id, version = parse_model_ref(ref)
    if not model_id:
        raise ValueError("model must name a model, e.g. 'my-model' or 'my-model:2'")
    if version is not None:
        return resolve_target(db, model_id, str(version))
    return deployment_service.resolve_alias(db, model_id, "prod")


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
) -> GenerationResult:
    """Generate a completion for `prompt` from `model_version`'s adapter. `max_tokens`/
    `temperature` are optional per-request sampling overrides (issue #179). Raises the
    backend's `InferenceError` on upstream failure (translated to 502 by the router).

    Returns a `GenerationResult` rather than bare text so the `usage` block the OpenAI
    contract requires can be filled from real numbers (issue #228).
    """
    return backend.generate(
        prompt,
        model_version.model_id,
        model_version.version,
        max_tokens=max_tokens,
        temperature=temperature,
    )
