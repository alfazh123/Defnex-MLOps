"""Serving boundary for deployed model artifacts (issue #40).

The `ServingBackend` Protocol is the only thing callers depend on -
`deployment_service.deploy` invokes `deploy`/`unload` and never touches the concrete
implementation. Two implementations exist:

- `MockServingBackend` records calls only, used for tests and local dev without a GPU
  (the default via `settings.serving_backend == "mock"`, matching the pre-#40 contract).
- `VLLMServingBackend` talks to a real vLLM Server / vLLM Runtime via its HTTP API:
  `POST /v1/load_lora_adapter` to make an adapter serve traffic and
  `POST /v1/unload_lora_adapter` to stop serving it (runtime LoRA load/unload, no service
  restart). Enabled with `settings.serving_backend == "vllm"`.

`serving.py` replaces the pre-#40 placeholder with a stateful backend that is built once per
process (`get_serving_backend`) so a real HTTP client is not recreated per call. Since #68 the
default (None/`default`) backend remains that process singleton, while a named environment
(e.g. `staging`/`production`) resolves to its own cached backend pointed at that environment's
URL from `vllm_url_by_env` (PRD §16.1) — letting a deploy target a different host per
environment (PRD §19.4).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import httpx
import structlog

from app.config import settings
from app.models.model import ModelVersion
from app.services.http_retry import request_sync_with_retry

logger = structlog.get_logger(__name__)


class ServingError(Exception):
    """The serving backend failed to load the requested model artifact. Raised by `deploy`
    only for the *load* of the version being deployed; unload failures are best-effort and
    never raise (see `VLLMServingBackend.unload`)."""


class InferenceError(Exception):
    """The serving backend could not produce a generation for a loaded adapter (issue #41):
    unreachable backend, HTTP error from vLLM, or a malformed/empty completion response.
    Raised by `generate`; callers (the inference endpoint, the deploy-time smoke test) treat
    it as "this adapter cannot serve traffic right now"."""


class BaseModelMismatchError(Exception):
    """The artifact being deployed was trained on a different base model than the one the
    serving stack is currently running (issue #65). Deploy must reject this *before* the
    pointer moves — a base-model change is a controlled recreate/redeploy (PRD §17.4), never
    a silent adapter hot-swap onto a mismatched base."""


def _validate_base_model(model_version: ModelVersion) -> None:
    """(issue #65) Reject a deploy whose artifact's recorded `base_model` does not match the
    base model the serving stack is actually running (`settings.served_base_model`).

    Runs before any adapter load or pointer move. When `served_base_model` is empty (default)
    the check is skipped so unconfigured / legacy setups keep deploying unchanged. When set,
    a mismatch raises `BaseModelMismatchError` and the deploy is refused — the operator must
    recreate the serving stack on the matching base rather than rely on a hot-swap (PRD §17.4).
    """
    served = settings.served_base_model
    if not served:
        return
    if model_version.base_model != served:
        raise BaseModelMismatchError(
            f"cannot deploy model {model_version.model_id} v{model_version.version}: "
            f"artifact base_model {model_version.base_model!r} does not match the served "
            f"base model {served!r}; recreate/redeploy the serving stack on the matching "
            "base model (PRD §17.4), do not hot-swap"
        )


class ServingBackend(Protocol):
    def deploy(self, model_version: ModelVersion) -> None:
        """Make `model_version`'s artifact serve traffic. Raises `ServingError` on failure
        (e.g. unreachable backend, adapter load rejected, or no adapter artifact)."""
        ...

    def unload(self, model_version: ModelVersion) -> None:
        """Stop serving `model_version`'s artifact. Idempotent and non-raising: a backend that
        cannot unload (already unloaded, unreachable, ...) must log rather than block the
        deploy that retires this version."""
        ...

    def generate(
        self,
        prompt: str,
        model_id: str,
        version: int,
        *,
        max_tokens: int | None = None,
        temperature: float | None = None,
    ) -> GenerationResult:
        """Generate a completion for `prompt` from the adapter `{model_id}-v{version}` that
        must already be loaded. `max_tokens`/`temperature` are optional per-request sampling
        overrides (issue #179); omitted means "use the backend's own default". Returns a
        `GenerationResult` carrying the text plus token usage (issue #228); raises
        `InferenceError` on upstream failure or a malformed/empty completion (issue #41)."""
        ...


@dataclass
class GenerationResult:
    """What a serving backend produced for one request.

    `usage` is not decoration: issue #228 found that vLLM already returns token counts and
    this layer threw them away, so there was no way to attribute cost or rate-limit by token
    volume. The three canonical fields are always present; the `*_details` sub-objects stay
    None because they are optional in the OpenAI spec and this layer has no source for
    them beyond what vLLM reports.
    """

    text: str
    prompt_tokens: int = 0
    completion_tokens: int = 0

    @property
    def total_tokens(self) -> int:
        return self.prompt_tokens + self.completion_tokens

    def to_usage(self) -> dict:
        return {
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "total_tokens": self.total_tokens,
        }


def _estimate_tokens(text: str) -> int:
    """Rough token count for backends that cannot report a real one.

    Deliberately labeled an estimate: the mock backend has no tokenizer, and a number that
    looks authoritative but is not would be worse than one that is documented as an
    approximation. ~4 characters per token is the usual English-text rule of thumb.
    """

    return max(1, (len(text) + 3) // 4) if text else 0


class MockServingBackend:
    """Placeholder serving backend for tests and no-GPU local dev. The pre-#40 docstring kept
    "the concrete stack is TBD"; issue #40 decides that stack is vLLM and adds
    `VLLMServingBackend` for production, leaving this mock behind for the test suite.

    Records its calls so tests (and the end-to-end lifecycle test) can assert the deploy and
    unload paths reached the serving boundary.
    """

    def __init__(self) -> None:
        self.deployed: list[tuple[str, int]] = []
        self.unloaded: list[tuple[str, int]] = []
        self.events: list[tuple[str, tuple[str, int]]] = []

    def deploy(self, model_version: ModelVersion) -> None:
        self.deployed.append((model_version.model_id, model_version.version))
        # Interleaved with `unload` events so tests can assert load-before-unload ordering.
        self.events.append(("load", (model_version.model_id, model_version.version)))

    def unload(self, model_version: ModelVersion) -> None:
        self.unloaded.append((model_version.model_id, model_version.version))
        self.events.append(("unload", (model_version.model_id, model_version.version)))

    def generate(
        self,
        prompt: str,
        model_id: str,
        version: int,
        *,
        max_tokens: int | None = None,
        temperature: float | None = None,
    ) -> GenerationResult:
        """Canned, non-empty generation so the deploy-time smoke test (issue #41) passes in
        tests and no-GPU local dev the same way a real vLLM adapter would. Sampling params
        are accepted (protocol compliance) but don't affect the canned text - there's no
        real model here to sample from.

        Token counts are estimates (`_estimate_tokens`): this backend has no tokenizer, and
        the point of the `usage` block (issue #228) is that callers get *something* to
        account with, not that a mock is as accurate as vLLM.
        """
        text = f"mock generation for {model_id}-v{version}"
        return GenerationResult(
            text=text,
            prompt_tokens=_estimate_tokens(prompt),
            completion_tokens=_estimate_tokens(text),
        )


def _lora_name(model_version: ModelVersion) -> str:
    """Deterministic adapter identity for the vLLM LoRA registry. Collapses to
    `{model_id}-v{version}` — shorter than the 3-part artifact name
    (`model_service.build_version_name`, `{model_id}-{base_model_slug}-v{N}`)
    because the registry keys on model_id + version only; base_model is redundant here
    since a version already binds a specific base_model."""
    return f"{model_version.model_id}-v{model_version.version}"


def _adapter_path(model_version: ModelVersion) -> str:
    """Local filesystem path of the adapter artifact to load, from `model_version.artifacts`
    (`{"type": "adapter", "uri": "file:///..."}`, app/schemas/model.py Artifact /
    model-artifact-versioning-lineage.md §8). A `file://` URI is converted to the path vLLM
    mounts; any other URI/tag is used verbatim (e.g. a Hugging Face repo id)."""
    for artifact in model_version.artifacts or []:
        if artifact.get("type") == "adapter":
            uri = artifact["uri"]
            return uri[len("file://") :] if uri.startswith("file://") else uri
    raise ServingError(
        f"model {model_version.model_id} v{model_version.version} has no 'adapter' "
        "artifact to load (artifacts: "
        f"{[a.get('type') for a in model_version.artifacts or []]})"
    )


def parse_vllm_url_by_env(raw: str) -> dict[str, str]:
    """Parse the `vllm_url_by_env` setting (`env:url` pairs, comma-separated) into a mapping
    (issue #68, PRD §16.1). The URL is split on its *first* colon only, so `http://…` URLs with
    more colons survive intact. Empty pairs are skipped; a malformed entry (no `:` or an empty
    side) raises ValueError so a misconfigured topology fails loudly rather than silently
    routing an environment to the wrong host."""
    mapping: dict[str, str] = {}
    for pair in raw.split(","):
        pair = pair.strip()
        if not pair:
            continue
        env, _, url = pair.partition(":")
        env = env.strip()
        url = url.strip()
        if not env or not url:
            raise ValueError(
                f"invalid VLLM_URL_BY_ENV entry {pair!r}; expected `env:url`"
            )
        mapping[env] = url
    return mapping


def _auth_header(api_key: str) -> dict[str, str]:
    headers: dict[str, str] = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    return headers


class VLLMServingBackend:
    """Real serving backend backed by a vLLM Server (OpenAI-compatible runtime API).

    `deploy` loads the model version's adapter at runtime (`/v1/load_lora_adapter`,
    requires vLLM launched with `--enable-lora` and `VLLM_ALLOW_RUNTIME_LORA_UPDATING=true`);
    `unload` removes it (`/v1/unload_lora_adapter`). Both go through the shared HTTP retry
    policy (retries on 5xx/connection errors, never on 4xx). Load failure raises
    `ServingError` so the DB transaction that retires the previous version is never started;
    unload failure is logged and swallowed (the newly deployed version already serves).
    """

    def __init__(
        self,
        base_url: str | None = None,
        api_key: str | None = None,
        timeout: float | None = None,
        client: httpx.Client | None = None,
    ) -> None:
        self.base_url = (base_url or settings.vllm_url).rstrip("/")
        self.api_key = api_key if api_key is not None else settings.vllm_api_key
        self._client = client or httpx.Client(
            timeout=httpx.Timeout(timeout or settings.vllm_timeout_seconds)
        )

    def _headers(self) -> dict[str, str]:
        return _auth_header(self.api_key)

    def deploy(self, model_version: ModelVersion) -> None:
        adapter_name = _lora_name(model_version)
        adapter_path = _adapter_path(model_version)
        try:
            request_sync_with_retry(
                self._client,
                "POST",
                f"{self.base_url}/v1/load_lora_adapter",
                json={"lora_name": adapter_name, "lora_path": adapter_path},
                headers=self._headers(),
                context=adapter_name,
                parse_json=False,
            )
        except httpx.HTTPStatusError as exc:
            raise ServingError(
                f"vLLM rejected adapter load {adapter_name} ({adapter_path}): "
                f"HTTP {exc.response.status_code} {exc.response.text[:200]}"
            ) from exc
        except httpx.RequestError as exc:
            raise ServingError(
                f"vLLM unreachable at {self.base_url} while loading {adapter_name}: {exc}"
            ) from exc
        logger.info(
            "vllm_adapter_loaded",
            lora_name=adapter_name,
            lora_path=adapter_path,
            model_id=model_version.model_id,
            version=model_version.version,
        )

    def unload(self, model_version: ModelVersion) -> None:
        adapter_name = _lora_name(model_version)
        try:
            request_sync_with_retry(
                self._client,
                "POST",
                f"{self.base_url}/v1/unload_lora_adapter",
                json={"lora_name": adapter_name},
                headers=self._headers(),
                context=adapter_name,
                parse_json=False,
            )
        except httpx.HTTPStatusError as exc:
            if exc.response.status_code == 404:
                logger.warning(
                    "vllm_adapter_not_loaded",
                    lora_name=adapter_name,
                    model_id=model_version.model_id,
                    version=model_version.version,
                )
                return
            logger.warning(
                "vllm_unload_failed",
                lora_name=adapter_name,
                status_code=exc.response.status_code,
                response_body=exc.response.text[:200],
                model_id=model_version.model_id,
                version=model_version.version,
            )
            return
        except httpx.RequestError as exc:
            logger.warning(
                "vllm_unload_unreachable",
                lora_name=adapter_name,
                base_url=self.base_url,
                error=str(exc),
                model_id=model_version.model_id,
                version=model_version.version,
            )
            return
        logger.info(
            "vllm_adapter_unloaded",
            lora_name=adapter_name,
            model_id=model_version.model_id,
            version=model_version.version,
        )

    def generate(
        self,
        prompt: str,
        model_id: str,
        version: int,
        *,
        max_tokens: int | None = None,
        temperature: float | None = None,
    ) -> GenerationResult:
        adapter_name = f"{model_id}-v{version}"
        payload = {
            "model": adapter_name,
            "prompt": prompt,
            "max_tokens": max_tokens
            if max_tokens is not None
            else settings.inference_max_tokens,
        }
        if temperature is not None:
            payload["temperature"] = temperature
        try:
            data = request_sync_with_retry(
                self._client,
                "POST",
                f"{self.base_url}/v1/completions",
                json=payload,
                headers=self._headers(),
                context=adapter_name,
            )
        except httpx.HTTPStatusError as exc:
            raise InferenceError(
                f"vLLM rejected generation for {adapter_name}: "
                f"HTTP {exc.response.status_code} {exc.response.text[:200]}"
            ) from exc
        except httpx.RequestError as exc:
            raise InferenceError(
                f"vLLM unreachable at {self.base_url} while generating for {adapter_name}: {exc}"
            ) from exc
        choices = data.get("choices") or []
        text = str(choices[0]["text"]) if choices and "text" in choices[0] else ""
        if not text:
            raise InferenceError(
                f"vLLM returned no completion text for {adapter_name} "
                f"(choices: {choices!r})"
            )
        # Issue #228: vLLM already returns this block; it used to be dropped here, leaving
        # no way to attribute token volume or cost per request. Read defensively -- an older
        # or proxied vLLM may omit it, and a missing usage block should not fail a
        # generation that succeeded.
        raw_usage = data.get("usage") or {}
        result = GenerationResult(
            text=text,
            prompt_tokens=int(raw_usage.get("prompt_tokens") or 0),
            completion_tokens=int(raw_usage.get("completion_tokens") or 0),
        )
        logger.info(
            "vllm_generation",
            model_id=model_id,
            version=version,
            adapter_name=adapter_name,
            output_chars=len(text),
            prompt_tokens=result.prompt_tokens,
            completion_tokens=result.completion_tokens,
        )
        return result


_backend: ServingBackend | None = None
_env_backends: dict[str, ServingBackend] = {}


def _build_backend(base_url: str | None = None) -> ServingBackend:
    """Construct the backend for `base_url` (or the process default when None). Mock never
    touches a host, so a vllm backend is only built when `serving_backend == "vllm"`."""
    if settings.serving_backend == "vllm":
        return VLLMServingBackend(base_url=base_url)
    return MockServingBackend()


def get_serving_backend(environment: str | None = None) -> ServingBackend:
    """The process-wide serving backend for `environment` (issue #68, PRD §16.1/§17.1).

    - `environment` None or `"default"` → the default singleton, created once and reused
      (issue #40): the mock was previously instantiated fresh per call via
      `(backend or MockServingBackend())` in `deployment_service.deploy`, so a real vLLM HTTP
      client would have been rebuilt (and its connection dropped) on every deploy/rollback.
    - a named environment (e.g. `"staging"`, `"production"`) → a per-environment backend cached
      in `_env_backends`, constructed with the environment's URL from `vllm_url_by_env`. This is
      how a staging/prod deploy can target a *different* host than the default (PRD §19.4):
      changing the host is a config change, not a code change.
    """
    global _backend
    if environment is None or environment == "default":
        if _backend is None:
            _backend = _build_backend()
        return _backend
    if environment not in _env_backends:
        urls = parse_vllm_url_by_env(settings.vllm_url_by_env)
        _env_backends[environment] = _build_backend(urls.get(environment))
    return _env_backends[environment]
