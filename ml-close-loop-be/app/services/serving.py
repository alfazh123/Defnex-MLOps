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
process (`get_serving_backend`) so a real HTTP client is not recreated per call.
"""

from __future__ import annotations

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

    def generate(self, prompt: str, model_id: str, version: int) -> str:
        """Generate a completion for `prompt` from the adapter `{model_id}-v{version}` that
        must already be loaded. Returns the generated text; raises `InferenceError` on
        upstream failure or a malformed/empty completion (issue #41)."""
        ...


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

    def generate(self, prompt: str, model_id: str, version: int) -> str:
        """Canned, non-empty generation so the deploy-time smoke test (issue #41) passes in
        tests and no-GPU local dev the same way a real vLLM adapter would."""
        return f"mock generation for {model_id}-v{version}"


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

    def generate(self, prompt: str, model_id: str, version: int) -> str:
        adapter_name = f"{model_id}-v{version}"
        try:
            data = request_sync_with_retry(
                self._client,
                "POST",
                f"{self.base_url}/v1/completions",
                json={
                    "model": adapter_name,
                    "prompt": prompt,
                    "max_tokens": settings.inference_max_tokens,
                },
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
        logger.info(
            "vllm_generation",
            model_id=model_id,
            version=version,
            adapter_name=adapter_name,
            output_chars=len(text),
        )
        return text


_backend: ServingBackend | None = None


def get_serving_backend() -> ServingBackend:
    """The process-wide serving backend, created once and reused.

    Issue #40: the mock was previously instantiated fresh per call via
    `(backend or MockServingBackend())` in `deployment_service.deploy`, so the real vLLM HTTP
    client would have been rebuilt (and its connection dropped) on every deploy/rollback. A
    singleton fixes that while keeping the default (`settings.serving_backend == "mock"`)
    identical to pre-#40 behavior for tests and no-GPU local dev.
    """
    global _backend
    if _backend is None:
        if settings.serving_backend == "vllm":
            _backend = VLLMServingBackend()
        else:
            _backend = MockServingBackend()
    return _backend
