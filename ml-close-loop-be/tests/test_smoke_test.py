"""Deploy-time smoke test (issue #41): `deployment_service.deploy` runs a real generation
against the just-loaded adapter, *before* the `prod` alias / DEPLOYED pointer moves.

Covers the gate against a real `VLLMServingBackend` wire handler (load + completions + unload)
rather than a mock: the gate's prompt/threshold come from config, a failing generation aborts the
deploy with the alias still on the old version and the just-loaded adapter unloaded again, the
gate is skippable, and the smoke failure surfaces through the API as `502 SMOKE_TEST_FAILED`
with the pointer unchanged - which is exactly the guarantee that inference arriving mid-deploy
always resolves to the old, still-smoke-tested adapter.
"""

import json

import httpx
import pytest

from app.services import deployment_service
from app.services.deployment_service import SmokeTestError
from app.services.serving import VLLMServingBackend

from tests.conftest import auth_header
from tests.test_deployment_api import (
    _promoted_model_version as _http_promoted_model_version,
)
from tests.test_deployment_service import _promoted_model_version


@pytest.fixture(autouse=True)
def _no_sleep(monkeypatch):
    """Run the sync retry backoff without actually sleeping (retry count is asserted where it
    matters, in tests/test_vllm_serving.py)."""
    monkeypatch.setattr("app.services.http_retry.time.sleep", lambda _seconds: None)


@pytest.fixture(autouse=True)
def _disable_promotion_gates(monkeypatch):
    """`_promoted_model_version` doesn't attach an eval-set reference, so without this the #43
    eval gate blocks every promotion with 409 before deploy is ever reached."""
    from app.config import settings

    for name in (
        "eval_gate_require_eval_set_reference",
        "eval_gate_require_qualitative_majority",
        "eval_gate_require_no_general_regression",
        "eval_gate_require_eval_loss_not_worse",
    ):
        monkeypatch.setattr(settings, name, False)


def _client_for(handler) -> httpx.Client:
    return httpx.Client(transport=httpx.MockTransport(handler))


def _vllm_backend(requests: list, completions: object) -> VLLMServingBackend:
    """VLLMServingBackend serving load/unload as 200 and `/v1/completions` as the given
    Response, recording every request so tests can assert what reached vLLM."""

    def handler(request):
        requests.append(request)
        if request.url.path.endswith("completions"):
            return completions
        return httpx.Response(200, json={"message": "ok"})

    return VLLMServingBackend(client=_client_for(handler))


def _completions_ok(text: str = "OK!") -> httpx.Response:
    return httpx.Response(200, json={"choices": [{"text": text}]})


def test_smoke_generation_runs_before_pointer_moves(db_session, monkeypatch):
    from app.config import settings

    monkeypatch.setattr(settings, "inference_smoke_prompt", "Say OK")
    model_version, _ = _promoted_model_version(db_session)
    requests: list[httpx.Request] = []
    backend = _vllm_backend(requests, _completions_ok())

    deployment, previous = deployment_service.deploy(
        db_session, model_version, backend=backend
    )
    db_session.commit()

    completions = [r for r in requests if r.url.path.endswith("completions")]
    assert len(completions) == 1
    assert json.loads(completions[0].content) == {
        "model": f"{model_version.model_id}-v{model_version.version}",
        "prompt": "Say OK",
        "max_tokens": settings.inference_max_tokens,
    }
    # the gate passed on the very first deploy, so nothing was unloaded
    assert not [r for r in requests if r.url.path.endswith("unload_lora_adapter")]
    assert model_version.status == "DEPLOYED"


def test_smoke_too_short_aborts_and_alias_stays_on_old_version(db_session, monkeypatch):
    from app.config import settings

    v1, dataset_version = _promoted_model_version(db_session)
    deployment_service.deploy(
        db_session, v1
    )  # mock backend - its generation always passes
    db_session.commit()

    v2, _ = _promoted_model_version(db_session, dataset_version)
    requests: list[httpx.Request] = []
    # "OK" is 2 chars, below the threshold we raise to 5
    backend = _vllm_backend(requests, _completions_ok(text="OK"))
    monkeypatch.setattr(settings, "inference_smoke_min_chars", 5)

    with pytest.raises(SmokeTestError, match="below the configured minimum"):
        deployment_service.deploy(db_session, v2, backend=backend)

    # the alias still resolves to v1 - the smoke run happened before any pointer movement
    assert (
        deployment_service.resolve_alias(db_session, v1.model_id, "prod").version
        == v1.version
    )
    # inference "arriving" during/after the aborted switch resolves to the old adapter and
    # serves from it: the endpoint's target resolution and generation both pass through
    # deployment_service.resolve_alias / the ServingBackend (AC9)
    resolved = deployment_service.resolve_alias(db_session, v1.model_id, "prod")
    from app.services.serving import MockServingBackend

    generation = MockServingBackend().generate(
        "What is the capital of France?", resolved.model_id, resolved.version
    )
    assert generation.text == (
        f"mock generation for {resolved.model_id}-v{resolved.version}"
    )
    assert resolved.version == v1.version
    assert v1.status == "DEPLOYED"
    assert v2.status == "PROMOTED"
    # the just-loaded v2 adapter was unloaded again, leaving the serving backend as before
    unloads = [r for r in requests if r.url.path.endswith("unload_lora_adapter")]
    assert len(unloads) == 1
    assert json.loads(unloads[0].content)["lora_name"] == f"{v2.model_id}-v{v2.version}"


def test_smoke_generation_error_aborts_and_keeps_previous(db_session):
    v1, dataset_version = _promoted_model_version(db_session)
    deployment_service.deploy(db_session, v1)
    db_session.commit()

    v2, _ = _promoted_model_version(db_session, dataset_version)
    requests: list[httpx.Request] = []
    backend = _vllm_backend(requests, httpx.Response(500, json={"error": "OOM"}))

    with pytest.raises(SmokeTestError, match="generation error"):
        deployment_service.deploy(db_session, v2, backend=backend)

    assert v1.status == "DEPLOYED"
    assert v2.status == "PROMOTED"
    unloads = [r for r in requests if r.url.path.endswith("unload_lora_adapter")]
    assert len(unloads) == 1
    assert json.loads(unloads[0].content)["lora_name"] == f"{v2.model_id}-v{v2.version}"


def test_smoke_disabled_skips_generation(db_session, monkeypatch):
    from app.config import settings

    monkeypatch.setattr(settings, "inference_smoke_enabled", False)
    model_version, _ = _promoted_model_version(db_session)
    requests: list[httpx.Request] = []
    backend = _vllm_backend(requests, httpx.Response(500, json={"error": "OOM"}))

    deployment_service.deploy(db_session, model_version, backend=backend)
    db_session.commit()

    # the 500-servicing completions path was never hit
    assert not [r for r in requests if r.url.path.endswith("completions")]
    assert model_version.status == "DEPLOYED"


def test_deploy_returns_502_smoke_test_failed_and_pointer_unchanged(
    client, admin_token, monkeypatch
):
    from app.services import serving as serving_module

    requests: list[httpx.Request] = []
    backend = _vllm_backend(requests, httpx.Response(500, json={"error": "OOM"}))
    monkeypatch.setattr(serving_module, "_backend", backend)

    model_id, version = _http_promoted_model_version(client, admin_token)
    h = auth_header(admin_token)
    resp = client.post(
        f"/api/v1/models/{model_id}/versions/{version}/deploy", headers=h
    )

    assert resp.status_code == 502
    assert resp.json()["error"]["code"] == "SMOKE_TEST_FAILED"
    # the load ran, the smoke failed, the pointer never moved
    assert any(r.url.path.endswith("load_lora_adapter") for r in requests)
    assert (
        client.get(f"/api/v1/models/{model_id}/versions/{version}", headers=h).json()[
            "status"
        ]
        == "PROMOTED"
    )
    assert (
        client.get(f"/api/v1/models/{model_id}/deployment", headers=h).json()[
            "current_deployed_version"
        ]
        is None
    )
