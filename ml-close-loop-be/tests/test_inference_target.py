"""Issue #68: InferenceTarget abstraction per environment (PRD §16.1/§17.1/§19.4).

Proves that an `InferenceTarget` is represented as data/config (not hard-coded): a
per-environment vLLM host comes from `vllm_url_by_env` and `get_serving_backend(environment)`
resolves a cached backend pointed at that host. `deployment_service` routes a named-environment
deploy through that environment's backend, so staging and production can target *different*
hosts, and changing a host is a config change, not a code change.
"""

from datetime import datetime, timezone

import pytest
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.config import settings
from app.models.model import Model, ModelVersion
from app.services import deployment_service
from app.services import serving as serving_module
from app.services.serving import (
    MockServingBackend,
    VLLMServingBackend,
    parse_vllm_url_by_env,
)


@pytest.fixture(autouse=True)
def _restore_caches(monkeypatch):
    """Keep the module-level backend caches isolated between tests so a cached mock/vllm
    backend from another test never leaks into this suite."""
    monkeypatch.setattr(serving_module, "_backend", None)
    monkeypatch.setattr(serving_module, "_env_backends", {})


@pytest.fixture(autouse=True)
def _no_smoke(monkeypatch):
    monkeypatch.setattr(settings, "inference_smoke_enabled", False)


def _seed(db: Session, model_id: str, versions: list[int]) -> None:
    db.add(Model(model_id=model_id))
    db.flush()
    for ver in versions:
        db.add(
            ModelVersion(
                model_id=model_id,
                version=ver,
                status="PROMOTED",
                training_run_id=f"tr-{ver}",
                base_model="base",
                training_config={},
                artifacts=[
                    {
                        "type": "adapter",
                        "uri": f"file:///data/adapters/{model_id}-v{ver}",
                    }
                ],
                created_at=datetime.now(timezone.utc),
            )
        )
    db.commit()


def _v(db: Session, model_id: str, version: int) -> ModelVersion:
    return db.scalars(
        select(ModelVersion).where(
            ModelVersion.model_id == model_id, ModelVersion.version == version
        )
    ).one()


# ---------------------------------------------------------------------------
# vllm_url_by_env parsing (the data/config representation, PRD §16.1/§19.4)
# ---------------------------------------------------------------------------


def test_parse_vllm_url_by_env_pairs():
    assert parse_vllm_url_by_env(
        "staging:http://staging-vllm:8001,production:http://prod-vllm:8001"
    ) == {
        "staging": "http://staging-vllm:8001",
        "production": "http://prod-vllm:8001",
    }


def test_parse_vllm_url_by_env_skips_empty_pairs_and_blank():
    assert parse_vllm_url_by_env("staging:http://s:1,,   ,production:http://p:2") == {
        "staging": "http://s:1",
        "production": "http://p:2",
    }
    assert parse_vllm_url_by_env("") == {}


def test_parse_vllm_url_by_env_rejects_malformed():
    with pytest.raises(ValueError, match="expected `env:url`"):
        parse_vllm_url_by_env("staging:http://s:1,no-url")
    with pytest.raises(ValueError, match="expected `env:url`"):
        parse_vllm_url_by_env(":http://s:1")


# ---------------------------------------------------------------------------
# get_serving_backend(environment) resolves env-specific hosts (PRD §16.1)
# ---------------------------------------------------------------------------


def test_get_serving_backend_environment_builds_vllm_at_env_url(monkeypatch):
    monkeypatch.setattr(settings, "serving_backend", "vllm")
    monkeypatch.setattr(
        settings,
        "vllm_url_by_env",
        "staging:http://staging-vllm:8001,production:http://prod-vllm:8001",
    )

    staging = serving_module.get_serving_backend("staging")
    production = serving_module.get_serving_backend("production")

    assert isinstance(staging, VLLMServingBackend)
    assert isinstance(production, VLLMServingBackend)
    assert staging.base_url == "http://staging-vllm:8001"
    assert production.base_url == "http://prod-vllm:8001"
    assert staging is not production
    # caching: the same env returns the same backend instance
    assert serving_module.get_serving_backend("staging") is staging


def test_get_serving_backend_environment_falls_back_to_default_url(monkeypatch):
    monkeypatch.setattr(settings, "serving_backend", "vllm")
    monkeypatch.setattr(settings, "vllm_url_by_env", "staging:http://s:8001")

    production = serving_module.get_serving_backend("production")

    assert isinstance(production, VLLMServingBackend)
    assert production.base_url == settings.vllm_url


def test_get_serving_backend_none_and_default_are_mock_singleton():
    first = serving_module.get_serving_backend()
    second = serving_module.get_serving_backend(None)
    third = serving_module.get_serving_backend("default")
    assert all(isinstance(b, MockServingBackend) for b in (first, second, third))
    assert first is second is third


def test_get_serving_backend_env_is_mock_in_mock_mode(monkeypatch):
    monkeypatch.setattr(settings, "serving_backend", "mock")
    monkeypatch.setattr(settings, "vllm_url_by_env", "staging:http://s:8001")

    staging = serving_module.get_serving_backend("staging")

    assert isinstance(staging, MockServingBackend)


def test_get_serving_backend_unconfigured_env_is_mock(monkeypatch):
    monkeypatch.setattr(settings, "serving_backend", "mock")

    canary = serving_module.get_serving_backend("canary")

    assert isinstance(canary, MockServingBackend)


# ---------------------------------------------------------------------------
# deploy routes a named environment through its backend (PRD §16.1/§19.4)
# ---------------------------------------------------------------------------


def test_deploy_resolves_backend_from_environment(monkeypatch, db_session):
    """A deploy that names an environment without an explicit backend resolves the backend via
    `get_serving_backend(environment)` — the single resolution point (PRD §17.1)."""

    def fake_get_serving_backend(environment):
        assert environment == "production"
        return MockServingBackend()

    monkeypatch.setattr(serving_module, "get_serving_backend", fake_get_serving_backend)

    _seed(db_session, "m-env-target", [1])
    deployment, _ = deployment_service.deploy(
        db_session, _v(db_session, "m-env-target", 1), environment="production"
    )

    assert deployment.environment == "production"


def test_deploy_passes_explicit_backend_through(monkeypatch, db_session):
    """An explicit backend always wins over environment resolution (calls that hand a backend
    in—e.g. promotion_service.rollback—are unaffected, PRD §17.1)."""

    def fake_get_serving_backend(environment):
        raise AssertionError("explicit backend should bypass environment resolution")

    monkeypatch.setattr(serving_module, "get_serving_backend", fake_get_serving_backend)

    _seed(db_session, "m-explicit", [1])
    explicit = MockServingBackend()
    deployment, _ = deployment_service.deploy(
        db_session,
        _v(db_session, "m-explicit", 1),
        backend=explicit,
        environment="production",
    )

    assert deployment.environment == "production"
    assert explicit.deployed == [("m-explicit", 1)]


def test_deploy_without_environment_uses_default_backend(monkeypatch, db_session):
    """A None/`default` environment stays on the process default backend (backward compatible
    with pre-#67 callers that pass neither environment nor backend)."""

    def fake_get_serving_backend(environment):
        assert environment is None or environment == "default"
        return MockServingBackend()

    monkeypatch.setattr(serving_module, "get_serving_backend", fake_get_serving_backend)

    _seed(db_session, "m-nil-env", [1])
    deployment, _ = deployment_service.deploy(
        db_session, _v(db_session, "m-nil-env", 1)
    )

    assert deployment.environment == "default"


def test_staging_and_production_resolve_to_different_hosts(monkeypatch, db_session):
    """Core AC (§38.4/§19.4): a deploy to staging vs production targets a different host, both
    derived from config (`vllm_url_by_env`) and resolved through the single
    `get_serving_backend(environment)` point (PRD §17.1)."""
    monkeypatch.setattr(settings, "serving_backend", "vllm")
    monkeypatch.setattr(
        settings,
        "vllm_url_by_env",
        "staging:http://staging-vllm:8001,production:http://prod-vllm:8001",
    )

    staging = serving_module.get_serving_backend("staging")
    production = serving_module.get_serving_backend("production")

    assert staging.base_url == "http://staging-vllm:8001"
    assert production.base_url == "http://prod-vllm:8001"
    assert staging is not production
