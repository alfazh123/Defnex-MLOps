"""Issue #65: base-model compatibility at deploy time.

Before the deployment pointer moves, `deployment_service.deploy` verifies that the
artifact's recorded `base_model` matches the base model the serving stack is actually
running (`settings.served_base_model`, the config equivalent of the compose
`VLLM_MODEL_NAME`). A mismatch rejects the deploy with `BaseModelMismatchError`
(mapped to `409 BASE_MODEL_MISMATCH` by the API) — no hot-swap, no pointer move, no
adapter load; the operator recreates the serving stack on the matching base
(PRD §17.4). When `served_base_model` is unset (default) the check is skipped so
unconfigured/legacy deploys behave exactly as before.
"""

import pytest

from app.config import settings
from app.services import deployment_service
from app.services.artifact_storage import ArtifactChecksumError
from app.services.serving import BaseModelMismatchError, MockServingBackend

from tests.conftest import auth_header
from tests.test_deployment_api import _promoted_model_version as _api_promoted_version
from tests.test_deployment_service import (
    _actively_finalized_model_version,
    _promoted_model_version,
)


@pytest.fixture(autouse=True)
def _unset_served_base_model(monkeypatch):
    """Default state: the compatibility check is off (`served_base_model` unset), so the
    pre-#65 deploy behavior is preserved unless a test explicitly configures it."""
    monkeypatch.setattr(settings, "served_base_model", "")


@pytest.fixture(autouse=True)
def _disable_promotion_gates(monkeypatch):
    """These tests exercise the deploy-time base-model guard, not the #43 eval gate.
    `_promoted_model_version` (imported from both test_deployment_service and
    test_deployment_api) doesn't attach an eval-set reference; without disabling the gate
    its PROMOTED decision raises EvalGateBlocked before deploy is ever reached."""
    for name in (
        "eval_gate_require_eval_set_reference",
        "eval_gate_require_qualitative_majority",
        "eval_gate_require_no_general_regression",
        "eval_gate_require_eval_loss_not_worse",
    ):
        monkeypatch.setattr(settings, name, False)


def test_deploy_matches_served_base_model(db_session, monkeypatch):
    """Issue #65 happy path: artifact base_model == settings.served_base_model deploys
    normally (pointer moves, backend loads, version becomes DEPLOYED)."""
    monkeypatch.setattr(settings, "served_base_model", "Qwen/Qwen3.8-27B")
    model_version, _ = _promoted_model_version(db_session)
    backend = MockServingBackend()

    deployment, previous = deployment_service.deploy(
        db_session, model_version, backend=backend
    )

    assert previous is None
    assert deployment.model_id == "qwen-sft-domain-x"
    assert deployment.model_version == model_version.version
    assert model_version.status == "DEPLOYED"
    assert backend.deployed == [("qwen-sft-domain-x", model_version.version)]


def test_deploy_with_unset_served_base_model_still_works(db_session):
    """Backward compatibility: with `served_base_model` empty (default) the check is a
    no-op and a deploy whose base_model differs from nothing still proceeds."""
    model_version, _ = _promoted_model_version(db_session)
    backend = MockServingBackend()

    deployment, previous = deployment_service.deploy(
        db_session, model_version, backend=backend
    )

    assert previous is None
    assert model_version.status == "DEPLOYED"
    assert backend.deployed == [("qwen-sft-domain-x", model_version.version)]


def test_deploy_rejects_base_model_mismatch_before_pointer_moves(
    db_session, monkeypatch
):
    """Issue #65 failure path: base_model mismatch raises BaseModelMismatchError and — like
    checksum/smoke failures — happens before any adapter load or DB mutation."""
    monkeypatch.setattr(settings, "served_base_model", "unsloth/Qwen2.5-7B")
    model_version, _ = _promoted_model_version(db_session)
    backend = MockServingBackend()

    with pytest.raises(
        BaseModelMismatchError, match="does not match the served base model"
    ):
        deployment_service.deploy(db_session, model_version, backend=backend)

    assert model_version.status == "PROMOTED"  # pointer never moved
    assert backend.deployed == []  # adapter never loaded
    assert backend.unloaded == []


def test_deploy_mismatch_applies_to_rollback_path(db_session, monkeypatch):
    """Rollback routes through `deploy`, so the base-model guard protects it too: a RETIRED
    version trained on another base model cannot be hot-swapped back into a mismatched
    serving stack."""
    monkeypatch.setattr(settings, "served_base_model", "Qwen/Qwen3.8-27B")
    v1, _ = _promoted_model_version(db_session)
    deployment_service.deploy(db_session, v1)

    monkeypatch.setattr(settings, "served_base_model", "unsloth/Qwen2.5-7B")

    with pytest.raises(
        BaseModelMismatchError, match="does not match the served base model"
    ):
        deployment_service.deploy(db_session, v1)

    assert v1.status == "DEPLOYED"  # unchanged, still serving (nothing was unloaded)
    assert (
        deployment_service.get_deployment_status(
            db_session, v1.model_id
        ).current_deployed_version
        == v1.version
    )


def test_deploy_mismatch_does_not_disturb_checksum_verification(
    db_session, tmp_path, monkeypatch
):
    """Both pre-deploy guards coexist: the base-model check runs *after* the checksum check,
    so a corrupted artifact still fails with ArtifactChecksumError first even when the base
    model also mismatches."""
    from app.services.artifact_storage import _uri_to_path

    monkeypatch.setattr(settings, "served_base_model", "unsloth/Qwen2.5-7B")
    model_version = _actively_finalized_model_version(db_session, tmp_path)
    (  # corrupt the on-disk payload so BOTH guards would fire
        _uri_to_path(model_version.artifacts[0]["uri"]) / "adapter_model.safetensors"
    ).write_bytes(b"tampered")
    backend = MockServingBackend()

    with pytest.raises(ArtifactChecksumError, match="checksum mismatch"):
        deployment_service.deploy(db_session, model_version, backend=backend)
    assert backend.deployed == []


def test_deploy_api_returns_409_base_model_mismatch(client, admin_token, monkeypatch):
    """The mismatch surfaces at the API as a clear 409 BASE_MODEL_MISMATCH (a client-side
    config conflict the operator resolves by recreating on the matching base, per PRD §17.4)."""
    monkeypatch.setattr(settings, "served_base_model", "unsloth/Qwen2.5-7B")
    h = auth_header(admin_token)
    model_id, version = _api_promoted_version(client, admin_token)

    response = client.post(
        f"/api/v1/models/{model_id}/versions/{version}/deploy", headers=h
    )

    assert response.status_code == 409
    assert response.json()["error"]["code"] == "BASE_MODEL_MISMATCH"
    assert "does not match the served base model" in response.json()["error"]["message"]
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


def test_deploy_api_ok_when_base_model_matches(client, admin_token, monkeypatch):
    """The API happy path with the guard enabled: matching base_model deploys as before."""
    monkeypatch.setattr(settings, "served_base_model", "Qwen/Qwen3.8-27B")
    h = auth_header(admin_token)
    model_id, version = _api_promoted_version(client, admin_token)

    response = client.post(
        f"/api/v1/models/{model_id}/versions/{version}/deploy", headers=h
    )

    assert response.status_code == 200
    assert response.json()["current_deployed_version"] == version
