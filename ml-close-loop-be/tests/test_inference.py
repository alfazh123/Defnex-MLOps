"""Inference endpoint + service (issue #41): `POST /api/v1/models/{model_id}/inference` with a
`target` of `prod` (alias) or an explicit version number, returning the concrete version that
served and the generation.

Alias resolution must delegate to the *single* resolution function (deployment_service.resolve_alias,
issue #36) - the endpoint/service contain no second alias-resolution logic. Explicit versions must
be the currently DEPLOYED one (the only adapter guaranteed loaded and smoke-tested). Every failure
case is an explicit error code, never a 500.
"""

from datetime import datetime, timezone

import pytest
from sqlalchemy import select

from app.models.model import Model, ModelVersion
from app.services.inference_service import resolve_target
from app.services.serving import InferenceError

from tests.conftest import auth_header
from tests.test_deployment_api import _promoted_model_version


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


# ---------------------------------------------------------------------------
# resolve_target (service layer)
# ---------------------------------------------------------------------------


def _seed(db, model_id: str, versions: list[int]) -> None:
    now = datetime.now(timezone.utc)
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
                created_at=now,
            )
        )
    db.commit()


def _v(db, model_id: str, version: int) -> ModelVersion:
    return db.scalars(
        select(ModelVersion).where(
            ModelVersion.model_id == model_id, ModelVersion.version == version
        )
    ).one()


def test_resolve_target_alias_and_explicit_version_both_point_at_deployed(db_session):
    _seed(db_session, "m-inf", [1, 2])
    v2 = _v(db_session, "m-inf", 2)
    v2.status = "DEPLOYED"
    db_session.commit()

    assert resolve_target(db_session, "m-inf", "prod") is v2
    assert resolve_target(db_session, "m-inf", "2") is v2


def test_resolve_target_explicit_version_must_be_deployed(db_session):
    _seed(db_session, "m-inf", [1, 2])
    _v(db_session, "m-inf", 1).status = "DEPLOYED"
    db_session.commit()

    with pytest.raises(ValueError, match="version 2 is PROMOTED"):
        resolve_target(db_session, "m-inf", "2")


def test_resolve_target_failure_cases_are_explicit(db_session):
    _seed(db_session, "m-inf", [1])

    with pytest.raises(ValueError, match="must be a deployment alias"):
        resolve_target(db_session, "m-inf", "staging")
    with pytest.raises(ValueError, match="no deployed version"):
        resolve_target(db_session, "m-inf", "prod")
    with pytest.raises(ValueError, match="version 99 not found"):
        resolve_target(db_session, "m-inf", "99")
    with pytest.raises(ValueError, match="must be a deployment alias"):
        resolve_target(db_session, "m-inf", "latest")


# ---------------------------------------------------------------------------
# inference endpoint (API layer)
# ---------------------------------------------------------------------------


def _deployed_model(client, admin_token) -> tuple[str, int]:
    model_id, version = _promoted_model_version(client, admin_token)
    h = auth_header(admin_token)
    resp = client.post(
        f"/api/v1/models/{model_id}/versions/{version}/deploy", headers=h
    )
    assert resp.status_code == 200, resp.text
    return model_id, version


def test_inference_prod_alias_and_explicit_version(client, admin_token):
    model_id, version = _deployed_model(client, admin_token)
    h = auth_header(admin_token)

    for target in ("prod", str(version)):
        resp = client.post(
            f"/api/v1/models/{model_id}/inference",
            json={"target": target, "prompt": "What is the capital of France?"},
            headers=h,
        )
        assert resp.status_code == 200, resp.text
        assert resp.json() == {
            "model_id": model_id,
            "version": version,
            "generation": f"mock generation for {model_id}-v{version}",
        }


def test_inference_409_when_explicit_version_is_not_deployed(client, admin_token):
    model_id, version = _promoted_model_version(
        client, admin_token
    )  # PROMOTED, not deployed
    h = auth_header(admin_token)

    resp = client.post(
        f"/api/v1/models/{model_id}/inference",
        json={"target": str(version), "prompt": "hi"},
        headers=h,
    )

    assert resp.status_code == 409
    assert resp.json()["error"]["code"] == "INFERENCE_NOT_ALLOWED"


def test_inference_404_unknown_alias(client, admin_token):
    model_id, _ = _deployed_model(client, admin_token)
    h = auth_header(admin_token)

    resp = client.post(
        f"/api/v1/models/{model_id}/inference",
        json={"target": "staging", "prompt": "hi"},
        headers=h,
    )

    assert resp.status_code == 404
    assert resp.json()["error"]["code"] == "UNKNOWN_DEPLOYMENT_ALIAS"


def test_inference_404_known_alias_but_never_deployed(client, admin_token):
    model_id, _ = _promoted_model_version(client, admin_token)
    h = auth_header(admin_token)

    resp = client.post(
        f"/api/v1/models/{model_id}/inference",
        json={"target": "prod", "prompt": "hi"},
        headers=h,
    )

    assert resp.status_code == 404
    assert resp.json()["error"]["code"] == "DEPLOYMENT_NOT_FOUND"


def test_inference_404_unknown_model_and_missing_version(client, admin_token):
    h = auth_header(admin_token)
    model_id, _ = _deployed_model(client, admin_token)

    resp = client.post(
        "/api/v1/models/ghost/inference",
        json={"target": "prod", "prompt": "hi"},
        headers=h,
    )
    assert resp.status_code == 404
    assert resp.json()["error"]["code"] == "MODEL_NOT_FOUND"

    resp = client.post(
        f"/api/v1/models/{model_id}/inference",
        json={"target": "99", "prompt": "hi"},
        headers=h,
    )
    assert resp.status_code == 404
    assert resp.json()["error"]["code"] == "MODEL_NOT_FOUND"


def test_inference_502_upstream_failure(client, admin_token, monkeypatch):
    from app.services import serving as serving_module

    model_id, _ = _deployed_model(client, admin_token)

    class _FailingBackend:
        def deploy(self, model_version) -> None: ...

        def unload(self, model_version) -> None: ...

        def generate(self, prompt: str, model_id: str, version: int) -> str:
            raise InferenceError("vLLM down")

    monkeypatch.setattr(serving_module, "_backend", _FailingBackend())

    resp = client.post(
        f"/api/v1/models/{model_id}/inference",
        json={"target": "prod", "prompt": "hi"},
        headers=auth_header(admin_token),
    )

    assert resp.status_code == 502
    assert resp.json()["error"]["code"] == "INFERENCE_FAILED"


def test_inference_requires_auth_and_prompt(client, admin_token):
    model_id, _ = _deployed_model(client, admin_token)

    resp = client.post(
        f"/api/v1/models/{model_id}/inference",
        json={"target": "prod", "prompt": "hi"},
    )
    assert resp.status_code == 401

    h = auth_header(admin_token)
    resp = client.post(
        f"/api/v1/models/{model_id}/inference", json={"target": "prod"}, headers=h
    )
    assert resp.status_code == 422

    resp = client.post(
        f"/api/v1/models/{model_id}/inference",
        json={"target": "prod", "prompt": ""},
        headers=h,
    )
    assert resp.status_code == 422
