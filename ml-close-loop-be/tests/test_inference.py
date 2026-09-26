"""OpenAI-compatible inference surface (issues #226, #227, #228, #229).

Covers `POST /v1/chat/completions` and `GET /v1/models`: the standard envelope, `messages[]`
flattening (including the `developer` role), `usage` accounting, model addressing by
`model_id` or `model_id:version`, and every failure being an explicit 4xx rather than a 500.
"""

from datetime import datetime, timezone

import pytest

from app.models.model import Model, ModelVersion
from app.services.inference_service import parse_model_ref, resolve_target
from app.services.serving import GenerationResult, InferenceError

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


def _deployed_model(client, admin_token) -> tuple[str, int]:
    """Promote a model version and actually deploy it, so the `prod` alias resolves to it.

    Promotion alone leaves the version PROMOTED, not DEPLOYED, and the alias -- like an
    explicit version -- only ever serves the DEPLOYED one.
    """

    model_id, version = _promoted_model_version(client, admin_token)
    resp = client.post(
        f"/api/v1/models/{model_id}/versions/{version}/deploy",
        headers=auth_header(admin_token),
    )
    assert resp.status_code == 200, resp.text
    return model_id, version


class _RecordingBackend:
    """Records sampling params and returns a `GenerationResult` with real usage numbers."""

    def __init__(self, text: str = "ok", usage=(7, 11)):
        self.calls: list[dict] = []
        self._text = text
        self._usage = usage

    def deploy(self, model_version) -> None: ...

    def unload(self, model_version) -> None: ...

    def generate(
        self, prompt, model_id, version, *, max_tokens=None, temperature=None
    ) -> GenerationResult:
        self.calls.append(
            {
                "prompt": prompt,
                "max_tokens": max_tokens,
                "temperature": temperature,
            }
        )
        return GenerationResult(
            text=self._text,
            prompt_tokens=self._usage[0],
            completion_tokens=self._usage[1],
        )


def _install(monkeypatch, backend):
    from app.services import serving as serving_module

    monkeypatch.setattr(serving_module, "_backend", backend)
    return backend


# ── model reference parsing ───────────────────────────────────────────────────


class TestParseModelRef:
    def test_bare_name_has_no_version(self):
        assert parse_model_ref("my-model") == ("my-model", None)

    def test_name_colon_version(self):
        assert parse_model_ref("my-model:3") == ("my-model", 3)

    def test_only_the_last_colon_separates(self):
        """A model_id containing a colon must not be truncated at the wrong place."""
        assert parse_model_ref("org/model:v2") == ("org/model:v2", None)
        assert parse_model_ref("org/model:v2:4") == ("org/model:v2", 4)

    def test_non_numeric_suffix_is_part_of_the_name(self):
        assert parse_model_ref("my-model:latest") == ("my-model:latest", None)


# ── service layer: resolve_target still works for explicit versions ───────────


def test_resolve_target_explicit_version(db_session):
    _seed(db_session, "m1", [1])
    db_session.query(ModelVersion).filter_by(
        model_id="m1", version=1
    ).one().status = "DEPLOYED"
    db_session.commit()
    assert resolve_target(db_session, "m1", "1").version == 1


def test_resolve_target_rejects_non_deployed(db_session):
    _seed(db_session, "m1", [1])
    with pytest.raises(ValueError, match="only the DEPLOYED version"):
        resolve_target(db_session, "m1", "1")


# ── the standard envelope ────────────────────────────────────────────────────


class TestChatCompletions:
    def test_response_matches_the_openai_envelope(
        self, client, admin_token, monkeypatch
    ):
        """The whole point of issue #226: an off-the-shelf client parses this unchanged."""
        model_id, version = _deployed_model(client, admin_token)
        _install(monkeypatch, _RecordingBackend("Paris."))

        resp = client.post(
            "/v1/chat/completions",
            json={
                "model": model_id,
                "messages": [{"role": "user", "content": "Capital of France?"}],
            },
            headers=auth_header(admin_token),
        )

        assert resp.status_code == 200, resp.text
        body = resp.json()
        assert body["object"] == "chat.completion"
        assert body["id"].startswith("chatcmpl-")
        assert isinstance(body["created"], int)
        # The concrete reference, not the alias: "which version answered" is the first thing
        # anyone debugging a completion needs.
        assert body["model"] == f"{model_id}:{version}"
        assert len(body["choices"]) == 1
        choice = body["choices"][0]
        assert choice["index"] == 0
        assert choice["message"] == {"role": "assistant", "content": "Paris."}
        assert choice["finish_reason"] == "stop"
        assert set(body["usage"]) == {
            "prompt_tokens",
            "completion_tokens",
            "total_tokens",
        }

    def test_usage_reports_the_backends_numbers(self, client, admin_token, monkeypatch):
        """Issue #228: vLLM already returned usage and this layer discarded it."""
        model_id, _ = _deployed_model(client, admin_token)
        _install(monkeypatch, _RecordingBackend(usage=(13, 29)))

        resp = client.post(
            "/v1/chat/completions",
            json={"model": model_id, "messages": [{"role": "user", "content": "hi"}]},
            headers=auth_header(admin_token),
        )
        assert resp.json()["usage"] == {
            "prompt_tokens": 13,
            "completion_tokens": 29,
            "total_tokens": 42,
        }

    def test_bare_model_name_serves_the_deployed_version(
        self, client, admin_token, monkeypatch
    ):
        model_id, version = _deployed_model(client, admin_token)
        _install(monkeypatch, _RecordingBackend())
        resp = client.post(
            "/v1/chat/completions",
            json={"model": model_id, "messages": [{"role": "user", "content": "hi"}]},
            headers=auth_header(admin_token),
        )
        assert resp.status_code == 200
        assert resp.json()["model"] == f"{model_id}:{version}"

    def test_explicit_version_is_honoured(self, client, admin_token, monkeypatch):
        model_id, version = _deployed_model(client, admin_token)
        _install(monkeypatch, _RecordingBackend())
        resp = client.post(
            "/v1/chat/completions",
            json={
                "model": f"{model_id}:{version}",
                "messages": [{"role": "user", "content": "hi"}],
            },
            headers=auth_header(admin_token),
        )
        assert resp.status_code == 200
        assert resp.json()["model"] == f"{model_id}:{version}"

    def test_sampling_params_reach_the_backend(self, client, admin_token, monkeypatch):
        model_id, _ = _deployed_model(client, admin_token)
        backend = _install(monkeypatch, _RecordingBackend())

        resp = client.post(
            "/v1/chat/completions",
            json={
                "model": model_id,
                "messages": [{"role": "user", "content": "hi"}],
                "max_tokens": 64,
                "temperature": 0.7,
            },
            headers=auth_header(admin_token),
        )
        assert resp.status_code == 200
        assert backend.calls[0]["max_tokens"] == 64
        assert backend.calls[0]["temperature"] == 0.7

    def test_stream_true_is_refused_clearly(self, client, admin_token, monkeypatch):
        """Issue #231 has not landed. Answering a `stream: true` request with a
        non-streaming body would make an SDK fail while parsing SSE, so it is refused."""
        model_id, _ = _deployed_model(client, admin_token)
        _install(monkeypatch, _RecordingBackend())
        resp = client.post(
            "/v1/chat/completions",
            json={
                "model": model_id,
                "messages": [{"role": "user", "content": "hi"}],
                "stream": True,
            },
            headers=auth_header(admin_token),
        )
        assert resp.status_code == 501
        assert resp.json()["error"]["code"] == "STREAMING_NOT_SUPPORTED"

    def test_old_endpoint_is_gone(self, client, admin_token):
        """Issue #226 is the explicitly breaking change: the old shape must not linger
        alongside the new one."""
        model_id, _ = _deployed_model(client, admin_token)
        resp = client.post(
            f"/api/v1/models/{model_id}/inference",
            json={"target": "prod", "prompt": "hi"},
            headers=auth_header(admin_token),
        )
        assert resp.status_code == 404

    def test_requires_auth(self, client):
        resp = client.post(
            "/v1/chat/completions",
            json={"model": "m", "messages": [{"role": "user", "content": "hi"}]},
        )
        assert resp.status_code == 401


# ── messages[] flattening (issue #227) ───────────────────────────────────────


class TestMessageFlattening:
    def _prompt_sent(self, client, admin_token, monkeypatch, messages):
        model_id, _ = _deployed_model(client, admin_token)
        backend = _install(monkeypatch, _RecordingBackend())
        resp = client.post(
            "/v1/chat/completions",
            json={"model": model_id, "messages": messages},
            headers=auth_header(admin_token),
        )
        assert resp.status_code == 200, resp.text
        return backend.calls[0]["prompt"]

    def test_developer_role_is_accepted(self, client, admin_token, monkeypatch):
        """The OpenAI spec's current examples use `developer`; `system` predates it and is
        still supported. Both must work."""
        prompt = self._prompt_sent(
            client,
            admin_token,
            monkeypatch,
            [
                {"role": "developer", "content": "You are terse."},
                {"role": "user", "content": "Hello"},
            ],
        )
        assert "You are terse." in prompt
        assert "User: Hello" in prompt

    def test_system_role_is_still_accepted(self, client, admin_token, monkeypatch):
        prompt = self._prompt_sent(
            client,
            admin_token,
            monkeypatch,
            [
                {"role": "system", "content": "You are helpful."},
                {"role": "user", "content": "Hello"},
            ],
        )
        assert "You are helpful." in prompt

    def test_both_system_and_developer_concatenate(
        self, client, admin_token, monkeypatch
    ):
        prompt = self._prompt_sent(
            client,
            admin_token,
            monkeypatch,
            [
                {"role": "system", "content": "Policy A."},
                {"role": "developer", "content": "Style B."},
                {"role": "user", "content": "Hello"},
            ],
        )
        assert (
            prompt.index("Policy A.") < prompt.index("Style B.") < prompt.index("User:")
        )

    def test_multi_turn_transcript_is_ordered(self, client, admin_token, monkeypatch):
        prompt = self._prompt_sent(
            client,
            admin_token,
            monkeypatch,
            [
                {"role": "user", "content": "first question"},
                {"role": "assistant", "content": "first answer"},
                {"role": "user", "content": "second question"},
            ],
        )
        assert (
            prompt.index("first question")
            < prompt.index("first answer")
            < prompt.index("second question")
        )

    def test_unknown_role_is_rejected_by_the_schema(
        self, client, admin_token, monkeypatch
    ):
        model_id, _ = _deployed_model(client, admin_token)
        _install(monkeypatch, _RecordingBackend())
        resp = client.post(
            "/v1/chat/completions",
            json={
                "model": model_id,
                "messages": [{"role": "wizard", "content": "hi"}],
            },
            headers=auth_header(admin_token),
        )
        assert resp.status_code == 422

    def test_empty_messages_list_is_rejected(self, client, admin_token, monkeypatch):
        model_id, _ = _deployed_model(client, admin_token)
        _install(monkeypatch, _RecordingBackend())
        resp = client.post(
            "/v1/chat/completions",
            json={"model": model_id, "messages": []},
            headers=auth_header(admin_token),
        )
        assert resp.status_code == 422


# ── error paths: explicit codes, never a 500 ──────────────────────────────────


class TestErrors:
    def test_unknown_model_is_404(self, client, admin_token, monkeypatch):
        _install(monkeypatch, _RecordingBackend())
        resp = client.post(
            "/v1/chat/completions",
            json={"model": "nope", "messages": [{"role": "user", "content": "hi"}]},
            headers=auth_header(admin_token),
        )
        assert resp.status_code == 404
        assert resp.json()["error"]["code"] == "MODEL_NOT_FOUND"

    def test_known_model_never_deployed_is_404(self, client, admin_token, monkeypatch):
        _install(monkeypatch, _RecordingBackend())
        from sqlalchemy.orm import Session

        with Session(client.engine) as session:
            session.add(Model(model_id="never-deployed"))
            session.commit()

        resp = client.post(
            "/v1/chat/completions",
            json={
                "model": "never-deployed",
                "messages": [{"role": "user", "content": "hi"}],
            },
            headers=auth_header(admin_token),
        )
        assert resp.status_code == 404
        assert resp.json()["error"]["code"] == "DEPLOYMENT_NOT_FOUND"

    def test_explicit_version_not_deployed_is_409(
        self, client, admin_token, monkeypatch
    ):
        """The version exists but is not the loaded one -- a different failure from "no such
        version", and the code says which."""
        model_id, _ = _deployed_model(client, admin_token)
        _install(monkeypatch, _RecordingBackend())
        from sqlalchemy.orm import Session

        from app.models.model import ModelVersion as MV

        with Session(client.engine) as session:
            deployed = (
                session.query(MV)
                .filter_by(model_id=model_id)
                .order_by(MV.version.desc())
                .first()
            )
            other_version = deployed.version + 99
            session.add(
                MV(
                    model_id=model_id,
                    version=other_version,
                    status="REGISTERED",
                    training_run_id="tr-x",
                    base_model="base",
                    training_config={},
                    created_at=datetime.now(timezone.utc),
                )
            )
            session.commit()

        resp = client.post(
            "/v1/chat/completions",
            json={
                "model": f"{model_id}:{other_version}",
                "messages": [{"role": "user", "content": "hi"}],
            },
            headers=auth_header(admin_token),
        )
        assert resp.status_code == 409
        assert resp.json()["error"]["code"] == "INFERENCE_NOT_ALLOWED"

    def test_missing_explicit_version_is_404(self, client, admin_token, monkeypatch):
        model_id, _ = _deployed_model(client, admin_token)
        _install(monkeypatch, _RecordingBackend())
        resp = client.post(
            "/v1/chat/completions",
            json={
                "model": f"{model_id}:4242",
                "messages": [{"role": "user", "content": "hi"}],
            },
            headers=auth_header(admin_token),
        )
        assert resp.status_code == 404
        assert resp.json()["error"]["code"] == "MODEL_NOT_FOUND"

    def test_out_of_range_temperature_is_rejected(
        self, client, admin_token, monkeypatch
    ):
        model_id, _ = _deployed_model(client, admin_token)
        _install(monkeypatch, _RecordingBackend())
        resp = client.post(
            "/v1/chat/completions",
            json={
                "model": model_id,
                "messages": [{"role": "user", "content": "hi"}],
                "temperature": 5.0,
            },
            headers=auth_header(admin_token),
        )
        assert resp.status_code == 422

    def test_upstream_failure_is_502_not_500(self, client, admin_token, monkeypatch):
        class _FailingBackend(_RecordingBackend):
            def generate(self, prompt, model_id, version, **kwargs):
                raise InferenceError("vLLM unreachable")

        model_id, _ = _deployed_model(client, admin_token)
        _install(monkeypatch, _FailingBackend())
        resp = client.post(
            "/v1/chat/completions",
            json={"model": model_id, "messages": [{"role": "user", "content": "hi"}]},
            headers=auth_header(admin_token),
        )
        assert resp.status_code == 502
        assert resp.json()["error"]["code"] == "INFERENCE_FAILED"


# ── GET /v1/models (issue #229) ──────────────────────────────────────────────


class TestListModels:
    def test_lists_only_deployed_versions(self, client, admin_token):
        """A client picking a `model` value should not be offered a version that is
        guaranteed to 409."""
        model_id, version = _deployed_model(client, admin_token)
        resp = client.get("/v1/models", headers=auth_header(admin_token))
        assert resp.status_code == 200
        body = resp.json()
        assert body["object"] == "list"
        ids = [m["id"] for m in body["data"]]
        assert f"{model_id}:{version}" in ids
        assert all(m["object"] == "model" for m in body["data"])
        assert all(m["owned_by"] == "defnex" for m in body["data"])
        assert all(isinstance(m["created"], int) for m in body["data"])

    def test_listed_ids_are_directly_usable(self, client, admin_token, monkeypatch):
        """The `id` a client copies out of the list must work verbatim in `model`."""
        _deployed_model(client, admin_token)
        _install(monkeypatch, _RecordingBackend())
        listed = client.get("/v1/models", headers=auth_header(admin_token)).json()[
            "data"
        ]
        assert listed
        resp = client.post(
            "/v1/chat/completions",
            json={
                "model": listed[0]["id"],
                "messages": [{"role": "user", "content": "hi"}],
            },
            headers=auth_header(admin_token),
        )
        assert resp.status_code == 200, resp.text

    def test_empty_list_is_valid(self, client, admin_token):
        resp = client.get("/v1/models", headers=auth_header(admin_token))
        assert resp.status_code == 200
        assert resp.json() == {"object": "list", "data": []}

    def test_requires_auth(self, client):
        assert client.get("/v1/models").status_code == 401
