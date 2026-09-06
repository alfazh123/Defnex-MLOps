"""Issue #40: real vLLM serving backend.

Covers the acceptance criteria not already exercised elsewhere: `VLLMServingBackend` talks to
the vLLM runtime API (`load_lora_adapter` / `unload_lora_adapter`) through a real `httpx.Client`
with a `httpx.MockTransport` handler (so request building - URL, headers, JSON payload - runs
against real httpx code, only the wire is mocked); `deployment_service.deploy` loads before any
DB mutation (a failed load leaves the registry untouched), unloads the superseded version
best-effort, and cleans up its own adapter when it loses the concurrent deploy race; and the
process-wide backend singleton is created once. The API layer maps `ServingError` to
`502 DEPLOY_FAILED` with the promise that the pointer did not move.
"""

import json
import threading
from datetime import datetime, timezone

import httpx
import pytest
from sqlalchemy import create_engine, select
from sqlalchemy.orm import Session

from app.db.base import Base
from app.models.deployment import Deployment
from app.models.model import Model, ModelVersion
from app.schemas.promotion import RollbackRequest
from app.services import deployment_service, promotion_service
from app.services.serving import (
    MockServingBackend,
    ServingError,
    VLLMServingBackend,
    _adapter_path,
    get_serving_backend,
)

from tests.conftest import auth_header
from tests.test_deployment_api import _promoted_model_version

CONFLICT_MESSAGE = "another version of this model is already DEPLOYED"


@pytest.fixture(autouse=True)
def _no_sleep(monkeypatch):
    """Run the sync retry backoff without actually sleeping (retry policy itself is asserted
    via the call count)."""
    monkeypatch.setattr("app.services.http_retry.time.sleep", lambda _seconds: None)


def _client_for(handler) -> httpx.Client:
    return httpx.Client(transport=httpx.MockTransport(handler))


def _seed(db: Session, model_id: str, versions: list[int]) -> None:
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
                artifacts=[
                    {
                        "type": "adapter",
                        "uri": f"file:///data/adapters/{model_id}-v{ver}",
                    }
                ],
                created_at=now,
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
# VLLMServingBackend wire format
# ---------------------------------------------------------------------------


def test_load_payload_reaches_vllm_and_pointer_moves(db_session):
    _seed(db_session, "m-wire", [1])
    v1 = _v(db_session, "m-wire", 1)
    requests = []

    def handler(request):
        requests.append(request)
        return httpx.Response(200, json={"message": "Adapter loaded"})

    backend = VLLMServingBackend(
        base_url="http://vllm:8000",
        api_key="secret-key",
        client=_client_for(handler),
    )

    deployment_service.deploy(db_session, v1, backend=backend)
    db_session.commit()

    assert len(requests) == 1
    assert requests[0].url == "http://vllm:8000/v1/load_lora_adapter"
    assert requests[0].headers["Authorization"] == "Bearer secret-key"
    assert json.loads(requests[0].content) == {
        "lora_name": "m-wire-v1",
        "lora_path": "/data/adapters/m-wire-v1",
    }
    assert v1.status == "DEPLOYED"
    assert db_session.query(Deployment).count() == 1


def test_adapter_path_file_uri_and_missing_adapter():
    v = ModelVersion(
        model_id="m",
        version=1,
        artifacts=[{"type": "adapter", "uri": "file:///data/adapters/m-v1"}],
    )
    assert _adapter_path(v) == "/data/adapters/m-v1"

    naive = ModelVersion(
        model_id="m", version=1, artifacts=[{"type": "adapter", "uri": "org/repo"}]
    )
    assert _adapter_path(naive) == "org/repo"

    flaky = ModelVersion(model_id="m", version=1, artifacts=[{"type": "merged"}])
    with pytest.raises(ServingError, match="no 'adapter' artifact"):
        _adapter_path(flaky)


def test_deploy_json_uses_verbatim_non_file_uri(db_session):
    now = datetime.now(timezone.utc)
    db_session.add(Model(model_id="m-hf"))
    db_session.flush()
    db_session.add(
        ModelVersion(
            model_id="m-hf",
            version=1,
            status="PROMOTED",
            training_run_id="tr-1",
            base_model="base",
            training_config={},
            artifacts=[{"type": "adapter", "uri": "my-org/my-adapter"}],
            created_at=now,
        )
    )
    db_session.commit()
    v1 = _v(db_session, "m-hf", 1)

    def handler(request):
        return httpx.Response(200, json={"message": "ok"})

    backend = VLLMServingBackend(client=_client_for(handler))
    deployment_service.deploy(db_session, v1, backend=backend)
    db_session.commit()
    assert v1.status == "DEPLOYED"


# ---------------------------------------------------------------------------
# load failures leave the DB untouched
# ---------------------------------------------------------------------------


def test_load_failure_raises_serving_error_and_db_stays_unchanged(db_session):
    _seed(db_session, "m-fail", [1])
    v1 = _v(db_session, "m-fail", 1)
    calls = {"n": 0}

    def handler(request):
        calls["n"] += 1
        return httpx.Response(500, json={"error": "OOM"})

    backend = VLLMServingBackend(client=_client_for(handler))

    with pytest.raises(ServingError, match="vLLM rejected adapter load"):
        deployment_service.deploy(db_session, v1, backend=backend)

    assert calls["n"] == 3  # retries exhausted (sleep disabled by fixture)
    assert v1.status == "PROMOTED"
    assert db_session.query(Deployment).count() == 0
    assert deployment_service.get_deployment_status(db_session, "m-fail").status is None


def test_connection_error_retries_then_raises_serving_error(db_session):
    _seed(db_session, "m-conn", [1])
    v1 = _v(db_session, "m-conn", 1)
    calls = {"n": 0}

    def handler(request):
        calls["n"] += 1
        raise httpx.ConnectError("vllm down")

    backend = VLLMServingBackend(client=_client_for(handler))

    with pytest.raises(ServingError, match="vLLM unreachable"):
        deployment_service.deploy(db_session, v1, backend=backend)

    assert calls["n"] == 3
    assert v1.status == "PROMOTED"


# ---------------------------------------------------------------------------
# unload is idempotent / best-effort
# ---------------------------------------------------------------------------


def test_unload_missing_adapter_404_does_not_raise():
    v = ModelVersion(model_id="m-u", version=2)

    def handler(request):
        assert request.url == "http://localhost:8001/v1/unload_lora_adapter"
        assert json.loads(request.content) == {"lora_name": "m-u-v2"}
        return httpx.Response(404, json={"message": "Requested LoRA not found"})

    backend = VLLMServingBackend(client=_client_for(handler))
    backend.unload(v)  # must not raise


def test_unload_server_error_is_swallowed():
    v = ModelVersion(model_id="m-u2", version=1)

    def handler(request):
        return httpx.Response(500, json={"error": "boom"})

    backend = VLLMServingBackend(client=_client_for(handler))
    backend.unload(v)


# ---------------------------------------------------------------------------
# supersession / rollback order (load new, then unload superseded)
# ---------------------------------------------------------------------------


def test_supersede_loads_new_then_unloads_previous(db_session):
    _seed(db_session, "m-sup", [1, 2])
    v1 = _v(db_session, "m-sup", 1)
    v2 = _v(db_session, "m-sup", 2)
    backend = MockServingBackend()

    deployment_service.deploy(db_session, v1, backend=backend)
    db_session.commit()
    deployment_service.deploy(db_session, v2, backend=backend)
    db_session.commit()

    # asserts on the interleaved event log, not on per-type lists: a regression that swapped
    # the order (unload v1 before loading v2) would keep `deployed`/`unloaded` identical but
    # change the event order.
    assert backend.events == [
        ("load", ("m-sup", 1)),
        ("load", ("m-sup", 2)),
        ("unload", ("m-sup", 1)),
    ]
    assert v1.status == "RETIRED"
    assert v2.status == "DEPLOYED"


def test_rollback_unloads_the_superseded_version(db_session, monkeypatch):
    from app.services import serving as serving_module

    _seed(db_session, "m-rb", [1, 2])
    v1 = _v(db_session, "m-rb", 1)
    v2 = _v(db_session, "m-rb", 2)
    deployment_service.deploy(db_session, v1)
    deployment_service.deploy(db_session, v2)
    db_session.commit()

    backend = MockServingBackend()
    monkeypatch.setattr(serving_module, "_backend", backend)

    promotion_service.rollback(
        db_session,
        v1,
        RollbackRequest(
            rollback_of_version=v1.version,
            decided_by="reviewer-1",
            rationale="Regression.",
        ),
    )
    db_session.commit()

    # the rollback is a deploy of v1 whose previous version (v2) is unloaded - v1 must be
    # loaded before v2 is unloaded (asserted via the interleaved event log)
    assert backend.events == [
        ("load", ("m-rb", 1)),
        ("unload", ("m-rb", 2)),
    ]
    assert v1.status == "DEPLOYED"
    assert v2.status == "RETIRED"


# ---------------------------------------------------------------------------
# process-wide singleton
# ---------------------------------------------------------------------------


def test_get_serving_backend_is_a_singleton_and_defaults_to_mock(monkeypatch):
    from app.services import serving as serving_module

    monkeypatch.setattr(serving_module, "_backend", None)
    first = get_serving_backend()
    second = get_serving_backend()
    assert first is second
    assert isinstance(first, MockServingBackend)


# ---------------------------------------------------------------------------
# concurrent deploy race: only the winner's adapter stays resident in vLLM
# ---------------------------------------------------------------------------


def test_concurrent_deploy_race_leaves_only_winner_adapter_loaded(tmp_path):
    engine = create_engine(
        f"sqlite:///{tmp_path / 'race.db'}", connect_args={"check_same_thread": False}
    )
    Base.metadata.create_all(engine)
    with Session(engine) as db:
        _seed(db, "m-race", [1, 2])

    events = []
    lock = threading.Lock()

    def handler(request):
        body = json.loads(request.content)
        event = "unload" if request.url.path.endswith("unload_lora_adapter") else "load"
        with lock:
            events.append((event, body["lora_name"]))
        return httpx.Response(200, json={"message": "ok"})

    backend = VLLMServingBackend(client=_client_for(handler))

    # Deterministic interleaving: thread B reads "no previous deployment" and parks inside
    # backend.deploy() before its conflict flush; thread A then deploys and commits v1; only after
    # that is B released, so B's flush of DEPLOYED v2 is guaranteed to hit the partial unique
    # index and lose. A spontaneous barrier loses the race only sometimes, and worse, right after
    # a failed flush SQLAlchemy drops the loser into PendingRollbackError territory - both make the
    # assertion above timing-fragile.
    parked = threading.Event()
    released = threading.Event()
    orig_deploy = backend.deploy

    def paused_deploy(model_version):
        if model_version.version == 2:
            parked.set()
            released.wait(timeout=10)
        return orig_deploy(model_version)

    backend.deploy = paused_deploy
    results = {}

    def worker(name, version):
        eng = create_engine(engine.url, connect_args={"check_same_thread": False})
        try:
            with Session(eng) as db:
                deployment_service.deploy(
                    db, _v(db, "m-race", version), backend=backend
                )
                db.commit()
            results[name] = "OK"
        except Exception as exc:  # noqa: BLE001 - capture any race outcome
            results[name] = exc

    loser_thread = threading.Thread(target=worker, args=("B", 2))
    loser_thread.start()
    assert parked.wait(timeout=10)

    winner_thread = threading.Thread(target=worker, args=("A", 1))
    winner_thread.start()
    winner_thread.join(timeout=15)
    assert results["A"] == "OK"

    released.set()
    loser_thread.join(timeout=15)

    assert isinstance(results["B"], ValueError)
    assert str(results["B"]).startswith(CONFLICT_MESSAGE)

    with Session(engine) as db:
        deployed = db.scalars(
            select(ModelVersion).where(
                ModelVersion.model_id == "m-race",
                ModelVersion.status == "DEPLOYED",
            )
        ).all()
        assert len(deployed) == 1
        winner_version = deployed[0].version

    # exactly one adapter stays loaded: the winner's. The loser loaded its own then unloaded it
    # again in the race-cleanup, so its net residency is zero.
    loads = [name for kind, name in events if kind == "load"]
    unloads = [name for kind, name in events if kind == "unload"]
    assert [n for n in loads if n not in unloads] == [f"m-race-v{winner_version}"]
    assert unloads == [n for n in loads if n != f"m-race-v{winner_version}"]
    assert len(unloads) == 1


# ---------------------------------------------------------------------------
# API layer: ServingError surfaces as 502 DEPLOY_FAILED, pointer unchanged
# ---------------------------------------------------------------------------


def test_deploy_returns_502_when_vllm_rejects_load(client, admin_token, monkeypatch):
    from app.services import serving as serving_module

    calls = {"n": 0}

    def handler(request):
        calls["n"] += 1
        return httpx.Response(500, json={"error": "OOM"})

    vllm_backend = VLLMServingBackend(client=_client_for(handler))
    monkeypatch.setattr(serving_module, "_backend", vllm_backend)

    h = auth_header(admin_token)
    model_id, version = _promoted_model_version(client, admin_token)
    resp = client.post(
        f"/api/v1/models/{model_id}/versions/{version}/deploy", headers=h
    )

    assert resp.status_code == 502
    assert resp.json()["error"]["code"] == "DEPLOY_FAILED"
    assert calls["n"] == 3
    # the load runs before any DB mutation, so the registry/pointer did not move
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
