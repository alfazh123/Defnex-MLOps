"""Issue #36: deployment single source of truth, `prod` alias, and race protection.

Covers the acceptance criteria not already exercised by test_deployment_service.py /
test_deployment_api.py: registry status is the single source of truth (the `deployments`
table is history), the partial unique index makes two DEPLOYED versions impossible at the DB
level, alias resolution is a single function that errors instead of propagating None, and
concurrent deploys yield exactly one winner with a clear error for the loser.

Concurrency is exercised at the service layer against a file-backed SQLite with two real
threads/sessions - the FastAPI TestClient is single-threaded and the in-memory per-test engine
cannot run two requests in parallel, so HTTP-level concurrency is not meaningfully testable;
the service layer plus the DB index is where the race is actually decided.
"""

import threading
from datetime import datetime, timezone

import pytest
from sqlalchemy import create_engine, select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from app.db.base import Base
from app.models.deployment import Deployment
from app.models.model import Model, ModelVersion
from app.schemas.promotion import RollbackRequest
from app.services import deployment_service, promotion_service

from tests.conftest import auth_header
from tests.test_deployment_api import _promoted_model_version

CONFLICT_MESSAGE = "another version of this model is already DEPLOYED"


@pytest.fixture(autouse=True)
def _disable_promotion_gates(monkeypatch):
    """These tests exercise deployment mechanics, not the #43 eval gate."""
    from app.config import settings

    for name in (
        "eval_gate_require_eval_set_reference",
        "eval_gate_require_qualitative_majority",
        "eval_gate_require_no_general_regression",
        "eval_gate_require_eval_loss_not_worse",
    ):
        monkeypatch.setattr(settings, name, False)


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def _seed_model(engine, versions, model_id="m1", status="PROMOTED"):
    now = datetime.now(timezone.utc)
    with Session(engine) as db:
        db.add(Model(model_id=model_id))
        db.flush()
        for ver in versions:
            db.add(
                ModelVersion(
                    model_id=model_id,
                    version=ver,
                    status=status,
                    training_run_id=f"tr-{ver}",
                    base_model="base",
                    training_config={},
                    created_at=now,
                )
            )
        db.commit()


def _version(db, version, model_id="m1"):
    return db.scalars(
        select(ModelVersion).where(
            ModelVersion.model_id == model_id, ModelVersion.version == version
        )
    ).one()


def _deployed_versions(db, model_id="m1"):
    return db.scalars(
        select(ModelVersion).where(
            ModelVersion.model_id == model_id, ModelVersion.status == "DEPLOYED"
        )
    ).all()


def _rollback(db, target, rationale="Regression."):
    return promotion_service.rollback(
        db,
        target,
        RollbackRequest(
            rollback_of_version=target.version,
            decided_by="reviewer-1",
            rationale=rationale,
        ),
    )


def _fresh_engine(tmp_path, name, versions):
    engine = create_engine(
        f"sqlite:///{tmp_path / name}", connect_args={"check_same_thread": False}
    )
    Base.metadata.create_all(engine)
    _seed_model(engine, versions)
    return engine


def _run_race(engine, versions):
    """Deploy `versions[0]` and `versions[1]` concurrently, each in its own thread/session/
    engine against the same file-backed SQLite. Returns {name: "OK" | exception}."""
    barrier = threading.Barrier(2)
    results = {}

    def worker(name, version):
        eng = create_engine(engine.url, connect_args={"check_same_thread": False})
        try:
            with Session(eng) as db:
                barrier.wait(timeout=10)
                deployment_service.deploy(db, _version(db, version))
                db.commit()
            results[name] = "OK"
        except Exception as exc:  # noqa: BLE001 - capture any race outcome
            results[name] = exc

    threads = [
        threading.Thread(target=worker, args=(name, version))
        for name, version in (("A", versions[0]), ("B", versions[1]))
    ]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=15)
    return results


# ---------------------------------------------------------------------------
# service layer: single source of truth + alias
# ---------------------------------------------------------------------------


def test_deploy_v2_four_sources_agree(db_session):
    _seed_model(db_session.bind.engine, [1, 2])
    v1 = _version(db_session, 1)
    v2 = _version(db_session, 2)

    deployment_service.deploy(db_session, v1)
    deployment_service.deploy(db_session, v2)
    db_session.commit()

    assert v1.status == "RETIRED"
    assert v2.status == "DEPLOYED"
    assert (
        deployment_service.get_deployment_status(
            db_session, "m1"
        ).current_deployed_version
        == 2
    )
    assert deployment_service.resolve_alias(db_session, "m1", "prod").version == 2
    # deployments rows are append-only history (both recorded as DEPLOYED at their time), not
    # a second claim about the current production version.
    assert [d.status for d in db_session.query(Deployment).all()] == [
        "DEPLOYED",
        "DEPLOYED",
    ]


def test_rollback_four_sources_agree(db_session):
    _seed_model(db_session.bind.engine, [1, 2])
    v1 = _version(db_session, 1)
    v2 = _version(db_session, 2)
    deployment_service.deploy(db_session, v1)
    deployment_service.deploy(db_session, v2)
    db_session.commit()

    _rollback(db_session, v1)
    db_session.commit()

    assert v1.status == "DEPLOYED"
    assert v2.status == "RETIRED"
    assert (
        deployment_service.get_deployment_status(
            db_session, "m1"
        ).current_deployed_version
        == 1
    )
    assert deployment_service.resolve_alias(db_session, "m1", "prod").version == 1
    # history grows on rollback too (the pointer follows the registry status, not history).
    assert db_session.query(Deployment).count() == 3


def test_get_deployment_status_reads_registry_not_latest_history_row(db_session):
    """If the newest deployments row belongs to a version that is no longer DEPLOYED in the
    registry, the answer must come from the registry - never from history."""
    _seed_model(db_session.bind.engine, [1, 2])
    v1 = _version(db_session, 1)
    deployment_service.deploy(db_session, v1)
    db_session.flush()
    v1.status = "RETIRED"
    db_session.commit()

    status = deployment_service.get_deployment_status(db_session, "m1")
    assert status.current_deployed_version is None
    assert status.status is None


def test_db_index_forbids_second_deployed_version():
    """DB-level guarantee: two DEPLOYED rows for one model_id are impossible even when the
    application layer is bypassed entirely."""
    engine = create_engine("sqlite://", connect_args={"check_same_thread": False})
    Base.metadata.create_all(engine)
    _seed_model(engine, [1, 2])
    with Session(engine) as db:
        _version(db, 1).status = "DEPLOYED"
        db.commit()
        _version(db, 2).status = "DEPLOYED"
        with pytest.raises(IntegrityError):
            db.commit()


def test_resolve_alias_unsupported_alias_raises(db_session):
    _seed_model(db_session.bind.engine, [1])
    with pytest.raises(ValueError, match="unknown deployment alias 'staging'"):
        deployment_service.resolve_alias(db_session, "m1", "staging")


def test_resolve_alias_no_deployment_raises_explicit(db_session):
    """Resolving `prod` for a never-deployed model errors explicitly - never a propagating
    None."""
    _seed_model(db_session.bind.engine, [1])
    with pytest.raises(
        ValueError, match="has no deployed version to resolve alias 'prod'"
    ):
        deployment_service.resolve_alias(db_session, "m1", "prod")


# ---------------------------------------------------------------------------
# concurrency (service layer, real threads + file-backed SQLite)
# ---------------------------------------------------------------------------


def test_concurrent_same_version_deploy_keeps_single_deployed(tmp_path):
    engine = _fresh_engine(tmp_path, "same.db", [1])
    results = _run_race(engine, [1, 1])

    assert all(
        v == "OK" for v in results.values()
    )  # redeploying the same version is safe
    with Session(engine) as db:
        assert len(_deployed_versions(db)) == 1


def test_concurrent_different_versions_one_winner(tmp_path):
    engine = _fresh_engine(tmp_path, "diff.db", [1, 2])
    results = _run_race(engine, [1, 2])

    oks = [name for name, outcome in results.items() if outcome == "OK"]
    assert len(oks) == 1  # exactly one winner
    loser = next(outcome for outcome in results.values() if outcome != "OK")
    assert isinstance(loser, ValueError)
    assert str(loser).startswith(
        CONFLICT_MESSAGE
    )  # loser gets a clear error, not fake success

    with Session(engine) as db:
        deployed = _deployed_versions(db)
        assert len(deployed) == 1
        # GET .../deployment (registry-backed) points at that single winner.
        status = deployment_service.get_deployment_status(db, "m1")
        assert status.current_deployed_version == deployed[0].version
        assert deployed[0].version == (1 if oks[0] == "A" else 2)


def test_concurrent_deploy_and_rollback_consistent(tmp_path):
    """A concurrent deploy (v2) and rollback (to v1) must leave registry status and the
    deployments history consistent: exactly one DEPLOYED, and the status resource agrees."""
    engine = _fresh_engine(tmp_path, "rollback.db", [1, 2])
    with Session(engine) as db:
        deployment_service.deploy(db, _version(db, 1))
        db.commit()

    barrier = threading.Barrier(2)
    results = {}

    def do_deploy_v2():
        eng = create_engine(engine.url, connect_args={"check_same_thread": False})
        try:
            with Session(eng) as db:
                barrier.wait(timeout=10)
                deployment_service.deploy(db, _version(db, 2))
                db.commit()
            results["deploy"] = "OK"
        except Exception as exc:  # noqa: BLE001
            results["deploy"] = exc

    def do_rollback_v1():
        eng = create_engine(engine.url, connect_args={"check_same_thread": False})
        try:
            with Session(eng) as db:
                barrier.wait(timeout=10)
                _rollback(db, _version(db, 1), "Concurrent rollback.")
                db.commit()
            results["rollback"] = "OK"
        except Exception as exc:  # noqa: BLE001
            results["rollback"] = exc

    t1 = threading.Thread(target=do_deploy_v2)
    t2 = threading.Thread(target=do_rollback_v1)
    t1.start()
    t2.start()
    t1.join(timeout=15)
    t2.join(timeout=15)

    with Session(engine) as db:
        deployed = _deployed_versions(db)
        assert len(deployed) == 1  # never two DEPLOYED under any interleaving
        status = deployment_service.get_deployment_status(db, "m1")
        assert status.current_deployed_version == deployed[0].version
    assert (
        sum(1 for v in results.values() if v == "OK") == 1
    )  # one winner, one clear error


# ---------------------------------------------------------------------------
# API layer: alias endpoint + consistency after promote/deploy/rollback
# ---------------------------------------------------------------------------

DATASET_CREATE_REQUEST = {
    "source_type": "huggingface",
    "source_dataset": "HuggingFaceH4/no_robots",
    "source_commit_or_snapshot_date": "2026-08-01",
    "source_format": "chatml",
}
TRAINING_RUN_CREATE_REQUEST = {
    "dataset_id": "no_robots",
    "dataset_version": 1,
    "model_id": "qwen-sft-domain-x",
    "base_model": "Qwen/Qwen3.8-27B",
    "training_config": {
        "peft_method": "lora",
        "load_in_4bit": False,
        "lora_r": 16,
        "lora_alpha": 16,
        "learning_rate": None,
        "epochs": 2,
        "max_seq_length": 4096,
    },
    "triggered_by": "user-1",
}


def test_alias_returns_404_when_model_missing(client, admin_token):
    resp = client.get(
        "/api/v1/models/no-such-model/deployment/prod",
        headers=auth_header(admin_token),
    )
    assert resp.status_code == 404
    assert resp.json()["error"]["code"] == "MODEL_NOT_FOUND"


def test_alias_returns_404_when_model_never_deployed(client, admin_token):
    from app.services import model_service, training_service
    from app.workers.mock_runner import MockTrainingRunner

    h = auth_header(admin_token)
    client.post(
        "/api/v1/datasets/no_robots/versions", json=DATASET_CREATE_REQUEST, headers=h
    )
    validate_resp = client.post(
        "/api/v1/datasets/no_robots/versions/1/validate",
        json={
            "records": [
                {
                    "id": "r1",
                    "messages": [
                        {"role": "user", "content": "What is the capital of France?"},
                        {
                            "role": "assistant",
                            "content": " ".join(f"word{i}" for i in range(25)),
                        },
                    ],
                    "metadata": {"source_dataset": "no_robots", "source_id": "r1"},
                }
            ]
        },
        headers=h,
    )
    assert validate_resp.status_code == 201
    assert validate_resp.json()["gate_decision"] == "PASS"
    created = client.post(
        "/api/v1/training-runs", json=TRAINING_RUN_CREATE_REQUEST, headers=h
    ).json()
    with Session(client.engine) as db:
        run = training_service.get_training_run(db, created["training_run_id"])
        training_service.start_training_run(db, run)
        training_service.complete_training_run(
            db, run, artifact_uri=MockTrainingRunner().run(run)
        )
        model_service.register_model_version(db, run)
        db.commit()

    resp = client.get("/api/v1/models/qwen-sft-domain-x/deployment/prod", headers=h)
    assert resp.status_code == 404
    assert resp.json()["error"]["code"] == "DEPLOYMENT_NOT_FOUND"


def test_alias_returns_422_for_unknown_alias(client, admin_token):
    h = auth_header(admin_token)
    model_id, version = _promoted_model_version(client, admin_token)
    client.post(f"/api/v1/models/{model_id}/versions/{version}/deploy", headers=h)

    resp = client.get(f"/api/v1/models/{model_id}/deployment/staging", headers=h)
    assert resp.status_code == 422
    assert resp.json()["error"]["code"] == "UNKNOWN_DEPLOYMENT_ALIAS"


def test_alias_prod_resolves_deployed_version(client, admin_token):
    h = auth_header(admin_token)
    model_id, v1 = _promoted_model_version(client, admin_token)
    client.post(f"/api/v1/models/{model_id}/versions/{v1}/deploy", headers=h)
    _, v2 = _promoted_model_version(client, admin_token)
    client.post(f"/api/v1/models/{model_id}/versions/{v2}/deploy", headers=h)

    resp = client.get(f"/api/v1/models/{model_id}/deployment/prod", headers=h)
    assert resp.status_code == 200
    assert resp.json()["current_deployed_version"] == v2
    assert resp.json()["status"] == "DEPLOYED"


def test_consistency_after_promote_deploy_rollback(client, admin_token):
    """ModelVersion.status of both versions, GET .../deployment, and GET .../deployment/prod
    all agree after promote, supersession deploy, and rollback."""
    h = auth_header(admin_token)
    model_id, v1 = _promoted_model_version(client, admin_token)

    client.post(f"/api/v1/models/{model_id}/versions/{v1}/deploy", headers=h)
    assert (
        client.get(f"/api/v1/models/{model_id}/versions/{v1}", headers=h).json()[
            "status"
        ]
        == "DEPLOYED"
    )
    assert (
        client.get(f"/api/v1/models/{model_id}/deployment", headers=h).json()[
            "current_deployed_version"
        ]
        == v1
    )
    assert (
        client.get(f"/api/v1/models/{model_id}/deployment/prod", headers=h).json()[
            "current_deployed_version"
        ]
        == v1
    )

    _, v2 = _promoted_model_version(client, admin_token)
    client.post(f"/api/v1/models/{model_id}/versions/{v2}/deploy", headers=h)
    assert (
        client.get(f"/api/v1/models/{model_id}/versions/{v1}", headers=h).json()[
            "status"
        ]
        == "RETIRED"
    )
    assert (
        client.get(f"/api/v1/models/{model_id}/versions/{v2}", headers=h).json()[
            "status"
        ]
        == "DEPLOYED"
    )
    assert (
        client.get(f"/api/v1/models/{model_id}/deployment", headers=h).json()[
            "current_deployed_version"
        ]
        == v2
    )
    assert (
        client.get(f"/api/v1/models/{model_id}/deployment/prod", headers=h).json()[
            "current_deployed_version"
        ]
        == v2
    )

    client.post(
        f"/api/v1/models/{model_id}/rollback",
        json={
            "rollback_of_version": v1,
            "decided_by": "reviewer-1",
            "rationale": "Prod regression.",
        },
        headers=h,
    )
    assert (
        client.get(f"/api/v1/models/{model_id}/versions/{v1}", headers=h).json()[
            "status"
        ]
        == "DEPLOYED"
    )
    assert (
        client.get(f"/api/v1/models/{model_id}/versions/{v2}", headers=h).json()[
            "status"
        ]
        == "RETIRED"
    )
    assert (
        client.get(f"/api/v1/models/{model_id}/deployment", headers=h).json()[
            "current_deployed_version"
        ]
        == v1
    )
    assert (
        client.get(f"/api/v1/models/{model_id}/deployment/prod", headers=h).json()[
            "current_deployed_version"
        ]
        == v1
    )
