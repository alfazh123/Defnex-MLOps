"""Issue #124 — durable (Postgres-backed) idempotency-key cache tests.

Follow-up to #81. The old in-process `dict[str, tuple[dict, float]]` caches in
`deployment_service.py` / `intake_validate.py` are gone; both now go through
`app.services.idempotency_service`, which reads/writes the `idempotency_keys` table. These tests
therefore exercise `idempotency_service` directly (unit-level: get/store/expiry/cleanup, plus the
"two worker processes" and "restart" scenarios the issue calls out), and
`deployment_service.check_idempotency`/`store_idempotency` as the thin per-model-version wrapper
around it. HTTP-level replay coverage for the two endpoints lives in
`test_deployment_api.py::test_deploy_replays_cached_result_for_same_idempotency_key` and
`test_intake_validate.py::test_idempotency_key_returns_cached` /
`test_idempotency_key_replay_does_not_double_create_dataset_version`.
"""

from datetime import datetime, timedelta, timezone

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.models.idempotency import IdempotencyKey
from app.services import deployment_service, idempotency_service


def _utc_now() -> datetime:
    """Naive UTC "now", matching `idempotency_service._utc_now` (see its docstring for why
    naive-but-always-UTC: SQLite round-trips a tz-aware `DateTime` value unreliably). The test
    machine's local time is not UTC (WIB, UTC+7) - using bare `datetime.now()` here would make
    the "expired" fixtures below wrong by 7 hours on this machine."""
    return datetime.now(timezone.utc).replace(tzinfo=None)


class TestIdempotencyServiceBasics:
    def test_get_cached_response_returns_none_when_no_key(self, db_session):
        assert idempotency_service.get_cached_response(db_session, None) is None

    def test_get_cached_response_returns_none_when_not_cached(self, db_session):
        assert (
            idempotency_service.get_cached_response(db_session, "unknown-key") is None
        )

    def test_store_then_get_returns_cached_response(self, db_session):
        body = {"model_id": "m1", "current_deployed_version": 1}
        idempotency_service.store_response(
            db_session, "key-1", endpoint="deploy", status=200, body=body
        )
        cached = idempotency_service.get_cached_response(db_session, "key-1")
        assert cached is not None
        assert cached.status == 200
        assert cached.body == body

    def test_store_response_none_key_is_noop(self, db_session):
        idempotency_service.store_response(
            db_session, None, endpoint="deploy", status=200, body={}
        )
        rows = db_session.scalars(select(IdempotencyKey)).all()
        assert rows == []


class TestIdempotencyExpiry:
    def test_expired_row_returns_none_and_is_deleted(self, db_session):
        now = _utc_now()
        db_session.add(
            IdempotencyKey(
                key="expired-key",
                endpoint="deploy",
                response_status=200,
                response_body_json="{}",
                created_at=now - timedelta(hours=2),
                expires_at=now - timedelta(hours=1),  # 1h past TTL
            )
        )
        db_session.commit()

        assert (
            idempotency_service.get_cached_response(db_session, "expired-key") is None
        )
        db_session.commit()  # the lookup only flushes the delete; commit it to assert durably

        remaining = db_session.scalars(
            select(IdempotencyKey).where(IdempotencyKey.key == "expired-key")
        ).all()
        assert remaining == []

    def test_expired_row_can_be_replaced_by_a_fresh_store(self, db_session):
        """A caller re-using the same key after expiry (get -> miss -> store) must not collide
        with the dead row's primary key."""
        now = _utc_now()
        db_session.add(
            IdempotencyKey(
                key="reused-key",
                endpoint="deploy",
                response_status=200,
                response_body_json="{}",
                created_at=now - timedelta(hours=2),
                expires_at=now - timedelta(hours=1),
            )
        )
        db_session.commit()

        assert idempotency_service.get_cached_response(db_session, "reused-key") is None
        idempotency_service.store_response(
            db_session,
            "reused-key",
            endpoint="deploy",
            status=200,
            body={"fresh": True},
        )
        db_session.commit()

        cached = idempotency_service.get_cached_response(db_session, "reused-key")
        assert cached.body == {"fresh": True}


class TestCleanupExpired:
    def test_cleanup_expired_deletes_only_expired_rows(self, db_session):
        now = _utc_now()
        db_session.add(
            IdempotencyKey(
                key="dead",
                endpoint="deploy",
                response_status=200,
                response_body_json="{}",
                created_at=now - timedelta(hours=2),
                expires_at=now - timedelta(minutes=1),
            )
        )
        db_session.add(
            IdempotencyKey(
                key="alive",
                endpoint="deploy",
                response_status=200,
                response_body_json="{}",
                created_at=now,
                expires_at=now + timedelta(hours=1),
            )
        )
        db_session.commit()

        deleted = idempotency_service.cleanup_expired(db_session)
        db_session.commit()

        assert deleted == 1
        remaining_keys = {
            row.key for row in db_session.scalars(select(IdempotencyKey)).all()
        }
        assert remaining_keys == {"alive"}

    def test_cleanup_expired_returns_zero_when_nothing_expired(self, db_session):
        assert idempotency_service.cleanup_expired(db_session) == 0


class TestRestartAndMultiWorkerDurability:
    """AC: 'restarting the API process must not lose the idempotency guarantee' and 'sending the
    same request twice from two separate DB connections (simulating two worker processes) yields
    the same result and does not double-create the job.'

    Caveat (documented per the batch instructions): `client.engine` uses SQLAlchemy's
    `StaticPool`, which hands out the *same* underlying SQLite connection to every `Session`
    opened against it - unlike real Postgres, where two worker processes hold genuinely separate
    connections and can each start an independent, concurrently-uncommitted transaction. SQLite
    also serializes writers globally (a second writer blocks/raises rather than truly
    interleaving). So this cannot reproduce a real two-Postgres-connection race with both sides
    mid-transaction at once; what it *does* faithfully exercise is the actual correctness
    guarantee - the `key` primary key / unique constraint - by having two independent `Session`
    objects (no shared Python state, which is the point: nothing here is a process-local dict)
    race to insert the same key, sequentially committing. The second commit must fail on the PK
    and be handled as "already exists", not silently double-write.
    """

    def test_no_module_level_cache_holds_state(self):
        """The old bug: an in-process dict survives only as long as the Python process. Assert
        the replacement isn't one either - there is no module-level mutable cache to reset."""
        assert not hasattr(idempotency_service, "_DEPLOY_IDEMPOTENCY_CACHE")
        assert not any(
            isinstance(v, dict) and v
            for k, v in vars(idempotency_service).items()
            if not k.startswith("__")
        )

    def test_response_survives_a_brand_new_session_object(self, client):
        """Simulates "the process restarted": nothing is read back from the `Session` that wrote
        it - a completely new `Session`/connection-handle is opened afterward, mirroring how a
        freshly-started worker process would look up the key with no memory of the earlier one."""
        with Session(client.engine) as writer:
            idempotency_service.store_response(
                writer, "restart-key", endpoint="deploy", status=200, body={"v": 1}
            )
            writer.commit()

        with Session(client.engine) as reader:
            cached = idempotency_service.get_cached_response(reader, "restart-key")
        assert cached is not None
        assert cached.body == {"v": 1}

    def test_two_worker_connections_race_on_same_key_second_does_not_double_write(
        self, client
    ):
        """Two separate `Session`s (two "workers") both try to store the *same* key after
        computing the *same* result independently (the realistic race: both received the retried
        request before either had written its cache row yet). Only one row may end up
        persisted, and it must not be a silent double-create."""
        from sqlalchemy.exc import IntegrityError

        worker_a = Session(client.engine)
        worker_b = Session(client.engine)
        try:
            idempotency_service.store_response(
                worker_a, "race-key", endpoint="deploy", status=200, body={"job": "A"}
            )
            worker_a.commit()

            # Worker B didn't see worker A's write (it computed its own result before
            # checking), and now tries to store under the same key.
            raised = False
            try:
                idempotency_service.store_response(
                    worker_b,
                    "race-key",
                    endpoint="deploy",
                    status=200,
                    body={"job": "B"},
                )
                worker_b.commit()
            except IntegrityError:
                raised = True
                worker_b.rollback()
            assert raised, (
                "the unique primary key on idempotency_keys.key must reject the second "
                "worker's insert instead of silently creating a duplicate job"
            )

            # Exactly one row exists, and it's worker A's - the first writer's result is what
            # every caller replays, never a second/duplicate job.
            with Session(client.engine) as verifier:
                rows = verifier.scalars(
                    select(IdempotencyKey).where(IdempotencyKey.key == "race-key")
                ).all()
                assert len(rows) == 1
                cached = idempotency_service.get_cached_response(verifier, "race-key")
                assert cached.body == {"job": "A"}
        finally:
            worker_a.close()
            worker_b.close()


class TestDeploymentServiceIdempotencyWrapper:
    """`deployment_service.check_idempotency`/`store_idempotency` namespace the caller's key by
    `model_version_id` (unchanged contract from #81) and now delegate to `idempotency_service`."""

    def test_check_idempotency_returns_none_when_no_key(self, db_session):
        assert deployment_service.check_idempotency(db_session, None, 1) is None

    def test_check_idempotency_returns_none_when_not_cached(self, db_session):
        assert deployment_service.check_idempotency(db_session, "key-abc", 999) is None

    def test_store_and_check_idempotency_returns_cached_result(self, db_session):
        result = {"model_id": "m1", "current_deployed_version": 1}
        deployment_service.store_idempotency(db_session, "key-1", 10, result)
        db_session.commit()
        cached = deployment_service.check_idempotency(db_session, "key-1", 10)
        assert cached == result

    def test_different_model_versions_get_independent_results(self, db_session):
        r1 = {"model_id": "m1", "current_deployed_version": 1}
        r2 = {"model_id": "m1", "current_deployed_version": 2}
        deployment_service.store_idempotency(db_session, "key-x", 10, r1)
        deployment_service.store_idempotency(db_session, "key-x", 20, r2)
        db_session.commit()
        assert deployment_service.check_idempotency(db_session, "key-x", 10) == r1
        assert deployment_service.check_idempotency(db_session, "key-x", 20) == r2

    def test_expired_key_returns_none(self, db_session):
        now = _utc_now()
        db_session.add(
            IdempotencyKey(
                key="expired-key:5",
                endpoint=deployment_service._DEPLOY_IDEMPOTENCY_ENDPOINT,
                response_status=200,
                response_body_json='{"model_id": "m1", "current_deployed_version": 3}',
                created_at=now - timedelta(hours=2),
                expires_at=now - timedelta(hours=1),
            )
        )
        db_session.commit()
        assert (
            deployment_service.check_idempotency(db_session, "expired-key", 5) is None
        )

    def test_store_idempotency_none_key_is_noop(self, db_session):
        deployment_service.store_idempotency(db_session, None, 1, {})
        assert db_session.scalars(select(IdempotencyKey)).all() == []
