"""Notification tests (issue #130): job/deployment lifecycle events must produce an in-app
notification for the relevant user (job owner) or every admin (events with no single owner).
"""

from sqlalchemy import select
from sqlalchemy.orm import Session

import pytest

from app.models.notification import Notification
from tests.conftest import auth_header
from tests.test_deployment_api import (
    DATASET_CREATE_REQUEST,
    TRAINING_RUN_CREATE_REQUEST,
    _promoted_model_version,
)
from tests.test_training_api import _failed_run_via_api


@pytest.fixture(autouse=True)
def _disable_promotion_gates(monkeypatch):
    from app.config import settings

    for name in (
        "eval_gate_require_eval_set_reference",
        "eval_gate_require_qualitative_majority",
        "eval_gate_require_no_general_regression",
        "eval_gate_require_eval_loss_not_worse",
    ):
        monkeypatch.setattr(settings, name, False)


def _register_and_login(client, username="owner1", password="Owner1234", role="user"):
    client.post(
        "/api/v1/auth/register",
        json={"username": username, "password": password, "role": role},
    )
    resp = client.post(
        "/api/v1/auth/login", json={"username": username, "password": password}
    )
    return resp.json()["access_token"]


def _notifications_for(client, *, user_id=None, type_=None) -> list[Notification]:
    with Session(client.engine) as db:
        query = select(Notification)
        if user_id is not None:
            query = query.where(Notification.user_id == user_id)
        if type_ is not None:
            query = query.where(Notification.type == type_)
        return list(db.scalars(query).all())


def _pass_validation(client, admin_token, records):
    from app.models.dataset import DatasetVersion

    h = auth_header(admin_token)
    with Session(client.engine) as session:
        row = session.scalar(
            select(DatasetVersion).where(
                DatasetVersion.dataset_id == "no_robots",
                DatasetVersion.version == 1,
            )
        )
        row.status = "PROCESSED"
        session.commit()
    return client.post(
        "/api/v1/datasets/no_robots/versions/1/validate",
        json={"records": records},
        headers=h,
    )


def _owner_training_run(client, admin_token, owner_username):
    h = auth_header(admin_token)
    client.post(
        "/api/v1/datasets/no_robots/versions", json=DATASET_CREATE_REQUEST, headers=h
    )
    _pass_validation(
        client,
        admin_token,
        [
            {
                "id": "r1",
                "messages": [
                    {"role": "user", "content": "hi"},
                    {
                        "role": "assistant",
                        "content": " ".join(f"w{i}" for i in range(25)),
                    },
                ],
                "metadata": {"source_dataset": "no_robots", "source_id": "sq-1"},
            }
        ],
    )
    body = dict(TRAINING_RUN_CREATE_REQUEST, triggered_by=owner_username)
    created = client.post("/api/v1/training-runs", json=body, headers=h).json()
    return created["training_run_id"]


def test_training_completed_notifies_job_owner(client, admin_token):
    _register_and_login(client, "owner1")
    run_id = _owner_training_run(client, admin_token, "owner1")

    from app.services import training_service
    from app.workers.mock_runner import MockTrainingRunner

    with Session(client.engine) as db:
        run = training_service.get_training_run(db, run_id)
        training_service.start_training_run(db, run)
        artifact_uri = MockTrainingRunner().run(db, run)
        training_service.complete_training_run(db, run, artifact_uri=artifact_uri)
        db.commit()

    with Session(client.engine) as db:
        from app.services import auth_service

        owner = auth_service.get_user_by_username(db, "owner1")
        owner_id = owner.id

    rows = _notifications_for(client, user_id=owner_id, type_="TRAINING_COMPLETED")
    assert len(rows) == 1
    assert run_id in rows[0].message
    assert rows[0].resource_ref == run_id


def test_training_failed_notifies_job_owner(client, admin_token):
    _register_and_login(client, "owner2")
    run_id = _owner_training_run(client, admin_token, "owner2")

    from app.services import training_service

    with Session(client.engine) as db:
        run = training_service.get_training_run(db, run_id)
        assert training_service.claim_training_run(db, run)
        training_service.fail_training_run(db, run, error_message="boom: cuda oom")
        db.commit()

    with Session(client.engine) as db:
        from app.services import auth_service

        owner = auth_service.get_user_by_username(db, "owner2")
        owner_id = owner.id

    rows = _notifications_for(client, user_id=owner_id, type_="TRAINING_FAILED")
    assert len(rows) == 1
    assert "boom: cuda oom" in rows[0].message
    assert rows[0].resource_ref == run_id


def test_retry_via_api_does_not_duplicate_training_notifications(client, admin_token):
    """Sanity check: retrying a failed run creates a new PENDING run and must not itself fire a
    training-completed/failed notification (it hasn't run yet)."""
    failed_id = _failed_run_via_api(client, admin_token)
    h = auth_header(admin_token)
    resp = client.post(f"/api/v1/training-runs/{failed_id}/retry", headers=h)
    assert resp.status_code == 201

    assert _notifications_for(client, type_="TRAINING_COMPLETED") == []


def test_validation_failed_notifies_dataset_owner(client, admin_token):
    _register_and_login(client, "dsowner")
    h = auth_header(admin_token)
    client.post(
        "/api/v1/datasets/no_robots/versions",
        json=DATASET_CREATE_REQUEST,
        headers=h,
    )
    from app.models.dataset import DatasetVersion

    # NOTE (pre-existing, unrelated gap found while writing this test): the create-version
    # request schema accepts `created_by`, but dataset_service.create_dataset_version hardcodes
    # `created_by=None` and never reads it off the request - so it must be set directly here,
    # the same way this fixture already reaches into the DB to flip `status` to PROCESSED.
    with Session(client.engine) as session:
        row = session.scalar(
            select(DatasetVersion).where(
                DatasetVersion.dataset_id == "no_robots",
                DatasetVersion.version == 1,
            )
        )
        row.status = "PROCESSED"
        row.created_by = "dsowner"
        session.commit()

    # H8 leakage: a record whose user content exactly matches an eval record fails the gate.
    shared_user_content = "What is the capital of France?"
    resp = client.post(
        "/api/v1/datasets/no_robots/versions/1/validate",
        json={
            "records": [
                {
                    "id": "r1",
                    "messages": [
                        {"role": "user", "content": shared_user_content},
                        {"role": "assistant", "content": "Paris."},
                    ],
                    "metadata": {"source_dataset": "no_robots", "source_id": "sq-1"},
                }
            ],
            "eval_records": [
                {
                    "id": "e1",
                    "messages": [
                        {"role": "user", "content": shared_user_content},
                        {"role": "assistant", "content": "Paris."},
                    ],
                    "metadata": {"source_dataset": "eval", "source_id": "e-1"},
                }
            ],
        },
        headers=h,
    )
    assert resp.status_code == 201
    assert resp.json()["gate_decision"] == "FAIL"

    with Session(client.engine) as db:
        from app.services import auth_service

        owner = auth_service.get_user_by_username(db, "dsowner")
        owner_id = owner.id

    rows = _notifications_for(client, user_id=owner_id, type_="VALIDATION_FAILED")
    assert len(rows) == 1
    assert rows[0].resource_ref == "no_robots:v1"


def _admin_ids(client) -> set[int]:
    with Session(client.engine) as db:
        from app.models.user import User

        return {
            u.id for u in db.scalars(select(User).where(User.role == "admin")).all()
        }


def test_approval_required_notifies_every_admin(client, admin_token):
    model_id, version = _promoted_model_version(client, admin_token)
    # _promoted_model_version already drives submit_evaluation once (REGISTERED -> EVALUATED)
    # before promoting - the APPROVAL_REQUIRED notification fires at that transition.
    admin_ids = _admin_ids(client)
    assert admin_ids  # sanity: at least the fixture admin exists

    rows = _notifications_for(client, type_="APPROVAL_REQUIRED")
    assert {r.user_id for r in rows} == admin_ids
    assert all(f"{model_id!r}" in r.message for r in rows)


def test_production_deploy_failure_notifies_admins(client, admin_token, monkeypatch):
    from app.config import settings

    # Force the mock backend's fixed-length generation below the smoke threshold, regardless
    # of environment - no network/backend mocking needed (issue #41's mock keeps tests GPU-free).
    monkeypatch.setattr(settings, "inference_smoke_min_chars", 10_000)

    model_id, version = _promoted_model_version(client, admin_token)
    h = auth_header(admin_token)
    resp = client.post(
        f"/api/v1/models/{model_id}/versions/{version}/deploy",
        json={"environment": "production"},
        headers=h,
    )
    assert resp.status_code == 502
    assert resp.json()["error"]["code"] == "SMOKE_TEST_FAILED"

    admin_ids = _admin_ids(client)
    rows = _notifications_for(client, type_="PRODUCTION_DEPLOYMENT_FAILED")
    assert {r.user_id for r in rows} == admin_ids
    assert all(f"{model_id}" in r.message for r in rows)


def test_notifications_endpoint_lists_only_own_and_marks_read(client, admin_token):
    owner_token = _register_and_login(client, "listowner")
    run_id = _owner_training_run(client, admin_token, "listowner")

    from app.services import training_service
    from app.workers.mock_runner import MockTrainingRunner

    with Session(client.engine) as db:
        run = training_service.get_training_run(db, run_id)
        training_service.start_training_run(db, run)
        artifact_uri = MockTrainingRunner().run(db, run)
        training_service.complete_training_run(db, run, artifact_uri=artifact_uri)
        db.commit()

    owner_h = auth_header(owner_token)
    listed = client.get("/api/v1/notifications", headers=owner_h)
    assert listed.status_code == 200
    body = listed.json()
    assert body["total"] == 1
    entry = body["items"][0]
    assert entry["type"] == "TRAINING_COMPLETED"
    assert entry["read_at"] is None

    # admin never triggered/owns this notification and must not see it in their own list.
    admin_h = auth_header(admin_token)
    admin_listed = client.get("/api/v1/notifications", headers=admin_h).json()
    assert all(item["id"] != entry["id"] for item in admin_listed["items"])

    mark = client.post(f"/api/v1/notifications/{entry['id']}/read", headers=owner_h)
    assert mark.status_code == 200
    assert mark.json()["read_at"] is not None

    unread = client.get(
        "/api/v1/notifications?unread_only=true", headers=owner_h
    ).json()
    assert unread["total"] == 0


def test_mark_read_returns_404_for_other_users_notification(client, admin_token):
    owner_token = _register_and_login(client, "listowner2")
    run_id = _owner_training_run(client, admin_token, "listowner2")

    from app.services import training_service
    from app.workers.mock_runner import MockTrainingRunner

    with Session(client.engine) as db:
        run = training_service.get_training_run(db, run_id)
        training_service.start_training_run(db, run)
        artifact_uri = MockTrainingRunner().run(db, run)
        training_service.complete_training_run(db, run, artifact_uri=artifact_uri)
        db.commit()

    owner_h = auth_header(owner_token)
    entry = client.get("/api/v1/notifications", headers=owner_h).json()["items"][0]

    admin_h = auth_header(admin_token)
    resp = client.post(f"/api/v1/notifications/{entry['id']}/read", headers=admin_h)
    assert resp.status_code == 404
