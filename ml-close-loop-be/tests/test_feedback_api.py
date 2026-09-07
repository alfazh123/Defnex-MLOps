import threading

from sqlalchemy import create_engine
from sqlalchemy.orm import Session

from app.db.base import Base
from tests.conftest import auth_header
from tests.test_deployment_api import _registered_model_version


def _submit(client, admin_token, **overrides):
    model_id, version = _registered_model_version(client, admin_token)
    body = {
        "model_id": model_id,
        "version": version,
        "prompt": "What is the capital of France?",
        "response": "Paris.",
        "rating": 4,
    }
    body.update(overrides)
    resp = client.post("/api/v1/feedback", json=body, headers=auth_header(admin_token))
    return resp, model_id, version


def test_submit_feedback_returns_pending_with_model_reference(client, admin_token):
    resp, model_id, version = _submit(client, admin_token)

    assert resp.status_code == 201
    body = resp.json()
    assert body["curation_status"] == "PENDING"
    assert body["model_id"] == model_id
    assert body["version"] == version
    assert body["prompt"] == "What is the capital of France?"
    assert body["response"] == "Paris."
    assert body["correction"] is None


def test_submit_feedback_missing_model_version_returns_404(client, admin_token):
    resp = client.post(
        "/api/v1/feedback",
        json={
            "model_id": "no-such-model",
            "version": 1,
            "prompt": "hi",
            "response": "hello",
            "rating": 3,
        },
        headers=auth_header(admin_token),
    )

    assert resp.status_code == 404
    assert resp.json()["error"]["code"] == "MODEL_NOT_FOUND"


def test_submit_feedback_rating_out_of_range_returns_422(client, admin_token):
    resp, _, _ = _submit(client, admin_token)
    model_id, version = resp.json()["model_id"], resp.json()["version"]

    bad = client.post(
        "/api/v1/feedback",
        json={
            "model_id": model_id,
            "version": version,
            "prompt": "hi",
            "response": "hello",
            "rating": 6,
        },
        headers=auth_header(admin_token),
    )

    assert bad.status_code == 422


def test_approve_feedback_appears_in_candidates(client, admin_token):
    resp, _, _ = _submit(client, admin_token, correction="Paris, France.")
    feedback_id = resp.json()["feedback_id"]
    h = auth_header(admin_token)

    approved = client.post(f"/api/v1/feedback/{feedback_id}/approve", headers=h)
    assert approved.status_code == 200
    assert approved.json()["curation_status"] == "APPROVED"

    candidates = client.get("/api/v1/feedback/candidates", headers=h).json()
    assert any(c["id"] == feedback_id for c in candidates)
    record = next(c for c in candidates if c["id"] == feedback_id)
    # correction wins over response as the assistant ground truth.
    assert record["messages"][1]["content"] == "Paris, France."


def test_reject_feedback_does_not_appear_in_candidates(client, admin_token):
    resp, _, _ = _submit(client, admin_token)
    feedback_id = resp.json()["feedback_id"]
    h = auth_header(admin_token)

    rejected = client.post(f"/api/v1/feedback/{feedback_id}/reject", headers=h)
    assert rejected.status_code == 200
    assert rejected.json()["curation_status"] == "REJECTED"

    candidates = client.get("/api/v1/feedback/candidates", headers=h).json()
    assert not any(c["id"] == feedback_id for c in candidates)


def test_approve_already_rejected_feedback_returns_409(client, admin_token):
    resp, _, _ = _submit(client, admin_token)
    feedback_id = resp.json()["feedback_id"]
    h = auth_header(admin_token)
    client.post(f"/api/v1/feedback/{feedback_id}/reject", headers=h)

    second = client.post(f"/api/v1/feedback/{feedback_id}/approve", headers=h)

    assert second.status_code == 409
    assert second.json()["error"]["code"] == "FEEDBACK_NOT_PENDING"
    still_rejected = client.get("/api/v1/feedback", headers=h).json()
    row = next(i for i in still_rejected["items"] if i["feedback_id"] == feedback_id)
    assert row["curation_status"] == "REJECTED"


def test_approve_nonexistent_feedback_returns_404(client, admin_token):
    resp = client.post(
        "/api/v1/feedback/feedback-nope/approve", headers=auth_header(admin_token)
    )

    assert resp.status_code == 404
    assert resp.json()["error"]["code"] == "FEEDBACK_NOT_FOUND"


def test_non_admin_cannot_approve_or_reject(client, admin_token, user_token):
    resp, _, _ = _submit(client, admin_token)
    feedback_id = resp.json()["feedback_id"]
    h_user = auth_header(user_token)

    approve = client.post(f"/api/v1/feedback/{feedback_id}/approve", headers=h_user)
    reject = client.post(f"/api/v1/feedback/{feedback_id}/reject", headers=h_user)

    assert approve.status_code == 403
    assert reject.status_code == 403


def test_feedback_without_correction_is_valid_and_curatable(client, admin_token):
    resp, _, _ = _submit(client, admin_token)
    assert resp.json()["correction"] is None
    feedback_id = resp.json()["feedback_id"]

    approved = client.post(
        f"/api/v1/feedback/{feedback_id}/approve", headers=auth_header(admin_token)
    )

    assert approved.status_code == 200
    assert approved.json()["curation_status"] == "APPROVED"


def test_concurrent_approve_only_one_wins(tmp_path):
    """Issue #42 required test: two concurrent approve calls on the same feedback ->
    exactly one succeeds, no duplicate candidate. Same pattern as
    test_concurrent_workers_run_single_pending_job_once (issue #33): two real threads,
    each with its own engine, racing over one row in a shared file-backed SQLite DB.
    """
    from app.models.model import Model, ModelVersion
    from app.models.feedback import Feedback
    from app.services import feedback_service
    from datetime import datetime, timezone

    db_path = str(tmp_path / "race.db")
    engine = create_engine(f"sqlite:///{db_path}", connect_args={"timeout": 30})
    Base.metadata.create_all(engine)

    with Session(engine) as setup:
        setup.add(Model(model_id="m1"))
        setup.flush()
        mv = ModelVersion(
            model_id="m1",
            version=1,
            training_run_id="run-1",
            base_model="base",
            training_config={},
            created_at=datetime.now(timezone.utc),
        )
        setup.add(mv)
        setup.flush()
        fb = Feedback(
            feedback_id="feedback-race1",
            model_version_id=mv.id,
            prompt="p",
            response="r",
            rating=5,
            curation_status="PENDING",
            submitted_at=datetime.now(timezone.utc),
        )
        setup.add(fb)
        setup.commit()

    results: dict[str, str] = {}

    def approve(name):
        eng = create_engine(f"sqlite:///{db_path}", connect_args={"timeout": 30})
        try:
            with Session(eng) as session:
                feedback_service.approve_feedback(session, "feedback-race1", name)
                session.commit()
            results[name] = "OK"
        except ValueError:
            results[name] = "LOST"
        finally:
            eng.dispose()

    threads = [threading.Thread(target=approve, args=(n,)) for n in ("A", "B")]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=15)

    assert sorted(results.values()) == ["LOST", "OK"]
    with Session(engine) as check:
        row = check.get(Feedback, "feedback-race1")
        assert row.curation_status == "APPROVED"
        assert row.curated_by in ("A", "B")


def test_create_dataset_version_from_approved_feedback_records_provenance(
    client, admin_token
):
    h = auth_header(admin_token)
    ids = []
    for i in range(2):
        resp, _, _ = _submit(client, admin_token, prompt=f"q{i}", response=f"a{i}")
        fid = resp.json()["feedback_id"]
        client.post(f"/api/v1/feedback/{fid}/approve", headers=h)
        ids.append(fid)

    created = client.post(
        "/api/v1/datasets/feedback-curated/versions/from-feedback",
        json={"feedback_ids": ids, "source_format": "chatml"},
        headers=h,
    )

    assert created.status_code == 201
    body = created.json()
    assert body["manifest"]["row_count"] == 2
    assert body["manifest"]["source_type"] == "feedback"
    assert sorted(body["manifest"]["source_feedback_ids"]) == sorted(ids)


def test_create_dataset_version_from_feedback_rejects_unapproved(client, admin_token):
    h = auth_header(admin_token)
    resp, _, _ = _submit(client, admin_token)
    pending_id = resp.json()["feedback_id"]

    created = client.post(
        "/api/v1/datasets/feedback-curated/versions/from-feedback",
        json={"feedback_ids": [pending_id], "source_format": "chatml"},
        headers=h,
    )

    assert created.status_code == 409
    assert created.json()["error"]["code"] == "FEEDBACK_NOT_APPROVED"
    listing = client.get("/api/v1/datasets/feedback-curated/versions", headers=h)
    assert listing.status_code == 404
