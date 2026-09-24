from sqlalchemy.orm import Session

from tests.conftest import auth_header

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


VALID_RECORD = {
    "id": "r1",
    "messages": [
        {"role": "user", "content": "What is the capital of France?"},
        {"role": "assistant", "content": " ".join(f"word{i}" for i in range(25))},
    ],
    "metadata": {"source_dataset": "no_robots", "source_id": "sq-1"},
}


def _pass_validation(client, admin_token):
    from sqlalchemy import select

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
    report = client.post(
        "/api/v1/datasets/no_robots/versions/1/validate",
        json={"records": [VALID_RECORD]},
        headers=h,
    )
    assert report.status_code == 201, report.text
    assert report.json()["gate_decision"] == "PASS"


def _registered_model_version(client, admin_token):
    """Drive a training run through to COMPLETED via the mock worker, then register it."""
    from app.services import model_service, training_service
    from app.workers.mock_runner import MockTrainingRunner

    h = auth_header(admin_token)
    client.post(
        "/api/v1/datasets/no_robots/versions", json=DATASET_CREATE_REQUEST, headers=h
    )
    _pass_validation(client, admin_token)
    created = client.post(
        "/api/v1/training-runs", json=TRAINING_RUN_CREATE_REQUEST, headers=h
    ).json()

    with Session(client.engine) as db:
        training_run = training_service.get_training_run(db, created["training_run_id"])
        training_service.start_training_run(db, training_run)
        artifact_uri = MockTrainingRunner().run(db, training_run)
        training_service.complete_training_run(
            db, training_run, artifact_uri=artifact_uri
        )
        model_version = model_service.register_model_version(db, training_run)
        db.commit()
        return model_version.model_id, model_version.version


def test_get_model_version_returns_404_when_missing(client, admin_token):
    response = client.get(
        "/api/v1/models/no-such-model/versions/1", headers=auth_header(admin_token)
    )

    assert response.status_code == 404
    assert response.json()["error"]["code"] == "MODEL_NOT_FOUND"


def test_get_model_version_returns_full_lineage(client, admin_token):
    model_id, version = _registered_model_version(client, admin_token)

    response = client.get(
        f"/api/v1/models/{model_id}/versions/{version}",
        headers=auth_header(admin_token),
    )

    assert response.status_code == 200
    body = response.json()
    assert body["model_id"] == model_id
    assert body["version"] == version
    assert body["status"] == "REGISTERED"
    assert body["training_run_id"]
    assert body["base_model"] == "Qwen/Qwen3.8-27B"
    assert body["dataset_id"] == "no_robots"
    assert body["dataset_version"] == 1
    assert body["artifacts"][0]["type"] == "adapter"
    assert body["promotion_decision_ref"] is None


def test_list_models_returns_latest_version_and_status(client, admin_token):
    model_id, version = _registered_model_version(client, admin_token)

    response = client.get("/api/v1/models", headers=auth_header(admin_token))

    assert response.status_code == 200
    body = response.json()
    assert {
        "model_id": model_id,
        "latest_version": version,
        "status": "REGISTERED",
    } in body["items"]


def test_list_models_filters_by_status(client, admin_token):
    _registered_model_version(client, admin_token)

    response = client.get(
        "/api/v1/models",
        params={"status": "PROMOTED"},
        headers=auth_header(admin_token),
    )

    assert response.status_code == 200
    assert response.json()["items"] == []


def test_get_evaluation_returns_404_when_missing(client, admin_token):
    response = client.get(
        "/api/v1/models/no-such-model/versions/1/evaluation",
        headers=auth_header(admin_token),
    )

    assert response.status_code == 404
    assert response.json()["error"]["code"] == "MODEL_NOT_FOUND"


def test_get_evaluation_is_all_null_before_any_submission(client, admin_token):
    model_id, version = _registered_model_version(client, admin_token)

    response = client.get(
        f"/api/v1/models/{model_id}/versions/{version}/evaluation",
        headers=auth_header(admin_token),
    )

    assert response.status_code == 200
    assert response.json() == {
        "eval_loss_trend": None,
        "qualitative_comparison": None,
        "general_domain_regression_check": None,
    }


def _eval_set_version(
    client, admin_token, eval_set_id="domain-benchmark", records=None
):
    """Create a golden/eval set version via the real API (issue #43) for trigger tests."""
    if records is None:
        records = [{"messages": [{"role": "user", "content": "eval-probe-1"}]}]
    resp = client.post(
        f"/api/v1/eval-sets/{eval_set_id}/versions",
        json={"records": records},
        headers=auth_header(admin_token),
    )
    assert resp.status_code == 201, resp.text
    return resp.json()["version"]


def test_submit_evaluation_returns_404_when_missing(client, admin_token):
    response = client.post(
        "/api/v1/models/no-such-model/versions/1/evaluation",
        json={},
        headers=auth_header(admin_token),
    )

    assert response.status_code == 404
    assert response.json()["error"]["code"] == "MODEL_NOT_FOUND"


def test_submit_evaluation_rejects_manual_signal_payload(client, admin_token):
    """Issue #128: the old jalur lama (manual caller-supplied signal numbers) must be
    rejected, not silently applied - `EvaluationTriggerRequest` forbids those fields."""
    model_id, version = _registered_model_version(client, admin_token)

    response = client.post(
        f"/api/v1/models/{model_id}/versions/{version}/evaluation",
        json={"eval_loss_trend": {"this_version_eval_loss": 0.01}},
        headers=auth_header(admin_token),
    )

    assert response.status_code == 422
    # Nothing was applied: the evaluation stays untouched and the status unchanged.
    evaluation = client.get(
        f"/api/v1/models/{model_id}/versions/{version}/evaluation",
        headers=auth_header(admin_token),
    ).json()
    assert evaluation["eval_loss_trend"] is None
    lineage = client.get(
        f"/api/v1/models/{model_id}/versions/{version}",
        headers=auth_header(admin_token),
    ).json()
    assert lineage["status"] == "REGISTERED"


def test_submit_evaluation_rejects_manual_qualitative_payload(client, admin_token):
    model_id, version = _registered_model_version(client, admin_token)

    response = client.post(
        f"/api/v1/models/{model_id}/versions/{version}/evaluation",
        json={
            "qualitative_comparison": {
                "question_table_version": 1,
                "wins": 999,
                "losses": 0,
                "ties": 0,
                "total": 999,
            }
        },
        headers=auth_header(admin_token),
    )

    assert response.status_code == 422


def test_trigger_evaluation_requires_eval_set_reference(client, admin_token):
    model_id, version = _registered_model_version(client, admin_token)

    response = client.post(
        f"/api/v1/models/{model_id}/versions/{version}/evaluation",
        json={},
        headers=auth_header(admin_token),
    )

    assert response.status_code == 400
    assert response.json()["error"]["code"] == "EVAL_SET_REQUIRED"


def test_trigger_evaluation_stays_registered_until_worker_runs(client, admin_token):
    """Triggering is async (issue #128): the response reflects the pre-trigger state, not a
    synchronously-computed result."""
    model_id, version = _registered_model_version(client, admin_token)
    eval_version = _eval_set_version(client, admin_token)

    response = client.post(
        f"/api/v1/models/{model_id}/versions/{version}/evaluation",
        json={"eval_set_id": "domain-benchmark", "eval_set_version": eval_version},
        headers=auth_header(admin_token),
    )

    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "REGISTERED"
    assert body["evaluation"]["eval_loss_trend"] is None

    with Session(client.engine) as db:
        from sqlalchemy import select

        from app.models.model import ModelVersion

        row = db.scalar(
            select(ModelVersion).where(
                ModelVersion.model_id == model_id, ModelVersion.version == version
            )
        )
        assert row.evaluation_requested is True
        assert row.eval_set_id == "domain-benchmark"
        assert row.eval_set_version == eval_version


def test_trigger_evaluation_worker_computes_signals_and_transitions_to_evaluated(
    client, admin_token
):
    """End-to-end: trigger via the API, run the evaluation worker (same shape the
    training_worker.process_next_job poll uses), and assert the server-computed signals -
    real generation via MockServingBackend against the stored golden set - drove the
    REGISTERED -> EVALUATED transition."""
    model_id, version = _registered_model_version(client, admin_token)
    eval_version = _eval_set_version(
        client,
        admin_token,
        records=[
            {"messages": [{"role": "user", "content": "q1"}]},
            {"messages": [{"role": "user", "content": "q2"}]},
        ],
    )
    h = auth_header(admin_token)

    trigger = client.post(
        f"/api/v1/models/{model_id}/versions/{version}/evaluation",
        json={"eval_set_id": "domain-benchmark", "eval_set_version": eval_version},
        headers=h,
    )
    assert trigger.status_code == 200

    with Session(client.engine) as db:
        from app.workers.evaluation_worker import process_next_evaluation

        processed = process_next_evaluation(db)
        db.commit()
        assert processed is not None

    lineage = client.get(
        f"/api/v1/models/{model_id}/versions/{version}", headers=h
    ).json()
    assert lineage["status"] == "EVALUATED"
    assert lineage["evaluation"]["qualitative_comparison"]["wins"] == 2
    assert lineage["evaluation"]["qualitative_comparison"]["losses"] == 0
    assert lineage["evaluation"]["general_domain_regression_check"]["checked"] is True
    assert (
        lineage["evaluation"]["general_domain_regression_check"]["regressions_found"]
        == []
    )
    # MockTrainingRunner reports a fake training eval_loss (issue #128); the trigger endpoint
    # itself never accepted this number from the caller.
    assert lineage["evaluation"]["eval_loss_trend"]["this_version_eval_loss"] == 0.84


def test_submit_evaluation_returns_409_once_promoted(client, admin_token):
    model_id, version = _registered_model_version(client, admin_token)

    with Session(client.engine) as db:
        from app.services import model_service

        model_version = model_service.get_model_version(db, model_id, version)
        model_version.status = "PROMOTED"
        db.commit()

    response = client.post(
        f"/api/v1/models/{model_id}/versions/{version}/evaluation",
        json={},
        headers=auth_header(admin_token),
    )

    assert response.status_code == 409
    assert response.json()["error"]["code"] == "EVALUATION_NOT_EDITABLE"


def test_list_models_empty(client, admin_token):
    response = client.get("/api/v1/models", headers=auth_header(admin_token))

    assert response.status_code == 200
    assert response.json()["items"] == []


def test_list_models_respects_pagination_params(client, admin_token):
    """Issue #176: GET /models is now paginated like every other list endpoint."""
    _registered_model_version(client, admin_token)

    page2 = client.get(
        "/api/v1/models",
        params={"page": 2, "size": 1},
        headers=auth_header(admin_token),
    )

    assert page2.status_code == 200
    body = page2.json()
    assert body["items"] == []
    assert body["total"] == 1
    assert body["page"] == 2
    assert body["size"] == 1
    assert body["pages"] == 1


def test_list_model_versions_returns_404_when_model_missing(client, admin_token):
    response = client.get(
        "/api/v1/models/no-such-model/versions", headers=auth_header(admin_token)
    )

    assert response.status_code == 404
    assert response.json()["error"]["code"] == "MODEL_NOT_FOUND"


def test_list_model_versions_paginated_full_record(client, admin_token):
    model_id, version = _registered_model_version(client, admin_token)
    h = auth_header(admin_token)

    response = client.get(f"/api/v1/models/{model_id}/versions", headers=h)

    assert response.status_code == 200
    body = response.json()
    assert set(body) == {"items", "total", "page", "size", "pages"}
    assert body["total"] == 1
    assert body["page"] == 1
    assert body["size"] == 20
    assert body["pages"] == 1
    item = body["items"][0]
    assert set(item) == {
        "model_id",
        "version",
        "name",
        "status",
        "training_run_id",
        "base_model",
        "dataset_id",
        "dataset_version",
        "dataset_validation_report_ref",
        "training_config",
        "training_config_hash",
        "created_at",
        "created_by",
        "evaluation",
        "eval_set_id",
        "eval_set_version",
        "artifacts",
        "promotion_decision_ref",
        "previous_model_id",
        "is_seed_data",
    }
    assert item["model_id"] == model_id
    assert item["version"] == version


def test_list_model_versions_includes_retired_status(client, admin_token):
    model_id, version = _registered_model_version(client, admin_token)
    with Session(client.engine) as db:
        from app.services import model_service

        model_version = model_service.get_model_version(db, model_id, version)
        model_version.status = "RETIRED"
        db.commit()

    response = client.get(
        f"/api/v1/models/{model_id}/versions", headers=auth_header(admin_token)
    )

    assert response.status_code == 200
    items = response.json()["items"]
    assert [v["status"] for v in items] == ["RETIRED"]


def test_list_model_versions_pagination_and_ordering(client, admin_token):
    model_id, _ = _registered_model_version(client, admin_token)
    h = auth_header(admin_token)

    from app.services import training_service
    from app.workers.mock_runner import MockTrainingRunner

    for _ in range(3):
        created = client.post(
            "/api/v1/training-runs", json=TRAINING_RUN_CREATE_REQUEST, headers=h
        ).json()
        with Session(client.engine) as db:
            training_run = training_service.get_training_run(
                db, created["training_run_id"]
            )
            training_service.start_training_run(db, training_run)
            artifact_uri = MockTrainingRunner().run(db, training_run)
            training_service.complete_training_run(
                db, training_run, artifact_uri=artifact_uri
            )
            from app.services import model_service

            model_service.register_model_version(db, training_run)
            db.commit()

    page1 = client.get(
        f"/api/v1/models/{model_id}/versions", params={"page": 1, "size": 2}, headers=h
    ).json()
    assert page1["total"] == 4
    assert page1["pages"] == 2
    assert [v["version"] for v in page1["items"]] == [1, 2]

    page2 = client.get(
        f"/api/v1/models/{model_id}/versions", params={"page": 2, "size": 2}, headers=h
    ).json()
    assert page2["page"] == 2
    assert [v["version"] for v in page2["items"]] == [3, 4]
