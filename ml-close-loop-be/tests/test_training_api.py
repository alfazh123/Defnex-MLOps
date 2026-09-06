import pytest
from sqlalchemy import func, select
from sqlalchemy.orm import Session

from app.models.dataset import DatasetVersion
from app.models.training import TrainingRun
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


def _mark_processed(client, dataset_id="no_robots", version=1):
    from sqlalchemy.orm import Session

    with Session(client.engine) as session:
        row = session.scalar(
            select(DatasetVersion).where(
                DatasetVersion.dataset_id == dataset_id,
                DatasetVersion.version == version,
            )
        )
        row.status = "PROCESSED"
        session.commit()


def _pass_validation(client, admin_token):
    """Create the dataset, mark it processed, and run a PASS validation report."""
    h = auth_header(admin_token)
    client.post(
        "/api/v1/datasets/no_robots/versions", json=DATASET_CREATE_REQUEST, headers=h
    )
    _mark_processed(client)
    response = client.post(
        "/api/v1/datasets/no_robots/versions/1/validate",
        json={"records": [_valid_record()]},
        headers=h,
    )
    assert response.status_code == 201, response.text
    assert response.json()["gate_decision"] == "PASS"


def test_create_training_run_returns_404_when_dataset_version_missing(
    client, admin_token
):
    response = client.post(
        "/api/v1/training-runs",
        json=TRAINING_RUN_CREATE_REQUEST,
        headers=auth_header(admin_token),
    )

    assert response.status_code == 404
    assert response.json()["error"]["code"] == "DATASET_NOT_FOUND"


def test_create_training_run_returns_201_queued(client, admin_token):
    h = auth_header(admin_token)
    _pass_validation(client, admin_token)

    response = client.post(
        "/api/v1/training-runs", json=TRAINING_RUN_CREATE_REQUEST, headers=h
    )

    assert response.status_code == 201
    body = response.json()
    assert body["training_run_id"].startswith("run-")
    assert body["status"] == "PENDING"
    assert body["dataset_id"] == "no_robots"
    assert body["dataset_version"] == 1
    assert body["model_id"] == "qwen-sft-domain-x"
    assert body["base_model"] == "Qwen/Qwen3.8-27B"
    assert body["training_config"]["peft_method"] == "lora"
    assert body["model_version"] is None


def test_create_training_run_rejects_invalid_body(client, admin_token):
    response = client.post(
        "/api/v1/training-runs",
        json={"dataset_id": "no_robots"},
        headers=auth_header(admin_token),
    )

    assert response.status_code == 422


@pytest.mark.parametrize("method", ["dora", "qdora", "none"])
def test_create_training_run_rejects_unservable_peft_method(
    client, admin_token, method
):
    h = auth_header(admin_token)
    client.post(
        "/api/v1/datasets/no_robots/versions", json=DATASET_CREATE_REQUEST, headers=h
    )
    request = dict(TRAINING_RUN_CREATE_REQUEST)
    request["training_config"] = dict(TRAINING_RUN_CREATE_REQUEST["training_config"])
    request["training_config"]["peft_method"] = method

    response = client.post("/api/v1/training-runs", json=request, headers=h)

    assert response.status_code == 422
    detail = response.json()["detail"][0]["msg"]
    assert method in detail
    assert "vLLM serving path" in detail
    with Session(client.engine) as db:
        count = db.scalar(select(func.count()).select_from(TrainingRun))
    assert count == 0


def test_create_training_run_peft_method_defaults_to_lora(client, admin_token):
    h = auth_header(admin_token)
    _pass_validation(client, admin_token)
    request = dict(TRAINING_RUN_CREATE_REQUEST)
    request["training_config"] = {"epochs": 2}

    response = client.post("/api/v1/training-runs", json=request, headers=h)

    assert response.status_code == 201
    assert response.json()["training_config"]["peft_method"] == "lora"


def test_create_training_run_accepts_rslora(client, admin_token):
    h = auth_header(admin_token)
    _pass_validation(client, admin_token)
    request = dict(TRAINING_RUN_CREATE_REQUEST)
    request["training_config"] = dict(TRAINING_RUN_CREATE_REQUEST["training_config"])
    request["training_config"]["peft_method"] = "rslora"

    response = client.post("/api/v1/training-runs", json=request, headers=h)

    assert response.status_code == 201
    assert response.json()["training_config"]["peft_method"] == "rslora"


def test_get_training_run_returns_404_when_missing(client, admin_token):
    response = client.get(
        "/api/v1/training-runs/run-doesnotexist", headers=auth_header(admin_token)
    )

    assert response.status_code == 404
    assert response.json()["error"]["code"] == "TRAINING_RUN_NOT_FOUND"


def test_get_training_run_returns_created_run(client, admin_token):
    h = auth_header(admin_token)
    _pass_validation(client, admin_token)
    created = client.post(
        "/api/v1/training-runs", json=TRAINING_RUN_CREATE_REQUEST, headers=h
    ).json()

    response = client.get(
        f"/api/v1/training-runs/{created['training_run_id']}", headers=h
    )

    assert response.status_code == 200
    assert response.json() == created


def test_create_training_run_does_not_start_training_inline(client, admin_token):
    """Create must only queue the run (PENDING); the worker owns execution."""
    from unittest.mock import AsyncMock, patch

    h = auth_header(admin_token)
    _pass_validation(client, admin_token)

    start_training = AsyncMock(return_value={"job_id": "job-123"})
    with patch(
        "app.api.training.unsloth_client.start_training", new=start_training
    ) as mocked:
        created = client.post(
            "/api/v1/training-runs", json=TRAINING_RUN_CREATE_REQUEST, headers=h
        ).json()

    assert created["status"] == "PENDING"
    mocked.assert_not_called()


def test_create_training_run_does_not_set_artifact_uri(client, admin_token):
    """artifact_uri stays NULL on create; only the worker populates it on COMPLETED."""
    h = auth_header(admin_token)
    _pass_validation(client, admin_token)
    created = client.post(
        "/api/v1/training-runs", json=TRAINING_RUN_CREATE_REQUEST, headers=h
    ).json()

    with Session(client.engine) as db:
        row = db.scalar(
            select(TrainingRun).where(
                TrainingRun.training_run_id == created["training_run_id"]
            )
        )
        assert row.status == "PENDING"
        assert row.artifact_uri is None


def _create_runs(client, admin_token, count):
    h = auth_header(admin_token)
    _pass_validation(client, admin_token)
    for _ in range(count):
        client.post(
            "/api/v1/training-runs", json=TRAINING_RUN_CREATE_REQUEST, headers=h
        )


def test_create_training_run_rejects_unsafe_model_id(client, admin_token):
    """Issue #38: model_id becomes part of the immutable version name `{model_id}-{base_model}-v{N}`
    and a filesystem directory, so path/URL-unsafe characters are rejected (no more free-form ids)."""
    h = auth_header(admin_token)
    _pass_validation(client, admin_token)
    request = dict(TRAINING_RUN_CREATE_REQUEST)
    request["model_id"] = "bad model!@#/with spaces"

    response = client.post("/api/v1/training-runs", json=request, headers=h)

    assert response.status_code == 422
    assert "model_id" in response.json()["detail"][0]["loc"]


def test_create_training_run_returns_correct_status_field(client, admin_token):
    h = auth_header(admin_token)
    _pass_validation(client, admin_token)
    created = client.post(
        "/api/v1/training-runs", json=TRAINING_RUN_CREATE_REQUEST, headers=h
    ).json()

    fetched = client.get(
        f"/api/v1/training-runs/{created['training_run_id']}", headers=h
    )

    assert fetched.status_code == 200
    assert fetched.json()["status"] == "PENDING"
    assert fetched.json()["current_epoch"] is None


def test_get_training_run_shows_live_progress_while_running(client, admin_token):
    """Issue #38: while a run is still RUNNING, GET /training-runs/{id} returns the progress
    fields the training runner streams in — non-NULL mid-run, not only after COMPLETED."""
    from app.services import training_service as svc

    h = auth_header(admin_token)
    _pass_validation(client, admin_token)
    created = client.post(
        "/api/v1/training-runs", json=TRAINING_RUN_CREATE_REQUEST, headers=h
    ).json()

    with Session(client.engine) as db:
        run = svc.get_training_run(db, created["training_run_id"])
        assert svc.claim_training_run(db, run)
        svc.update_training_progress(db, run, epoch=1, current_step=42, train_loss=0.3)
        db.commit()

    fetched = client.get(
        f"/api/v1/training-runs/{created['training_run_id']}", headers=h
    )

    assert fetched.status_code == 200
    body = fetched.json()
    assert body["status"] == "RUNNING"
    assert body["current_epoch"] == 1
    assert body["current_step"] == 42
    assert body["train_loss"] == 0.3
    assert body["eval_loss"] is None


def test_list_training_runs_empty(client, admin_token):
    response = client.get("/api/v1/training-runs", headers=auth_header(admin_token))

    assert response.status_code == 200
    body = response.json()
    assert body["items"] == []
    assert body["total"] == 0


def test_list_training_runs_paginated_response_shape(client, admin_token):
    _create_runs(client, admin_token, count=2)

    response = client.get(
        "/api/v1/training-runs",
        params={"page": 1, "size": 1},
        headers=auth_header(admin_token),
    )

    assert response.status_code == 200
    body = response.json()
    assert set(body.keys()) == {"items", "total", "page", "size", "pages"}
    assert len(body["items"]) == 1
    assert body["total"] == 2
    assert body["page"] == 1
    assert body["size"] == 1
    assert body["pages"] == 2


GOOD_ANSWER = " ".join(f"word{i}" for i in range(25))


def _valid_record(record_id="r1", user="What is the capital of France?"):
    return {
        "id": record_id,
        "messages": [
            {"role": "user", "content": user},
            {"role": "assistant", "content": GOOD_ANSWER},
        ],
        "metadata": {"source_dataset": "no_robots", "source_id": record_id},
    }


def test_create_training_run_returns_409_without_validation_report(client, admin_token):
    h = auth_header(admin_token)
    client.post(
        "/api/v1/datasets/no_robots/versions", json=DATASET_CREATE_REQUEST, headers=h
    )
    _mark_processed(client)

    response = client.post(
        "/api/v1/training-runs", json=TRAINING_RUN_CREATE_REQUEST, headers=h
    )

    assert response.status_code == 409
    assert response.json()["error"]["code"] == "VALIDATION_REQUIRED"
    listing = client.get("/api/v1/training-runs", headers=h).json()
    assert listing["total"] == 0


def test_create_training_run_blocked_by_fail_gate_and_no_row_created(
    client, admin_token
):
    h = auth_header(admin_token)
    client.post(
        "/api/v1/datasets/no_robots/versions", json=DATASET_CREATE_REQUEST, headers=h
    )
    _mark_processed(client)
    leaked = _valid_record(user="pertanyaan rahasia")
    eval_record = {"messages": [{"role": "user", "content": "pertanyaan rahasia"}]}

    report = client.post(
        "/api/v1/datasets/no_robots/versions/1/validate",
        json={"records": [leaked], "eval_records": [eval_record]},
        headers=h,
    )
    assert report.status_code == 201
    assert report.json()["gate_decision"] == "FAIL"

    response = client.post(
        "/api/v1/training-runs", json=TRAINING_RUN_CREATE_REQUEST, headers=h
    )

    assert response.status_code == 409
    assert response.json()["error"]["code"] == "VALIDATION_FAILED"
    listing = client.get("/api/v1/training-runs", headers=h).json()
    assert listing["total"] == 0


def test_create_training_run_uses_latest_report_for_gate(client, admin_token):
    h = auth_header(admin_token)
    client.post(
        "/api/v1/datasets/no_robots/versions", json=DATASET_CREATE_REQUEST, headers=h
    )
    _mark_processed(client)

    # First run: PASS (no eval overlap).
    first = _valid_record()
    pass_report = client.post(
        "/api/v1/datasets/no_robots/versions/1/validate",
        json={"records": [first]},
        headers=h,
    )
    assert pass_report.status_code == 201
    assert pass_report.json()["gate_decision"] == "PASS"

    # Second run: FAIL (leakage against the supplied eval set).
    eval_record = {
        "messages": [{"role": "user", "content": "What is the capital of France?"}]
    }
    fail_report = client.post(
        "/api/v1/datasets/no_robots/versions/1/validate",
        json={"records": [first], "eval_records": [eval_record]},
        headers=h,
    )
    assert fail_report.status_code == 201
    assert fail_report.json()["gate_decision"] == "FAIL"

    # The gate must consult the latest report (FAIL), not the earlier PASS.
    response = client.post(
        "/api/v1/training-runs", json=TRAINING_RUN_CREATE_REQUEST, headers=h
    )

    assert response.status_code == 409
    assert response.json()["error"]["code"] == "VALIDATION_FAILED"
    listing = client.get("/api/v1/training-runs", headers=h).json()
    assert listing["total"] == 0


def test_create_training_run_blocked_by_stored_eval_set_leakage(client, admin_token):
    """Training is blocked when the stored eval set's content leaks into the validated
    training records (issue #43): the eval set is real storage, not an inline request list."""
    h = auth_header(admin_token)
    client.post(
        "/api/v1/datasets/no_robots/versions", json=DATASET_CREATE_REQUEST, headers=h
    )
    _mark_processed(client)
    created = client.post(
        "/api/v1/eval-sets/domain-benchmark/versions",
        json={
            "records": [
                {"messages": [{"role": "user", "content": "pertanyaan rahasia"}]}
            ]
        },
        headers=h,
    )
    assert created.status_code == 201

    leaked = _valid_record(user="pertanyaan rahasia")
    report = client.post(
        "/api/v1/datasets/no_robots/versions/1/validate",
        json={
            "records": [leaked],
            "eval_set_id": "domain-benchmark",
            "eval_set_version": created.json()["version"],
        },
        headers=h,
    )
    assert report.status_code == 201, report.text
    assert report.json()["gate_decision"] == "FAIL"
    assert report.json()["dataset_statistics"]["leakage_check"]["checked_against"] == [
        "domain-benchmark@1"
    ]

    response = client.post(
        "/api/v1/training-runs", json=TRAINING_RUN_CREATE_REQUEST, headers=h
    )

    assert response.status_code == 409
    assert response.json()["error"]["code"] == "VALIDATION_FAILED"
    listing = client.get("/api/v1/training-runs", headers=h).json()
    assert listing["total"] == 0
