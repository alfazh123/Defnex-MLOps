"""End-to-end integration test for the closed-loop lifecycle (PRD §19 Integration Tests, §25).

Dataset -> Validation -> Training Run -> Model Registry -> Evaluation -> Promotion -> Deployment,
driven through the HTTP API with the mock training runner — no GPU, no VM (PRD §20).

One step has no HTTP trigger in this codebase and is driven directly instead:

* The worker loop. `process_next_job` is called once instead of running `run_forever` in a
  thread — same code path the `worker` compose service runs, without the polling sleep.

Note: Dataset versions are created as PROCESSED directly (no intake pipeline exists),
so validation is immediately available after dataset creation.
"""

from sqlalchemy import select
from sqlalchemy.orm import Session
from app.models.deployment import Deployment
from app.models.model import ModelVersion
from app.models.promotion import PromotionDecision
from app.services import deployment_service, model_service
from app.services.serving import MockServingBackend
from app.workers.mock_runner import MockTrainingRunner
from app.workers.training_worker import process_next_job
from tests.conftest import auth_header

DATASET_ID = "no_robots"
MODEL_ID = "qwen-sft-domain-x"

GOOD_ANSWER = " ".join(f"word{i}" for i in range(25))

VALID_RECORD = {
    "id": "r1",
    "messages": [
        {"role": "user", "content": "Apa itu quantum computing?"},
        {"role": "assistant", "content": GOOD_ANSWER},
    ],
    "metadata": {"source_dataset": DATASET_ID, "source_id": "sq-1"},
}

DATASET_CREATE_REQUEST = {
    "source_type": "huggingface",
    "source_dataset": "HuggingFaceH4/no_robots",
    "source_commit_or_snapshot_date": "2026-08-01",
    "source_format": "chatml",
}

TRAINING_RUN_CREATE_REQUEST = {
    "dataset_id": DATASET_ID,
    "dataset_version": 1,
    "model_id": MODEL_ID,
    "base_model": "Qwen/Qwen3.8-27B",
    "training_config": {
        "peft_method": "lora",
        "load_in_4bit": False,
        "lora_r": 16,
        "lora_alpha": 16,
        "epochs": 2,
        "max_seq_length": 4096,
    },
    "triggered_by": "user-1",
}


def _mark_processed(client, dataset_id, version):
    """Versions are created as PROCESSED directly; this is now a no-op."""


def _run_lifecycle(client, admin_token, serving_backend=None, training_request=None):
    """Walk the whole lifecycle through the API, asserting each step, and return the IDs it produced."""
    h = auth_header(admin_token)

    # --- Dataset ---------------------------------------------------------------------------
    response = client.post(
        f"/api/v1/datasets/{DATASET_ID}/versions",
        json=DATASET_CREATE_REQUEST,
        headers=h,
    )
    assert response.status_code == 201
    dataset_version = response.json()
    assert dataset_version["dataset_id"] == DATASET_ID
    assert dataset_version["version"] == 1

    _mark_processed(client, DATASET_ID, 1)

    # --- Validation ------------------------------------------------------------------------
    response = client.post(
        f"/api/v1/datasets/{DATASET_ID}/versions/1/validate",
        json={"records": [VALID_RECORD]},
        headers=h,
    )
    assert response.status_code == 201
    report = response.json()
    assert report["dataset_id"] == DATASET_ID
    assert report["dataset_version"] == 1
    assert report["record_count"] == 1
    assert report["status_counts"]["VALID"] == 1
    assert report["per_record_errors"] == [[]]
    assert report["gate_decision"] == "PASS"

    latest = client.get(
        f"/api/v1/datasets/{DATASET_ID}/versions/1/validation-reports/latest", headers=h
    )
    assert latest.status_code == 200
    assert latest.json() == report

    # --- Training run ----------------------------------------------------------------------
    response = client.post(
        "/api/v1/training-runs",
        json=training_request or TRAINING_RUN_CREATE_REQUEST,
        headers=h,
    )
    assert response.status_code == 201
    training_run = response.json()
    training_run_id = training_run["training_run_id"]
    assert training_run["status"] == "PENDING"
    assert training_run["model_version"] is None

    # The worker runs the job and issues the internal Register call, exactly as the `worker`
    # compose service does — mock runner, so no GPU is needed (PRD §20).
    with Session(client.engine) as db:
        processed = process_next_job(db, MockTrainingRunner())
        db.commit()
        assert processed.training_run_id == training_run_id

    response = client.get(f"/api/v1/training-runs/{training_run_id}", headers=h)
    assert response.status_code == 200
    completed = response.json()
    assert completed["status"] == "COMPLETED"
    assert completed["dataset_id"] == DATASET_ID
    assert completed["dataset_version"] == 1
    # The forward link training_run_id -> the model version the run produced (openapi.yaml).
    model_version = completed["model_version"]
    assert model_version == 1

    # --- Model registry --------------------------------------------------------------------
    response = client.get(
        f"/api/v1/models/{MODEL_ID}/versions/{model_version}", headers=h
    )
    assert response.status_code == 200
    record = response.json()
    assert record["status"] == "REGISTERED"
    assert record["training_run_id"] == training_run_id
    assert record["dataset_id"] == DATASET_ID
    assert record["dataset_version"] == 1
    assert record["artifacts"][0]["uri"] is not None

    assert client.get("/api/v1/models", headers=h).json() == [
        {"model_id": MODEL_ID, "latest_version": model_version, "status": "REGISTERED"}
    ]

    # --- Evaluation ------------------------------------------------------------------------
    evaluation_url = f"/api/v1/models/{MODEL_ID}/versions/{model_version}/evaluation"
    partial = client.post(
        evaluation_url,
        json={"eval_loss_trend": {"this_version_eval_loss": 0.84}},
        headers=h,
    )
    assert (
        partial.json()["status"] == "REGISTERED"
    )  # partial evaluation does not qualify
    client.post(
        evaluation_url,
        json={
            "qualitative_comparison": {
                "question_table_version": 1,
                "wins": 13,
                "losses": 5,
                "ties": 2,
                "total": 20,
            }
        },
        headers=h,
    )
    response = client.post(
        evaluation_url,
        json={
            "general_domain_regression_check": {
                "checked": True,
                "regressions_found": [],
            }
        },
        headers=h,
    )
    assert response.status_code == 200
    assert response.json()["status"] == "EVALUATED"

    # --- Promotion -------------------------------------------------------------------------
    response = client.post(
        f"/api/v1/models/{MODEL_ID}/versions/{model_version}/decisions",
        json={
            "decision": "PROMOTED",
            "decided_by": "reviewer-1",
            "rationale": "All three signals aligned.",
        },
        headers=h,
    )
    assert response.status_code == 201
    decision = response.json()
    decision_id = decision["decision_id"]
    assert decision["model_id"] == MODEL_ID
    assert decision["version"] == model_version
    # The evidence snapshot is frozen from the evaluation submitted above.
    assert (
        decision["evidence_snapshot"]["eval_loss_trend"]["this_version_eval_loss"]
        == 0.84
    )

    record = client.get(
        f"/api/v1/models/{MODEL_ID}/versions/{model_version}", headers=h
    ).json()
    assert record["status"] == "PROMOTED"
    assert record["promotion_decision_ref"] == decision_id

    # --- Deployment ------------------------------------------------------------------------
    if serving_backend is None:
        response = client.post(
            f"/api/v1/models/{MODEL_ID}/versions/{model_version}/deploy", headers=h
        )
        assert response.status_code == 200
        assert response.json() == {
            "model_id": MODEL_ID,
            "current_deployed_version": model_version,
            "previous_deployed_version": None,
        }
    else:
        # Same service call the endpoint makes, with an injectable backend, to assert the
        # lifecycle actually reached the serving boundary.
        with Session(client.engine) as db:
            deployment_service.deploy(
                db,
                model_service.get_model_version(db, MODEL_ID, model_version),
                serving_backend,
            )
            db.commit()

    response = client.get(f"/api/v1/models/{MODEL_ID}/deployment", headers=h)
    assert response.status_code == 200
    status = response.json()
    assert status["model_id"] == MODEL_ID
    assert status["current_deployed_version"] == model_version
    assert status["status"] == "DEPLOYED"
    assert status["deployed_at"] is not None

    assert (
        client.get(
            f"/api/v1/models/{MODEL_ID}/versions/{model_version}", headers=h
        ).json()["status"]
        == "DEPLOYED"
    )

    return {
        "dataset_id": DATASET_ID,
        "dataset_version": 1,
        "training_run_id": training_run_id,
        "model_id": MODEL_ID,
        "model_version": model_version,
        "decision_id": decision_id,
    }


def test_full_lifecycle_runs_end_to_end_through_the_api(client, admin_token):
    ids = _run_lifecycle(client, admin_token)

    assert ids["training_run_id"].startswith("run-")
    assert ids["decision_id"].startswith("decision-")


def test_full_lifecycle_reaches_the_serving_backend(client, admin_token):
    backend = MockServingBackend()

    ids = _run_lifecycle(client, admin_token, serving_backend=backend)

    assert backend.deployed == [(ids["model_id"], ids["model_version"])]


def test_full_lifecycle_persists_a_complete_lineage_chain(client, admin_token):
    ids = _run_lifecycle(client, admin_token)

    with Session(client.engine) as db:
        model_version = db.scalar(
            select(ModelVersion).where(
                ModelVersion.model_id == ids["model_id"],
                ModelVersion.version == ids["model_version"],
            )
        )

        # dataset_id/version <- training_run <- model version
        training_run = model_version.training_run
        assert training_run.training_run_id == ids["training_run_id"]
        assert training_run.dataset_version.dataset_id == ids["dataset_id"]
        assert training_run.dataset_version.version == ids["dataset_version"]
        # ...and the same link read forwards, which is what GET /training-runs reports.
        assert [v.version for v in training_run.model_versions] == [
            ids["model_version"]
        ]
        # The dataset version that was validated is the one that was trained on.
        assert [
            r.dataset_version_id
            for r in training_run.dataset_version.validation_reports
        ] == [training_run.dataset_version_id]

        # evaluation -> promotion_decision_ref
        assert model_version.eval_loss_trend is not None
        assert model_version.qualitative_comparison is not None
        assert model_version.general_domain_regression_check is not None
        assert model_version.promotion_decision_ref == ids["decision_id"]

        decision = db.scalar(
            select(PromotionDecision).where(
                PromotionDecision.decision_id == ids["decision_id"]
            )
        )
        assert decision.model_version_id == model_version.id

        # -> deployment_id
        deployment = db.scalar(
            select(Deployment).where(Deployment.status == "DEPLOYED")
        )
        assert deployment.deployment_id.startswith("deployment-")
        assert deployment.model_id == ids["model_id"]
        assert deployment.model_version == ids["model_version"]
        assert model_version.status == "DEPLOYED"


class _RaisingRunner:
    def run(self, training_run):
        raise RuntimeError("boom: cuda oom")


def test_lifecycle_training_failure_path(client, admin_token):
    h = auth_header(admin_token)
    client.post(
        f"/api/v1/datasets/{DATASET_ID}/versions",
        json=DATASET_CREATE_REQUEST,
        headers=h,
    )
    _mark_processed(client, DATASET_ID, 1)
    report = client.post(
        f"/api/v1/datasets/{DATASET_ID}/versions/1/validate",
        json={"records": [VALID_RECORD]},
        headers=h,
    )
    assert report.status_code == 201
    assert report.json()["gate_decision"] == "PASS"
    created = client.post(
        "/api/v1/training-runs", json=TRAINING_RUN_CREATE_REQUEST, headers=h
    )
    assert created.status_code == 201
    training_run_id = created.json()["training_run_id"]

    with Session(client.engine) as db:
        processed = process_next_job(db, _RaisingRunner())
        db.commit()
        assert processed.training_run_id == training_run_id
        assert processed.status == "FAILED"
        assert processed.error_message == "boom: cuda oom"


def test_lifecycle_double_deploy_same_version(client, admin_token):
    ids = _run_lifecycle(client, admin_token)

    # The HTTP endpoint gates on PROMOTED, so the second deploy of an already-DEPLOYED
    # version goes through the service directly — the pointer-move logic has no such guard.
    with Session(client.engine) as db:
        model_version = db.scalar(
            select(ModelVersion).where(
                ModelVersion.model_id == MODEL_ID,
                ModelVersion.version == ids["model_version"],
            )
        )
        deployment, previous = deployment_service.deploy(db, model_version)
        db.commit()
        rows = db.scalars(
            select(Deployment).where(Deployment.model_id == MODEL_ID)
        ).all()
        record = db.scalar(
            select(ModelVersion).where(
                ModelVersion.model_id == MODEL_ID,
                ModelVersion.version == ids["model_version"],
            )
        )
        assert previous is None
        assert deployment.model_version == ids["model_version"]
        assert record.status == "DEPLOYED"
        assert len(rows) == 2


def test_lifecycle_with_custom_training_config(client, admin_token):
    custom = dict(TRAINING_RUN_CREATE_REQUEST)
    custom["training_config"] = dict(TRAINING_RUN_CREATE_REQUEST["training_config"])
    custom["training_config"]["peft_method"] = "lora"
    custom["training_config"]["lora_r"] = 32

    ids = _run_lifecycle(client, admin_token, training_request=custom)

    run = client.get(
        f"/api/v1/training-runs/{ids['training_run_id']}",
        headers=auth_header(admin_token),
    ).json()
    assert run["training_config"]["peft_method"] == "lora"
    assert run["training_config"]["lora_r"] == 32
    record = client.get(
        f"/api/v1/models/{MODEL_ID}/versions/{ids['model_version']}",
        headers=auth_header(admin_token),
    ).json()
    assert record["training_config"]["lora_r"] == 32


def test_lifecycle_sequential_multi_version(client, admin_token):
    h = auth_header(admin_token)
    model_version_of = {}

    for dataset_version in (1, 2):
        client.post(
            f"/api/v1/datasets/{DATASET_ID}/versions",
            json=DATASET_CREATE_REQUEST,
            headers=h,
        )
        _mark_processed(client, DATASET_ID, dataset_version)
        report = client.post(
            f"/api/v1/datasets/{DATASET_ID}/versions/{dataset_version}/validate",
            json={"records": [VALID_RECORD]},
            headers=h,
        )
        assert report.status_code == 201
        assert report.json()["gate_decision"] == "PASS"
        request = dict(TRAINING_RUN_CREATE_REQUEST)
        request["dataset_version"] = dataset_version
        created = client.post("/api/v1/training-runs", json=request, headers=h)
        assert created.status_code == 201

        with Session(client.engine) as db:
            processed = process_next_job(db, MockTrainingRunner())
            db.commit()
            model_version_of[dataset_version] = processed.model_versions[-1].version

    assert model_version_of == {1: 1, 2: 2}
    for version in (1, 2):
        record = client.get(
            f"/api/v1/models/{MODEL_ID}/versions/{version}", headers=h
        ).json()
        assert record["dataset_version"] == version
        assert record["status"] == "REGISTERED"

    assert client.get("/api/v1/models", headers=h).json() == [
        {"model_id": MODEL_ID, "latest_version": 2, "status": "REGISTERED"}
    ]
