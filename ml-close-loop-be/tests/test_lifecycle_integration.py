"""End-to-end integration test for the closed-loop lifecycle (PRD §19 Integration Tests, §25).

Dataset -> Validation -> Training Run -> Model Registry -> Evaluation -> Promotion -> Deployment,
driven through the HTTP API with the mock training runner — no GPU, no VM (PRD §20).

Two steps have no HTTP trigger in this codebase and are driven directly instead:

* `DatasetVersion.status` PROCESSING -> PROCESSED. No intake/normalization pipeline exists in
  this PRD's story list (see progress.txt US-002/US-006), so nothing transitions it.
* The worker loop. `process_next_job` is called once instead of running `run_forever` in a
  thread — same code path the `worker` compose service runs, without the polling sleep.
"""

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine, select
from sqlalchemy.orm import Session
from sqlalchemy.pool import StaticPool

from app.db.base import Base
from app.db.session import get_db
from app.main import app
from app.models.dataset import DatasetVersion
from app.models.deployment import Deployment
from app.models.model import ModelVersion
from app.models.promotion import PromotionDecision
from app.services import deployment_service, model_service
from app.services.serving import MockServingBackend
from app.workers.mock_runner import MockTrainingRunner
from app.workers.training_worker import process_next_job

DATASET_ID = "no_robots"
MODEL_ID = "qwen-sft-domain-x"

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
        "peft_method": "dora",
        "load_in_4bit": False,
        "lora_r": 16,
        "lora_alpha": 16,
        "epochs": 2,
        "max_seq_length": 4096,
    },
    "triggered_by": "user-1",
}


@pytest.fixture
def client():
    engine = create_engine("sqlite://", connect_args={"check_same_thread": False}, poolclass=StaticPool)
    Base.metadata.create_all(engine)

    def override_get_db():
        with Session(engine) as session:
            yield session

    app.dependency_overrides[get_db] = override_get_db
    test_client = TestClient(app)
    test_client.engine = engine
    yield test_client
    app.dependency_overrides.clear()
    engine.dispose()


def _mark_processed(client, dataset_id, version):
    """No intake pipeline exists to do this — see module docstring."""

    with Session(client.engine) as db:
        row = db.scalar(
            select(DatasetVersion).where(
                DatasetVersion.dataset_id == dataset_id, DatasetVersion.version == version
            )
        )
        row.status = "PROCESSED"
        db.commit()


def _run_lifecycle(client, serving_backend=None):
    """Walk the whole lifecycle through the API, asserting each step, and return the IDs it produced."""

    # --- Dataset ---------------------------------------------------------------------------
    response = client.post(f"/datasets/{DATASET_ID}/versions", json=DATASET_CREATE_REQUEST)
    assert response.status_code == 201
    dataset_version = response.json()
    assert dataset_version["dataset_id"] == DATASET_ID
    assert dataset_version["version"] == 1

    _mark_processed(client, DATASET_ID, 1)

    # --- Validation ------------------------------------------------------------------------
    response = client.post(f"/datasets/{DATASET_ID}/versions/1/validate")
    assert response.status_code == 201
    report = response.json()
    assert report["dataset_id"] == DATASET_ID
    assert report["dataset_version"] == 1
    assert report["gate_decision"] == "PASS"

    latest = client.get(f"/datasets/{DATASET_ID}/versions/1/validation-reports/latest")
    assert latest.status_code == 200
    assert latest.json() == report

    # --- Training run ----------------------------------------------------------------------
    response = client.post("/training-runs", json=TRAINING_RUN_CREATE_REQUEST)
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

    response = client.get(f"/training-runs/{training_run_id}")
    assert response.status_code == 200
    completed = response.json()
    assert completed["status"] == "COMPLETED"
    assert completed["dataset_id"] == DATASET_ID
    assert completed["dataset_version"] == 1
    # The forward link training_run_id -> the model version the run produced (openapi.yaml).
    model_version = completed["model_version"]
    assert model_version == 1

    # --- Model registry --------------------------------------------------------------------
    response = client.get(f"/models/{MODEL_ID}/versions/{model_version}")
    assert response.status_code == 200
    record = response.json()
    assert record["status"] == "REGISTERED"
    assert record["training_run_id"] == training_run_id
    assert record["dataset_id"] == DATASET_ID
    assert record["dataset_version"] == 1
    assert record["artifacts"][0]["uri"] is not None

    assert client.get("/models").json() == [
        {"model_id": MODEL_ID, "latest_version": model_version, "status": "REGISTERED"}
    ]

    # --- Evaluation ------------------------------------------------------------------------
    evaluation_url = f"/models/{MODEL_ID}/versions/{model_version}/evaluation"
    partial = client.post(evaluation_url, json={"eval_loss_trend": {"this_version_eval_loss": 0.84}})
    assert partial.json()["status"] == "REGISTERED"  # partial evaluation does not qualify
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
    )
    response = client.post(
        evaluation_url, json={"general_domain_regression_check": {"checked": True, "regressions_found": []}}
    )
    assert response.status_code == 200
    assert response.json()["status"] == "EVALUATED"

    # --- Promotion -------------------------------------------------------------------------
    response = client.post(
        f"/models/{MODEL_ID}/versions/{model_version}/decisions",
        json={"decision": "PROMOTED", "decided_by": "reviewer-1", "rationale": "All three signals aligned."},
    )
    assert response.status_code == 201
    decision = response.json()
    decision_id = decision["decision_id"]
    assert decision["model_id"] == MODEL_ID
    assert decision["version"] == model_version
    # The evidence snapshot is frozen from the evaluation submitted above.
    assert decision["evidence_snapshot"]["eval_loss_trend"]["this_version_eval_loss"] == 0.84

    record = client.get(f"/models/{MODEL_ID}/versions/{model_version}").json()
    assert record["status"] == "PROMOTED"
    assert record["promotion_decision_ref"] == decision_id

    # --- Deployment ------------------------------------------------------------------------
    if serving_backend is None:
        response = client.post(f"/models/{MODEL_ID}/versions/{model_version}/deploy")
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
                db, model_service.get_model_version(db, MODEL_ID, model_version), serving_backend
            )
            db.commit()

    response = client.get(f"/models/{MODEL_ID}/deployment")
    assert response.status_code == 200
    status = response.json()
    assert status["model_id"] == MODEL_ID
    assert status["current_deployed_version"] == model_version
    assert status["status"] == "DEPLOYED"
    assert status["deployed_at"] is not None

    assert client.get(f"/models/{MODEL_ID}/versions/{model_version}").json()["status"] == "DEPLOYED"

    return {
        "dataset_id": DATASET_ID,
        "dataset_version": 1,
        "training_run_id": training_run_id,
        "model_id": MODEL_ID,
        "model_version": model_version,
        "decision_id": decision_id,
    }


def test_full_lifecycle_runs_end_to_end_through_the_api(client):
    ids = _run_lifecycle(client)

    assert ids["training_run_id"].startswith("run-")
    assert ids["decision_id"].startswith("decision-")


def test_full_lifecycle_reaches_the_serving_backend(client):
    backend = MockServingBackend()

    ids = _run_lifecycle(client, serving_backend=backend)

    assert backend.deployed == [(ids["model_id"], ids["model_version"])]


def test_full_lifecycle_persists_a_complete_lineage_chain(client):
    ids = _run_lifecycle(client)

    with Session(client.engine) as db:
        model_version = db.scalar(
            select(ModelVersion).where(
                ModelVersion.model_id == ids["model_id"], ModelVersion.version == ids["model_version"]
            )
        )

        # dataset_id/version <- training_run <- model version
        training_run = model_version.training_run
        assert training_run.training_run_id == ids["training_run_id"]
        assert training_run.dataset_version.dataset_id == ids["dataset_id"]
        assert training_run.dataset_version.version == ids["dataset_version"]
        # ...and the same link read forwards, which is what GET /training-runs reports.
        assert [v.version for v in training_run.model_versions] == [ids["model_version"]]
        # The dataset version that was validated is the one that was trained on.
        assert [r.dataset_version_id for r in training_run.dataset_version.validation_reports] == [
            training_run.dataset_version_id
        ]

        # evaluation -> promotion_decision_ref
        assert model_version.eval_loss_trend is not None
        assert model_version.qualitative_comparison is not None
        assert model_version.general_domain_regression_check is not None
        assert model_version.promotion_decision_ref == ids["decision_id"]

        decision = db.scalar(
            select(PromotionDecision).where(PromotionDecision.decision_id == ids["decision_id"])
        )
        assert decision.model_version_id == model_version.id

        # -> deployment_id
        deployment = db.scalar(select(Deployment).where(Deployment.status == "DEPLOYED"))
        assert deployment.deployment_id.startswith("deployment-")
        assert deployment.model_id == ids["model_id"]
        assert deployment.model_version == ids["model_version"]
        assert model_version.status == "DEPLOYED"
