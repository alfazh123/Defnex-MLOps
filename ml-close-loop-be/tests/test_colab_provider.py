"""ColabProvider tests (issue #77, PRD §9.6/§9.7/§9.8).

Tests the ColabProvider (ephemeral on-demand training) and the runner-link endpoint.
"""

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import Session

from app.config import settings
from app.db.base import Base
from app.models.compute_resource import ComputeResource
from app.providers.training_provider import ColabProvider
from app.schemas.dataset import DatasetVersionCreateRequest
from app.schemas.training import TrainingConfig, TrainingRunCreateRequest
from app.services import dataset_service, training_service

from tests.conftest import auth_header


@pytest.fixture(autouse=True)
def _artifact_sandbox(tmp_path, monkeypatch):
    monkeypatch.setattr(settings, "artifact_storage_dir", str(tmp_path))


@pytest.fixture
def db(tmp_path):
    engine = create_engine(
        f"sqlite:///{tmp_path / 'colab.db'}", connect_args={"timeout": 30}
    )
    Base.metadata.create_all(engine)
    with Session(engine) as session:
        yield session
    engine.dispose()


@pytest.fixture
def colab_resource(db):
    res = ComputeResource(
        name="colab-test",
        provider_type="colab",
        notebook_url="https://colab.research.google.com/drive/abc123",
    )
    db.add(res)
    db.flush()
    return res


def _create_run(db, resource_id=None):
    dataset_version = dataset_service.create_dataset_version(
        db,
        "test_ds",
        DatasetVersionCreateRequest(
            source_type="huggingface",
            source_dataset="test/dataset",
            source_commit_or_snapshot_date="2026-08-01",
            source_format="chatml",
        ),
    )
    return training_service.create_training_run(
        db,
        dataset_version,
        TrainingRunCreateRequest(
            dataset_id="test_ds",
            dataset_version=1,
            model_id="qwen-test",
            base_model="Qwen/Qwen3.8-27B",
            training_config=TrainingConfig(),
            compute_resource_id=resource_id,
        ),
    )


# --- ColabProvider submit ---


def test_colab_submit_creates_claim_token(db, colab_resource):
    """submit() generates a UUID claim token as external_job_id."""
    run = _create_run(db, colab_resource.id)
    provider = ColabProvider()

    external_job_id = provider.submit(db, run)

    import uuid

    uuid.UUID(external_job_id)  # raises if invalid
    assert run.external_job_id == external_job_id


# --- ColabProvider get_status ---


def test_colab_get_status_returns_running_when_not_claimed(db, colab_resource):
    """get_status returns RUNNING for a newly submitted job (waiting for notebook)."""
    run = _create_run(db, colab_resource.id)
    provider = ColabProvider()
    eid = provider.submit(db, run)

    status = provider.get_status(eid)
    assert status.status == "RUNNING"


def test_colab_get_status_completed(db, colab_resource):
    """get_status returns COMPLETED when notebook reports success."""
    run = _create_run(db, colab_resource.id)
    provider = ColabProvider()
    eid = provider.submit(db, run)

    provider.mark_completed(eid, artifact_uri="s3://artifacts/run-123")

    status = provider.get_status(eid)
    assert status.status == "COMPLETED"


def test_colab_get_status_failed(db, colab_resource):
    """get_status returns FAILED when notebook reports failure."""
    run = _create_run(db, colab_resource.id)
    provider = ColabProvider()
    eid = provider.submit(db, run)

    provider.mark_failed(eid, error_message="Out of memory")

    status = provider.get_status(eid)
    assert status.status == "FAILED"
    assert "Out of memory" in status.error_message


# --- ColabProvider cancel ---


def test_colab_cancel_sets_failed(db, colab_resource):
    """cancel() sets the claim to FAILED."""
    run = _create_run(db, colab_resource.id)
    provider = ColabProvider()
    eid = provider.submit(db, run)

    provider.cancel(eid)

    status = provider.get_status(eid)
    assert status.status == "FAILED"
    assert "Cancelled" in status.error_message


# --- ColabProvider collect_result ---


def test_colab_collect_result_returns_artifact_uri(db, colab_resource):
    """collect_result returns the MinIO URI for completed jobs."""
    run = _create_run(db, colab_resource.id)
    provider = ColabProvider()
    eid = provider.submit(db, run)

    provider.mark_completed(eid, artifact_uri="s3://artifacts/run-456")

    result = provider.collect_result(eid)
    assert result == "s3://artifacts/run-456"


def test_colab_collect_result_raises_on_pending(db, colab_resource):
    """collect_result raises for non-completed jobs."""
    run = _create_run(db, colab_resource.id)
    provider = ColabProvider()
    eid = provider.submit(db, run)

    with pytest.raises(RuntimeError, match="not yet completed"):
        provider.collect_result(eid)


# --- Runner link endpoint ---


def test_runner_link_returns_notebook_url(client, admin_token):
    """GET runner-link returns notebook_url for a colab resource."""
    resp = client.post(
        "/api/v1/compute-resources",
        json={
            "name": "colab-api-test",
            "provider_type": "colab",
            "notebook_url": "https://colab.research.google.com/drive/xyz",
        },
        headers=auth_header(admin_token),
    )
    assert resp.status_code == 201
    resource_id = resp.json()["id"]

    resp = client.get(
        f"/api/v1/compute-resources/{resource_id}/runner-link",
        headers=auth_header(admin_token),
    )
    assert resp.status_code == 200
    data = resp.json()
    assert data["notebook_url"] == "https://colab.research.google.com/drive/xyz"
    assert data["resource_id"] == resource_id


def test_runner_link_rejects_non_colab_resource(client, admin_token):
    """Runner link returns 400 for non-Colab resources."""
    resp = client.post(
        "/api/v1/compute-resources",
        json={"name": "local-test", "provider_type": "local"},
        headers=auth_header(admin_token),
    )
    resource_id = resp.json()["id"]

    resp = client.get(
        f"/api/v1/compute-resources/{resource_id}/runner-link",
        headers=auth_header(admin_token),
    )
    assert resp.status_code == 400
    assert resp.json()["error"]["code"] == "NOT_COLAB_RESOURCE"


def test_runner_link_rejects_no_notebook_url(client, admin_token):
    """Runner link returns 400 when notebook_url is not configured."""
    resp = client.post(
        "/api/v1/compute-resources",
        json={"name": "colab-no-url", "provider_type": "colab"},
        headers=auth_header(admin_token),
    )
    resource_id = resp.json()["id"]

    resp = client.get(
        f"/api/v1/compute-resources/{resource_id}/runner-link",
        headers=auth_header(admin_token),
    )
    assert resp.status_code == 400
    assert resp.json()["error"]["code"] == "NO_NOTEBOOK_URL"


def test_runner_link_404(client, admin_token):
    """Runner link returns 404 for non-existent resource."""
    resp = client.get(
        "/api/v1/compute-resources/99999/runner-link",
        headers=auth_header(admin_token),
    )
    assert resp.status_code == 404


# --- Provider factory ---


def test_provider_factory_selects_colab_by_type():
    """get_provider_for_resource returns ColabProvider for colab type."""
    from app.workers.training_worker import get_provider_for_resource

    res = ComputeResource(name="test", provider_type="colab")
    provider = get_provider_for_resource(res)
    assert isinstance(provider, ColabProvider)


# --- AC: No Google password stored ---


def test_no_google_password_stored(client, admin_token):
    """AC: Password Google TIDAK pernah disimpan (PRD §9.7)."""
    resp = client.post(
        "/api/v1/compute-resources",
        json={
            "name": "colab-security",
            "provider_type": "colab",
            "notebook_url": "https://colab.research.google.com/drive/abc",
        },
        headers=auth_header(admin_token),
    )
    assert resp.status_code == 201
    data = resp.json()
    assert "google" not in (data.get("credential_ref") or "").lower()
    assert data.get("ssh_host") is None
    assert data.get("ssh_username") is None


# --- AC: Colab = on-demand ---


def test_colab_is_on_demand_not_permanent():
    """AC: Colab dimodelkan sebagai on-demand, bukan worker permanen (PRD §9.6)."""
    provider = ColabProvider()
    assert not hasattr(provider, "_host")
    assert not hasattr(provider, "_ssh")
    assert not hasattr(provider, "_jobs")
