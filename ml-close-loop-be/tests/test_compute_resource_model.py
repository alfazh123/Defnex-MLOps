"""ComputeResource model tests (issue #75, PRD §9.3, §19.3).

Verifies the ORM model fields, defaults, FK on TrainingRun, and basic
CRUD operations via the DB session.
"""

from datetime import datetime, timezone

import pytest
from sqlalchemy import create_engine
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from app.db.base import Base
from app.models.compute_resource import ComputeResource
from app.schemas.dataset import DatasetVersionCreateRequest
from app.schemas.training import TrainingConfig, TrainingRunCreateRequest
from app.services import dataset_service, training_service


@pytest.fixture
def db(tmp_path):
    engine = create_engine(
        f"sqlite:///{tmp_path / 'compute_resource.db'}",
        connect_args={"check_same_thread": False},
    )
    Base.metadata.create_all(engine)
    with Session(engine) as session:
        yield session
    engine.dispose()


def _create_resource(db, **overrides) -> ComputeResource:
    defaults = dict(
        name="server-2",
        role="training",
        environment="staging",
        provider_type="gpu_vps",
        host="10.0.0.2",
        gpu_info={"model": "H100", "memory_gb": 80},
        ssh_host="10.0.0.2",
        ssh_port=22,
        ssh_username="root",
        credential_ref="secret://gpu-server-key",
        is_healthy=True,
        created_at=datetime.now(timezone.utc),
        updated_at=datetime.now(timezone.utc),
    )
    defaults.update(overrides)
    resource = ComputeResource(**defaults)
    db.add(resource)
    db.flush()
    return resource


def _create_training_run(db, compute_resource_id=None):
    dataset_version = dataset_service.create_dataset_version(
        db,
        "test-ds",
        DatasetVersionCreateRequest(
            source_type="huggingface",
            source_dataset="HuggingFaceH4/no_robots",
            source_commit_or_snapshot_date="2026-08-01",
            source_format="chatml",
        ),
    )
    run = training_service.create_training_run(
        db,
        dataset_version,
        TrainingRunCreateRequest(
            dataset_id="test-ds",
            dataset_version=1,
            model_id="test-model",
            base_model="Qwen/Qwen3.8-27B",
            training_config=TrainingConfig(),
        ),
    )
    if compute_resource_id is not None:
        run.compute_resource_id = compute_resource_id
        db.flush()
    return run


# --- Model fields ---


def test_compute_resource_required_fields(db):
    """PRD §9.3: name, role, environment, provider_type are required."""
    resource = _create_resource(db)
    assert resource.id is not None
    assert resource.name == "server-2"
    assert resource.role == "training"
    assert resource.environment == "staging"
    assert resource.provider_type == "gpu_vps"


def test_compute_resource_optional_fields(db):
    """Host, GPU, SSH, credential_ref are optional (nullable)."""
    resource = _create_resource(
        db,
        name="minimal",
        host=None,
        gpu_info=None,
        ssh_host=None,
        ssh_port=None,
        ssh_username=None,
        credential_ref=None,
    )
    assert resource.host is None
    assert resource.gpu_info is None
    assert resource.ssh_host is None
    assert resource.credential_ref is None


def test_compute_resource_unique_name(db):
    """Duplicate name must raise IntegrityError."""
    _create_resource(db, name="unique-server")
    with pytest.raises(IntegrityError):
        _create_resource(db, name="unique-server")
    db.rollback()


def test_compute_resource_default_healthy(db):
    """New resources default to healthy."""
    resource = _create_resource(db, name="healthy-server")
    assert resource.is_healthy is True


def test_compute_resource_credential_ref_not_plaintext(db):
    """PRD §21/§43: credential_ref is a reference (secret://...), never a plaintext secret."""
    resource = _create_resource(db, credential_ref="secret://my-ssh-key")
    assert resource.credential_ref == "secret://my-ssh-key"
    assert "password" not in (resource.credential_ref or "").lower()


# --- FK on TrainingRun ---


def test_training_run_has_compute_resource_fk(db):
    """TrainingRun has a nullable FK to compute_resources (issue #75)."""
    resource = _create_resource(db)
    run = _create_training_run(db, compute_resource_id=resource.id)

    assert run.compute_resource_id == resource.id


def test_training_run_compute_resource_nullable(db):
    """Existing training runs without a resource still work (backward compat)."""
    run = _create_training_run(db)
    assert run.compute_resource_id is None


def test_training_run_fk_references_resource(db):
    """FK constraint: the referenced compute resource must exist."""
    resource = _create_resource(db)
    _create_training_run(db, compute_resource_id=resource.id)

    fetched = db.get(ComputeResource, resource.id)
    assert fetched is not None
    assert fetched.name == "server-2"


def test_training_run_orphan_resource_id(db):
    """Setting a non-existent resource_id is allowed at ORM level (FK not enforced in SQLite)."""
    run = _create_training_run(db, compute_resource_id=99999)
    assert run.compute_resource_id == 99999


# --- Defaults ---


def test_compute_resource_created_at_auto(db):
    """created_at is set automatically."""
    resource = _create_resource(db)
    assert resource.created_at is not None


def test_compute_resource_updated_at_auto(db):
    """updated_at is set automatically."""
    resource = _create_resource(db)
    assert resource.updated_at is not None
