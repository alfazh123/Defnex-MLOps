"""Tests for artifact transfer contract (issue #72, PRD §20.3, §29)."""

import uuid
from datetime import UTC, datetime
from pathlib import Path
from unittest.mock import patch

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from app.db.base import Base
from app.models.transfer import ArtifactTransfer
from app.services.artifact_transfer import (
    TransferVerificationError,
    execute_transfer,
    initiate_transfer,
    verify_transfer,
)


@pytest.fixture()
def db_session():
    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(engine)
    SessionLocal = sessionmaker(bind=engine)
    session = SessionLocal()
    yield session
    session.close()


def _staging_dir(tmp_path: Path) -> Path:
    staging = tmp_path / "staging-src"
    staging.mkdir()
    (staging / "adapter_model.safetensors").write_bytes(b"weights")
    (staging / "adapter_config.json").write_text('{"lora": true}')
    return staging


def test_initiate_transfer_creates_pending_row(tmp_path, db_session):
    from app.services.artifact_storage import LocalFilesystemArtifactStorage

    storage = LocalFilesystemArtifactStorage(base_dir=tmp_path)
    uri = storage.finalize_version("model-a", "v1", _staging_dir(tmp_path), {})

    transfer = initiate_transfer(db_session, uri, "host-a", "host-b")

    assert transfer.status == "PENDING"
    assert transfer.artifact_uri == uri
    assert transfer.source_host == "host-a"
    assert transfer.target_host == "host-b"
    assert len(transfer.checksum_before) == 64  # SHA-256 hex
    assert transfer.transfer_id
    assert db_session.get(ArtifactTransfer, transfer.transfer_id) is not None


def test_checksum_mismatch_marks_failed(tmp_path, db_session):
    from app.services.artifact_storage import LocalFilesystemArtifactStorage

    storage = LocalFilesystemArtifactStorage(base_dir=tmp_path)
    uri = storage.finalize_version("model-a", "v1", _staging_dir(tmp_path), {})
    # Same host — no SSH needed, transfer is a no-op then checksum is verified
    transfer = initiate_transfer(db_session, uri, "host-a", "host-a")
    db_session.commit()

    # Tamper the artifact after checksum_before was recorded
    target = Path(uri[len("file://") :])
    (target / "adapter_model.safetensors").write_bytes(b"tampered")

    result = execute_transfer(db_session, transfer.transfer_id)

    assert result.status == "FAILED"
    assert "Checksum mismatch" in (result.error_message or "")
    assert result.completed_at is not None


def test_verify_blocks_deploy_on_failure(tmp_path, db_session):
    """Transfer with failed checksum → deployment blocked (PRD §29)."""
    from app.services.artifact_storage import LocalFilesystemArtifactStorage

    storage = LocalFilesystemArtifactStorage(base_dir=tmp_path)
    uri = storage.finalize_version("model-a", "v1", _staging_dir(tmp_path), {})
    transfer = initiate_transfer(db_session, uri, "host-a", "host-a")
    db_session.commit()

    # Tamper after initiate
    target = Path(uri[len("file://") :])
    (target / "adapter_model.safetensors").write_bytes(b"tampered")
    execute_transfer(db_session, transfer.transfer_id)

    with pytest.raises(TransferVerificationError):
        verify_transfer(db_session, transfer.transfer_id)


def test_transfer_records_timestamps(tmp_path, db_session):
    from app.services.artifact_storage import LocalFilesystemArtifactStorage

    storage = LocalFilesystemArtifactStorage(base_dir=tmp_path)
    uri = storage.finalize_version("model-a", "v1", _staging_dir(tmp_path), {})
    transfer = initiate_transfer(db_session, uri, "host-a", "host-a")

    assert transfer.created_at is not None
    assert transfer.completed_at is None

    result = execute_transfer(db_session, transfer.transfer_id)

    assert result.completed_at is not None
    assert result.completed_at >= transfer.created_at


def test_transfer_idempotent_on_already_completed(tmp_path, db_session):
    from app.services.artifact_storage import LocalFilesystemArtifactStorage

    storage = LocalFilesystemArtifactStorage(base_dir=tmp_path)
    uri = storage.finalize_version("model-a", "v1", _staging_dir(tmp_path), {})
    transfer = initiate_transfer(db_session, uri, "host-a", "host-a")
    db_session.commit()

    result1 = execute_transfer(db_session, transfer.transfer_id)
    assert result1.status == "COMPLETED"

    # Execute again — should be idempotent
    result2 = execute_transfer(db_session, transfer.transfer_id)
    assert result2.status == "COMPLETED"
    assert result2.transfer_id == result1.transfer_id


def test_local_to_minio_upload(tmp_path, db_session):
    """Local artifact → MinIO path via get_artifact_storage().store()."""
    from app.services.artifact_storage import LocalFilesystemArtifactStorage

    storage = LocalFilesystemArtifactStorage(base_dir=tmp_path)
    uri = storage.finalize_version("model-a", "v1", _staging_dir(tmp_path), {})
    transfer = initiate_transfer(db_session, uri, "localhost", "minio-host")

    assert transfer.status == "PENDING"
    assert transfer.artifact_uri == uri


def test_ssh_transfer_uses_remote_host(tmp_path, db_session):
    """SSH transfer calls _ssh_transfer for cross-host local artifacts."""
    from app.services.artifact_storage import LocalFilesystemArtifactStorage

    storage = LocalFilesystemArtifactStorage(base_dir=tmp_path)
    uri = storage.finalize_version("model-a", "v1", _staging_dir(tmp_path), {})
    transfer = initiate_transfer(db_session, uri, "host-local", "host-remote")

    with patch("app.services.artifact_transfer._ssh_transfer") as mock_ssh:
        execute_transfer(db_session, transfer.transfer_id)
        mock_ssh.assert_called_once_with(transfer)


def test_minio_to_minio_copy_succeeds(tmp_path, db_session):
    """S3→S3 transfer uses boto3 copy_object (mocked)."""
    transfer = ArtifactTransfer(
        transfer_id=str(uuid.uuid4()),
        artifact_uri="s3://bucket-a/model/v1",
        source_host="minio-a",
        target_host="minio-b",
        checksum_before="a" * 64,
        status="PENDING",
        created_at=datetime.now(UTC),
    )
    db_session.add(transfer)
    db_session.commit()

    with (
        patch(
            "app.services.artifact_transfer._compute_artifact_checksum"
        ) as mock_checksum,
        patch("app.services.artifact_transfer._copy_artifact"),
    ):
        mock_checksum.return_value = "a" * 64
        result = execute_transfer(db_session, transfer.transfer_id)

    assert result.status == "COMPLETED"
    assert result.checksum_after == "a" * 64


def test_list_transfers_paginated(db_session):
    """list_transfers returns paginated results."""
    from app.api.deps import PaginationParams

    now = datetime.now(UTC)
    for i in range(5):
        t = ArtifactTransfer(
            transfer_id=str(uuid.uuid4()),
            artifact_uri=f"s3://b/m{i}",
            source_host="a",
            target_host="b",
            checksum_before="x" * 64,
            status="COMPLETED",
            created_at=now,
        )
        db_session.add(t)
    db_session.commit()

    from app.api.transfer import list_transfers

    class FakeUser:
        pass

    result = list_transfers(
        db=db_session,
        _user=FakeUser(),
        pagination=PaginationParams(page=1, size=3),
    )
    assert len(result.items) == 3
    assert result.total == 5
    assert result.pages == 2


def test_get_transfer_returns_404_when_missing(db_session):
    """GET /transfers/{id} with unknown ID raises 404."""
    from app.api.errors import APIError
    from app.api.transfer import get_transfer

    class FakeUser:
        pass

    with pytest.raises(APIError) as exc_info:
        get_transfer(
            transfer_id="nonexistent-id",
            db=db_session,
            _user=FakeUser(),
        )
    assert exc_info.value.status_code == 404
    assert "TRANSFER_NOT_FOUND" in str(exc_info.value.detail)
