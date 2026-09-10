"""Tests for artifact transfer service (issue #72, PRD §20.3)."""

import pytest
from unittest.mock import patch
from datetime import UTC, datetime

from app.db.base import Base
from app.models.transfer import ArtifactTransfer
from app.services.artifact_transfer import (
    TransferVerificationError,
    initiate_transfer,
    execute_transfer,
    verify_transfer,
    list_transfers,
)


@pytest.fixture()
def db_session():
    from sqlalchemy import create_engine
    from sqlalchemy.orm import sessionmaker

    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(engine)
    Session = sessionmaker(bind=engine)
    session = Session()
    yield session
    session.close()


@pytest.fixture()
def sample_transfer(db_session):
    transfer = ArtifactTransfer(
        transfer_id="test-001",
        artifact_uri="file:///tmp/test-artifact",
        source_host="host-a",
        target_host="host-b",
        checksum_before="abc123",
        status="PENDING",
        created_at=datetime.now(UTC),
    )
    db_session.add(transfer)
    db_session.commit()
    return transfer


class TestInitiateTransfer:
    @patch(
        "app.services.artifact_transfer._compute_artifact_checksum",
        return_value="sha256hash",
    )
    def test_creates_pending_record(self, mock_checksum, db_session):
        transfer = initiate_transfer(db_session, "file:///tmp/art", "host-a", "host-b")
        assert transfer.status == "PENDING"
        assert transfer.checksum_before == "sha256hash"
        assert transfer.source_host == "host-a"
        assert transfer.target_host == "host-b"
        assert transfer.transfer_id is not None

    @patch(
        "app.services.artifact_transfer._compute_artifact_checksum", return_value="hash"
    )
    def test_records_timestamp(self, mock_checksum, db_session):
        transfer = initiate_transfer(db_session, "file:///tmp/art", "a", "b")
        assert transfer.created_at is not None


class TestExecuteTransfer:
    def test_idempotent_on_completed(self, db_session, sample_transfer):
        sample_transfer.status = "COMPLETED"
        db_session.commit()
        result = execute_transfer(db_session, "test-001")
        assert result.status == "COMPLETED"

    @patch("app.services.artifact_transfer._copy_artifact")
    @patch(
        "app.services.artifact_transfer._compute_artifact_checksum",
        return_value="newhash",
    )
    def test_checksum_mismatch_marks_failed(
        self, mock_checksum, mock_copy, db_session, sample_transfer
    ):
        result = execute_transfer(db_session, "test-001")
        assert result.status == "FAILED"
        assert "Checksum mismatch" in result.error_message

    @patch("app.services.artifact_transfer._copy_artifact")
    @patch(
        "app.services.artifact_transfer._compute_artifact_checksum",
        return_value="abc123",
    )
    def test_checksum_match_marks_completed(
        self, mock_checksum, mock_copy, db_session, sample_transfer
    ):
        result = execute_transfer(db_session, "test-001")
        assert result.status == "COMPLETED"
        assert result.completed_at is not None

    def test_not_found_raises(self, db_session):
        with pytest.raises(ValueError, match="not found"):
            execute_transfer(db_session, "nonexistent")


class TestVerifyTransfer:
    @patch("app.services.artifact_transfer._compute_artifact_checksum", return_value="abc123")
    def test_completed_transfer_passes(self, mock_checksum, db_session, sample_transfer):
        sample_transfer.status = "COMPLETED"
        db_session.commit()
        result = verify_transfer(db_session, "test-001")
        assert result.transfer_id == "test-001"

    def test_non_completed_raises(self, db_session, sample_transfer):
        with pytest.raises(TransferVerificationError, match="not COMPLETED"):
            verify_transfer(db_session, "test-001")

    @patch(
        "app.services.artifact_transfer._compute_artifact_checksum",
        return_value="different",
    )
    def test_checksum_mismatch_raises(self, mock_checksum, db_session, sample_transfer):
        sample_transfer.status = "COMPLETED"
        db_session.commit()
        with pytest.raises(TransferVerificationError, match="mismatch"):
            verify_transfer(db_session, "test-001")


class TestListTransfers:
    def test_returns_paginated(self, db_session, sample_transfer):
        transfers, total = list_transfers(db_session, limit=10, offset=0)
        assert total == 1
        assert len(transfers) == 1

    def test_empty_list(self, db_session):
        transfers, total = list_transfers(db_session)
        assert total == 0
        assert transfers == []
