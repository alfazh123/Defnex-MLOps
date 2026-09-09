"""Artifact transfer service (PRD §20.3, issue #72).

Tracks cross-host or cross-storage artifact transfers with checksum
verification.  Deployment is blocked when verification fails (PRD §29).
"""

from __future__ import annotations

import uuid
from datetime import UTC, datetime

import structlog
from sqlalchemy.orm import Session

from app.models.transfer import ArtifactTransfer
from app.services.artifact_storage import (
    MinioArtifactStorage,
    _compute_checksum,
    _uri_to_path,
    _uri_to_s3,
)

logger = structlog.get_logger(__name__)


class TransferVerificationError(Exception):
    """Raised when a transfer's checksum verification fails (PRD §29)."""


def _compute_artifact_checksum(uri: str) -> str:
    """Compute SHA-256 checksum for an artifact at the given URI."""
    if uri.startswith("s3://"):
        storage = MinioArtifactStorage()
        bucket, prefix = _uri_to_s3(uri)
        client = storage._client()
        if not prefix.endswith("/"):
            prefix += "/"
        entries: list[tuple[str, bytes]] = []
        paginator = client.get_paginator("list_objects_v2")
        for page in paginator.paginate(Bucket=bucket, Prefix=prefix):
            for obj in page.get("Contents", []):
                key = obj["Key"]
                if key.endswith("/metadata.json"):
                    continue
                rel = key[len(prefix) :]
                if not rel:
                    continue
                body = client.get_object(Bucket=bucket, Key=key)["Body"].read()
                entries.append((rel, body))
        from app.services.artifact_storage import compute_checksum_from_bytes

        return compute_checksum_from_bytes(entries)
    else:
        path = _uri_to_path(uri)
        return _compute_checksum(path)


def initiate_transfer(
    db: Session,
    artifact_uri: str,
    source_host: str,
    target_host: str,
) -> ArtifactTransfer:
    """Compute checksum_before and create a PENDING transfer record."""
    checksum_before = _compute_artifact_checksum(artifact_uri)
    now = datetime.now(UTC)
    transfer = ArtifactTransfer(
        transfer_id=str(uuid.uuid4()),
        artifact_uri=artifact_uri,
        source_host=source_host,
        target_host=target_host,
        checksum_before=checksum_before,
        status="PENDING",
        created_at=now,
    )
    db.add(transfer)
    db.flush()
    logger.info(
        "transfer_initiated",
        transfer_id=transfer.transfer_id,
        artifact_uri=artifact_uri,
    )
    return transfer


def execute_transfer(db: Session, transfer_id: str) -> ArtifactTransfer:
    """Execute the transfer: copy artifact, verify checksum, update status."""
    transfer = db.get(ArtifactTransfer, transfer_id)
    if transfer is None:
        raise ValueError(f"Transfer {transfer_id} not found")
    if transfer.status == "COMPLETED":
        return transfer  # idempotent

    transfer.status = "TRANSFERRING"
    db.flush()

    try:
        _copy_artifact(transfer)
        checksum_after = _compute_artifact_checksum(transfer.artifact_uri)
        transfer.checksum_after = checksum_after

        if checksum_after != transfer.checksum_before:
            transfer.status = "FAILED"
            transfer.error_message = (
                f"Checksum mismatch: before={transfer.checksum_before}, "
                f"after={checksum_after}"
            )
            transfer.completed_at = datetime.now(UTC)
            db.flush()
            logger.warning(
                "transfer_checksum_mismatch",
                transfer_id=transfer.transfer_id,
            )
            return transfer

        transfer.status = "COMPLETED"
        transfer.completed_at = datetime.now(UTC)
        db.flush()
        logger.info(
            "transfer_completed",
            transfer_id=transfer.transfer_id,
        )
        return transfer

    except Exception as exc:
        transfer.status = "FAILED"
        transfer.error_message = str(exc)
        transfer.completed_at = datetime.now(UTC)
        db.flush()
        logger.error(
            "transfer_failed",
            transfer_id=transfer.transfer_id,
            error=str(exc),
        )
        return transfer


def _copy_artifact(transfer: ArtifactTransfer) -> None:
    """Copy artifact from source to target based on URI schemes."""
    src = transfer.artifact_uri
    src_is_s3 = src.startswith("s3://")
    same_host = transfer.source_host == transfer.target_host

    if same_host:
        # Same host — artifact is already at the URI, no copy needed
        return

    if src_is_s3:
        # S3-to-S3: server-side copy via boto3 copy_object (PRD §20.3)
        storage = MinioArtifactStorage()
        client = storage._client()
        src_bucket, src_prefix = _uri_to_s3(src)
        dest_key = f"transfers/{transfer.transfer_id}/artifact"
        client.copy_object(
            Bucket=storage._bucket,
            Key=dest_key,
            CopySource={"Bucket": src_bucket, "Key": src_prefix},
        )
    else:
        # Local-to-remote: SSH/SFTP fallback (PRD §20.2)
        _ssh_transfer(transfer)


def _ssh_transfer(transfer: ArtifactTransfer) -> None:
    """Transfer via SSH/SFTP for local→remote hosts (PRD §20.2)."""
    import os

    from app.services.ssh import (
        PasswordCredential,
        SecretRef,
        SSHRemoteHost,
        resolve_secret_ref,
    )

    src_path = _uri_to_path(transfer.artifact_uri)
    username = os.environ.get("SSH_TRANSFER_USER", "root")
    password = resolve_secret_ref(
        SecretRef(
            ref=os.environ.get(
                "SSH_TRANSFER_PASSWORD", "secret://SSH_TRANSFER_PASSWORD"
            )
        )
    )
    cred = PasswordCredential(username=username, password=password)
    with SSHRemoteHost(transfer.target_host, credential=cred) as host:
        host.upload(src_path, f"/tmp/transfer_{transfer.transfer_id}")


def verify_transfer(db: Session, transfer_id: str) -> ArtifactTransfer:
    """Re-verify a completed transfer's checksum (PRD §29)."""
    transfer = db.get(ArtifactTransfer, transfer_id)
    if transfer is None:
        raise ValueError(f"Transfer {transfer_id} not found")
    if transfer.status != "COMPLETED":
        raise TransferVerificationError(
            f"Transfer {transfer_id} status is {transfer.status}, not COMPLETED"
        )
    checksum_now = _compute_artifact_checksum(transfer.artifact_uri)
    if checksum_now != transfer.checksum_before:
        transfer.status = "FAILED"
        transfer.error_message = (
            f"Post-verify checksum mismatch: expected={transfer.checksum_before}, "
            f"got={checksum_now}"
        )
        transfer.completed_at = datetime.now(UTC)
        db.flush()
        raise TransferVerificationError(
            f"Checksum mismatch on re-verify for {transfer_id}"
        )
    return transfer
