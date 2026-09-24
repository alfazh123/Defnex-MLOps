"""Real-MinIO integration test for MinioArtifactStorage (issue #171).

Unlike tests/test_artifact_storage_minio.py (which mocks the S3 client everywhere), this
file talks to an actual MinIO server. It requires:
  - `boto3` installed (optional `s3` extra - not in [dev], so this whole module skips
    cleanly where it's absent, e.g. in CI today).
  - A reachable MinIO server at `settings.minio_endpoint` (default localhost:9000, matching
    `docker compose up -d minio` in this repo's docker-compose.yml). If nothing answers
    there, every test in this module skips with a message telling the developer how to
    start one, rather than failing.

Run locally with: docker compose up -d minio && pytest tests/test_artifact_storage_minio_integration.py
"""

import socket
import uuid

import pytest

boto3 = pytest.importorskip("boto3")

from app.config import settings  # noqa: E402
from app.services.artifact_storage import (  # noqa: E402
    ArtifactExistsError,
    MinioArtifactStorage,
)


def _minio_reachable() -> bool:
    host, _, port = settings.minio_endpoint.partition(":")
    try:
        with socket.create_connection((host, int(port or 9000)), timeout=1):
            return True
    except OSError:
        return False


pytestmark = pytest.mark.skipif(
    not _minio_reachable(),
    reason=(
        f"No MinIO reachable at {settings.minio_endpoint!r} - "
        "run `docker compose up -d minio` to exercise this test"
    ),
)


@pytest.fixture
def storage():
    """A fresh, uniquely-named bucket per test so tests don't collide or need cleanup
    logic beyond what a throwaway MinIO dev instance already tolerates."""
    bucket = f"defnex-test-{uuid.uuid4().hex[:12]}"
    return MinioArtifactStorage(bucket=bucket)


def _staging_dir(tmp_path):
    staging = tmp_path / f"staging-src-{uuid.uuid4().hex[:8]}"
    staging.mkdir()
    (staging / "adapter_model.safetensors").write_bytes(b"weights")
    (staging / "adapter_config.json").write_text('{"lora": true}')
    return staging


def test_store_round_trips_through_real_minio(storage):
    uri = storage.store("smoke-test.txt", "hello from a real MinIO server")

    assert uri.startswith("s3://")
    assert storage.read_metadata(uri) == {}  # no metadata.json at this key - expected


def test_finalize_version_round_trips_through_real_minio(storage, tmp_path):
    staging = _staging_dir(tmp_path)

    uri = storage.finalize_version(
        "qwen-sft-domain-x",
        "qwen-sft-domain-x-Qwen-Qwen3.8-27B-v1",
        staging,
        {"model_id": "qwen-sft-domain-x"},
    )

    assert (
        uri
        == f"s3://{storage._bucket}/qwen-sft-domain-x/qwen-sft-domain-x-Qwen-Qwen3.8-27B-v1/"
    )
    metadata = storage.read_metadata(uri)
    assert len(metadata["checksum"]) == 64  # sha256 hex
    assert storage.verify_checksum(uri) is True


def test_finalize_version_rejects_existing_path_against_real_minio(storage, tmp_path):
    name = "qwen-sft-domain-x-Qwen-Qwen3.8-27B-v1"
    storage.finalize_version("qwen-sft-domain-x", name, _staging_dir(tmp_path), {})

    with pytest.raises(ArtifactExistsError):
        storage.finalize_version("qwen-sft-domain-x", name, _staging_dir(tmp_path), {})


def test_verify_checksum_detects_tampering_via_real_minio(storage, tmp_path):
    name = "qwen-sft-domain-x-Qwen-Qwen3.8-27B-v1"
    uri = storage.finalize_version(
        "qwen-sft-domain-x", name, _staging_dir(tmp_path), {}
    )

    client = storage._client()
    client.put_object(
        Bucket=storage._bucket,
        Key=f"qwen-sft-domain-x/{name}/adapter_model.safetensors",
        Body=b"tampered",
    )

    assert storage.verify_checksum(uri) is False


def test_generate_presigned_upload_url_against_real_minio(storage):
    url = storage.generate_presigned_upload_url("some/key.bin")

    assert url.startswith("http")
    assert "some/key.bin" in url
