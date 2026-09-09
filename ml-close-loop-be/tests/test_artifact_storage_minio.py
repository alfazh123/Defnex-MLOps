"""MinIO/S3 artifact storage tests (issue #71).

The S3 client is mocked everywhere — tests never hit a real MinIO server.
"""

import json
from pathlib import Path

import pytest

from app.services.artifact_storage import (
    ArtifactExistsError,
    LocalFilesystemArtifactStorage,
    MinioArtifactStorage,
    get_artifact_storage,
)


class FakeS3:
    """Minimal in-memory S3 stub exposing just the methods MinioArtifactStorage calls."""

    def __init__(self):
        self.objects: dict[str, bytes] = {}
        self.buckets: set[str] = set()
        self.url = None

    class exceptions:
        class NoSuchKey(Exception):
            pass

    def list_buckets(self):
        return {"Buckets": [{"Name": b} for b in self.buckets]}

    def create_bucket(self, **kwargs):
        self.buckets.add(kwargs["Bucket"])

    def put_object(self, **kwargs):
        self.objects[kwargs["Key"]] = kwargs["Body"]

    def list_objects_v2(self, **kwargs):
        prefix = kwargs.get("Prefix", "")
        keys = [k for k in self.objects if k.startswith(prefix)]
        if kwargs.get("MaxKeys") is not None:
            return {
                "KeyCount": len(keys) > 0,
                "Contents": [{"Key": k} for k in keys[: kwargs["MaxKeys"]]],
            }
        return {"KeyCount": len(keys), "Contents": [{"Key": k} for k in keys]}

    def get_object(self, **kwargs):
        key = kwargs["Key"]
        if key not in self.objects:
            raise self.exceptions.NoSuchKey()
        return {"Body": _Bytes(self.objects[key])}

    def get_paginator(self, _name):
        return _Paginator(self)

    def generate_presigned_url(self, method, Params, ExpiresIn):
        self.url = (method, Params, ExpiresIn)
        return f"https://presigned.example/{Params['Key']}?expires={ExpiresIn}"


class _Bytes:
    def __init__(self, data: bytes):
        self._data = data

    def read(self):
        return self._data


class _Paginator:
    def __init__(self, s3):
        self._s3 = s3

    def paginate(self, **kwargs):
        prefix = kwargs.get("Prefix", "")
        keys = [k for k in self._s3.objects if k.startswith(prefix)]
        yield {"Contents": [{"Key": k} for k in keys]}


@pytest.fixture
def fake_s3(monkeypatch):
    s3 = FakeS3()
    monkeypatch.setattr(MinioArtifactStorage, "_client", lambda self: s3, raising=False)
    return s3


def _make_storage(**kwargs):
    kwargs.setdefault("bucket", "artifacts")
    return MinioArtifactStorage(**kwargs)


def _staging(tmp_path: Path) -> Path:
    staging = tmp_path / "staging"
    staging.mkdir(exist_ok=True)
    (staging / "adapter_model.safetensors").write_bytes(b"weights")
    (staging / "adapter_config.json").write_text('{"lora": true}')
    return staging


def test_store_puts_object_and_returns_s3_uri(fake_s3):
    storage = _make_storage()
    uri = storage.store("some/key", "hello")
    assert uri == "s3://artifacts/some/key"
    assert fake_s3.objects["some/key"] == b"hello"


def test_store_creates_bucket_when_missing(fake_s3):
    storage = _make_storage()
    assert "artifacts" not in fake_s3.buckets
    storage.store("a/b", "x")
    assert "artifacts" in fake_s3.buckets


def test_finalize_version_uploads_payload_and_metadata(fake_s3, tmp_path):
    storage = _make_storage()
    staging = _staging(tmp_path)
    uri = storage.finalize_version("m", "name", staging, {"model_id": "m"})
    assert uri == "s3://artifacts/m/name/"
    assert fake_s3.objects["m/name/adapter_model.safetensors"] == b"weights"
    assert fake_s3.objects["m/name/adapter_config.json"] == b'{"lora": true}'
    meta = json.loads(fake_s3.objects["m/name/metadata.json"])
    assert meta["model_id"] == "m"
    assert len(meta["checksum"]) == 64


def test_finalize_version_rejects_existing_object(fake_s3, tmp_path):
    storage = _make_storage()
    staging = _staging(tmp_path)
    storage.finalize_version("m", "name", staging, {})
    with pytest.raises(ArtifactExistsError):
        storage.finalize_version("m", "name", _staging(tmp_path), {})


def test_verify_checksum_passes_for_intact(fake_s3, tmp_path):
    storage = _make_storage()
    uri = storage.finalize_version("m", "name", _staging(tmp_path), {})
    assert storage.verify_checksum(uri) is True


def test_verify_checksum_detects_corruption(fake_s3, tmp_path):
    storage = _make_storage()
    uri = storage.finalize_version("m", "name", _staging(tmp_path), {})
    fake_s3.objects["m/name/adapter_model.safetensors"] = b"tampered"
    assert storage.verify_checksum(uri) is False


def test_verify_checksum_true_when_no_metadata(fake_s3):
    storage = _make_storage()
    assert storage.verify_checksum("s3://artifacts/m/name/") is True


def test_read_metadata_roundtrip(fake_s3, tmp_path):
    storage = _make_storage()
    uri = storage.finalize_version("m", "name", _staging(tmp_path), {"model_id": "m"})
    meta = storage.read_metadata(uri)
    assert meta["model_id"] == "m"
    assert "checksum" in meta


def test_read_metadata_empty_when_missing(fake_s3):
    storage = _make_storage()
    assert storage.read_metadata("s3://artifacts/m/name/") == {}


def test_generate_presigned_upload_url(fake_s3):
    storage = _make_storage()
    url = storage.generate_presigned_upload_url(
        "compute/direct.bin", expires_seconds=600
    )
    assert url == "https://presigned.example/compute/direct.bin?expires=600"
    assert fake_s3.url == (
        "put_object",
        {"Bucket": "artifacts", "Key": "compute/direct.bin"},
        600,
    )


def test_local_fallback_default(monkeypatch):
    """Default artifact_backend = local, so the factory returns the filesystem store."""
    precompute = get_artifact_storage.__globals__["settings"]
    monkeypatch.setattr(precompute, "artifact_backend", "local", raising=False)
    storage = get_artifact_storage()
    assert isinstance(storage, LocalFilesystemArtifactStorage)


def test_factory_selects_minio(monkeypatch, fake_s3):
    settings = get_artifact_storage.__globals__["settings"]
    monkeypatch.setattr(settings, "artifact_backend", "minio", raising=False)
    storage = get_artifact_storage()
    assert isinstance(storage, MinioArtifactStorage)


def test_minio_uses_config_defaults(monkeypatch, fake_s3):
    settings = get_artifact_storage.__globals__["settings"]
    storage = MinioArtifactStorage()
    monkeypatch.setattr(settings, "minio_bucket", "artifacts", raising=False)
    uri = storage.store("k", "v")
    assert uri == "s3://artifacts/k"
