"""Dataset bytes in object storage (issue #214, PRD §13.1/§13.2, §22).

The local backend is exercised through the real `DatasetStorage`; the MinIO backend is
exercised against a fake boto3 client so the tests stay network-free while still asserting
the S3-specific behaviour (URI shape, `head_object` guard, streamed download).
"""

import json
from pathlib import Path

import pytest

from app.config import settings
from app.services.artifact_storage import (
    ArtifactExistsError,
    LocalFilesystemArtifactStorage,
    MinioArtifactStorage,
)
from app.services.dataset_storage import DatasetStorage, dataset_version_key

RECORD = {
    "id": "r1",
    "messages": [
        {"role": "user", "content": "Apa itu DEFNEX?"},
        {
            "role": "assistant",
            "content": "DEFNEX adalah platform MLOps untuk integrasi model AI secara "
            "terpusat yang menyatukan empat sub-proyek universitas dalam satu arsitektur "
            "intelijen terpadu agar tim dapat mengelola dataset dan model dari satu tempat.",
        },
    ],
    "metadata": {"source_dataset": "a7", "source_id": "s1"},
}


def _storage(tmp_path, store=None):
    return DatasetStorage(
        tmp_path / "datasets",
        store=store or LocalFilesystemArtifactStorage(tmp_path / "artifacts"),
    )


def _stage(storage, name="train.jsonl", records=(RECORD,)):
    payload = ("\n".join(json.dumps(r) for r in records) + "\n").encode()
    return storage.stage_upload(name, payload)


class TestLocalBackend:
    def test_committed_bytes_land_under_the_dataset_version_prefix(self, tmp_path):
        store = _storage(tmp_path)
        staged = _stage(store)
        uri, filename, size = store.commit_file(staged["staging_id"], "ds-a7", 2)

        assert (
            uri == f"file://{tmp_path / 'artifacts' / 'datasets/ds-a7/v2/train.jsonl'}"
        )
        assert filename == "train.jsonl"
        assert size == staged["size_bytes"]
        assert Path(uri.removeprefix("file://")).read_bytes()

    def test_dataset_version_is_immutable(self, tmp_path):
        """Re-committing the same version must fail, not overwrite: a dataset version is
        immutable (`uq_dataset_version`), so a second write is a bug, not an update."""
        store = _storage(tmp_path)
        first = _stage(store)
        store.commit_file(first["staging_id"], "ds-a7", 1)

        second = _stage(store)
        with pytest.raises(ArtifactExistsError):
            store.commit_file(second["staging_id"], "ds-a7", 1)

    def test_staged_copy_is_removed_after_commit(self, tmp_path):
        """Otherwise the same bytes would be readable from staging and from the store, and a
        later re-commit could resurrect a superseded copy."""
        store = _storage(tmp_path)
        staged = _stage(store)
        store.commit_file(staged["staging_id"], "ds-a7", 1)
        assert store.resolve_staged(staged["staging_id"]) is None

    def test_sidecars_are_written_next_to_the_bytes(self, tmp_path):
        store = _storage(tmp_path)
        staged = _stage(store)
        uri, _, _ = store.commit_file(staged["staging_id"], "ds-a7", 1)
        store.write_sidecar("ds-a7", 1, "validation_report.json", {"row_count": 1})

        sidecar = Path(
            tmp_path
            / "artifacts"
            / "datasets"
            / "ds-a7"
            / "v1"
            / "validation_report.json"
        )
        assert sidecar.is_file()
        assert json.loads(sidecar.read_text()) == {"row_count": 1}
        # The dataset bytes are untouched by the sidecar write.
        assert Path(uri.removeprefix("file://")).is_file()

    def test_fetch_to_returns_a_private_copy(self, tmp_path):
        """The trainer gets its own file: deleting it must not damage the immutable original
        (which a remote worker, or the next training run, still needs)."""
        store = _storage(tmp_path)
        staged = _stage(store)
        uri, _, _ = store.commit_file(staged["staging_id"], "ds-a7", 1)

        local = store.fetch_to(uri, tmp_path / "work" / "train.jsonl")
        assert local.is_file()
        local.unlink()
        assert store.exists(uri)
        assert store.fetch_to(uri, tmp_path / "again.jsonl").is_file()

    def test_missing_object_raises_file_not_found(self, tmp_path):
        store = _storage(tmp_path)
        with pytest.raises(FileNotFoundError):
            store.fetch_to(f"file://{tmp_path / 'nope.jsonl'}", tmp_path / "out.jsonl")
        assert not store.exists(f"file://{tmp_path / 'nope.jsonl'}")

    def test_key_layout_is_one_prefix_per_version(self):
        """A whole version is a single S3 prefix so 'does this version exist?' is one
        list_objects_v2 call, and a future presigned GET (issue #254) can hand out a prefix."""
        assert (
            dataset_version_key("ds", 3, "train.jsonl") == "datasets/ds/v3/train.jsonl"
        )


class TestMinioBackend:
    """Fake boto3 client: the S3 code paths are asserted without a network or a MinIO
    container, which CI does not have."""

    @pytest.fixture
    def s3(self, monkeypatch):
        store = MinioArtifactStorage(
            endpoint="minio.test:9000", bucket="artifacts", secure=False
        )
        client = _FakeS3()
        monkeypatch.setattr(store, "_client", lambda: client)
        monkeypatch.setattr(store, "_ensure_bucket", lambda c: None)
        return client, store

    def test_committed_bytes_land_in_the_bucket(self, s3):
        client, store = s3
        storage = _storage(Path("/tmp/a7-does-not-matter"), store=store)
        staged = _stage(storage)
        uri, filename, size = storage.commit_file(staged["staging_id"], "ds-minio", 4)

        assert uri == "s3://artifacts/datasets/ds-minio/v4/train.jsonl"
        assert filename == "train.jsonl"
        assert (
            client.objects["datasets/ds-minio/v4/train.jsonl"]
            == ("\n".join(json.dumps(r) for r in [RECORD]) + "\n").encode()
        )

    def test_immutability_uses_head_object(self, s3):
        client, store = s3
        client.objects["datasets/ds-minio/v4/train.jsonl"] = b"already here"
        storage = _storage(Path("/tmp/a7-immutable"), store=store)
        staged = _stage(storage)
        with pytest.raises(ArtifactExistsError):
            storage.commit_file(staged["staging_id"], "ds-minio", 4)
        # The pre-existing object is left exactly as it was.
        assert client.objects["datasets/ds-minio/v4/train.jsonl"] == b"already here"

    def test_missing_bucket_error_is_not_mistaken_for_a_missing_object(self, s3):
        """A 404 on the *bucket* is a real misconfiguration and must surface, not read as
        'object absent' -- which would turn a broken deployment into a silent overwrite."""
        client, store = s3

        def _raise_no_such_bucket(*_a, **_k):
            raise client.exceptions.ClientError(
                {"Error": {"Code": "NoSuchBucket", "Message": "no bucket"}},
                "HeadObject",
            )

        client.head_object = _raise_no_such_bucket
        with pytest.raises(client.exceptions.ClientError):
            store.exists("s3://artifacts/whatever.jsonl")

    def test_get_bytes_raises_for_a_missing_key(self, s3):
        client, store = s3
        with pytest.raises(FileNotFoundError):
            store.get_bytes("s3://artifacts/datasets/ds/v1/absent.jsonl")

    def test_download_to_writes_the_object(self, s3, tmp_path):
        client, store = s3
        client.objects["datasets/ds/v1/train.jsonl"] = b'{"a":1}\n'
        dest = store.download_to(
            "s3://artifacts/datasets/ds/v1/train.jsonl", tmp_path / "out" / "t.jsonl"
        )
        assert dest.read_bytes() == b'{"a":1}\n'


class _FakeStream:
    def __init__(self, data: bytes):
        self._data = data

    def __enter__(self):
        return self

    def __exit__(self, *_exc):
        return False

    def iter_chunks(self, chunk_size=1024 * 1024):
        for i in range(0, len(self._data), chunk_size):
            yield self._data[i : i + chunk_size]


class _FakeExceptions:
    class NoSuchKey(Exception):
        pass

    class ClientError(Exception):
        def __init__(self, response, operation):
            self.response = response
            super().__init__(f"{operation}: {response}")


class _FakeS3:
    """Minimal stand-in for a boto3 S3 client, covering only what MinioArtifactStorage uses."""

    def __init__(self):
        self.objects: dict[str, bytes] = {}
        self.exceptions = _FakeExceptions

    def head_object(self, Bucket, Key):  # noqa: N803 - boto3's own casing
        if Key not in self.objects:
            raise self.exceptions.ClientError(
                {"Error": {"Code": "404", "Message": "Not Found"}}, "HeadObject"
            )
        return {"ContentLength": len(self.objects[Key])}

    def get_object(self, Bucket, Key):  # noqa: N803
        if Key not in self.objects:
            raise self.exceptions.NoSuchKey(f"no such key {Key}")
        return {"Body": _FakeStream(self.objects[Key])}

    def put_object(self, Bucket, Key, Body):  # noqa: N803
        self.objects[Key] = Body


def test_dataset_storage_uses_the_configured_backend(tmp_path, monkeypatch):
    """Issue #214's core requirement: the same `ArtifactStorage` abstraction serves datasets,
    selected by `artifact_backend` — not a second, dataset-only storage class."""
    from app.services.artifact_storage import get_artifact_storage

    monkeypatch.setattr(settings, "artifact_backend", "minio")
    assert isinstance(get_artifact_storage(), MinioArtifactStorage)

    monkeypatch.setattr(settings, "artifact_backend", "local")
    assert isinstance(get_artifact_storage(), LocalFilesystemArtifactStorage)
