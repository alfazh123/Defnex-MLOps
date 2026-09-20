import hashlib
import json
import os
import shutil
from pathlib import Path
from typing import Protocol, runtime_checkable

from app.config import settings


class ArtifactExistsError(Exception):
    """The target artifact version path already exists and must not be overwritten (issue #38)."""


class ArtifactChecksumError(Exception):
    """The artifact's recomputed SHA-256 does not match the immutable checksum recorded at
    finalize time (issue #62). Deploy/transfer must reject the artifact before any pointer moves."""


@runtime_checkable
class ArtifactStorage(Protocol):
    def store(self, key: str, content: str) -> str:
        """Persist `content` under `key` and return its URI."""
        ...

    def finalize_version(
        self, model_id: str, name: str, staging_dir: Path | str, metadata: dict
    ) -> str:
        """Move a training run's staged output into its immutable, versioned location."""
        ...

    def verify_checksum(self, uri: str) -> bool:
        """Return True if the artifact at `uri` passes its recorded checksum."""
        ...


def compute_checksum_from_bytes(entries: list[tuple[str, bytes]]) -> str:
    """Deterministic SHA-256 over a list of (relative_path, data) pairs.

    Each entry is prefixed with its relative path and byte length so the digest is
    order-correct and unambiguous between files.  This is the canonical algorithm shared
    by LocalFilesystemArtifactStorage and MinioArtifactStorage (issue #62)."""
    h = hashlib.sha256()
    for rel, data in sorted(entries, key=lambda e: e[0]):
        h.update(f"{rel}:{len(data)}:".encode())
        h.update(data)
    return h.hexdigest()


def _make_world_readable(target: Path) -> None:
    """Issue #164: artifacts written by the training worker are root-owned (UID 0) inside
    the container, so the worker itself (and any host user without sudo) can't read them
    back. chmod alone can't fix ownership, but it makes the bytes actually accessible -
    directories need the execute bit to be traversable, files just need read.
    """
    try:
        for p in target.rglob("*"):
            os.chmod(p, 0o755 if p.is_dir() else 0o644)
        os.chmod(target, 0o755)
    except OSError:
        # Best-effort: a filesystem that doesn't support chmod (or a permission we can't
        # change ourselves) shouldn't fail the whole artifact finalization.
        pass


def _compute_checksum(target: Path) -> str:
    """Deterministic SHA-256 over every payload file (excluding metadata.json) in a version dir.

    Files are hashed in sorted relative-path order; each file is prefixed with its relative
    path and byte length so the digest is order-correct and unambiguous between files."""
    entries: list[tuple[str, bytes]] = []
    for p in target.rglob("*"):
        if p.is_file() and p.name != "metadata.json":
            entries.append((p.relative_to(target).as_posix(), p.read_bytes()))
    return compute_checksum_from_bytes(entries)


def _uri_to_path(uri: str) -> Path:
    """Strip a `file://` prefix from an artifact URI (artifacts are stored at `file://{dir}`)."""
    return Path(uri[len("file://") :]) if uri.startswith("file://") else Path(uri)


def _uri_to_s3(uri: str) -> tuple[str, str]:
    """Parse an `s3://{bucket}/{key}` URI into (bucket, key)."""
    prefix = "s3://"
    if not uri.startswith(prefix):
        raise ValueError(f"not an s3:// URI: {uri!r}")
    without_prefix = uri[len(prefix) :]
    slash = without_prefix.find("/")
    if slash < 0:
        raise ValueError(f"s3:// URI has no key: {uri!r}")
    return without_prefix[:slash], without_prefix[slash + 1 :]


def _serialize_metadata(metadata: dict) -> bytes:
    return json.dumps(metadata, indent=2, sort_keys=True, default=str).encode()


class LocalFilesystemArtifactStorage:
    """Local-disk artifact storage behind a swappable interface.

    Issue #38 turned this from a throwaway temp-dir writer into a versioned, immutable store:
    each registered model version lands in a dedicated, never-overwritten directory and is
    accompanied by a `metadata.json` capturing its lineage. The base directory is configurable
    via the `ARTIFACT_STORAGE_DIR` env var (default `data/artifacts`, no longer a system temp
    dir). Callers depend only on the `ArtifactStorage` interface, so a real backend (object
    storage, HF Hub, etc.) can replace this later without changing them.
    """

    def __init__(self, base_dir: Path | str | None = None):
        self._base_dir = Path(base_dir or settings.artifact_storage_dir)

    def store(self, key: str, content: str) -> str:
        """Write a single flat file (mock/stub use only). Not immutable and not versioned."""
        self._base_dir.mkdir(parents=True, exist_ok=True)
        path = self._base_dir / key
        path.write_text(content)
        return f"file://{path}"

    def finalize_version(
        self, model_id: str, name: str, staging_dir: Path | str, metadata: dict
    ) -> str:
        """Move a training run's staged output into its immutable, versioned location.

        The trained output lands at `{base_dir}/{model_id}/{name}/` together with a
        `metadata.json`. If that path already exists, the write is REJECTED (immutability —
        never overwrite an existing version's artifact; issue #38).

        Raises:
            ArtifactExistsError: the version's directory already exists.
        """
        target = self._base_dir / model_id / name
        if target.exists():
            raise ArtifactExistsError(
                f"Refusing to overwrite existing immutable artifact at {target}"
            )
        self._base_dir.mkdir(parents=True, exist_ok=True)
        staging = Path(staging_dir)
        shutil.move(str(staging), str(target))
        metadata["checksum"] = _compute_checksum(target)
        self._write_metadata(target, metadata)
        # After metadata.json is written too, so it's covered by the chmod pass as well.
        _make_world_readable(target)
        return f"file://{target}"

    @staticmethod
    def _write_metadata(target: Path, metadata: dict) -> None:
        (target / "metadata.json").write_text(
            json.dumps(metadata, indent=2, sort_keys=True, default=str)
        )

    @staticmethod
    def read_metadata(uri: str) -> dict:
        """Load the metadata.json of a version dir given its `file://` URI. Returns an empty
        dict when no metadata.json exists (a pre-#62 artifact that predates checksum metadata)."""
        target = _uri_to_path(uri)
        meta_path = target / "metadata.json"
        if not meta_path.exists():
            return {}
        return json.loads(meta_path.read_text())

    def verify_checksum(self, uri: str) -> bool:
        """Recompute the SHA-256 of the artifact payload and compare against the checksum
        recorded in metadata.json at finalize time (issue #62). Returns False only when the
        byte content actually differs; a pre-#62 artifact with no recorded checksum is treated
        as verified (no baseline to compare against) so existing deploy paths keep working."""
        meta = self.read_metadata(uri)
        recorded = meta.get("checksum")
        if recorded is None:
            return True
        return _compute_checksum(_uri_to_path(uri)) == recorded


class MinioArtifactStorage:
    """MinIO/S3-backed artifact storage (issue #71, PRD §13.1, §13.2).

    Artifacts (adapter/model) and dataset bytes live in MinIO, not a server-local
    directory.  The URI scheme is ``s3://{bucket}/{key}``.  Callers interact through
    the same ``ArtifactStorage`` protocol as ``LocalFilesystemArtifactStorage``.
    """

    def __init__(
        self,
        *,
        endpoint: str | None = None,
        access_key: str | None = None,
        secret_key: str | None = None,
        bucket: str | None = None,
        secure: bool | None = None,
    ):
        self._endpoint = endpoint or settings.minio_endpoint
        self._access_key = access_key or settings.minio_access_key
        self._secret_key = secret_key or settings.minio_secret_key
        self._bucket = bucket or settings.minio_bucket
        self._secure = secure if secure is not None else settings.minio_secure

    def _client(self):
        """Lazy boto3 S3 client so boto3 is only imported when MinIO is actually used."""
        import boto3

        return boto3.client(
            "s3",
            endpoint_url=f"{'https' if self._secure else 'http'}://{self._endpoint}",
            aws_access_key_id=self._access_key,
            aws_secret_access_key=self._secret_key,
        )

    def _ensure_bucket(self, client) -> None:
        existing = [b["Name"] for b in client.list_buckets().get("Buckets", [])]
        if self._bucket not in existing:
            client.create_bucket(Bucket=self._bucket)

    def store(self, key: str, content: str) -> str:
        client = self._client()
        self._ensure_bucket(client)
        client.put_object(Bucket=self._bucket, Key=key, Body=content.encode())
        return f"s3://{self._bucket}/{key}"

    def finalize_version(
        self, model_id: str, name: str, staging_dir: Path | str, metadata: dict
    ) -> str:
        """Upload staging_dir contents to ``{model_id}/{name}/`` under the bucket,
        then write ``metadata.json`` with a SHA-256 checksum over all payload objects.

        Raises ArtifactExistsError if any object already exists at the target prefix
        (immutability — issue #38)."""
        client = self._client()
        self._ensure_bucket(client)

        prefix = f"{model_id}/{name}/"
        existing = client.list_objects_v2(Bucket=self._bucket, Prefix=prefix, MaxKeys=1)
        if existing.get("KeyCount", 0) > 0:
            raise ArtifactExistsError(
                f"Refusing to overwrite existing immutable artifact at "
                f"s3://{self._bucket}/{prefix}"
            )

        staging = Path(staging_dir)
        checksum_entries: list[tuple[str, bytes]] = []
        for p in sorted(staging.rglob("*")):
            if p.is_file():
                data = p.read_bytes()
                rel = p.relative_to(staging).as_posix()
                client.put_object(
                    Bucket=self._bucket,
                    Key=f"{prefix}{rel}",
                    Body=data,
                )
                checksum_entries.append((rel, data))

        metadata["checksum"] = compute_checksum_from_bytes(checksum_entries)
        client.put_object(
            Bucket=self._bucket,
            Key=f"{prefix}metadata.json",
            Body=_serialize_metadata(metadata),
        )
        return f"s3://{self._bucket}/{prefix}"

    def read_metadata(self, uri: str) -> dict:
        """Load metadata.json from the S3 prefix pointed to by `uri`.

        Returns an empty dict when no metadata.json exists (a pre-#62 artifact
        that predates checksum metadata)."""
        client = self._client()
        bucket, prefix = _uri_to_s3(uri)
        if not prefix.endswith("/"):
            prefix += "/"
        try:
            resp = client.get_object(Bucket=bucket, Key=f"{prefix}metadata.json")
            return json.loads(resp["Body"].read())
        except client.exceptions.NoSuchKey:
            return {}

    def verify_checksum(self, uri: str) -> bool:
        """Recompute the SHA-256 of the artifact payload and compare against the checksum
        recorded in metadata.json at finalize time (issue #62 / #71).  A pre-#62 artifact
        with no recorded checksum is treated as verified (no baseline to compare against)."""
        client = self._client()
        bucket, prefix = _uri_to_s3(uri)
        if not prefix.endswith("/"):
            prefix += "/"
        try:
            meta_resp = client.get_object(Bucket=bucket, Key=f"{prefix}metadata.json")
            meta = json.loads(meta_resp["Body"].read())
        except client.exceptions.NoSuchKey:
            return True
        recorded = meta.get("checksum")
        if recorded is None:
            return True
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
        return compute_checksum_from_bytes(entries) == recorded

    def generate_presigned_upload_url(
        self, key: str, expires_seconds: int = 3600
    ) -> str:
        """Generate a presigned PUT URL so compute can upload directly to MinIO without
        going through the FastAPI backend (PRD §13.3 — direct artifact upload)."""
        client = self._client()
        return client.generate_presigned_url(
            "put_object",
            Params={"Bucket": self._bucket, "Key": key},
            ExpiresIn=expires_seconds,
        )


def get_artifact_storage() -> ArtifactStorage:
    """Return the configured artifact storage backend (PRD §13.1).

    ``local`` → LocalFilesystemArtifactStorage (dev / CI).
    ``minio`` → MinioArtifactStorage (production object storage)."""
    if settings.artifact_backend == "minio":
        return MinioArtifactStorage()
    return LocalFilesystemArtifactStorage()
