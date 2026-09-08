import hashlib
import json
import shutil
from pathlib import Path
from typing import Protocol

from app.config import settings


class ArtifactExistsError(Exception):
    """The target artifact version path already exists and must not be overwritten (issue #38)."""


class ArtifactChecksumError(Exception):
    """The artifact's recomputed SHA-256 does not match the immutable checksum recorded at
    finalize time (issue #62). Deploy/transfer must reject the artifact before any pointer moves."""


class ArtifactStorage(Protocol):
    def store(self, key: str, content: str) -> str:
        """Persist `content` under `key` and return its URI."""
        ...


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
        # Issue #62: record a SHA-256 over the payload files (everything but metadata.json)
        # as part of the immutable metadata, so deploy/transfer can detect corruption.
        metadata["checksum"] = _compute_checksum(target)
        self._write_metadata(target, metadata)
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


def _compute_checksum(target: Path) -> str:
    """Deterministic SHA-256 over every payload file (excluding metadata.json) in a version dir.

    Files are hashed in sorted relative-path order; each file is prefixed with its relative
    path and byte length so the digest is order-correct and unambiguous between files."""
    h = hashlib.sha256()
    for rel in sorted(
        p.relative_to(target).as_posix()
        for p in target.rglob("*")
        if p.is_file() and p.name != "metadata.json"
    ):
        data = (target / rel).read_bytes()
        h.update(f"{rel}:{len(data)}:".encode())
        h.update(data)
    return h.hexdigest()


def _uri_to_path(uri: str) -> Path:
    """Strip a `file://` prefix from an artifact URI (artifacts are stored at `file://{dir}`)."""
    return Path(uri[len("file://") :]) if uri.startswith("file://") else Path(uri)
