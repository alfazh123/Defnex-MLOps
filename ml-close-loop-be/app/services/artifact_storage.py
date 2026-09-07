import json
import shutil
from pathlib import Path
from typing import Protocol

from app.config import settings


class ArtifactExistsError(Exception):
    """The target artifact version path already exists and must not be overwritten (issue #38)."""


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
        self._write_metadata(target, metadata)
        return f"file://{target}"

    def _write_metadata(self, target: Path, metadata: dict) -> None:
        (target / "metadata.json").write_text(
            json.dumps(metadata, indent=2, sort_keys=True, default=str)
        )
