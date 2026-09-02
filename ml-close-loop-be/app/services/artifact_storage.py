import tempfile
from pathlib import Path
from typing import Protocol


class ArtifactStorage(Protocol):
    def store(self, key: str, content: str) -> str:
        """Persist `content` under `key` and return its URI."""
        ...


class LocalFilesystemArtifactStorage:
    """Wraps the local-disk artifact writing used in Phase 0-era prototyping (`mock_runner.py`)
    behind a swappable interface. Persistent artifact storage location/backend is `[UNKNOWN]`
    (model-artifact-versioning-lineage.md §10.1) - this is a placeholder implementation, not a
    storage decision; callers depend only on `ArtifactStorage.store()`, so a real backend (object
    storage, HF Hub, etc.) can replace this later without changing them.
    """

    def __init__(self, base_dir: Path | None = None):
        self._base_dir = (
            base_dir or Path(tempfile.gettempdir()) / "defnex-mock-artifacts"
        )

    def store(self, key: str, content: str) -> str:
        self._base_dir.mkdir(parents=True, exist_ok=True)
        path = self._base_dir / key
        path.write_text(content)
        return f"file://{path}"
