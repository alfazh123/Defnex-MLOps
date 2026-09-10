"""Dataset file storage service for intake pipeline (issue #intake).

Handles staging, normalization, and permanent storage of dataset files.
Separate from ArtifactStorage (which handles model artifacts).
"""

from __future__ import annotations

import hashlib
import json
import uuid
from datetime import UTC, datetime
from pathlib import Path

import structlog

from app.config import settings

logger = structlog.get_logger(__name__)


class DatasetStorage:
    """Stores raw and canonical dataset files."""

    def __init__(self, base_dir: str | Path | None = None):
        self._base_dir = Path(base_dir or settings.dataset_storage_dir)
        self._staging_dir = self._base_dir / "_staging"
        self._staging_dir.mkdir(parents=True, exist_ok=True)

    def stage_upload(self, filename: str, content: bytes) -> dict:
        """Stage an uploaded file. Returns metadata dict."""
        staging_id = str(uuid.uuid4())
        staging_path = self._staging_dir / f"{staging_id}_{filename}"
        staging_path.write_bytes(content)

        checksum = hashlib.sha256(content).hexdigest()
        return {
            "staging_id": staging_id,
            "filename": filename,
            "file_size": len(content),
            "checksum_sha256": checksum,
            "path": str(staging_path),
        }

    def resolve_staged(self, staging_id: str) -> Path | None:
        """Find a staged file by staging_id prefix."""
        for p in self._staging_dir.glob(f"{staging_id}_*"):
            return p
        return None

    def read_records(self, path: Path) -> list[dict]:
        """Parse a JSONL file into list of dicts."""
        records = []
        for line in path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            records.append(json.loads(line))
        return records

    def compute_checksum(self, path: Path) -> str:
        """SHA-256 of file contents."""
        h = hashlib.sha256()
        h.update(path.read_bytes())
        return h.hexdigest()

    def commit_file(self, staging_id: str, dataset_id: str, version: int) -> str:
        """Move staged file to permanent storage. Returns canonical URI."""
        staged = self.resolve_staged(staging_id)
        if staged is None:
            raise ValueError(f"Staged file {staging_id} not found")

        perm_dir = self._base_dir / dataset_id / f"v{version}"
        perm_dir.mkdir(parents=True, exist_ok=True)

        dest = perm_dir / "dataset.jsonl"
        staged.rename(dest)

        manifest = {
            "dataset_id": dataset_id,
            "version": version,
            "format": "jsonl",
            "created_at": datetime.now(UTC).isoformat(),
            "checksum_sha256": self.compute_checksum(dest),
        }
        (perm_dir / "manifest.json").write_text(
            json.dumps(manifest, indent=2), encoding="utf-8"
        )

        return f"file://{dest}"

    def write_validation_report(
        self, dataset_id: str, version: int, report: dict
    ) -> None:
        """Write validation-report.json to permanent storage."""
        perm_dir = self._base_dir / dataset_id / f"v{version}"
        perm_dir.mkdir(parents=True, exist_ok=True)
        (perm_dir / "validation-report.json").write_text(
            json.dumps(report, indent=2, default=str), encoding="utf-8"
        )

    def write_schema(self, dataset_id: str, version: int, schema: dict) -> None:
        """Write schema.json to permanent storage."""
        perm_dir = self._base_dir / dataset_id / f"v{version}"
        perm_dir.mkdir(parents=True, exist_ok=True)
        (perm_dir / "schema.json").write_text(
            json.dumps(schema, indent=2), encoding="utf-8"
        )
