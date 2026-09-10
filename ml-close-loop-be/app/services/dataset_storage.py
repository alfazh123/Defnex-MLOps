"""Local filesystem storage for the dataset intake pipeline.

Handles staging → validation → commit lifecycle for uploaded dataset files.
Each dataset version lives under ``{base_dir}/{dataset_id}/v{version}/``.
"""

from __future__ import annotations

import csv
import hashlib
import io
import json
import shutil
import uuid
from pathlib import Path

from app.config import settings


class DatasetStorage:
    def __init__(self, base_dir: str | Path | None = None):
        self._base = Path(base_dir or settings.dataset_storage_dir)
        self._staging = self._base / "_staging"
        self._staging.mkdir(parents=True, exist_ok=True)

    def stage_upload(self, filename: str, content: bytes) -> dict:
        staging_id = uuid.uuid4().hex
        staging_dir = self._staging / staging_id
        staging_dir.mkdir(parents=True, exist_ok=True)
        dest = staging_dir / filename
        dest.write_bytes(content)
        return {
            "staging_id": staging_id,
            "path": str(dest),
            "filename": filename,
            "size_bytes": len(content),
        }

    def resolve_staged(self, staging_id: str) -> Path | None:
        d = self._staging / staging_id
        if not d.is_dir():
            return None
        files = [f for f in d.iterdir() if f.is_file()]
        return files[0] if files else None

    def read_records(self, path: Path, source_format: str = "jsonl") -> list[dict]:
        if source_format == "jsonl":
            records = []
            for line in path.read_text().splitlines():
                line = line.strip()
                if line:
                    records.append(json.loads(line))
            return records
        if source_format == "json":
            data = json.loads(path.read_text())
            if not isinstance(data, list):
                return []
            return [r for r in data if isinstance(r, dict)]
        if source_format == "csv":
            text = path.read_text()
            reader = csv.DictReader(io.StringIO(text))
            return [row for row in reader]
        if source_format == "xlsx":
            try:
                import openpyxl
            except ImportError:
                raise FileNotFoundError("openpyxl not installed")
            wb = openpyxl.load_workbook(str(path), read_only=True)
            ws = wb.active
            rows = list(ws.iter_rows(values_only=True))
            wb.close()
            if not rows:
                return []
            headers = [str(h) for h in rows[0]]
            return [dict(zip(headers, row)) for row in rows[1:]]
        raise ValueError(f"Unsupported format: {source_format}")

    def compute_checksum(self, path: Path) -> str:
        h = hashlib.sha256()
        h.update(path.read_bytes())
        return h.hexdigest()

    def commit_file(self, staging_id: str, dataset_id: str, version: int) -> str:
        src_dir = self._staging / staging_id
        dest_dir = self._base / dataset_id / f"v{version}"
        dest_dir.mkdir(parents=True, exist_ok=True)
        files = [f for f in src_dir.iterdir() if f.is_file()]
        if not files:
            raise FileNotFoundError(f"No files in staging {staging_id}")
        dest_file = dest_dir / files[0].name
        shutil.move(str(files[0]), str(dest_file))
        shutil.rmtree(src_dir, ignore_errors=True)
        return str(dest_file)

    def write_validation_report(
        self, dataset_id: str, version: int, report_dict: dict
    ) -> Path:
        d = self._base / dataset_id / f"v{version}"
        d.mkdir(parents=True, exist_ok=True)
        p = d / "validation_report.json"
        p.write_text(json.dumps(report_dict, indent=2, default=str))
        return p

    def write_schema(self, dataset_id: str, version: int, schema_dict: dict) -> Path:
        d = self._base / dataset_id / f"v{version}"
        d.mkdir(parents=True, exist_ok=True)
        p = d / "schema.json"
        p.write_text(json.dumps(schema_dict, indent=2))
        return p
