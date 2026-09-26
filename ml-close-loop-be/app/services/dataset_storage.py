"""Local filesystem storage for the dataset intake pipeline.

Handles staging → validation → commit lifecycle for uploaded dataset files.
Each dataset version lives under ``{base_dir}/{dataset_id}/v{version}/``.

Record parsing is delegated to :mod:`app.services.dataset_parsing` (issue #245) so this
module only owns paths, checksums and the sidecar metadata files.
"""

from __future__ import annotations

import hashlib
import json
import shutil
import uuid
from pathlib import Path

from app.config import settings
from app.services import dataset_parsing


class UnsafeFilenameError(ValueError):
    """A client-supplied filename cannot be reduced to a single safe path component.

    Raised for names that carry no usable basename at all (``""``, ``"."``, ``".."``) or
    embed a NUL byte. Names that merely *contain* directory components (``../../x.jsonl``,
    ``C:\\data\\x.csv``) are normalized to their basename instead of rejected — see
    :func:`sanitize_filename`.
    """


def sanitize_filename(filename: str) -> str:
    """Reduce a client-supplied filename to exactly one safe path component (issue #246).

    ``stage_upload`` used to write ``staging_dir / filename`` straight from the client's
    ``Content-Disposition`` name, so ``../../x.jsonl`` escaped the staging tree entirely
    (audit finding T7). Normalizing to the basename is what makes the containment
    property hold; the ``Path.name``-style split is done on both POSIX and Windows
    separators because a Windows client legitimately sends ``C:\\data\\train.csv`` and we
    do not want to break it.

    Surrounding whitespace is stripped so a name of ``"  "`` cannot survive as an
    effectively-empty file. Interior spaces and non-ASCII characters are preserved.
    """

    if "\x00" in filename:
        raise UnsafeFilenameError("Filename must not contain a NUL byte")
    # Treat "\" as a separator regardless of host OS: on POSIX it is a legal filename
    # character, but accepting "..\\..\\x.jsonl" here would store a name that becomes a
    # traversal the moment the staging tree is moved to a Windows worker.
    candidate = filename.replace("\\", "/").rsplit("/", 1)[-1].strip()
    if candidate in ("", ".", ".."):
        raise UnsafeFilenameError(
            f"Filename {filename!r} has no usable file name component"
        )
    return candidate


class DatasetStorage:
    def __init__(self, base_dir: str | Path | None = None):
        self._base = Path(base_dir or settings.dataset_storage_dir)
        self._staging = self._base / "_staging"
        self._staging.mkdir(parents=True, exist_ok=True)

    def stage_upload(self, filename: str, content: bytes) -> dict:
        staging_id = uuid.uuid4().hex
        staging_dir = self._staging / staging_id
        staging_dir.mkdir(parents=True, exist_ok=True)
        safe_name = sanitize_filename(filename)
        dest = staging_dir / safe_name
        # Belt and braces: `safe_name` is a single component by construction, so this can
        # only fire if that invariant is ever broken. Asserting the resolved path stays
        # under the staging dir is the property issue #246 actually cares about, so it is
        # checked rather than assumed.
        resolved_dir = staging_dir.resolve()
        if dest.resolve().parent != resolved_dir:
            shutil.rmtree(staging_dir, ignore_errors=True)
            raise UnsafeFilenameError(
                f"Refusing to stage {filename!r}: resolved path escapes {resolved_dir}"
            )
        dest.write_bytes(content)
        return {
            "staging_id": staging_id,
            "path": str(dest),
            "filename": safe_name,
            "size_bytes": len(content),
        }

    def resolve_staged(self, staging_id: str) -> Path | None:
        d = self._staging / staging_id
        if not d.is_dir():
            return None
        files = [f for f in d.iterdir() if f.is_file()]
        return files[0] if files else None

    def read_records(self, path: Path, source_format: str = "jsonl") -> list:
        """Parse staged bytes into records.

        Raises `app.services.dataset_parsing.DatasetParseError` for every malformed-input
        case; it never raises a bare `ValueError`/`JSONDecodeError`/`FileNotFoundError`
        for those, so callers have one exception type to map to a coded API error
        (issue #245).
        """

        return dataset_parsing.parse_file(path, source_format)

    def compute_checksum(self, path: Path) -> str:
        h = hashlib.sha256()
        with path.open("rb") as fh:
            for chunk in iter(lambda: fh.read(1024 * 1024), b""):
                h.update(chunk)
        return h.hexdigest()

    def commit_file(self, staging_id: str, dataset_id: str, version: int) -> str:
        src_dir = self._staging / staging_id
        dest_dir = self._base / dataset_id / f"v{version}"
        dest_dir.mkdir(parents=True, exist_ok=True)
        files = [f for f in src_dir.iterdir() if f.is_file()]
        if not files:
            raise FileNotFoundError(f"No files in staging {staging_id}")
        # `f.name` comes from a directory listing, never from client input, so it is a
        # single component by construction (issue #246).
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
