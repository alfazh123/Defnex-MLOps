"""Dataset byte storage for the intake pipeline (issue #214, PRD §13.1/§13.2).

Handles staging → validation → commit lifecycle for uploaded dataset files.

Dataset bytes used to live only in ``data/datasets`` on local disk, which contradicted
the PRD in three places:

- **§13.2** — "MinIO/S3 = dataset bytes, adapter/model artifacts, ..."
- **Principle 3** — "Model and dataset bytes live in object storage, not inside PostgreSQL."
- **§22** — "Object storage holds immutable datasets/artifacts so model bytes are not tied
  to a particular compute machine."

That last one is the operational cost: with bytes on one machine's disk, a remote training
provider (`gpu_vps` / `colab`) has no way to read the dataset it was asked to train on.

So committed dataset versions are now written through the *same* `ArtifactStorage`
abstraction that already serves model artifacts — deliberately not a second abstraction.
Staging stays on local disk on purpose: it is transient, and the intake handlers need to
read the file back to parse and validate it.

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
from app.services.artifact_storage import get_artifact_storage


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


def dataset_version_key(dataset_id: str, version: int, filename: str) -> str:
    """Object key for one file of an immutable dataset version.

    The layout is ``datasets/{dataset_id}/v{version}/{filename}`` so a whole version is one
    S3 prefix — which is what makes "does this version already exist?" a single
    ``list_objects_v2(Prefix=...)`` rather than a per-file walk, and lets a future presigned
    GET (issue #254) hand out one prefix at a time.
    """

    return f"datasets/{dataset_id}/v{version}/{filename}"


class DatasetStorage:
    def __init__(
        self,
        base_dir: str | Path | None = None,
        *,
        store=None,
    ):
        self._base = Path(base_dir or settings.dataset_storage_dir)
        self._staging = self._base / "_staging"
        self._staging.mkdir(parents=True, exist_ok=True)
        # Resolved lazily-ish (one call, cached) so a test can inject a fake store, and so
        # importing this module never constructs a boto3 client.
        self._store = store if store is not None else get_artifact_storage()

    # --- staging (local, transient) ------------------------------------------------

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

    # --- commit (object storage, immutable) ----------------------------------------

    def commit_file(
        self, staging_id: str, dataset_id: str, version: int
    ) -> tuple[str, str, int]:
        """Promote the staged file to an immutable dataset version in object storage.

        Returns `(uri, filename, size_bytes)` where `uri` is the storage URI
        (`s3://{bucket}/datasets/{id}/v{N}/{file}` under `artifact_backend=minio`, else
        `file://{path}`). The caller persists it on `DatasetVersion.canonical_file_uri`,
        which is what issue #208's trainer pin reads.

        Unlike the local-only implementation this does not move a file: the staged copy is
        uploaded and then removed, so the same version is never readable from two places.

        Raises:
            FileNotFoundError: the staging directory holds no file.
            ArtifactExistsError: the version already exists in the store. A dataset version
                is immutable, so re-committing over one is a bug, not an update.
        """

        src_dir = self._staging / staging_id
        files = (
            [f for f in src_dir.iterdir() if f.is_file()] if src_dir.is_dir() else []
        )
        if not files:
            raise FileNotFoundError(f"No files in staging {staging_id}")
        # `f.name` comes from a directory listing, never from client input, so it is a
        # single component by construction (issue #246).
        source = files[0]
        data = source.read_bytes()
        key = dataset_version_key(dataset_id, version, source.name)
        uri = self._store.put_immutable(key, data)
        shutil.rmtree(src_dir, ignore_errors=True)
        return uri, source.name, len(data)

    def write_sidecar(
        self, dataset_id: str, version: int, name: str, payload: dict
    ) -> str:
        """Write a derived-metadata sidecar (`validation_report.json` / `schema.json`).

        The sidecars go to object storage next to the bytes they describe rather than
        staying local: they are the evidence for *why* a dataset version was accepted, so
        they have to travel with the dataset when a remote worker pulls it. Otherwise the
        training side would see bytes with no provenance attached.

        Written with `store()` rather than `put_immutable()`: these are derived views that
        `commit` legitimately rewrites when a re-validation replaces the manifest, and they
        are not the immutable payload.
        """

        key = dataset_version_key(dataset_id, version, name)
        return self._store.store(key, json.dumps(payload, indent=2, default=str))

    # --- reads (for training and download) ------------------------------------------

    def fetch_to(self, uri: str, dest: Path) -> Path:
        """Materialize the object at `uri` as a local file at `dest`.

        This is the read side that unblocks issue #208: the trainer subprocess can only open
        a local path, while the pin may well be `s3://`.
        """

        return self._store.download_to(uri, dest)

    def exists(self, uri: str) -> bool:
        return self._store.exists(uri)
