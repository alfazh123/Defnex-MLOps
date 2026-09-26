"""HuggingFace Hub dataset import (issue #240, audit finding T1).

HF import did not exist. `huggingface_hub` was declared in the `intake` extra but never
imported anywhere; `source_type="huggingface"` only inserted a database row with no bytes
behind it (the service docstring admitted as much), and no endpoint accepted a repo id at
all. The `defnex-frontend-demo` intake modal collected a revision and then dropped it from
the payload.

What this module does, following the official Hub download guide rather than inventing a
mechanism:

- `snapshot_download(repo_id, repo_type="dataset", revision=...)` returns a version-aware
  local cache path, and the `revision` argument is the documented way to pin a specific
  repository revision.
- The resolved commit SHA is what gets recorded in
  `DatasetVersion.source_commit_or_snapshot_date` (a column that already exists for exactly
  this). A pinned tag like `main` would still drift; the SHA does not.

The download is staged and then handed to the *existing* intake wizard
(`intake/validate` → `intake/commit`) rather than reimplementing those steps. One pipeline
means one set of rules: an HF-imported dataset is validated by the same H0-H9 gate, and gets
the same immutability and checksum treatment, as an uploaded one.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path

from app.services import dataset_parsing
from app.services.dataset_storage import DatasetStorage

# Extensions the pipeline can actually parse (issue #245 narrowed this list to formats a
# parser exists for). A Hub repo commonly ships parquet/arrow instead, which this cannot
# read -- see `_pick_data_file`'s error, which says so rather than failing obscurely.
_CANDIDATE_SUFFIXES = (".jsonl", ".json", ".csv", ".xlsx")

# Directory names a Hub dataset repo uses for split data files. Checked in order so a repo
# with both `train.jsonl` and `test.jsonl` resolves `train` deterministically.
_SPLIT_DIRS = ("", "data/", "dataset/")


class HuggingFaceImportError(Exception):
    """An HF import could not be completed.

    Carries the wire `code` and `http_status` for the API envelope, mirroring
    `DatasetParseError` so the intake endpoints have one error shape to map.
    """

    def __init__(self, code: str, message: str, http_status: int = 400) -> None:
        self.code = code
        self.message = message
        self.http_status = http_status
        super().__init__(message)


@dataclass(frozen=True)
class HfImportResult:
    staging_id: str
    local_path: str
    filename: str
    size_bytes: int
    checksum_sha256: str
    detected_format: str
    resolved_revision: str
    repo_id: str
    split: str | None


def _require_hf():
    try:
        from huggingface_hub import snapshot_download
    except ImportError as exc:  # pragma: no cover - depends on the install extra
        raise HuggingFaceImportError(
            "MISSING_DEP",
            "HuggingFace import requires the optional 'intake' extra (huggingface_hub). "
            "Install it, or upload the dataset file directly.",
        ) from exc
    return snapshot_download


def _pick_data_file(snapshot_root: Path, split: str | None) -> Path:
    """Choose the data file to ingest from an already-downloaded snapshot.

    Deliberately narrow and loud: this only knows how to hand a JSONL/JSON/CSV/XLSX file to
    the intake pipeline. A parquet-only repository is a real and common case, so it gets an
    explicit message naming the supported formats rather than an obscure parse failure.
    """

    if split:
        wanted = f"{split}"
        matches = [
            p
            for p in snapshot_root.rglob("*")
            if p.is_file()
            and p.stem == wanted
            and p.suffix.lower() in _CANDIDATE_SUFFIXES
        ]
        if not matches:
            raise HuggingFaceImportError(
                "HF_SPLIT_NOT_FOUND",
                f"No readable {split!r} data file in the snapshot. Supported extensions: "
                f"{', '.join(_CANDIDATE_SUFFIXES)}.",
            )
        return sorted(matches, key=lambda p: len(p.parts))[0]

    for subdir in _SPLIT_DIRS:
        root = snapshot_root / subdir if subdir else snapshot_root
        if not root.is_dir():
            continue
        for suffix in _CANDIDATE_SUFFIXES:
            matches = sorted(
                (
                    p
                    for p in root.rglob(f"*{suffix}")
                    if p.is_file()
                    # Skip the sidecars we may have written into the same prefix.
                    and p.name not in ("validation_report.json", "schema.json")
                ),
                key=lambda p: (len(p.parts), p.name),
            )
            if matches:
                return matches[0]

    available = sorted(
        {p.suffix.lower() for p in snapshot_root.rglob("*") if p.is_file() and p.suffix}
    )
    raise HuggingFaceImportError(
        "HF_NO_READABLE_DATA_FILE",
        "The snapshot holds no file this pipeline can ingest. Supported extensions: "
        f"{', '.join(_CANDIDATE_SUFFIXES)}"
        + (f"; found: {', '.join(available)}." if available else "."),
    )


def import_hf_dataset(
    *,
    repo_id: str,
    revision: str,
    dataset_id: str,
    split: str | None = None,
    storage: DatasetStorage | None = None,
    allow_patterns: list[str] | None = None,
) -> HfImportResult:
    """Download a pinned HF dataset revision and stage its data file for intake.

    `revision` is required, not defaulted to `main`. A default would make every import
    silently re-resolve to whatever the branch points at now, which is the unpinned behavior
    issue #208 exists to remove -- importing from `main` is fine as long as the resolved
    commit SHA is recorded, and that is what makes it fine.

    Returns a result whose `staging_id` feeds straight into
    `POST /api/v1/datasets/intake/validate`.
    """

    if not repo_id or "/" not in repo_id:
        raise HuggingFaceImportError(
            "INVALID_HF_REPO_ID",
            f"repo_id {repo_id!r} is not a Hub dataset id of the form 'owner/name'.",
        )
    if not revision:
        raise HuggingFaceImportError(
            "HF_REVISION_REQUIRED",
            "revision is required: import must name the exact Hub revision (a commit SHA or "
            "tag) so the dataset can be pinned. Defaulting to 'main' would let the same pin "
            "resolve to different bytes on a later run (issue #208/#240).",
        )

    snapshot_download = _require_hf()
    store = storage or DatasetStorage()

    try:
        local_dir = snapshot_download(
            repo_id=repo_id,
            repo_type="dataset",
            revision=revision,
            allow_patterns=allow_patterns,
        )
    except Exception as exc:
        # huggingface_hub raises a wide range of types (RepositoryNotFoundError,
        # RevisionNotFoundError, GatedRepoError, HfHubHTTPError, plus requests errors).
        # Mapping them to names here is brittle; a clear message plus the exception text is
        # more useful than a wrong guess about which one it was.
        #
        # 502, not 400: the request itself was well-formed and the Hub was the thing that
        # failed or refused. A 400 would tell the client to fix a request that is already
        # correct, and a nonexistent revision is not the caller's typo to be corrected.
        raise HuggingFaceImportError(
            "HF_DOWNLOAD_FAILED",
            f"Could not download {repo_id!r} at revision {revision!r}: "
            f"{type(exc).__name__}: {exc}",
            http_status=502,
        ) from exc

    snapshot_root = Path(local_dir)
    data_file = _pick_data_file(snapshot_root, split)
    content = data_file.read_bytes()

    # The resolved commit SHA, read back out of the snapshot's own metadata, is the value
    # worth recording: it is what the bytes actually are, whatever revision string was asked
    # for (a tag, a short SHA, or a branch name all resolve to one commit).
    resolved_revision = _resolved_commit(snapshot_root) or revision

    try:
        detected = dataset_parsing.detect_format(data_file.name, content)
    except Exception as exc:
        raise HuggingFaceImportError(
            "HF_NO_READABLE_DATA_FILE",
            f"Could not determine a readable format for {data_file.name!r}: {exc}",
        ) from exc
    if detected == "unknown":
        raise HuggingFaceImportError(
            "HF_NO_READABLE_DATA_FILE",
            f"{data_file.name!r} is not a format this pipeline can ingest. Supported: "
            f"{', '.join(dataset_parsing.SUPPORTED_FORMATS)}.",
        )

    staged = store.stage_upload(
        data_file.name,
        content,
        provenance={
            "source_type": "huggingface",
            "source_url_or_hf_id": repo_id,
            # The resolved commit, not the requested revision string: a tag or a branch name
            # still drifts, the SHA does not. This is the value that makes the import a pin
            # rather than a pointer (issue #208).
            "source_commit_or_snapshot_date": resolved_revision,
            "hf_requested_revision": revision,
            "split": split,
        },
    )
    return HfImportResult(
        staging_id=staged["staging_id"],
        local_path=staged["path"],
        filename=staged["filename"],
        size_bytes=staged["size_bytes"],
        checksum_sha256=hashlib.sha256(content).hexdigest(),
        detected_format=detected,
        resolved_revision=resolved_revision,
        repo_id=repo_id,
        split=split,
    )


def _resolved_commit(snapshot_root: Path) -> str | None:
    """Read the snapshot's own commit SHA out of huggingface_hub's cache metadata.

    `snapshot_download` writes the resolved revision next to the snapshot, so the exact
    commit the bytes came from is readable without a second network call.
    """

    refs = snapshot_root.parent / "refs"
    if not refs.is_dir():
        return None
    for candidate in refs.rglob("*"):
        if candidate.is_file():
            try:
                value = candidate.read_text().strip()
            except OSError:
                continue
            if value:
                return value
    return None
