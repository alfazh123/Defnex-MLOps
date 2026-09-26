"""Dataset pinning for training runs (issues #208 / #209).

A training run already carried a foreign key to the dataset version it was created from
(`training_runs.dataset_version_id`), and the wizard even refused to create a run without a
PASS validation report. But the trainer never read that dataset's bytes: it loaded
`training_config.hf_dataset` — a local path or a bare HuggingFace Hub name with no revision
pin. So a run recorded "trained on version N" while the bytes it actually read were whatever
`hf_dataset` happened to point at, and `raw_file_uri` / `canonical_file_uri` were written but
never read by anyone. Pinning was cosmetic, and the meeting's fourth agenda item ("trace
artifact X: from which training, when, which dataset, what metadata, what config") could not
have been answered accurately.

This module is the fix, in two halves:

- :func:`snapshot_pin` (issue #209) freezes the dataset's identity into the run's
  `training_config` at creation time, so the record survives the dataset version being
  edited or deleted.
- :func:`resolve_pin_to_config` (issue #208) turns that pin into a local file the trainer
  subprocess can actually open, and refuses — loudly — when the pinned object is gone.

**No silent fallback.** A run with a pin never falls back to `hf_dataset`, and a run whose
pin cannot be resolved fails with an explicit message rather than quietly training on
something else. Training on bytes other than the ones the record names is the exact failure
this issue exists to prevent, so there is no code path that produces it.
"""

from __future__ import annotations

import shutil
from dataclasses import dataclass
from pathlib import Path

from sqlalchemy.orm import Session

from app.models.dataset import DatasetVersion as DatasetVersionModel
from app.services.dataset_storage import DatasetStorage

# Key under which the pin is stored in `TrainingRun.training_config`. Namespaced and
# prefixed so it cannot collide with a `TrainingConfig` knob (which is `extra="allow"`,
# meaning any key a client sends is accepted -- issue A9/B1 territory).
PIN_KEY = "dataset_pin"


class DatasetPinError(RuntimeError):
    """A training run's dataset pin could not be honoured.

    Raised instead of falling back, so the worker marks the run FAILED with this message
    (issue #208: "pin points at a missing object -> fail with a clear message").
    """


@dataclass(frozen=True)
class DatasetPin:
    """The frozen identity of the dataset a run trains on."""

    dataset_id: str
    dataset_version: int
    file_uri: str
    content_hash: str
    source_format: str
    row_count: int | None

    def to_dict(self) -> dict:
        return {
            "dataset_id": self.dataset_id,
            "dataset_version": self.dataset_version,
            "file_uri": self.file_uri,
            "content_hash": self.content_hash,
            "source_format": self.source_format,
            "row_count": self.row_count,
        }

    @classmethod
    def from_dict(cls, payload: dict) -> DatasetPin:
        """Parse a stored pin.

        Only `dataset_id` and `dataset_version` are required here. An empty `file_uri` is a
        legitimate state -- a dataset version created outside the intake wizard never gets
        `canonical_file_uri` written -- and it has its own, more specific diagnosis in
        `resolve_pin_to_file` ("its bytes were never committed"). Rejecting it at parse time
        would replace that actionable message with a generic one.
        """

        missing = [
            k for k in ("dataset_id", "dataset_version") if payload.get(k) in (None, "")
        ]
        if missing:
            raise DatasetPinError(
                f"training_config.{PIN_KEY} is missing required field(s): "
                f"{', '.join(missing)}"
            )
        return cls(
            dataset_id=str(payload["dataset_id"]),
            dataset_version=int(payload["dataset_version"]),
            file_uri=str(payload.get("file_uri") or ""),
            content_hash=str(payload.get("content_hash") or ""),
            source_format=str(payload.get("source_format") or "jsonl"),
            row_count=payload.get("row_count"),
        )


def snapshot_pin(db: Session, dataset_version: DatasetVersionModel) -> DatasetPin:
    """Freeze `dataset_version`'s identity for a training run (issue #209).

    `content_hash` comes from the version's latest validation report, which is the checksum
    of the exact bytes that were examined -- the same value `commit` verified against the
    staged file. A version with no report still yields a pin (the run-creation gate already
    requires one), just without the hash.
    """

    from app.services import validation_service

    report = validation_service.get_latest_validation_report(db, dataset_version)
    return DatasetPin(
        dataset_id=dataset_version.dataset_id,
        dataset_version=dataset_version.version,
        file_uri=dataset_version.canonical_file_uri or "",
        content_hash=(report.content_hash if report else "") or "",
        source_format=dataset_version.source_format or "jsonl",
        row_count=dataset_version.row_count,
    )


def with_pin(config: dict, pin: DatasetPin) -> dict:
    """Return a copy of `config` carrying the pin. Never mutates the caller's dict."""

    return {**config, PIN_KEY: pin.to_dict()}


def read_pin(training_config: dict) -> DatasetPin | None:
    """The pin stored on a run, or None for a run created before issue #209.

    A missing pin is not an error here — it marks the run as legacy, and
    :func:`resolve_pin_to_config` decides what to do about it.
    """

    payload = (training_config or {}).get(PIN_KEY)
    if not isinstance(payload, dict):
        return None
    return DatasetPin.from_dict(payload)


def _verify_checksum(path: Path, pin: DatasetPin) -> None:
    """Confirm the bytes we just materialized still hash to the pinned checksum.

    The pin exists so "the bytes this run trained on" is a checkable claim. Object storage
    is not supposed to mutate an object, but a re-commit, a manual copy, or a bug that points
    the pin at the wrong key would all turn a wrong-but-plausible training run into a
    silent one. Verifying here is what makes the claim true rather than hopeful.
    """

    import hashlib

    if not pin.content_hash:
        # No recorded checksum (a version validated before the hash was persisted). There is
        # nothing to compare against, so this is reported, not enforced.
        return
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            h.update(chunk)
    actual = h.hexdigest()
    if actual != pin.content_hash:
        raise DatasetPinError(
            f"pinned dataset bytes do not match the pinned checksum for "
            f"{pin.dataset_id} v{pin.dataset_version}: expected "
            f"sha256:{pin.content_hash}, got sha256:{actual}. Refusing to train on "
            "bytes the run does not claim to have used."
        )


def resolve_pin_to_file(
    pin: DatasetPin,
    work_dir: Path,
    *,
    storage: DatasetStorage | None = None,
) -> Path:
    """Materialize the pinned dataset as a local file inside `work_dir` (issue #208).

    Handles both pin shapes: a local `file://`/path (dev, `artifact_backend=local`) and
    `s3://` (MinIO). The local case is still copied rather than used in place, so the
    trainer cannot mutate the immutable stored copy.

    Raises:
        DatasetPinError: the pin has no URI, or the object it names is not there.
    """

    if not pin.file_uri:
        raise DatasetPinError(
            f"training run pinned to dataset {pin.dataset_id} v{pin.dataset_version}, but "
            "that version has no canonical_file_uri — its bytes were never committed. "
            "Re-run intake commit for this dataset version before training on it."
        )

    store = storage or DatasetStorage()
    dest_dir = work_dir / "dataset"
    dest_dir.mkdir(parents=True, exist_ok=True)
    dest = dest_dir / "train.jsonl"
    try:
        path = store.fetch_to(pin.file_uri, dest)
    except FileNotFoundError as exc:
        raise DatasetPinError(
            f"pinned dataset object {pin.file_uri!r} for {pin.dataset_id} "
            f"v{pin.dataset_version} is missing from {store._store.__class__.__name__}. "
            "The run cannot fall back to another dataset; re-import or re-commit the "
            "dataset version, or start a new run against one that exists."
        ) from exc
    except Exception as exc:
        raise DatasetPinError(
            f"could not read pinned dataset object {pin.file_uri!r} for {pin.dataset_id} "
            f"v{pin.dataset_version}: {exc}"
        ) from exc

    _verify_checksum(path, pin)
    return path


def resolve_pin_to_config(
    training_config: dict,
    work_dir: Path,
    *,
    storage: DatasetStorage | None = None,
) -> tuple[dict, str]:
    """Return `(config_for_trainer, source)` with the dataset resolved to a local file.

    `source` is `"pinned"` when the dataset came from the pin and `"legacy_hf_dataset"` when
    the run predates issue #209 and only has `hf_dataset` — recorded so the manifest says
    which path was used instead of leaving it to be guessed.

    A run that *has* a pin never falls back to `hf_dataset`. If the pinned object is
    unreadable the call raises `DatasetPinError`; training on different bytes while the
    record claims otherwise is the bug this replaces.
    """

    pin = read_pin(training_config)
    config = dict(training_config or {})

    if pin is None:
        hf_dataset = (training_config or {}).get("hf_dataset") or ""
        if not hf_dataset:
            raise DatasetPinError(
                "training run has no dataset pin and no legacy hf_dataset. Every run must "
                "name the dataset version it trains on (issue #208)."
            )
        # Pre-#209 run: the wizard recorded a version but nothing ever read its bytes, so
        # there is nothing better to fall back to. Kept working, explicitly labelled.
        config["dataset_local_path"] = hf_dataset
        return config, "legacy_hf_dataset"

    path = resolve_pin_to_file(pin, work_dir, storage=storage)
    config["dataset_local_path"] = str(path)
    # hf_dataset is cleared rather than left in place: it is an unpinned dataset reference,
    # and leaving it next to a resolved path invites a future refactor to prefer it.
    config.pop("hf_dataset", None)
    return config, "pinned"


def cleanup_workspace(work_dir: Path) -> None:
    """Remove the per-run dataset download. Best-effort, like the staging cleanup."""

    shutil.rmtree(work_dir, ignore_errors=True)
