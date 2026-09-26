"""Model version lineage (issue #236).

The meeting's fourth agenda item: *"trace artifact X — which training did it come from, when,
which dataset, what metadata, what config."*

Every piece of that information already existed, but spread across five tables, and
`DatasetVersion.canonical_file_uri`/`raw_file_uri` were write-only (nothing read them). So
answering the question meant a caller — including a frontend — had to join
`model_versions` → `training_runs` → `dataset_versions` → `validation_reports`, then read a
JSON sidecar off the object store for the rest. This module is the one place that does the
joining, so the answer is a single response and the joins cannot drift apart from each other.

Each sub-object names the fields that were previously reachable-but-scattered:

- **run** — which run, and the wall-clock bounds of the execution (was: 2 columns)
- **dataset** — id, version, URI, checksum, and the validation report that cleared it. The
  id/version pair was join-derived and not filterable; the checksum and report ref were
  unavailable at all (issue #237 wrote the ref)
- **config** — the full `training_config` plus its hash. Because issue #209 snapshots the
  dataset pin into the config, this object also identifies the dataset the bytes came from
  even if the dataset version row has since changed
- **metadata** — the artifact's `metadata.json`, still a sidecar (issue #235/E3 owns lifting
  it into a table)
- **artifact** — URI and recorded checksum per artifact, verified against the store

Gaps are reported rather than papered over: a field that cannot be resolved comes back in
`gaps[]` with a reason, so a caller can tell "we don't know" from "it is null".
"""

from __future__ import annotations

from sqlalchemy.orm import Session

from app.models.dataset import DatasetVersion
from app.models.model import ModelVersion
from app.models.training import TrainingRun
from app.services.artifact_storage import get_artifact_storage


class LineageError(ValueError):
    """The requested model version does not exist.

    The router turns this into a 404; every *content* gap below is reported in `gaps[]`
    instead, because "this artifact has no dataset checksum recorded" is an answer, not an
    error.
    """


def _gap(gaps: list[dict], field: str, reason: str) -> None:
    gaps.append({"field": field, "reason": reason})


def build_lineage(
    db: Session,
    model_id: str,
    version: int,
    *,
    include_artifact_metadata: bool = True,
    storage=None,
) -> dict:
    """Assemble the full lineage record for one model version.

    Raises:
        LineageError: no such `(model_id, version)`.
    """

    model_version: ModelVersion | None = (
        db.query(ModelVersion)
        .filter_by(model_id=model_id, version=version)
        .one_or_none()
    )
    if model_version is None:
        raise LineageError(f'model_id "{model_id}" version {version} not found')

    gaps: list[dict] = []
    run: TrainingRun | None = model_version.training_run

    run_block = {
        "training_run_id": run.training_run_id if run else None,
        "triggered_by": run.triggered_by if run else None,
        "started_at": model_version.training_started_at,
        "completed_at": model_version.training_completed_at,
        "status": run.status if run else None,
        "git_commit": model_version.git_commit,
        "retried_from": (run.retry_of if run else None),
        "retry_count": (run.retry_count if run else None),
    }
    if run is None:
        _gap(
            gaps,
            "training_run_id",
            f"model version {model_id}:{version} has no training_run_id; it was not "
            "produced by a training run on this system (seed data, or an import).",
        )

    dataset_block = _dataset_block(db, model_version, run, gaps)
    config_block = _config_block(model_version, run, gaps)
    artifact_block = _artifact_block(
        model_version, include_artifact_metadata, storage, gaps
    )

    return {
        "model_id": model_id,
        "version": version,
        "name": model_version.name,
        "status": model_version.status,
        "base_model": model_version.base_model,
        "created_at": model_version.created_at,
        "created_by": model_version.created_by,
        "run": run_block,
        "dataset": dataset_block,
        "config": config_block,
        "metadata": artifact_block["metadata"],
        "artifacts": artifact_block["artifacts"],
        "previous_model_id": model_version.previous_model_id,
        "gaps": gaps,
    }


def _dataset_block(
    db: Session,
    model_version: ModelVersion,
    run: TrainingRun | None,
    gaps: list[dict],
) -> dict:
    dataset_version: DatasetVersion | None = (
        run.dataset_version if run is not None else None
    )
    if dataset_version is None:
        _gap(
            gaps,
            "dataset",
            "the producing training run has no dataset version attached, so the dataset "
            "this model was trained on cannot be identified",
        )
        return {"dataset_id": None, "dataset_version": None}

    # Prefer the pin (issue #209) over the join: the pin is what the trainer actually read,
    # and it survives the live row being edited. The join is reported alongside it so a
    # caller can see whether the two still agree.
    pin = (model_version.training_config or {}).get("dataset_pin") or {}
    return {
        "dataset_id": pin.get("dataset_id") or dataset_version.dataset_id,
        "dataset_version": pin.get("dataset_version") or dataset_version.version,
        "pinned": bool(pin),
        "file_uri": pin.get("file_uri") or dataset_version.canonical_file_uri,
        "checksum_sha256": pin.get("content_hash") or None,
        "source_type": dataset_version.source_type,
        "source_format": dataset_version.source_format,
        "row_count": dataset_version.row_count,
        "license": dataset_version.license,
        "pin_matches_row": (
            pin.get("dataset_version") == dataset_version.version if pin else None
        ),
        "validation_report_ref": model_version.dataset_validation_report_ref,
    }


def _config_block(
    model_version: ModelVersion, run: TrainingRun | None, gaps: list[dict]
) -> dict:
    config = model_version.training_config
    if not config and run is not None:
        config = run.training_config
    if not config:
        _gap(
            gaps,
            "config",
            "no training_config on the model version or its training run",
        )
    return {
        "training_config": config or {},
        "training_config_hash": model_version.training_config_hash,
    }


def _artifact_block(
    model_version: ModelVersion,
    include_metadata: bool,
    storage,
    gaps: list[dict],
) -> dict:
    artifacts = []
    metadata: dict = {}
    for entry in model_version.artifacts or []:
        uri = entry.get("uri") if isinstance(entry, dict) else None
        record = {
            "type": entry.get("type") if isinstance(entry, dict) else None,
            "uri": uri,
            # `checksum` is written on the runner-driven registration path. A row registered
            # through the flat `artifact_uri` path has no checksum, and saying so beats
            # reporting null with no explanation.
            "checksum": entry.get("checksum") if isinstance(entry, dict) else None,
        }
        if not record["checksum"]:
            _gap(
                gaps,
                "artifacts[].checksum",
                f"artifact {uri!r} has no recorded checksum; it was registered without a "
                "staging directory, so nothing was hashed",
            )
        artifacts.append(record)
        if not uri or not include_metadata:
            continue
        # metadata.json is the artifact-sidecar (issue #235/E3 owns promoting it to a table).
        # Read failures are reported, never raised: the row-level lineage above is still a
        # complete answer, and a sidecar that cannot be read must not fail the endpoint.
        try:
            sidecar = (storage or get_artifact_storage()).read_metadata(uri)
        except Exception as exc:  # noqa: BLE001 - a sidecar read must not fail lineage
            _gap(
                gaps,
                "metadata",
                f"could not read metadata.json for {uri!r}: {type(exc).__name__}: {exc}",
            )
            continue
        if sidecar:
            metadata = sidecar
        else:
            _gap(
                gaps,
                "metadata",
                f"no metadata.json found for {uri!r} (an artifact registered before issue "
                "#62, or an incomplete one)",
            )

    return {"artifacts": artifacts, "metadata": metadata}
