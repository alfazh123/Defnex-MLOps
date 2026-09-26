"""Dataset pinning for training runs (issue #208 trainer pin, #209 identity snapshot).

The property under test throughout: **the bytes a run trains on are the bytes its record
names.** Anything that could make those diverge — a missing object, a mutated object, a
legacy `hf_dataset` that disagrees with the pin — must fail loudly rather than quietly
train on something else.
"""

import json
from datetime import UTC, datetime
from pathlib import Path

import pytest

from app.models.dataset import DatasetVersion
from app.schemas.training import TrainingConfig, TrainingRunCreateRequest
from app.services import dataset_pinning, dataset_service, training_service
from app.services.artifact_storage import LocalFilesystemArtifactStorage
from app.services.dataset_pinning import PIN_KEY, DatasetPin, DatasetPinError
from app.services.dataset_storage import DatasetStorage

RECORD = {
    "id": "r1",
    "messages": [
        {"role": "user", "content": "Apa itu DEFNEX?"},
        {
            "role": "assistant",
            "content": "DEFNEX adalah platform MLOps untuk integrasi model AI secara "
            "terpusat yang menyatukan empat sub-proyek universitas dalam satu arsitektur "
            "intelijen terpadu agar tim dapat mengelola dataset dan model dari satu tempat.",
        },
    ],
    "metadata": {"source_dataset": "pin", "source_id": "s1"},
}

PAYLOAD = ("\n".join(json.dumps(RECORD) for _ in range(2)) + "\n").encode()


@pytest.fixture
def store(tmp_path):
    return LocalFilesystemArtifactStorage(tmp_path / "artifacts")


@pytest.fixture
def dataset_storage(tmp_path, store):
    return DatasetStorage(tmp_path / "datasets", store=store)


def _committed_version(db, tmp_path, store, *, dataset_id="ds-pin", version=1, rows=2):
    """A DatasetVersion whose bytes are really in the store, plus a PASS validation report.

    Returns `(dataset_version, checksum)` so tests can assert against the same value the
    system recorded.
    """
    import hashlib

    payload = ("\n".join(json.dumps(RECORD) for _ in range(rows)) + "\n").encode()
    dataset_service.register_dataset(db, dataset_id)
    dv = DatasetVersion(
        dataset_id=dataset_id,
        version=version,
        status="PROCESSED",
        source_type="file_upload",
        source_format="jsonl",
        row_count=rows,
        created_at=datetime.now(UTC),
        canonical_file_uri=store.put_immutable(
            f"datasets/{dataset_id}/v{version}/train.jsonl", payload
        ),
    )
    db.add(dv)
    db.flush()

    from app.models.validation import ValidationReport

    report = ValidationReport(
        dataset_version_id=dv.id,
        rule_set_version="2.2.0",
        run_at=datetime.now(UTC),
        record_count=rows,
        content_hash=hashlib.sha256(payload).hexdigest(),
        status_counts={"VALID": rows, "INVALID": 0, "NEEDS_REVIEW": 0},
        gate_decision="PASS",
        gate_reason="ok",
    )
    db.add(report)
    db.commit()
    db.refresh(dv)
    return dv, report.content_hash


def _create_run(db, dv, dataset_id="ds-pin", version=1, **config):
    return training_service.create_training_run(
        db,
        dv,
        TrainingRunCreateRequest(
            dataset_id=dataset_id,
            dataset_version=version,
            model_id="pin-model",
            base_model="unsloth/Qwen3-0.6B",
            training_config=TrainingConfig(**config),
        ),
    )


# ── A2 / #209 — the identity snapshot ─────────────────────────────────────────


class TestIdentitySnapshot:
    def test_run_records_the_dataset_it_was_created_from(
        self, db_session, tmp_path, store
    ):
        """The FK alone points at a mutable row; the snapshot in `training_config` is what
        survives the dataset version being edited or deleted."""
        dv, checksum = _committed_version(db_session, tmp_path, store)
        run = _create_run(db_session, dv)

        pin = run.training_config[PIN_KEY]
        assert pin["dataset_id"] == "ds-pin"
        assert pin["dataset_version"] == 1
        assert pin["content_hash"] == checksum
        assert pin["file_uri"] == dv.canonical_file_uri
        assert pin["source_format"] == "jsonl"
        assert pin["row_count"] == 2

    def test_snapshot_does_not_follow_later_edits_to_the_version(
        self, db_session, tmp_path, store
    ):
        """The point of snapshotting: the run's record of "which dataset" is frozen, so a
        later edit to the live row cannot silently change what the run claims it used.

        (Deleting the row outright is not testable here: `training_runs.dataset_version_id`
        is NOT NULL with an ORM cascade, so a delete is rejected by the schema itself. The
        property that matters — the snapshot is independent data, not a live reference — is
        what this asserts.)
        """
        dv, checksum = _committed_version(db_session, tmp_path, store)
        run = _create_run(db_session, dv)
        pinned = dict(run.training_config[PIN_KEY])

        dv.row_count = 999
        dv.source_format = "csv"
        dv.canonical_file_uri = "file:///somewhere/else.jsonl"
        db_session.commit()

        assert run.training_config[PIN_KEY] == pinned
        assert run.training_config[PIN_KEY]["content_hash"] == checksum
        assert run.training_config[PIN_KEY]["row_count"] == 2
        assert run.training_config[PIN_KEY]["source_format"] == "jsonl"

    def test_snapshot_is_plain_data_that_outlives_the_orm_object(
        self, db_session, tmp_path, store
    ):
        """It lives in a JSON column, so it survives detachment — nothing about reading it
        later requires the DatasetVersion row to be loaded."""
        dv, _ = _committed_version(db_session, tmp_path, store)
        run = _create_run(db_session, dv)

        detached = json.loads(json.dumps(run.training_config))
        db_session.expunge_all()

        assert DatasetPin.from_dict(detached[PIN_KEY]).dataset_id == "ds-pin"
        assert DatasetPin.from_dict(detached[PIN_KEY]).file_uri.startswith("file://")

    def test_snapshot_does_not_mutate_the_request_config(
        self, db_session, tmp_path, store
    ):
        dv, _ = _committed_version(db_session, tmp_path, store)
        request = TrainingRunCreateRequest(
            dataset_id="ds-pin",
            dataset_version=1,
            model_id="pin-model",
            base_model="unsloth/Qwen3-0.6B",
            training_config=TrainingConfig(epochs=2),
        )
        training_service.create_training_run(db_session, dv, request)
        # The caller's request object is not polluted with our internal key.
        assert PIN_KEY not in request.training_config.model_dump()

    def test_pin_is_readable_back(self, db_session, tmp_path, store):
        dv, _ = _committed_version(db_session, tmp_path, store)
        run = _create_run(db_session, dv)
        assert dataset_pinning.read_pin(run.training_config) == DatasetPin(
            dataset_id="ds-pin",
            dataset_version=1,
            file_uri=dv.canonical_file_uri,
            content_hash=dataset_pinning.read_pin(run.training_config).content_hash,
            source_format="jsonl",
            row_count=2,
        )


# ── A1 / #208 — the trainer loads the pinned bytes ────────────────────────────


class TestPinResolution:
    def test_resolves_to_a_local_file_with_the_pinned_bytes(
        self, db_session, tmp_path, store, dataset_storage
    ):
        dv, _ = _committed_version(db_session, tmp_path, store)
        run = _create_run(db_session, dv)

        config, source = dataset_pinning.resolve_pin_to_config(
            run.training_config, tmp_path / "work", storage=dataset_storage
        )
        assert source == "pinned"
        path = Path(config["dataset_local_path"])
        assert path.is_file()
        assert path.read_bytes() == PAYLOAD

    def test_works_for_an_s3_pin(self, tmp_path, monkeypatch, dataset_storage):
        """The pin may be `s3://` after issue #214; the trainer can only open a local path."""
        payload = PAYLOAD
        store = dataset_storage._store
        monkeypatch.setattr(
            store,
            "get_bytes",
            lambda uri: payload,
        )
        monkeypatch.setattr(store, "exists", lambda uri: True)
        pin = DatasetPin(
            dataset_id="ds-s3",
            dataset_version=2,
            file_uri="s3://artifacts/datasets/ds-s3/v2/train.jsonl",
            content_hash="",
            source_format="jsonl",
            row_count=2,
        )
        path = dataset_pinning.resolve_pin_to_file(
            pin, tmp_path / "work", storage=dataset_storage
        )
        assert path.read_bytes() == payload
        assert "s3://artifacts/datasets/ds-s3/v2/train.jsonl" in str(pin.file_uri)

    def test_hf_dataset_is_never_used_when_a_pin_exists(
        self, db_session, tmp_path, store, dataset_storage
    ):
        """A pin and an unpinned `hf_dataset` disagreeing is exactly the ambiguity that made
        pinning cosmetic. The pin wins, and the stale field is dropped."""
        dv, _ = _committed_version(db_session, tmp_path, store)
        run = _create_run(db_session, dv)
        run.training_config = {
            **run.training_config,
            "hf_dataset": "some/other-hub-dataset",
        }
        db_session.flush()

        config, source = dataset_pinning.resolve_pin_to_config(
            run.training_config, tmp_path / "work", storage=dataset_storage
        )
        assert source == "pinned"
        assert "hf_dataset" not in config
        assert Path(config["dataset_local_path"]).read_bytes() == PAYLOAD

    def test_missing_object_fails_with_a_clear_message(
        self, db_session, tmp_path, store, dataset_storage
    ):
        """Acceptance: "pin points at a missing object -> fail with a clear message"."""
        dv, _ = _committed_version(db_session, tmp_path, store)
        run = _create_run(db_session, dv)
        run.training_config = {
            **run.training_config,
            PIN_KEY: {
                **run.training_config[PIN_KEY],
                "file_uri": "file:///nonexistent/ds-pin/v1/train.jsonl",
            },
        }

        with pytest.raises(DatasetPinError) as excinfo:
            dataset_pinning.resolve_pin_to_config(
                run.training_config, tmp_path / "work", storage=dataset_storage
            )
        message = str(excinfo.value)
        assert "is missing" in message
        assert "cannot fall back" in message
        assert "ds-pin" in message

    def test_missing_uri_explains_the_version_was_never_committed(
        self, db_session, tmp_path, dataset_storage
    ):
        """A version created outside the intake wizard has no `canonical_file_uri`. The
        message has to say that, not complain about a missing field."""
        dataset_service.register_dataset(db_session, "ds-nobytes")
        dv = DatasetVersion(
            dataset_id="ds-nobytes",
            version=1,
            status="PENDING",
            source_type="seed",
            source_format="jsonl",
            created_at=datetime.now(UTC),
        )
        db_session.add(dv)
        db_session.commit()
        run = _create_run(db_session, dv, dataset_id="ds-nobytes")

        with pytest.raises(DatasetPinError) as excinfo:
            dataset_pinning.resolve_pin_to_config(
                run.training_config, tmp_path / "work", storage=dataset_storage
            )
        assert "never committed" in str(excinfo.value)
        assert "ds-nobytes" in str(excinfo.value)

    def test_checksum_mismatch_refuses_to_train(
        self, db_session, tmp_path, store, dataset_storage
    ):
        """The pin makes "these are the bytes" a *checkable* claim. If the object no longer
        hashes to the recorded value, the run must not proceed on it."""
        dv, checksum = _committed_version(db_session, tmp_path, store)
        run = _create_run(db_session, dv)
        run.training_config = {
            **run.training_config,
            PIN_KEY: {**run.training_config[PIN_KEY], "content_hash": "0" * 64},
        }

        with pytest.raises(DatasetPinError) as excinfo:
            dataset_pinning.resolve_pin_to_config(
                run.training_config, tmp_path / "work", storage=dataset_storage
            )
        message = str(excinfo.value)
        assert "do not match the pinned checksum" in message
        assert checksum[:12] in message

    def test_legacy_run_without_a_pin_still_works_and_is_labelled(
        self, tmp_path, dataset_storage
    ):
        """A run created before issue #209 has no pin and nothing better to fall back to, so
        it keeps working — but reports which path it used instead of leaving it to be
        guessed."""
        legacy = {"hf_dataset": str(tmp_path / "old.jsonl")}
        config, source = dataset_pinning.resolve_pin_to_config(
            legacy, tmp_path / "work", storage=dataset_storage
        )
        assert source == "legacy_hf_dataset"
        assert config["dataset_local_path"] == str(tmp_path / "old.jsonl")

    def test_run_with_neither_pin_nor_hf_dataset_is_refused(
        self, tmp_path, dataset_storage
    ):
        with pytest.raises(DatasetPinError) as excinfo:
            dataset_pinning.resolve_pin_to_config(
                {}, tmp_path / "work", storage=dataset_storage
            )
        assert "no dataset pin" in str(excinfo.value)

    def test_malformed_pin_is_rejected_with_the_field_named(
        self, tmp_path, dataset_storage
    ):
        with pytest.raises(DatasetPinError) as excinfo:
            dataset_pinning.resolve_pin_to_config(
                {PIN_KEY: {"dataset_version": 1}},
                tmp_path / "work",
                storage=dataset_storage,
            )
        assert "dataset_id" in str(excinfo.value)

    def test_resolution_leaves_the_persisted_config_untouched(
        self, db_session, tmp_path, store, dataset_storage
    ):
        """The local path is per-attempt scratch; persisting it would be meaningless on the
        next retry and would leak a temp path into the run's permanent record."""
        dv, _ = _committed_version(db_session, tmp_path, store)
        run = _create_run(db_session, dv)
        before = json.dumps(run.training_config, sort_keys=True)

        dataset_pinning.resolve_pin_to_config(
            run.training_config, tmp_path / "work", storage=dataset_storage
        )
        assert json.dumps(run.training_config, sort_keys=True) == before
        assert "dataset_local_path" not in run.training_config

    def test_cleanup_removes_the_workspace(self, tmp_path):
        """The per-attempt dataset download is scratch; a retried run (`retry_of`) would
        otherwise leave a full-size copy behind on every attempt."""
        work = tmp_path / "work"
        (work / "dataset").mkdir(parents=True)
        (work / "dataset" / "train.jsonl").write_bytes(b"x" * 1024)
        dataset_pinning.cleanup_workspace(work)
        assert not work.exists()
