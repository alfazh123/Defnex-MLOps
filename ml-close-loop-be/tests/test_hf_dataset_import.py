"""HuggingFace Hub dataset import (issue #240, audit finding T1).

Every test here runs with `snapshot_download` faked, so CI needs no network and no Hub
account. What is actually asserted is the behavior the issue cares about: `revision` is
mandatory and passed through verbatim, the *resolved* commit SHA is what gets recorded, and
the result enters the same intake pipeline an upload does.
"""

import json
from pathlib import Path

import pytest

from app.services import hf_dataset_import
from app.services.artifact_storage import LocalFilesystemArtifactStorage
from app.services.dataset_storage import DatasetStorage
from app.services.hf_dataset_import import HuggingFaceImportError

from tests.conftest import auth_header

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
    "metadata": {"source_dataset": "hf", "source_id": "s1"},
}


def _records(count: int) -> bytes:
    """Distinct records: byte-identical ones would trip H7_duplicate, and with the
    tightened gate (issue #241) 2 of 3 duplicates is a FAIL, not a PASS."""

    return (
        "\n".join(
            json.dumps(
                {
                    **RECORD,
                    "id": f"r{i}",
                    "messages": [
                        {"role": "user", "content": f"Pertanyaan {i} tentang DEFNEX?"},
                        {
                            "role": "assistant",
                            "content": RECORD["messages"][1]["content"]
                            + f" Bagian {i}.",
                        },
                    ],
                    "metadata": {**RECORD["metadata"], "source_id": f"s{i}"},
                }
            )
            for i in range(count)
        )
        + "\n"
    ).encode()


PAYLOAD = _records(3)

RESOLVED_SHA = "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"


@pytest.fixture
def store(tmp_path):
    return LocalFilesystemArtifactStorage(tmp_path / "artifacts")


@pytest.fixture
def storage(tmp_path, store):
    return DatasetStorage(tmp_path / "datasets", store=store)


def _fake_snapshot_download(root: Path, calls: list):
    """Stand-in for `huggingface_hub.snapshot_download` that records how it was called."""

    def _download(*, repo_id, repo_type, revision, allow_patterns=None, **kwargs):
        calls.append(
            {
                "repo_id": repo_id,
                "repo_type": repo_type,
                "revision": revision,
                "allow_patterns": allow_patterns,
            }
        )
        return str(root)

    return _download


def _build_snapshot(tmp_path, files: dict[str, bytes], *, commit: str = RESOLVED_SHA):
    """A snapshot directory shaped like huggingface_hub's cache, refs commit included."""
    root = tmp_path / "snapshots" / "abc123"
    for rel, data in files.items():
        p = root / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_bytes(data)
    refs = root.parent / "refs" / "main"
    refs.parent.mkdir(parents=True, exist_ok=True)
    refs.write_text(commit + "\n")
    return root


@pytest.fixture
def fake_hub(monkeypatch, tmp_path):
    """Patch the module's huggingface_hub import; yields the list of recorded calls."""

    def _install(files=None, *, commit=RESOLVED_SHA, raises=None):
        root = _build_snapshot(
            tmp_path,
            files if files is not None else {"data/train.jsonl": PAYLOAD},
            commit=commit,
        )
        calls: list = []
        if raises is not None:

            def _boom(**kwargs):
                calls.append(kwargs)
                raise raises

            fn = _boom
        else:
            fn = _fake_snapshot_download(root, calls)
        monkeypatch.setattr(hf_dataset_import, "_require_hf", lambda: fn, raising=True)
        return calls

    return _install


# ── the documented mechanism ──────────────────────────────────────────────────


class TestSnapshotDownload:
    def test_calls_snapshot_download_with_dataset_type_and_revision(
        self, fake_hub, storage
    ):
        """The official Hub API: `repo_type="dataset"` plus an explicit `revision`. A
        `model` type or a missing revision would fetch the wrong thing entirely."""
        calls = fake_hub()
        result = hf_dataset_import.import_hf_dataset(
            repo_id="HuggingFaceH4/no_robots",
            revision=RESOLVED_SHA,
            dataset_id="ds-hf",
            storage=storage,
        )
        assert calls[0]["repo_id"] == "HuggingFaceH4/no_robots"
        assert calls[0]["repo_type"] == "dataset"
        assert calls[0]["revision"] == RESOLVED_SHA
        assert result.detected_format == "jsonl"

    def test_revision_is_required(self, storage):
        """A default of `main` would make the same pin resolve to different bytes later --
        exactly the unpinned behavior issue #208 removes."""
        with pytest.raises(HuggingFaceImportError) as excinfo:
            hf_dataset_import.import_hf_dataset(
                repo_id="owner/name", revision="", dataset_id="ds", storage=storage
            )
        assert excinfo.value.code == "HF_REVISION_REQUIRED"

    def test_repo_id_must_be_owner_slash_name(self, storage):
        with pytest.raises(HuggingFaceImportError) as excinfo:
            hf_dataset_import.import_hf_dataset(
                repo_id="norepo", revision="v1", dataset_id="ds", storage=storage
            )
        assert excinfo.value.code == "INVALID_HF_REPO_ID"

    def test_resolved_commit_is_recorded_not_the_requested_tag(self, fake_hub, storage):
        """Asking for a tag is fine; recording the tag is not, because it drifts. The commit
        SHA the bytes actually came from is what gets stored."""
        fake_hub(commit=RESOLVED_SHA)
        result = hf_dataset_import.import_hf_dataset(
            repo_id="owner/name",
            revision="refs/convert/parquet",
            dataset_id="ds-hf",
            storage=storage,
        )
        assert result.resolved_revision == RESOLVED_SHA
        assert result.resolved_revision != "refs/convert/parquet"

    def test_download_failure_is_a_coded_error_not_a_500(self, fake_hub, storage):
        class RevisionNotFoundError(Exception):
            pass

        fake_hub(raises=RevisionNotFoundError("no such revision"))
        with pytest.raises(HuggingFaceImportError) as excinfo:
            hf_dataset_import.import_hf_dataset(
                repo_id="owner/name",
                revision="deadbeef",
                dataset_id="ds",
                storage=storage,
            )
        assert excinfo.value.code == "HF_DOWNLOAD_FAILED"
        assert "RevisionNotFoundError" in excinfo.value.message
        assert excinfo.value.http_status == 502

    def test_missing_huggingface_hub_is_reported_as_a_missing_dep(
        self, monkeypatch, storage
    ):
        monkeypatch.setattr(hf_dataset_import, "_require_hf", _raise_import_error)
        with pytest.raises(HuggingFaceImportError) as excinfo:
            hf_dataset_import.import_hf_dataset(
                repo_id="owner/name", revision="v1", dataset_id="ds", storage=storage
            )
        assert excinfo.value.code == "MISSING_DEP"


def _raise_import_error():
    raise HuggingFaceImportError(
        "MISSING_DEP", "HuggingFace import requires the optional 'intake' extra."
    )


# ── picking the data file ────────────────────────────────────────────────────


class TestDataFileSelection:
    def test_prefers_jsonl(self, fake_hub, storage):
        fake_hub({"data/train.jsonl": PAYLOAD, "data/train.csv": b"id,q\n1,hi\n"})
        result = hf_dataset_import.import_hf_dataset(
            repo_id="owner/name", revision="v1", dataset_id="ds", storage=storage
        )
        assert result.detected_format == "jsonl"

    def test_explicit_split_is_honoured(self, fake_hub, storage):
        fake_hub(
            {
                "data/train.jsonl": PAYLOAD,
                "data/test.jsonl": PAYLOAD,
            }
        )
        result = hf_dataset_import.import_hf_dataset(
            repo_id="owner/name",
            revision="v1",
            dataset_id="ds",
            split="test",
            storage=storage,
        )
        assert result.filename == "test.jsonl"

    def test_missing_split_names_the_supported_formats(self, fake_hub, storage):
        fake_hub({"data/train.jsonl": PAYLOAD})
        with pytest.raises(HuggingFaceImportError) as excinfo:
            hf_dataset_import.import_hf_dataset(
                repo_id="owner/name",
                revision="v1",
                dataset_id="ds",
                split="validation",
                storage=storage,
            )
        assert excinfo.value.code == "HF_SPLIT_NOT_FOUND"
        assert ".jsonl" in excinfo.value.message

    def test_parquet_only_repo_says_so_explicitly(self, fake_hub, storage):
        """A parquet-only repo is the common case, and an obscure "unsupported format" is a
        much worse answer than naming what the pipeline can and cannot read."""
        fake_hub({"data/train-00000.parquet": b"PAR1not-really"})
        with pytest.raises(HuggingFaceImportError) as excinfo:
            hf_dataset_import.import_hf_dataset(
                repo_id="owner/name", revision="v1", dataset_id="ds", storage=storage
            )
        assert excinfo.value.code == "HF_NO_READABLE_DATA_FILE"
        assert ".parquet" in excinfo.value.message
        assert ".jsonl" in excinfo.value.message

    def test_our_own_sidecars_are_not_mistaken_for_data(self, fake_hub, storage):
        """`validation_report.json` is JSON and would otherwise be picked as a dataset."""
        fake_hub(
            {
                "datasets/ds/v1/validation_report.json": b'{"row_count": 1}',
                "data/train.jsonl": PAYLOAD,
            }
        )
        result = hf_dataset_import.import_hf_dataset(
            repo_id="owner/name", revision="v1", dataset_id="ds", storage=storage
        )
        assert result.filename == "train.jsonl"


# ── the staged result enters the existing pipeline ────────────────────────────


class TestStaging:
    def test_provenance_travels_with_the_bytes(self, fake_hub, storage):
        """Recorded when staged rather than re-sent on `validate`, so a client cannot claim
        a different origin for a file than the one recorded with it."""
        fake_hub()
        result = hf_dataset_import.import_hf_dataset(
            repo_id="owner/name",
            revision="refs/convert/parquet",
            dataset_id="ds-hf",
            storage=storage,
        )
        provenance = storage.read_provenance(result.staging_id)
        assert provenance["source_type"] == "huggingface"
        assert provenance["source_url_or_hf_id"] == "owner/name"
        assert provenance["source_commit_or_snapshot_date"] == RESOLVED_SHA
        assert provenance["hf_requested_revision"] == "refs/convert/parquet"

    def test_provenance_sidecar_is_not_mistaken_for_the_dataset(
        self, fake_hub, storage
    ):
        """The staging dir now holds two files; `resolve_staged` must still return the
        payload, not `_provenance.json`."""
        fake_hub()
        result = hf_dataset_import.import_hf_dataset(
            repo_id="owner/name", revision="v1", dataset_id="ds", storage=storage
        )
        resolved = storage.resolve_staged(result.staging_id)
        assert resolved.name == "train.jsonl"
        assert resolved.read_bytes() == PAYLOAD

    def test_checksum_is_of_the_staged_bytes(self, fake_hub, storage):
        import hashlib

        fake_hub()
        result = hf_dataset_import.import_hf_dataset(
            repo_id="owner/name", revision="v1", dataset_id="ds", storage=storage
        )
        assert result.checksum_sha256 == hashlib.sha256(PAYLOAD).hexdigest()
        assert Path(result.local_path).read_bytes() == PAYLOAD

    def test_plain_upload_has_no_provenance(self, storage):
        """A file staged before issue #240 simply has no origin beyond 'an upload'."""
        staged = storage.stage_upload("train.jsonl", PAYLOAD)
        assert storage.read_provenance(staged["staging_id"]) == {}


# ── the endpoint ─────────────────────────────────────────────────────────────


class TestEndpoint:
    def test_import_hf_then_validate_and_commit(
        self, client, admin_token, fake_hub, storage, monkeypatch
    ):
        """End to end: import -> the *existing* validate/commit wizard -> a PROCESSED
        DatasetVersion carrying the resolved HF commit."""
        fake_hub()
        monkeypatch.setattr("app.api.intake.DatasetStorage", lambda: storage)
        monkeypatch.setattr("app.api.intake_validate.DatasetStorage", lambda: storage)

        imp = client.post(
            "/api/v1/datasets/intake/import-hf",
            json={
                "dataset_id": "ds-hf-e2e",
                "repo_id": "HuggingFaceH4/no_robots",
                "revision": RESOLVED_SHA,
            },
            headers=auth_header(admin_token),
        )
        assert imp.status_code == 200, imp.text
        body = imp.json()
        assert body["repo_id"] == "HuggingFaceH4/no_robots"
        assert body["revision"] == RESOLVED_SHA
        assert body["resolved_revision"] == RESOLVED_SHA
        assert body["detected_format"] == "jsonl"
        assert body["next_step"] == "/api/v1/datasets/intake/validate"

        val = client.post(
            "/api/v1/datasets/intake/validate",
            json={"staging_id": body["staging_id"], "dataset_id": "ds-hf-e2e"},
            headers=auth_header(admin_token),
        )
        assert val.status_code == 200, val.text
        assert val.json()["status"] == "PASS"

        commit = client.post(
            "/api/v1/datasets/intake/commit",
            json={
                "staging_id": body["staging_id"],
                "dataset_id": "ds-hf-e2e",
                # Deliberately NOT passing source_type: the request default is
                # "file_upload", and the sidecar has to win or the import is mislabelled.
                "validation_report_id": val.json()["validation_report_id"],
            },
            headers=auth_header(admin_token),
        )
        assert commit.status_code == 200, commit.text
        assert commit.json()["status"] == "PROCESSED"

        from sqlalchemy.orm import Session

        from app.models.dataset import DatasetVersion

        with Session(client.engine) as session:
            dv = session.query(DatasetVersion).filter_by(dataset_id="ds-hf-e2e").one()
            assert dv.source_type == "huggingface"
            assert dv.source_url_or_hf_id == "HuggingFaceH4/no_robots"
            assert dv.source_commit_or_snapshot_date == RESOLVED_SHA
            assert dv.canonical_file_uri.startswith("file://")

    def test_missing_revision_is_rejected_by_the_schema(
        self, client, admin_token, fake_hub
    ):
        """No default on the field: omitting it is a 422, not a silent import of `main`."""
        fake_hub()
        resp = client.post(
            "/api/v1/datasets/intake/import-hf",
            json={"dataset_id": "ds", "repo_id": "owner/name"},
            headers=auth_header(admin_token),
        )
        assert resp.status_code == 422

    def test_requires_admin(self, client, admin_token, fake_hub):
        fake_hub()
        client.post(
            "/api/v1/auth/register",
            json={"username": "bob", "password": "Bob123456", "role": "user"},
        )
        login = client.post(
            "/api/v1/auth/login", json={"username": "bob", "password": "Bob123456"}
        )
        token = login.json()["access_token"]
        resp = client.post(
            "/api/v1/datasets/intake/import-hf",
            json={
                "dataset_id": "ds",
                "repo_id": "owner/name",
                "revision": RESOLVED_SHA,
            },
            headers=auth_header(token),
        )
        assert resp.status_code == 403

    def test_download_failure_is_502_not_500(self, client, admin_token, fake_hub):
        fake_hub(raises=OSError("connection reset"))
        resp = client.post(
            "/api/v1/datasets/intake/import-hf",
            json={
                "dataset_id": "ds",
                "repo_id": "owner/name",
                "revision": "v1",
            },
            headers=auth_header(admin_token),
        )
        assert resp.status_code == 502
        assert resp.json()["error"]["code"] == "HF_DOWNLOAD_FAILED"
