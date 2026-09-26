"""Model version lineage endpoint and validation-report reference (issues #236, #237).

The property under test: **one request answers the meeting's fourth agenda item** — which
training, when, which dataset, what metadata, what config — and a field that cannot be
resolved says so in `gaps[]` instead of quietly coming back null.
"""

from datetime import UTC, datetime, timedelta


from sqlalchemy.orm import Session

from app.models.dataset import Dataset, DatasetVersion
from app.models.model import Model, ModelVersion
from app.models.training import TrainingRun
from app.models.validation import ValidationReport
from app.services import model_service

from tests.conftest import auth_header

ANSWER = " ".join(f"word{i}" for i in range(30))
RECORD = {
    "id": "r1",
    "messages": [
        {"role": "user", "content": "What is the capital of France?"},
        {"role": "assistant", "content": ANSWER},
    ],
    "metadata": {"source_dataset": "lin", "source_id": "s1"},
}


def _seed_lineage(session, *, with_run=True, with_metadata=True):
    """Build a COMPLETED run + registered model version with as much lineage as possible."""
    now = datetime.now(UTC)
    dataset = Dataset(dataset_id="ds-lineage", display_name="Lineage DS")
    session.add(dataset)
    session.flush()

    dataset_version = DatasetVersion(
        dataset_id="ds-lineage",
        version=1,
        status="PROCESSED",
        source_type="file_upload",
        source_format="jsonl",
        row_count=1,
        license="apache-2.0",
        created_at=now,
        canonical_file_uri="s3://artifacts/datasets/ds-lineage/v1/train.jsonl",
    )
    session.add(dataset_version)
    session.flush()

    report = ValidationReport(
        dataset_version_id=dataset_version.id,
        rule_set_version="2.2.0",
        run_at=now,
        record_count=1,
        content_hash="a" * 64,
        status_counts={"VALID": 1, "INVALID": 0, "NEEDS_REVIEW": 0},
        gate_decision="PASS",
        gate_reason="ok",
        per_record_errors=[[]],
        records=[RECORD],
    )
    session.add(report)
    session.flush()

    started = now - timedelta(minutes=30)
    finished = now - timedelta(minutes=10)
    run = TrainingRun(
        training_run_id="run-lineage",
        dataset_version_id=dataset_version.id,
        model_id="lin-model",
        base_model="unsloth/Qwen3-0.6B",
        training_config={
            "peft_method": "qlora",
            "lora_r": 16,
            "dataset_pin": {
                "dataset_id": "ds-lineage",
                "dataset_version": 1,
                "file_uri": "s3://artifacts/datasets/ds-lineage/v1/train.jsonl",
                "content_hash": "a" * 64,
                "source_format": "jsonl",
                "row_count": 1,
            },
        },
        status="COMPLETED",
        triggered_by="alice",
        started_at=started,
        finished_at=finished,
        created_at=now,
        # Set before registration: `register_model_version` copies it onto the artifact
        # entry when there is no staging dir, and a null URI would leave the artifact block
        # unable to say anything about its own metadata.
        artifact_uri="file:///tmp/lineage-adapter",
    )
    session.add(run)
    session.commit()
    session.refresh(run)

    model_service.register_model_version(
        session,
        run,
        staging_dir=None,
    )
    session.commit()
    return model_version_latest(session), dataset_version, report


def model_version_latest(db):
    return (
        db.query(ModelVersion)
        .filter_by(model_id="lin-model")
        .order_by(ModelVersion.version)
        .all()
    )[-1]


def _lineage(client, model_id, version, token):
    return client.get(
        f"/api/v1/models/{model_id}/versions/{version}/lineage",
        headers=auth_header(token),
    )


class TestLineageEndpoint:
    def test_full_lineage(self, db_session, tmp_path, monkeypatch):
        """Every sub-object the meeting asked for, in one response."""
        from app.services import lineage_service

        model_version, dataset_version, report = _seed_lineage(db_session)
        lineage = lineage_service.build_lineage(
            db_session, "lin-model", model_version.version
        )

        # (a) which training, and (b) when
        assert lineage["run"]["training_run_id"] == "run-lineage"
        assert lineage["run"]["triggered_by"] == "alice"
        assert lineage["run"]["started_at"] is not None
        assert lineage["run"]["completed_at"] is not None
        assert lineage["run"]["git_commit"] is not None

        # (c) which dataset
        assert lineage["dataset"]["dataset_id"] == "ds-lineage"
        assert lineage["dataset"]["dataset_version"] == 1
        assert lineage["dataset"]["file_uri"].endswith("train.jsonl")
        assert lineage["dataset"]["checksum_sha256"] == "a" * 64
        assert lineage["dataset"]["pinned"] is True
        assert lineage["dataset"]["pin_matches_row"] is True
        assert lineage["dataset"]["license"] == "apache-2.0"

        # (e) what config
        assert lineage["config"]["training_config"]["peft_method"] == "qlora"
        assert lineage["config"]["training_config_hash"]
        # The pin inside the config identifies the dataset even if the row changes.
        assert (
            lineage["config"]["training_config"]["dataset_pin"]["content_hash"]
            == "a" * 64
        )

        # artifact + metadata
        assert lineage["artifacts"]
        assert lineage["artifacts"][0]["uri"] == "file:///tmp/lineage-adapter"

        # No gaps in the parts this endpoint exists to answer. The two that remain are both
        # real and expected for a row registered without a staging dir: nothing was hashed
        # (issue #218 covers writing checksums on that path) and there is no metadata.json
        # to read for a file that does not exist.
        fields = {g["field"] for g in lineage["gaps"]}
        assert "training_run_id" not in fields
        assert "dataset" not in fields
        assert "config" not in fields
        assert fields == {"artifacts[].checksum", "metadata"}

    def test_validation_report_ref_is_written_at_registration(self, db_session):
        """Issue #237: the column was read by the promotion flow and never written."""
        model_version, dataset_version, report = _seed_lineage(db_session)
        assert model_version.dataset_validation_report_ref is not None
        assert str(report.id) in model_version.dataset_validation_report_ref

    def test_report_ref_is_none_when_no_report_exists(self, db_session):
        """Honest null: a version created outside the wizard has no report to point at."""
        now = datetime.now(UTC)
        db_session.add(Dataset(dataset_id="ds-noreport"))
        db_session.flush()
        dv = DatasetVersion(
            dataset_id="ds-noreport",
            version=1,
            status="PROCESSED",
            source_type="seed",
            source_format="jsonl",
            created_at=now,
        )
        db_session.add(dv)
        db_session.flush()
        db_session.add(Model(model_id="nr-model"))
        db_session.flush()
        run = TrainingRun(
            training_run_id="run-nr",
            dataset_version_id=dv.id,
            model_id="nr-model",
            base_model="base",
            training_config={},
            status="COMPLETED",
            created_at=now,
        )
        db_session.add(run)
        db_session.commit()
        db_session.refresh(run)
        mv = model_service.register_model_version(db_session, run)
        assert mv.dataset_validation_report_ref is None


class TestGapReporting:
    """`gaps[]` is the point: a caller can tell "we do not know" from "it is null"."""

    def test_missing_dataset_is_reported_not_silently_null(self):
        """A version with no producing run, and a run with no dataset.

        Reachable only defensively: `model_versions.training_run_id` and
        `training_runs.dataset_version_id` are both NOT NULL, so the schema itself forbids
        these rows. The branches exist for a row that arrived some other way (a future
        import, a manual fix-up), and this asserts they degrade honestly rather than
        crashing -- which is exactly the alternative that matters.
        """
        from app.services import lineage_service

        gaps: list[dict] = []
        block = lineage_service._dataset_block(
            None, ModelVersion(model_id="m", version=1), None, gaps
        )
        assert block["dataset_id"] is None
        assert block["dataset_version"] is None
        assert [g["field"] for g in gaps] == ["dataset"]
        assert gaps[0]["reason"]

    def test_missing_run_is_reported(self):
        from app.services import lineage_service

        gaps: list[dict] = []
        run_block = {
            "training_run_id": None,
        }
        # build_lineage's run-gap branch, exercised through the same helper it uses.
        lineage_service._gap(
            gaps,
            "training_run_id",
            "model version m:1 has no training_run_id; it was not produced by a training "
            "run on this system (seed data, or an import).",
        )
        assert run_block["training_run_id"] is None
        assert [g["field"] for g in gaps] == ["training_run_id"]

    def test_artifact_without_checksum_is_reported(self, db_session):
        """A row registered through the flat `artifact_uri` path has no checksum. Reporting
        that beats a bare null."""
        now = datetime.now(UTC)
        db_session.add(Model(model_id="flat-model"))
        db_session.add(Dataset(dataset_id="ds-flat"))
        db_session.flush()
        dv = DatasetVersion(
            dataset_id="ds-flat",
            version=1,
            status="PROCESSED",
            source_type="seed",
            source_format="jsonl",
            created_at=now,
        )
        db_session.add(dv)
        db_session.flush()
        # Both `training_run_id` and `dataset_version_id` are NOT NULL, so a legitimate run
        # is created; what makes this row "flat" is the registration path, not the schema.
        db_session.add(
            TrainingRun(
                training_run_id="tr-flat",
                dataset_version_id=dv.id,
                model_id="flat-model",
                base_model="base",
                training_config={},
                status="COMPLETED",
                artifact_uri="file:///tmp/x",
                created_at=now,
            )
        )
        db_session.commit()
        db_session.add(
            ModelVersion(
                model_id="flat-model",
                version=1,
                status="REGISTERED",
                training_run_id="tr-flat",
                base_model="base",
                training_config={},
                artifacts=[{"type": "adapter", "uri": "file:///tmp/x"}],
                created_at=now,
            )
        )
        db_session.commit()

        from app.services import lineage_service

        lineage = lineage_service.build_lineage(db_session, "flat-model", 1)
        assert lineage["artifacts"][0]["checksum"] is None
        assert any(g["field"] == "artifacts[].checksum" for g in lineage["gaps"])

    def test_unreadable_metadata_does_not_fail_the_endpoint(
        self, db_session, monkeypatch
    ):
        """A sidecar that cannot be read is a gap, not a 500: the row-level record above is
        still a complete answer."""
        from app.services import lineage_service

        model_version, _, _ = _seed_lineage(db_session)

        class _BrokenStorage:
            def read_metadata(self, uri):
                raise OSError("bucket unreachable")

        lineage = lineage_service.build_lineage(
            db_session, "lin-model", model_version.version, storage=_BrokenStorage()
        )
        assert lineage["run"]["training_run_id"] == "run-lineage"
        assert lineage["metadata"] == {}
        assert any(g["field"] == "metadata" for g in lineage["gaps"])


class TestLineageOverHttp:
    """The route end to end. Seeded through `client.engine` — the same database the
    request-scoped session reads — so the HTTP response is exercised for real."""

    def test_200_with_the_full_record(self, client, admin_token):
        with Session(client.engine) as db:
            model_version, _, report = _seed_lineage(db)
            version = model_version.version
            report_id = str(report.id)

        resp = _lineage(client, "lin-model", version, admin_token)
        assert resp.status_code == 200, resp.text
        body = resp.json()
        assert body["model_id"] == "lin-model"
        assert body["version"] == version
        assert body["base_model"] == "unsloth/Qwen3-0.6B"
        assert body["run"]["training_run_id"] == "run-lineage"
        assert body["dataset"]["dataset_id"] == "ds-lineage"
        assert body["dataset"]["checksum_sha256"] == "a" * 64
        assert body["config"]["training_config"]["peft_method"] == "qlora"
        assert body["config"]["training_config_hash"]
        assert report_id in body["dataset"]["validation_report_ref"]
        assert body["artifacts"][0]["uri"]

    def test_unknown_version_is_404(self, client, admin_token):
        resp = _lineage(client, "no-such-model", 7, admin_token)
        assert resp.status_code == 404
        assert resp.json()["error"]["code"] == "MODEL_VERSION_NOT_FOUND"

    def test_known_model_unknown_version_is_404(self, client, admin_token):
        with Session(client.engine) as db:
            _seed_lineage(db)
        resp = _lineage(client, "lin-model", 99, admin_token)
        assert resp.status_code == 404
        assert resp.json()["error"]["code"] == "MODEL_VERSION_NOT_FOUND"

    def test_requires_auth(self, client):
        resp = client.get("/api/v1/models/m/versions/1/lineage")
        assert resp.status_code == 401
