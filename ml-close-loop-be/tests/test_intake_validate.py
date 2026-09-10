"""Tests for the dataset intake validate + commit endpoints."""

import json
import tempfile
from pathlib import Path
from unittest.mock import patch

import pytest

from tests.conftest import auth_header

from app.services.dataset_storage import DatasetStorage


VALID_RECORD = {
    "id": "rec-001",
    "messages": [
        {"role": "user", "content": "Apa itu defnex?"},
        {
            "role": "assistant",
            "content": "DEFNEX adalah platform MLOps untuk integrasi model AI secara terpusat.",
        },
    ],
    "metadata": {"source_dataset": "test_ds", "source_id": "s-001"},
}

LEAKAGE_RECORD = {
    "id": "rec-leak",
    "messages": [
        {"role": "user", "content": "eval_question_1"},
        {"role": "assistant", "content": "This is the answer to eval question one."},
    ],
    "metadata": {"source_dataset": "test_ds", "source_id": "s-leak"},
}

EVAL_RECORD = {
    "id": "eval-001",
    "messages": [
        {"role": "user", "content": "eval_question_1"},
        {"role": "assistant", "content": "Golden answer."},
    ],
    "metadata": {"source_dataset": "eval", "source_id": "e-001"},
}


def _write_jsonl(path: Path, records: list[dict]) -> None:
    with open(path, "w") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")


class _FakeStorage:
    """In-memory stand-in for DatasetStorage backed by tmp_path."""

    def __init__(self, tmp_path: Path):
        self._base = tmp_path / "datasets"
        self._base.mkdir()
        self._staging = tmp_path / "staging"
        self._staging.mkdir()
        self._files: dict[str, Path] = {}

    def stage_upload(self, filename: str, content: bytes) -> dict:
        import uuid

        sid = uuid.uuid4().hex
        d = self._staging / sid
        d.mkdir()
        p = d / filename
        p.write_bytes(content)
        self._files[sid] = p
        return {
            "staging_id": sid,
            "path": str(p),
            "filename": filename,
            "size_bytes": len(content),
        }

    def resolve_staged(self, staging_id: str) -> Path | None:
        return self._files.get(staging_id)

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
            import csv
            import io

            text = path.read_text()
            reader = csv.DictReader(io.StringIO(text))
            return [row for row in reader]
        if source_format == "xlsx":
            import io

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
        import hashlib

        return hashlib.sha256(path.read_bytes()).hexdigest()

    def commit_file(self, staging_id: str, dataset_id: str, version: int) -> str:
        src = self._files.pop(staging_id, None)
        if src is None:
            raise FileNotFoundError(f"staging {staging_id} not found")
        dest_dir = self._base / dataset_id / f"v{version}"
        dest_dir.mkdir(parents=True, exist_ok=True)
        dest = dest_dir / src.name
        import shutil

        shutil.move(str(src), str(dest))
        return str(dest)

    def write_validation_report(self, dataset_id: str, version: int, report_dict: dict):
        pass

    def write_schema(self, dataset_id: str, version: int, schema_dict: dict):
        pass


@pytest.fixture
def fake_storage(tmp_path):
    return _FakeStorage(tmp_path)


@pytest.fixture
def regular_user_token(client):
    """Register an admin first, then a regular user and return their JWT."""
    client.post(
        "/api/v1/auth/register",
        json={"username": "admin", "password": "Admin1234", "role": "admin"},
    )
    client.post(
        "/api/v1/auth/register",
        json={"username": "bob", "password": "Bob123456", "role": "user"},
    )
    resp = client.post(
        "/api/v1/auth/login", json={"username": "bob", "password": "Bob123456"}
    )
    return resp.json()["access_token"]


def _stage_records(client, admin_token, storage, records, filename="data.jsonl"):
    tmp = Path(tempfile.mktemp(suffix=".jsonl"))
    _write_jsonl(tmp, records)
    info = storage.stage_upload(filename, tmp.read_bytes())
    tmp.unlink()
    return info


def test_validate_happy_path(client, admin_token, fake_storage):
    info = _stage_records(client, admin_token, fake_storage, [VALID_RECORD])

    with patch("app.api.intake_validate.DatasetStorage", return_value=fake_storage):
        resp = client.post(
            "/api/v1/datasets/intake/validate",
            json={
                "staging_id": info["staging_id"],
                "dataset_id": "test_ds",
            },
            headers=auth_header(admin_token),
        )

    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "PASS"
    assert body["total_records"] == 1
    assert body["valid_records"] == 1
    assert body["blocking_error_count"] == 0
    assert len(body["checks"]) == 6
    assert body["checks"][0]["name"] == "parse"
    assert body["staging_id"] == info["staging_id"]
    assert "validation_report_id" in body


def test_validate_staging_not_found(client, admin_token, fake_storage):
    with patch("app.api.intake_validate.DatasetStorage", return_value=fake_storage):
        resp = client.post(
            "/api/v1/datasets/intake/validate",
            json={"staging_id": "nonexistent", "dataset_id": "test_ds"},
            headers=auth_header(admin_token),
        )

    assert resp.status_code == 404
    assert resp.json()["error"]["code"] == "STAGING_NOT_FOUND"


def test_validate_empty_dataset(client, admin_token, fake_storage):
    info = _stage_records(client, admin_token, fake_storage, [])

    with patch("app.api.intake_validate.DatasetStorage", return_value=fake_storage):
        resp = client.post(
            "/api/v1/datasets/intake/validate",
            json={"staging_id": info["staging_id"], "dataset_id": "test_ds"},
            headers=auth_header(admin_token),
        )

    assert resp.status_code == 400
    assert resp.json()["error"]["code"] == "EMPTY_DATASET"


def test_validate_requires_admin(client, regular_user_token, fake_storage):
    info = _stage_records(client, regular_user_token, fake_storage, [VALID_RECORD])

    with patch("app.api.intake_validate.DatasetStorage", return_value=fake_storage):
        resp = client.post(
            "/api/v1/datasets/intake/validate",
            json={"staging_id": info["staging_id"], "dataset_id": "test_ds"},
            headers=auth_header(regular_user_token),
        )

    assert resp.status_code == 403


def test_validate_leakage_detected(client, admin_token, fake_storage):
    info = _stage_records(
        client, admin_token, fake_storage, [LEAKAGE_RECORD], filename="leak.jsonl"
    )

    from app.services import eval_set_service

    with (
        patch("app.api.intake_validate.DatasetStorage", return_value=fake_storage),
        patch.object(eval_set_service, "get_eval_set_version") as mock_get,
    ):
        from types import SimpleNamespace

        mock_get.return_value = SimpleNamespace(records=[EVAL_RECORD])

        resp = client.post(
            "/api/v1/datasets/intake/validate",
            json={
                "staging_id": info["staging_id"],
                "dataset_id": "test_ds",
                "eval_set_id": "benchmark",
                "eval_set_version": 1,
            },
            headers=auth_header(admin_token),
        )

    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "FAIL"
    leakage_check = next(c for c in body["checks"] if c["name"] == "leakage")
    assert leakage_check["status"] == "FAIL"


def test_commit_happy_path(client, admin_token, fake_storage):
    info = _stage_records(client, admin_token, fake_storage, [VALID_RECORD])

    with patch("app.api.intake_validate.DatasetStorage", return_value=fake_storage):
        val_resp = client.post(
            "/api/v1/datasets/intake/validate",
            json={"staging_id": info["staging_id"], "dataset_id": "test_ds"},
            headers=auth_header(admin_token),
        )
        report_id = val_resp.json()["validation_report_id"]

    info2 = _stage_records(client, admin_token, fake_storage, [VALID_RECORD])

    with patch("app.api.intake_validate.DatasetStorage", return_value=fake_storage):
        resp = client.post(
            "/api/v1/datasets/intake/commit",
            json={
                "staging_id": info2["staging_id"],
                "dataset_id": "test_ds",
                "validation_report_id": report_id,
            },
            headers=auth_header(admin_token),
        )

    assert resp.status_code == 200
    body = resp.json()
    assert body["dataset_id"] == "test_ds"
    assert body["version"] == 2
    assert body["status"] == "PROCESSED"


def test_commit_staging_not_found(client, admin_token, fake_storage):
    with patch("app.api.intake_validate.DatasetStorage", return_value=fake_storage):
        resp = client.post(
            "/api/v1/datasets/intake/commit",
            json={
                "staging_id": "nonexistent",
                "dataset_id": "test_ds",
                "validation_report_id": 1,
            },
            headers=auth_header(admin_token),
        )

    assert resp.status_code == 404
    assert resp.json()["error"]["code"] == "STAGING_NOT_FOUND"


def test_commit_validation_report_not_found(client, admin_token, fake_storage):
    info = _stage_records(client, admin_token, fake_storage, [VALID_RECORD])

    with patch("app.api.intake_validate.DatasetStorage", return_value=fake_storage):
        resp = client.post(
            "/api/v1/datasets/intake/commit",
            json={
                "staging_id": info["staging_id"],
                "dataset_id": "test_ds",
                "validation_report_id": 99999,
            },
            headers=auth_header(admin_token),
        )

    assert resp.status_code == 404
    assert resp.json()["error"]["code"] == "VALIDATION_REPORT_NOT_FOUND"


def test_commit_requires_admin(client, regular_user_token, fake_storage):
    info = _stage_records(client, regular_user_token, fake_storage, [VALID_RECORD])

    with patch("app.api.intake_validate.DatasetStorage", return_value=fake_storage):
        resp = client.post(
            "/api/v1/datasets/intake/commit",
            json={
                "staging_id": info["staging_id"],
                "dataset_id": "test_ds",
                "validation_report_id": 1,
            },
            headers=auth_header(regular_user_token),
        )

    assert resp.status_code == 403


def test_commit_sets_display_name(client, admin_token, fake_storage):
    info = _stage_records(client, admin_token, fake_storage, [VALID_RECORD])

    with patch("app.api.intake_validate.DatasetStorage", return_value=fake_storage):
        val_resp = client.post(
            "/api/v1/datasets/intake/validate",
            json={"staging_id": info["staging_id"], "dataset_id": "ds_with_name"},
            headers=auth_header(admin_token),
        )
        report_id = val_resp.json()["validation_report_id"]

    info2 = _stage_records(client, admin_token, fake_storage, [VALID_RECORD])

    with patch("app.api.intake_validate.DatasetStorage", return_value=fake_storage):
        resp = client.post(
            "/api/v1/datasets/intake/commit",
            json={
                "staging_id": info2["staging_id"],
                "dataset_id": "ds_with_name",
                "validation_report_id": report_id,
                "display_name": "My Dataset",
                "description": "A test dataset",
            },
            headers=auth_header(admin_token),
        )

    assert resp.status_code == 200

    from sqlalchemy.orm import Session
    from app.models.dataset import Dataset

    with Session(client.engine) as session:
        ds = session.get(Dataset, "ds_with_name")
        assert ds.display_name == "My Dataset"
        assert ds.description == "A test dataset"


def test_idempotency_key_returns_cached(client, admin_token, fake_storage):
    info = _stage_records(client, admin_token, fake_storage, [VALID_RECORD])

    with patch("app.api.intake_validate.DatasetStorage", return_value=fake_storage):
        val_resp = client.post(
            "/api/v1/datasets/intake/validate",
            json={"staging_id": info["staging_id"], "dataset_id": "idempotent_ds"},
            headers=auth_header(admin_token),
        )
        report_id = val_resp.json()["validation_report_id"]

    info2 = _stage_records(client, admin_token, fake_storage, [VALID_RECORD])

    with patch("app.api.intake_validate.DatasetStorage", return_value=fake_storage):
        resp1 = client.post(
            "/api/v1/datasets/intake/commit",
            json={
                "staging_id": info2["staging_id"],
                "dataset_id": "idempotent_ds",
                "validation_report_id": report_id,
            },
            headers={**auth_header(admin_token), "X-Idempotency-Key": "key-abc"},
        )

    assert resp1.status_code == 200

    info3 = _stage_records(client, admin_token, fake_storage, [VALID_RECORD])

    with patch("app.api.intake_validate.DatasetStorage", return_value=fake_storage):
        resp2 = client.post(
            "/api/v1/datasets/intake/commit",
            json={
                "staging_id": info3["staging_id"],
                "dataset_id": "idempotent_ds",
                "validation_report_id": report_id,
            },
            headers={**auth_header(admin_token), "X-Idempotency-Key": "key-abc"},
        )

    assert resp2.status_code == 200
    assert resp2.json() == resp1.json()


def test_staging_uses_directory_layout(client, admin_token):
    """P0-1: verify stage_upload creates _staging/{id}/file, not _staging/{id}_file."""
    storage = DatasetStorage()
    data = b'{"q":"a","a":"b"}\n'
    info = storage.stage_upload("test.jsonl", data)

    from pathlib import Path

    staged_path = Path(info["path"])
    assert staged_path.is_file()
    assert staged_path.parent.name == info["staging_id"]
    assert staged_path.name == "test.jsonl"
    assert not (storage._staging / f"{info['staging_id']}_test.jsonl").exists()


def test_validate_csv_records(client, admin_token, fake_storage):
    csv_content = b"id,q,a\nrec-1,what is defnex,it is a platform\nrec-2,hello,hi\n"
    info = fake_storage.stage_upload("data.csv", csv_content)

    with patch("app.api.intake_validate.DatasetStorage", return_value=fake_storage):
        resp = client.post(
            "/api/v1/datasets/intake/validate",
            json={
                "staging_id": info["staging_id"],
                "dataset_id": "csv_ds",
                "source_format": "csv",
            },
            headers=auth_header(admin_token),
        )

    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "PASS"
    assert body["total_records"] == 2
    assert body["preview"][0]["id"] == "rec-1"


def test_validate_json_records(client, admin_token, fake_storage):
    json_content = json.dumps([VALID_RECORD, VALID_RECORD]).encode()
    info = fake_storage.stage_upload("data.json", json_content)

    with patch("app.api.intake_validate.DatasetStorage", return_value=fake_storage):
        resp = client.post(
            "/api/v1/datasets/intake/validate",
            json={
                "staging_id": info["staging_id"],
                "dataset_id": "json_ds",
                "source_format": "json",
            },
            headers=auth_header(admin_token),
        )

    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "PASS"
    assert body["total_records"] == 2


def test_validate_xlsx_records(client, admin_token, fake_storage):
    import io

    import openpyxl

    wb = openpyxl.Workbook()
    ws = wb.active
    ws.append(["id", "q", "a"])
    ws.append(["rec-1", "what is defnex", "it is a platform"])
    buf = io.BytesIO()
    wb.save(buf)
    buf.seek(0)
    xlsx_bytes = buf.read()
    wb.close()

    info = fake_storage.stage_upload("data.xlsx", xlsx_bytes)

    with patch("app.api.intake_validate.DatasetStorage", return_value=fake_storage):
        resp = client.post(
            "/api/v1/datasets/intake/validate",
            json={
                "staging_id": info["staging_id"],
                "dataset_id": "xlsx_ds",
                "source_format": "xlsx",
            },
            headers=auth_header(admin_token),
        )

    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "PASS"
    assert body["total_records"] == 1
    assert body["preview"][0]["id"] == "rec-1"
