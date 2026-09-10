"""Tests for dataset intake inspect endpoint."""

import io
import json
import uuid

from tests.conftest import auth_header


def _upload(filename, content, token, client):
    return client.post(
        "/api/v1/datasets/intake/inspect",
        files={"file": (filename, io.BytesIO(content), "application/octet-stream")},
        headers=auth_header(token),
    )


class TestInspectJsonl:
    def test_returns_metadata(self, client, admin_token):
        data = b'{"q":"a","a":"b"}\n{"q":"c","a":"d"}\n'
        resp = _upload("train.jsonl", data, admin_token, client)
        assert resp.status_code == 200
        body = resp.json()
        assert body["detected_format"] == "jsonl"
        assert body["normalized_format"] == "jsonl"
        assert body["detected_sample_count"] == 2
        assert body["parse_status"] == "ok"
        assert len(body["checksum_sha256"]) == 64
        assert len(body["staging_id"]) > 0


class TestInspectJson:
    def test_returns_metadata(self, client, admin_token):
        data = json_bytes([{"a": 1}, {"a": 2}, {"a": 3}])
        resp = _upload("data.json", data, admin_token, client)
        assert resp.status_code == 200
        assert resp.json()["detected_format"] == "json"
        assert resp.json()["detected_sample_count"] == 3


class TestInspectCsv:
    def test_returns_metadata(self, client, admin_token):
        data = b"name,age\nAlice,30\nBob,25\nCharlie,35\n"
        resp = _upload("data.csv", data, admin_token, client)
        assert resp.status_code == 200
        assert resp.json()["detected_format"] == "csv"
        assert resp.json()["detected_sample_count"] == 3


class TestInspectErrors:
    def test_empty_file_returns_400(self, client, admin_token):
        resp = _upload("empty.jsonl", b"", admin_token, client)
        assert resp.status_code == 400
        assert resp.json()["error"]["code"] == "EMPTY_FILE"

    def test_unsupported_format_returns_400(self, client, admin_token):
        resp = _upload("data.xml", b"<root/>", admin_token, client)
        assert resp.status_code == 400
        assert resp.json()["error"]["code"] == "UNSUPPORTED_FORMAT"

    def test_empty_dataset_returns_400(self, client, admin_token):
        resp = _upload("empty.jsonl", b"\n\n", admin_token, client)
        assert resp.status_code == 400
        assert resp.json()["error"]["code"] == "EMPTY_DATASET"

    def test_file_too_large_returns_413(self, client, admin_token):
        resp = _upload("big.jsonl", b"x" * (100 * 1024 * 1024 + 1), admin_token, client)
        assert resp.status_code == 413


class TestInspectAuth:
    def test_requires_admin(self, client, admin_token):
        uname = f"alice_{uuid.uuid4().hex[:6]}"
        client.post(
            "/api/v1/auth/register",
            json={"username": uname, "password": "Alice1234", "role": "user"},
        )
        resp = client.post(
            "/api/v1/auth/login", json={"username": uname, "password": "Alice1234"}
        )
        token = resp.json()["access_token"]

        resp = _upload("train.jsonl", b'{"a":1}\n', token, client)
        assert resp.status_code == 403
        assert resp.json()["error"]["code"] == "FORBIDDEN"


def json_bytes(records):
    return json.dumps(records).encode()
