import io

from tests.conftest import auth_header


def _upload(filename: str, content: bytes, token: str, client):
    return client.post(
        "/api/v1/datasets/intake/inspect",
        files={"file": (filename, io.BytesIO(content), "application/octet-stream")},
        headers=auth_header(token),
    )


def test_inspect_jsonl_returns_metadata(client, admin_token):
    data = b'{"q":"a","a":"b"}\n{"q":"c","a":"d"}\n'
    resp = _upload("train.jsonl", data, admin_token, client)
    assert resp.status_code == 200
    body = resp.json()
    assert body["detected_format"] == "jsonl"
    assert body["detected_sample_count"] == 2
    assert body["parse_status"] == "ok"
    assert body["staging_id"]
    assert body["checksum_sha256"]


def test_inspect_json_returns_metadata(client, admin_token):
    data = b'[{"q":"a","a":"b"},{"q":"c","a":"d"}]'
    resp = _upload("data.json", data, admin_token, client)
    assert resp.status_code == 200
    body = resp.json()
    assert body["detected_format"] == "json"
    assert body["detected_sample_count"] == 2


def test_inspect_csv_returns_metadata(client, admin_token):
    data = b"question,answer\na,b\nc,d\n"
    resp = _upload("data.csv", data, admin_token, client)
    assert resp.status_code == 200
    body = resp.json()
    assert body["detected_format"] == "csv"
    assert body["detected_sample_count"] == 2


def test_inspect_xlsx_returns_metadata(client, admin_token):
    import openpyxl

    wb = openpyxl.Workbook()
    ws = wb.active
    ws.append(["question", "answer"])
    ws.append(["a", "b"])
    ws.append(["c", "d"])
    buf = io.BytesIO()
    wb.save(buf)
    wb.close()
    resp = _upload("data.xlsx", buf.getvalue(), admin_token, client)
    assert resp.status_code == 200
    body = resp.json()
    assert body["detected_format"] == "xlsx"
    assert body["detected_sample_count"] == 2


def test_inspect_empty_file_returns_400(client, admin_token):
    resp = _upload("empty.jsonl", b"", admin_token, client)
    assert resp.status_code == 400
    assert resp.json()["error"]["code"] == "EMPTY_FILE"


def test_inspect_unsupported_format_returns_400(client, admin_token):
    resp = _upload("data.exe", b"MZ\x90\x00", admin_token, client)
    assert resp.status_code == 400
    assert resp.json()["error"]["code"] == "UNSUPPORTED_FORMAT"


def test_inspect_malformed_json_returns_400(client, admin_token):
    resp = _upload("bad.json", b"{not json", admin_token, client)
    assert resp.status_code == 400
    assert resp.json()["error"]["code"] == "UNPARSEABLE_FILE"


def test_inspect_empty_dataset_returns_400(client, admin_token):
    resp = _upload("empty.jsonl", b"\n\n\n", admin_token, client)
    assert resp.status_code == 400
    assert resp.json()["error"]["code"] == "EMPTY_DATASET"


def test_inspect_requires_admin(client, admin_token):
    # admin_token already registers the first user (auto-admin).
    # Now register a non-admin user.
    import uuid
    uname = f"alice_{uuid.uuid4().hex[:6]}"
    client.post("/api/v1/auth/register", json={"username": uname, "password": "Alice1234", "role": "user"})
    resp = client.post("/api/v1/auth/login", json={"username": uname, "password": "Alice1234"})
    token = resp.json()["access_token"]

    data = b'{"q":"a","a":"b"}\n'
    resp = _upload("train.jsonl", data, token, client)
    assert resp.status_code == 403
    assert resp.json()["error"]["code"] == "FORBIDDEN"
