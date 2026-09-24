from fastapi.testclient import TestClient

from app.main import app
from app.middleware.request_id import REQUEST_ID_HEADER
from tests.conftest import auth_header

client = TestClient(app)


def test_generates_request_id_when_absent():
    response = client.get("/api/v1/health")

    assert REQUEST_ID_HEADER in response.headers
    assert len(response.headers[REQUEST_ID_HEADER]) > 0


def test_echoes_inbound_request_id_unchanged():
    response = client.get(
        "/api/v1/health", headers={REQUEST_ID_HEADER: "caller-supplied-id-123"}
    )

    assert response.headers[REQUEST_ID_HEADER] == "caller-supplied-id-123"


def test_two_requests_without_header_get_different_ids():
    first = client.get("/api/v1/health")
    second = client.get("/api/v1/health")

    assert first.headers[REQUEST_ID_HEADER] != second.headers[REQUEST_ID_HEADER]


def test_error_envelope_includes_matching_request_id(client, admin_token):
    headers = {**auth_header(admin_token), REQUEST_ID_HEADER: "err-id-456"}
    response = client.get("/api/v1/training-runs/missing", headers=headers)

    assert response.status_code == 404
    body = response.json()
    assert body["error"]["request_id"] == "err-id-456"
    assert response.headers[REQUEST_ID_HEADER] == "err-id-456"
