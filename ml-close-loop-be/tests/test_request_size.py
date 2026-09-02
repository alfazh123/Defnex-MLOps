"""Tests for request size limiting middleware (US-002)."""


def test_oversized_body_rejected(client):
    """POST with body > 1MB must return 413."""
    large_body = "x" * (1_048_576 + 1)
    resp = client.post(
        "/api/v1/auth/register",
        content=large_body,
        headers={"Content-Type": "text/plain"},
    )
    assert resp.status_code == 413
    body = resp.json()
    assert body["error"]["code"] == "REQUEST_TOO_LARGE"


def test_undersized_body_accepted(client):
    """POST with body < 1MB must be accepted (201 or other success)."""
    small_body = "x" * 100
    resp = client.post(
        "/api/v1/auth/register",
        content=small_body,
        headers={"Content-Type": "application/json"},
    )
    assert resp.status_code != 413


def test_exact_limit_body_accepted(client):
    """POST with body exactly at 1MB limit must be accepted."""
    body = "x" * 1_048_576
    resp = client.post(
        "/api/v1/auth/register",
        content=body,
        headers={"Content-Type": "application/json"},
    )
    assert resp.status_code != 413


def test_content_length_header_checked(client):
    """Request with Content-Length > max is rejected without reading body."""
    resp = client.post(
        "/api/v1/auth/register",
        content=b"",
        headers={
            "Content-Type": "text/plain",
            "Content-Length": "2097152",
        },
    )
    assert resp.status_code == 413
    body = resp.json()
    assert body["error"]["code"] == "REQUEST_TOO_LARGE"
