"""Tests for the security-headers middleware (issue #131)."""


def test_security_headers_present_on_success_response(client):
    resp = client.get("/api/v1/health")
    assert resp.status_code == 200
    assert (
        resp.headers["Strict-Transport-Security"]
        == "max-age=31536000; includeSubDomains"
    )
    assert resp.headers["X-Content-Type-Options"] == "nosniff"
    assert resp.headers["X-Frame-Options"] == "DENY"
    assert resp.headers["Referrer-Policy"] == "strict-origin-when-cross-origin"


def test_security_headers_present_on_error_response(client):
    resp = client.get("/api/v1/does-not-exist")
    assert resp.status_code == 404
    assert resp.headers["X-Content-Type-Options"] == "nosniff"
    assert resp.headers["X-Frame-Options"] == "DENY"
    assert resp.headers["Referrer-Policy"] == "strict-origin-when-cross-origin"


def test_security_headers_present_on_legacy_redirect(client):
    resp = client.get("/health", follow_redirects=False)
    assert resp.status_code == 301
    assert resp.headers["X-Content-Type-Options"] == "nosniff"
    assert resp.headers["X-Frame-Options"] == "DENY"
