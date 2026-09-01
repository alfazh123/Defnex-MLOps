"""Tests for rate limiting on auth endpoints (US-001)."""


def test_register_rate_limit_triggers_after_threshold(client):
    """POST /auth/register must return 429 after 3 requests/minute."""
    for i in range(3):
        resp = client.post(
            "/auth/register", json={"username": f"u{i}", "password": "Pass1234"}
        )
        assert resp.status_code == 201, f"Request {i + 1} should succeed"

    resp = client.post(
        "/auth/register", json={"username": "u3", "password": "Pass1234"}
    )
    assert resp.status_code == 429
    body = resp.json()
    assert body["error"]["code"] == "RATE_LIMIT_EXCEEDED"


def test_login_rate_limit_triggers_after_threshold(client):
    """POST /auth/login must return 429 after 5 requests/minute."""
    from app.limiter import limiter

    for i in range(5):
        client.post(
            "/auth/register", json={"username": f"u{i}", "password": "Pass1234"}
        )
        limiter.reset()

    for i in range(5):
        resp = client.post(
            "/auth/login", json={"username": f"u{i}", "password": "Pass1234"}
        )
        assert resp.status_code == 200, f"Login {i + 1} should succeed"

    resp = client.post("/auth/login", json={"username": "u0", "password": "Pass1234"})
    assert resp.status_code == 429
    body = resp.json()
    assert body["error"]["code"] == "RATE_LIMIT_EXCEEDED"


def test_rate_limit_returns_headers(client):
    """Rate limit responses must include X-RateLimit-Limit and X-RateLimit-Remaining headers."""
    for i in range(3):
        resp = client.post(
            "/auth/register", json={"username": f"h{i}", "password": "Pass1234"}
        )
        assert resp.status_code == 201

    resp = client.post(
        "/auth/register", json={"username": "h3", "password": "Pass1234"}
    )
    assert resp.status_code == 429
    assert "X-RateLimit-Limit" in resp.headers
    assert "X-RateLimit-Remaining" in resp.headers
    assert "X-RateLimit-Reset" in resp.headers


def test_rate_limit_resets_after_window(client):
    """After rate limit triggers, new requests succeed once the window resets (simulated via reset)."""
    from app.limiter import limiter

    for i in range(3):
        client.post(
            "/auth/register", json={"username": f"r{i}", "password": "Pass1234"}
        )

    resp = client.post(
        "/auth/register", json={"username": "r3", "password": "Pass1234"}
    )
    assert resp.status_code == 429

    limiter.reset()

    resp = client.post(
        "/auth/register", json={"username": "r4", "password": "Pass1234"}
    )
    assert resp.status_code == 201
