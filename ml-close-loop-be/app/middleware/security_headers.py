class SecurityHeadersMiddleware:
    """Sets baseline security headers on every response (issue #131).

    Plain ASGI middleware rather than BaseHTTPMiddleware: nested
    BaseHTTPMiddleware layers can drop headers on responses that an inner
    BaseHTTPMiddleware short-circuits (e.g. ApiVersionRedirectMiddleware's
    301s), so this wraps `send` directly instead.

    CSRF middleware is intentionally not part of this: the API authenticates
    with a Bearer JWT in the Authorization header, not a browser-managed
    session cookie, so there is no ambient credential for a cross-site
    request to ride on and CSRF protection does not apply here.
    """

    _HEADERS = [
        (b"strict-transport-security", b"max-age=31536000; includeSubDomains"),
        (b"x-content-type-options", b"nosniff"),
        (b"x-frame-options", b"DENY"),
        (b"referrer-policy", b"strict-origin-when-cross-origin"),
    ]

    def __init__(self, app):
        self.app = app

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        async def send_wrapper(message):
            if message["type"] == "http.response.start":
                message["headers"] = list(message.get("headers", [])) + self._HEADERS
            await send(message)

        await self.app(scope, receive, send_wrapper)
