import uuid

import structlog
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request

REQUEST_ID_HEADER = "X-Request-Id"


class RequestIdMiddleware(BaseHTTPMiddleware):
    """Assigns/propagates a per-request id (issue #168).

    Reuses the inbound `X-Request-Id` header if the caller sent one, otherwise generates a
    UUID4. Exposed on `request.state.request_id` for other middleware/handlers (e.g.
    AccessLogMiddleware, the error-envelope handlers in app/main.py), bound into structlog
    context so every log line for this request carries it, and echoed back on the response.
    """

    async def dispatch(self, request: Request, call_next):
        request_id = request.headers.get(REQUEST_ID_HEADER) or str(uuid.uuid4())
        request.state.request_id = request_id
        structlog.contextvars.bind_contextvars(request_id=request_id)
        response = await call_next(request)
        response.headers[REQUEST_ID_HEADER] = request_id
        return response
