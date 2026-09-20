import time

import structlog
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request

logger = structlog.get_logger(__name__)


class AccessLogMiddleware(BaseHTTPMiddleware):
    """Logs one structured entry per request (issue #160).

    Previously the only structlog calls tied to HTTP requests fired from
    exception handlers, so successful (2xx/3xx) requests left no trace at all -
    no latency baseline, no traffic-volume signal outside of errors.
    """

    async def dispatch(self, request: Request, call_next):
        start = time.perf_counter()
        response = await call_next(request)
        duration_ms = round((time.perf_counter() - start) * 1000, 2)
        logger.info(
            "http_request",
            method=request.method,
            path=request.url.path,
            status_code=response.status_code,
            duration_ms=duration_ms,
            request_id=getattr(request.state, "request_id", None),
        )
        return response
