import time

import structlog
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request

from app.telemetry import record_http_request

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
        duration_seconds = time.perf_counter() - start
        duration_ms = round(duration_seconds * 1000, 2)
        # The route *template* (e.g. "/models/{model_id}/versions" - note this doesn't
        # include the /api/v1 mount prefix in this FastAPI version), not the resolved path -
        # using the resolved path as a Prometheus label would create one time series per
        # model_id/version ever requested (unbounded cardinality). Falls back to the raw
        # path for unmatched routes (404s from a totally unknown path), where no route was
        # resolved at all.
        route = request.scope.get("route")
        metric_path = getattr(route, "path", request.url.path)
        logger.info(
            "http_request",
            method=request.method,
            path=request.url.path,
            status_code=response.status_code,
            duration_ms=duration_ms,
            request_id=getattr(request.state, "request_id", None),
        )
        record_http_request(
            request.method, metric_path, response.status_code, duration_seconds
        )
        return response
