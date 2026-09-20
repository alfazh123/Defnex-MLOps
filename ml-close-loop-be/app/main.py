from collections.abc import Awaitable, Callable
from contextlib import asynccontextmanager

import structlog
from fastapi import FastAPI, Request, APIRouter
from fastapi.exceptions import HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse
from slowapi.errors import RateLimitExceeded
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.responses import RedirectResponse, Response

from app.api.audit import router as audit_router
from app.api.auth import router as auth_router
from app.api.compute_resources import router as compute_resources_router
from app.api.datasets import router as datasets_router
from app.api.deployment import router as deployment_router
from app.api.eval_sets import router as eval_sets_router
from app.api.feedback import router as feedback_router
from app.api.health import router as health_router
from app.db.session import engine
from app.limiter import limiter
from app.api.inference import router as inference_router
from app.api.intake import router as intake_router
from app.api.intake_validate import router as intake_validate_router
from app.api.models import router as models_router
from app.api.notifications import router as notifications_router
from app.api.promotion import router as promotion_router
from app.api.training import router as training_router
from app.api.transfer import router as transfer_router
from app.api.users import router as users_router
from app.api.validation import router as validation_router
from app.config import settings
from app.logging import configure_logging
from app.middleware.access_log import AccessLogMiddleware
from app.middleware.request_size import RequestSizeLimitMiddleware
from app.middleware.security_headers import SecurityHeadersMiddleware
from app.services.deployment_service import SmokeTestError
from app.services.serving import BaseModelMismatchError, ServingError

logger = structlog.get_logger(__name__)


@asynccontextmanager
async def lifespan(_app: FastAPI):
    configure_logging(log_level=settings.log_level, debug=settings.debug)
    from app.telemetry import setup_telemetry

    setup_telemetry(_app, engine=engine)
    logger.info("application_starting")
    yield
    logger.info("application_shutting_down")
    engine.dispose()


app = FastAPI(title="DEFNEX MLOps Backend", version="0.1.0", lifespan=lifespan)
app.state.limiter = limiter

app.add_middleware(
    CORSMiddleware,
    allow_origins=[o.strip() for o in settings.cors_origins.split(",")],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.add_middleware(
    RequestSizeLimitMiddleware,
    max_body_size=settings.max_request_body_size,
)


class ApiVersionRedirectMiddleware(BaseHTTPMiddleware):
    """Redirect legacy /<path> to /api/v1/<path> with 301 Moved Permanently."""

    _LEGACY_PREFIXES = (
        "/health",
        "/auth",
        "/users",
        "/datasets",
        "/eval-sets",
        "/feedback",
        "/training-runs",
        "/models",
        "/deployments",
        "/promotions",
        "/transfers",
        "/compute-resources",
        "/audit-logs",
        "/notifications",
    )

    async def dispatch(
        self, request: Request, call_next: Callable[[Request], Awaitable[Response]]
    ) -> Response:
        path = request.url.path
        if not path.startswith("/api/v1/") and any(
            path.startswith(p) for p in self._LEGACY_PREFIXES
        ):
            return RedirectResponse(url=f"/api/v1{path}", status_code=301)
        return await call_next(request)


app.add_middleware(ApiVersionRedirectMiddleware)

# Added last: Starlette's add_middleware() inserts at the front of the stack,
# so the last-added middleware ends up outermost and sees every response —
# including CORS preflights, body-size 413s, and legacy-prefix 301s.
app.add_middleware(SecurityHeadersMiddleware)

# Outermost of all (added very last): logs every request/response that reaches
# the app, including ones the middlewares above short-circuit (issue #160).
app.add_middleware(AccessLogMiddleware)

v1_router = APIRouter(prefix="/api/v1")

v1_router.include_router(health_router)
v1_router.include_router(audit_router)
v1_router.include_router(auth_router)
v1_router.include_router(users_router)
v1_router.include_router(datasets_router)
v1_router.include_router(eval_sets_router)
v1_router.include_router(feedback_router)
v1_router.include_router(validation_router)
v1_router.include_router(training_router)
v1_router.include_router(transfer_router)
v1_router.include_router(compute_resources_router)
v1_router.include_router(models_router)
v1_router.include_router(notifications_router)
v1_router.include_router(promotion_router)
v1_router.include_router(deployment_router)
v1_router.include_router(inference_router)
v1_router.include_router(intake_router)
v1_router.include_router(intake_validate_router)

app.include_router(v1_router)


@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException) -> JSONResponse:
    """Pass through APIError's already-enveloped detail (PRD §18); wrap anything else the same way."""

    content = (
        exc.detail
        if isinstance(exc.detail, dict) and "error" in exc.detail
        else {"detail": exc.detail}
    )

    log_kwargs: dict = {
        "method": request.method,
        "path": request.url.path,
        "status_code": exc.status_code,
    }

    error_code = ""
    if isinstance(content, dict) and "error" in content:
        error_code = content["error"].get("code", "")

    log_kwargs["error_code"] = error_code

    if exc.status_code >= 500:
        logger.error("http_exception", **log_kwargs, exc_info=True)
    elif exc.status_code >= 400:
        logger.warning("http_exception", **log_kwargs)

    return JSONResponse(status_code=exc.status_code, content=content)


@app.exception_handler(ServingError)
async def serving_error_handler(request: Request, exc: ServingError) -> JSONResponse:
    """The serving backend could not load the adapter being deployed (issue #40). The deploy
    transaction never started, so 502 is a clean "upstream serving rejected/refused the load"."""
    logger.error(
        "serving_error", method=request.method, path=request.url.path, error=str(exc)
    )
    return JSONResponse(
        status_code=502,
        content={"error": {"code": "DEPLOY_FAILED", "message": str(exc)}},
    )


@app.exception_handler(SmokeTestError)
async def smoke_test_error_handler(
    request: Request, exc: SmokeTestError
) -> JSONResponse:
    """The just-loaded adapter failed the deploy-time smoke test (issue #41). The deploy aborted
    *before* the pointer moved, so the alias still points at the old (still-serving) version and
    the failure is already recorded as a warning log line. The 502 is a clean "upstream serving
    could not prove the adapter generates" - the transaction stays uncommitted by the caller."""
    logger.error(
        "smoke_test_error", method=request.method, path=request.url.path, error=str(exc)
    )
    return JSONResponse(
        status_code=502,
        content={"error": {"code": "SMOKE_TEST_FAILED", "message": str(exc)}},
    )


@app.exception_handler(BaseModelMismatchError)
async def base_model_mismatch_handler(
    request: Request, exc: BaseModelMismatchError
) -> JSONResponse:
    """(issue #65) The artifact being deployed is trained on a different base model than the
    serving stack is running, and the deploy was rejected before the pointer moved. 409 is a
    client-side configuration conflict (misaligned served_base_model) that the operator resolves
    by recreating/redeploying on the matching base (PRD §17.4) - never a server fault."""
    logger.error(
        "base_model_mismatch",
        method=request.method,
        path=request.url.path,
        error=str(exc),
    )
    return JSONResponse(
        status_code=409,
        content={"error": {"code": "BASE_MODEL_MISMATCH", "message": str(exc)}},
    )


@app.exception_handler(RateLimitExceeded)
async def rate_limit_exception_handler(
    request: Request, exc: RateLimitExceeded
) -> JSONResponse:
    logger.warning(
        "rate_limit_exceeded",
        method=request.method,
        path=request.url.path,
        client_host=request.client.host if request.client else None,
    )
    response = JSONResponse(
        status_code=429,
        content={"error": {"code": "RATE_LIMIT_EXCEEDED", "message": str(exc.detail)}},
    )
    view_rate_limit = getattr(request.state, "view_rate_limit", None)
    if view_rate_limit is not None:
        limit_item, args = view_rate_limit
        response.headers["X-RateLimit-Limit"] = str(limit_item.amount)
        response.headers["X-RateLimit-Remaining"] = "0"
        window_stats = limiter.limiter.get_window_stats(limit_item, *args)
        response.headers["X-RateLimit-Reset"] = str(1 + window_stats[0])
    return response


@app.get("/scalar", include_in_schema=False)
def scalar_docs() -> HTMLResponse:
    """Scalar API reference reading the same live OpenAPI spec as /docs and /redoc -- a
    friendlier alternative to Swagger UI, no separate spec file to keep in sync."""
    return HTMLResponse(
        """<!doctype html>
<html>
  <head>
    <title>DEFNEX MLOps API Reference</title>
    <meta charset="utf-8" />
    <meta name="viewport" content="width=device-width, initial-scale=1" />
  </head>
  <body>
    <script id="api-reference" data-url="/openapi.json"></script>
    <script src="https://cdn.jsdelivr.net/npm/@scalar/api-reference"></script>
  </body>
</html>"""
    )


@app.get("/metrics")
def metrics_endpoint() -> Response:
    """Prometheus metrics endpoint (PRD §28)."""
    from app.telemetry import get_prometheus_metrics

    try:
        from prometheus_client import CONTENT_TYPE_LATEST

        content_type = CONTENT_TYPE_LATEST
    except ImportError:
        content_type = "text/plain; charset=utf-8"

    return Response(content=get_prometheus_metrics(), media_type=content_type)
