from collections.abc import Awaitable, Callable
from contextlib import asynccontextmanager

import structlog
from fastapi import FastAPI, Request, APIRouter
from fastapi.exceptions import HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from slowapi.errors import RateLimitExceeded
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.responses import RedirectResponse, Response

from app.api.auth import router as auth_router
from app.api.datasets import router as datasets_router
from app.api.deployment import router as deployment_router
from app.api.health import router as health_router
from app.db.session import engine
from app.limiter import limiter
from app.api.models import router as models_router
from app.api.promotion import router as promotion_router
from app.api.training import router as training_router
from app.api.users import router as users_router
from app.api.validation import router as validation_router
from app.config import settings
from app.logging import configure_logging
from app.middleware.request_size import RequestSizeLimitMiddleware

logger = structlog.get_logger(__name__)


@asynccontextmanager
async def lifespan(_app: FastAPI):
    configure_logging(log_level=settings.log_level, debug=settings.debug)
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
        "/training-runs",
        "/models",
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

v1_router = APIRouter(prefix="/api/v1")

v1_router.include_router(health_router)
v1_router.include_router(auth_router)
v1_router.include_router(users_router)
v1_router.include_router(datasets_router)
v1_router.include_router(validation_router)
v1_router.include_router(training_router)
v1_router.include_router(models_router)
v1_router.include_router(promotion_router)
v1_router.include_router(deployment_router)

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
