from fastapi import FastAPI, Request
from fastapi.exceptions import HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from slowapi.errors import RateLimitExceeded

from app.api.auth import router as auth_router
from app.api.datasets import router as datasets_router
from app.api.deployment import router as deployment_router
from app.api.health import router as health_router
from app.limiter import limiter
from app.api.models import router as models_router
from app.api.promotion import router as promotion_router
from app.api.training import router as training_router
from app.api.users import router as users_router
from app.api.validation import router as validation_router
from app.config import settings
from app.middleware.request_size import RequestSizeLimitMiddleware

app = FastAPI(title="DEFNEX MLOps Backend", version="0.1.0")
app.state.limiter = limiter

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",
        "http://localhost:8888",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.add_middleware(
    RequestSizeLimitMiddleware,
    max_body_size=settings.max_request_body_size,
)

app.include_router(health_router)
app.include_router(auth_router)
app.include_router(users_router)
app.include_router(datasets_router)
app.include_router(validation_router)
app.include_router(training_router)
app.include_router(models_router)
app.include_router(promotion_router)
app.include_router(deployment_router)


@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException) -> JSONResponse:
    """Pass through APIError's already-enveloped detail (PRD §18); wrap anything else the same way."""

    content = (
        exc.detail
        if isinstance(exc.detail, dict) and "error" in exc.detail
        else {"detail": exc.detail}
    )
    return JSONResponse(status_code=exc.status_code, content=content)


@app.exception_handler(RateLimitExceeded)
async def rate_limit_exception_handler(
    request: Request, exc: RateLimitExceeded
) -> JSONResponse:
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
