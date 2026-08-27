from fastapi import FastAPI, Request
from fastapi.exceptions import HTTPException
from fastapi.responses import JSONResponse

from app.api.datasets import router as datasets_router
from app.api.health import router as health_router

app = FastAPI(title="DEFNEX MLOps Backend", version="0.1.0")

app.include_router(health_router)
app.include_router(datasets_router)


@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException) -> JSONResponse:
    """Pass through APIError's already-enveloped detail (PRD §18); wrap anything else the same way."""

    content = exc.detail if isinstance(exc.detail, dict) and "error" in exc.detail else {"detail": exc.detail}
    return JSONResponse(status_code=exc.status_code, content=content)
