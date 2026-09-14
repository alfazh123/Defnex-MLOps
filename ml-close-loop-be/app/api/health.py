from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session

from app.db.session import get_db

router = APIRouter()


@router.get("/health")
def health(db: Session = Depends(get_db)) -> dict:
    checks: dict[str, str] = {"db": "ok"}
    try:
        db.execute(__import__("sqlalchemy", fromlist=["text"]).text("SELECT 1"))
    except Exception:
        checks["db"] = "error"

    try:
        from app.services.artifact_storage import get_artifact_storage
        from app.config import settings

        if settings.artifact_backend == "minio":
            storage = get_artifact_storage()
            if hasattr(storage, "client"):
                storage.client.list_buckets()
            checks["minio"] = "ok"
    except Exception:
        checks["minio"] = "error"

    try:
        from app.config import settings

        if settings.serving_backend == "vllm":
            import httpx

            r = httpx.get(f"{settings.vllm_url}/health", timeout=5)
            checks["vllm"] = "ok" if r.status_code == 200 else f"error:{r.status_code}"
        else:
            checks["vllm"] = "skipped"
    except Exception:
        checks["vllm"] = "error"

    return {"status": "ok", "checks": checks}
