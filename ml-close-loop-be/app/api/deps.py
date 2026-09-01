from fastapi import Depends, Header
from sqlalchemy.orm import Session

from app.api.errors import APIError
from app.db.session import get_db
from app.models.model import ModelVersion
from app.models.user import User
from app.services import auth_service, model_service


def get_model_version_or_404(db: Session, model_id: str, version: int) -> ModelVersion:
    """Resolve a model version or raise the standard 404 error envelope (openapi.yaml)."""
    model_version = model_service.get_model_version(db, model_id, version)
    if model_version is None:
        raise APIError(404, "MODEL_NOT_FOUND", f'model_id "{model_id}" version {version} not found')
    return model_version


def get_current_user(authorization: str = Header(None), db: Session = Depends(get_db)) -> User:
    """Extract and validate JWT token from Authorization header."""
    if authorization is None or not authorization.startswith("Bearer "):
        raise APIError(401, "MISSING_TOKEN", "Authorization header must be: Bearer <token>")

    token = authorization.removeprefix("Bearer ").strip()
    payload = auth_service.decode_token(token)
    if payload is None:
        raise APIError(401, "INVALID_TOKEN", "Token is invalid or expired")

    user_id = payload.get("sub")
    if user_id is None:
        raise APIError(401, "INVALID_TOKEN", "Token payload is invalid")

    user = auth_service.get_user_by_id(db, int(user_id))
    if user is None:
        raise APIError(401, "USER_NOT_FOUND", "User from token no longer exists")

    return user


def require_admin(current_user: User = Depends(get_current_user)) -> User:
    """Require the current user to have admin role."""
    if current_user.role != "admin":
        raise APIError(403, "FORBIDDEN", "Admin access required")
    return current_user
