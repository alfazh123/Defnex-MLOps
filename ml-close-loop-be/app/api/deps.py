import math
from dataclasses import dataclass

from fastapi import Depends, Header, Query
from sqlalchemy.orm import Session

from app.api.errors import APIError
from app.db.session import get_db
from app.models.model import ModelVersion
from app.models.user import User
from app.rbac import has_permission
from app.services import auth_service, model_service


@dataclass
class PaginationParams:
    page: int = 1
    size: int = 20

    @property
    def offset(self) -> int:
        return (self.page - 1) * self.size

    @property
    def limit(self) -> int:
        return self.size

    @staticmethod
    def pages_from(total: int, size: int) -> int:
        return math.ceil(total / size) if size > 0 else 0


def get_pagination(
    page: int = Query(1, ge=1),
    size: int = Query(20, ge=1, le=100),
) -> PaginationParams:
    return PaginationParams(page=page, size=size)


@dataclass
class FilterParams:
    status: str | None = None
    search: str | None = None
    model: str | None = None


def get_filters(
    status: str | None = Query(None),
    search: str | None = Query(None),
    model: str | None = Query(None),
) -> FilterParams:
    return FilterParams(status=status, search=search, model=model)


def get_model_version_or_404(db: Session, model_id: str, version: int) -> ModelVersion:
    """Resolve a model version or raise the standard 404 error envelope (openapi.yaml)."""
    model_version = model_service.get_model_version(db, model_id, version)
    if model_version is None:
        raise APIError(
            404, "MODEL_NOT_FOUND", f'model_id "{model_id}" version {version} not found'
        )
    return model_version


def get_current_user(
    authorization: str = Header(None), db: Session = Depends(get_db)
) -> User:
    """Extract and validate JWT token from Authorization header."""
    if authorization is None or not authorization.startswith("Bearer "):
        raise APIError(
            401, "MISSING_TOKEN", "Authorization header must be: Bearer <token>"
        )

    token = authorization.removeprefix("Bearer ").strip()
    payload = auth_service.decode_token(token, db)
    if payload is None:
        raise APIError(401, "INVALID_TOKEN", "Token is invalid or expired")

    user_id = payload.get("sub")
    if user_id is None or not str(user_id).isdigit():
        raise APIError(401, "INVALID_TOKEN", "Token payload is invalid")

    user = auth_service.get_user_by_id(db, int(user_id))
    if user is None:
        raise APIError(401, "USER_NOT_FOUND", "User from token no longer exists")

    return user


def get_current_user_and_token(
    authorization: str = Header(None), db: Session = Depends(get_db)
) -> tuple[User, str]:
    """Extract user + raw token for endpoints that need to revoke the token (e.g. logout)."""
    if authorization is None or not authorization.startswith("Bearer "):
        raise APIError(
            401, "MISSING_TOKEN", "Authorization header must be: Bearer <token>"
        )

    token = authorization.removeprefix("Bearer ").strip()
    payload = auth_service.decode_token(token, db)
    if payload is None:
        raise APIError(401, "INVALID_TOKEN", "Token is invalid or expired")

    user_id = payload.get("sub")
    if user_id is None or not str(user_id).isdigit():
        raise APIError(401, "INVALID_TOKEN", "Token payload is invalid")

    user = auth_service.get_user_by_id(db, int(user_id))
    if user is None:
        raise APIError(401, "USER_NOT_FOUND", "User from token no longer exists")

    return user, token


def require_admin(current_user: User = Depends(get_current_user)) -> User:
    """Require the current user to have admin role."""
    if current_user.role != "admin":
        raise APIError(403, "FORBIDDEN", "Admin access required")
    return current_user


def require_permission(permission: str):
    """Return a dependency requiring the current user's role to hold `permission`
    (app.rbac.ROLE_PERMISSIONS), for the critical endpoints (promote, deploy,
    rollback, infra credential write) identified in issue #123. Replaces the binary
    `require_admin` on those endpoints so a role beyond "admin" can be authorized
    without touching the endpoint again."""

    def _check(current_user: User = Depends(get_current_user)) -> User:
        if not has_permission(current_user.role, permission):
            raise APIError(
                403,
                "FORBIDDEN",
                f'role "{current_user.role}" lacks permission "{permission}"',
            )
        return current_user

    return _check
