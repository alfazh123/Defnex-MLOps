from fastapi import APIRouter, Depends, Request
from sqlalchemy.orm import Session
import structlog

from app.api.deps import get_current_user_and_token
from app.api.errors import APIError
from app.db.session import get_db
from app.limiter import limiter
from app.models.user import User
from app.schemas.auth import (
    LoginRequest,
    RefreshRequest,
    TokenResponse,
    UserCreate,
    UserResponse,
)
from app.services import auth_service

logger = structlog.get_logger(__name__)

router = APIRouter(tags=["Auth"])


@router.post("/auth/login", response_model=TokenResponse)
@limiter.limit("5/minute")
def login(
    request: Request, body: LoginRequest, db: Session = Depends(get_db)
) -> TokenResponse:
    user = auth_service.get_user_by_username(db, body.username)
    if user is None or not auth_service.verify_password(
        body.password, user.hashed_password
    ):
        logger.warning(
            "login_failed",
            username=body.username,
            client_host=request.client.host if request.client else None,
            user_agent=request.headers.get("user-agent"),
        )
        raise APIError(401, "INVALID_CREDENTIALS", "Username or password is incorrect")

    logger.info("login_success", username=body.username)
    token = auth_service.create_access_token({"sub": str(user.id), "role": user.role})
    refresh = auth_service.create_refresh_token(
        {"sub": str(user.id), "role": user.role}
    )
    return TokenResponse(
        access_token=token,
        refresh_token=refresh,
        user=UserResponse(
            id=user.id,
            username=user.username,
            role=user.role,
            created_at=user.created_at,
        ),
    )


@router.post("/auth/refresh", response_model=TokenResponse)
def refresh(body: RefreshRequest, db: Session = Depends(get_db)) -> TokenResponse:
    payload = auth_service.decode_token(body.refresh_token)
    if payload is None or payload.get("type") != "refresh":
        raise APIError(401, "INVALID_REFRESH_TOKEN", "Invalid or expired refresh token")

    user_id = int(payload["sub"])
    user = auth_service.get_user_by_id(db, user_id)
    if user is None:
        raise APIError(401, "INVALID_REFRESH_TOKEN", "Invalid or expired refresh token")

    token = auth_service.create_access_token({"sub": str(user.id), "role": user.role})
    refresh = auth_service.create_refresh_token(
        {"sub": str(user.id), "role": user.role}
    )
    return TokenResponse(
        access_token=token,
        refresh_token=refresh,
        user=UserResponse(
            id=user.id,
            username=user.username,
            role=user.role,
            created_at=user.created_at,
        ),
    )


@router.post("/auth/register", response_model=UserResponse, status_code=201)
@limiter.limit("3/minute")
def register(
    request: Request, body: UserCreate, db: Session = Depends(get_db)
) -> UserResponse:
    existing = auth_service.get_user_by_username(db, body.username)
    if existing is not None:
        logger.warning(
            "register_username_taken",
            username=body.username,
            client_host=request.client.host if request.client else None,
            user_agent=request.headers.get("user-agent"),
        )
        raise APIError(
            409, "USERNAME_TAKEN", f'Username "{body.username}" is already taken'
        )

    if body.role not in ("admin", "user"):
        raise APIError(
            400, "INVALID_ROLE", f'Role must be "admin" or "user", got "{body.role}"'
        )

    # First user auto-becomes admin; subsequent registrations always get "user"
    from sqlalchemy import func, select
    from app.models.user import User

    user_count = db.scalar(select(func.count()).select_from(User))
    role = "admin" if user_count == 0 else "user"

    user = auth_service.create_user(db, body.username, body.password, role)
    db.commit()
    logger.info("register_success", username=body.username, role=role)
    return UserResponse(
        id=user.id, username=user.username, role=user.role, created_at=user.created_at
    )


@router.post("/auth/logout", status_code=204)
def logout(
    user_and_token: tuple[User, str] = Depends(get_current_user_and_token),
) -> None:
    """Revoke the current access token (#86 Security Hardening)."""
    user, token = user_and_token
    auth_service.revoke_token(token)
    logger.info("logout_success", user_id=user.id)
