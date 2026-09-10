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
    access_token, refresh_token = auth_service.issue_token_pair(user)
    return TokenResponse(
        access_token=access_token,
        refresh_token=refresh_token,
        user=UserResponse(
            id=user.id,
            username=user.username,
            role=user.role,
            created_at=user.created_at,
        ),
    )


@router.post("/auth/refresh", response_model=TokenResponse)
def refresh(body: RefreshRequest, db: Session = Depends(get_db)) -> TokenResponse:
    """Rotate the refresh token: the old one is invalidated and a new pair is
    issued. Reuse of an already-rotated token revokes the whole family (#125)."""
    result = auth_service.rotate_refresh_token(body.refresh_token, db)
    if result is None:
        db.commit()  # persist any reuse-detection revocation even on failure
        raise APIError(401, "INVALID_REFRESH_TOKEN", "Invalid or expired refresh token")

    user, access_token, refresh_token = result
    db.commit()
    return TokenResponse(
        access_token=access_token,
        refresh_token=refresh_token,
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
    db: Session = Depends(get_db),
) -> None:
    """Revoke the refresh-token family tied to the current access token (#125).

    The access token itself is short-lived and never revocation-checked (see
    auth_service.decode_token), so it stays technically valid until it expires
    naturally; logout's real effect is that the refresh token can no longer be
    used to mint new access tokens, forcing re-login once it expires.
    """
    user, token = user_and_token
    auth_service.revoke_refresh_family_from_access_token(token, db)
    db.commit()
    logger.info("logout_success", user_id=user.id)
