from fastapi import APIRouter, Depends, Request
from sqlalchemy.orm import Session

from app.api.errors import APIError
from app.db.session import get_db
from app.limiter import limiter
from app.schemas.auth import LoginRequest, TokenResponse, UserCreate, UserResponse
from app.services import auth_service

router = APIRouter(tags=["Auth"])


@router.post("/auth/login", response_model=TokenResponse)
@limiter.limit("5/minute")
def login(request: Request, body: LoginRequest, db: Session = Depends(get_db)) -> TokenResponse:
    user = auth_service.get_user_by_username(db, body.username)
    if user is None or not auth_service.verify_password(body.password, user.hashed_password):
        raise APIError(401, "INVALID_CREDENTIALS", "Username or password is incorrect")

    token = auth_service.create_access_token({"sub": str(user.id), "role": user.role})
    return TokenResponse(
        access_token=token,
        user=UserResponse(id=user.id, username=user.username, role=user.role, created_at=user.created_at),
    )


@router.post("/auth/register", response_model=UserResponse, status_code=201)
@limiter.limit("3/minute")
def register(request: Request, body: UserCreate, db: Session = Depends(get_db)) -> UserResponse:
    existing = auth_service.get_user_by_username(db, body.username)
    if existing is not None:
        raise APIError(409, "USERNAME_TAKEN", f'Username "{body.username}" is already taken')

    if body.role not in ("admin", "user"):
        raise APIError(400, "INVALID_ROLE", f'Role must be "admin" or "user", got "{body.role}"')

    # First user auto-becomes admin
    from sqlalchemy import func, select
    from app.models.user import User

    user_count = db.scalar(select(func.count()).select_from(User))
    role = "admin" if user_count == 0 else body.role

    user = auth_service.create_user(db, body.username, body.password, role)
    db.commit()
    return UserResponse(id=user.id, username=user.username, role=user.role, created_at=user.created_at)
