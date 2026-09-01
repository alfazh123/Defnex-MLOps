from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session

from app.api.errors import APIError
from app.db.session import get_db
from app.schemas.auth import LoginRequest, TokenResponse, UserCreate, UserResponse
from app.services import auth_service

router = APIRouter(tags=["Auth"])


@router.post("/auth/login", response_model=TokenResponse)
def login(request: LoginRequest, db: Session = Depends(get_db)) -> TokenResponse:
    user = auth_service.get_user_by_username(db, request.username)
    if user is None or not auth_service.verify_password(request.password, user.hashed_password):
        raise APIError(401, "INVALID_CREDENTIALS", "Username or password is incorrect")

    token = auth_service.create_access_token({"sub": str(user.id), "role": user.role})
    return TokenResponse(
        access_token=token,
        user=UserResponse(id=user.id, username=user.username, role=user.role, created_at=user.created_at),
    )


@router.post("/auth/register", response_model=UserResponse, status_code=201)
def register(request: UserCreate, db: Session = Depends(get_db)) -> UserResponse:
    existing = auth_service.get_user_by_username(db, request.username)
    if existing is not None:
        raise APIError(409, "USERNAME_TAKEN", f'Username "{request.username}" is already taken')

    if request.role not in ("admin", "user"):
        raise APIError(400, "INVALID_ROLE", f'Role must be "admin" or "user", got "{request.role}"')

    # First user auto-becomes admin
    from sqlalchemy import func, select
    from app.models.user import User

    user_count = db.scalar(select(func.count()).select_from(User))
    role = "admin" if user_count == 0 else request.role

    user = auth_service.create_user(db, request.username, request.password, role)
    db.commit()
    return UserResponse(id=user.id, username=user.username, role=user.role, created_at=user.created_at)
