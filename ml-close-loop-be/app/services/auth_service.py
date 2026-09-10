from datetime import datetime, timedelta, timezone
import uuid

from jose import JWTError, jwt
from passlib.context import CryptContext
from sqlalchemy import func, select
from sqlalchemy.orm import Session

from app.config import settings
from app.models.user import User

pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

_REVOKED_TOKENS: set[str] = set()


def hash_password(password: str) -> str:
    return pwd_context.hash(password)


def verify_password(plain: str, hashed: str) -> bool:
    return pwd_context.verify(plain, hashed)


def create_access_token(data: dict) -> str:
    to_encode = data.copy()
    expire = datetime.now(timezone.utc) + timedelta(minutes=settings.jwt_expire_minutes)
    to_encode.update({"exp": expire, "type": "access", "jti": uuid.uuid4().hex})
    return jwt.encode(to_encode, settings.jwt_secret, algorithm=settings.jwt_algorithm)


def create_refresh_token(data: dict) -> str:
    to_encode = data.copy()
    expire = datetime.now(timezone.utc) + timedelta(
        minutes=settings.jwt_refresh_expire_minutes
    )
    to_encode.update({"exp": expire, "type": "refresh", "jti": uuid.uuid4().hex})
    return jwt.encode(to_encode, settings.jwt_secret, algorithm=settings.jwt_algorithm)


def decode_token(token: str) -> dict | None:
    try:
        payload = jwt.decode(
            token, settings.jwt_secret, algorithms=[settings.jwt_algorithm]
        )
        if payload.get("jti") in _REVOKED_TOKENS:
            return None
        return payload
    except JWTError:
        return None


def revoke_token(token: str) -> None:
    """Add a token's JTI to the revocation set."""
    try:
        payload = jwt.decode(
            token, settings.jwt_secret, algorithms=[settings.jwt_algorithm]
        )
        if payload and "jti" in payload:
            _REVOKED_TOKENS.add(payload["jti"])
    except JWTError:
        pass


def is_token_revoked(token: str) -> bool:
    """Check if a token has been revoked."""
    try:
        payload = jwt.decode(
            token, settings.jwt_secret, algorithms=[settings.jwt_algorithm]
        )
        if payload and "jti" in payload:
            return payload["jti"] in _REVOKED_TOKENS
    except JWTError:
        pass
    return False


def get_user_by_username(db: Session, username: str) -> User | None:
    return db.scalar(select(User).where(User.username == username))


def get_user_by_id(db: Session, user_id: int) -> User | None:
    return db.get(User, user_id)


def create_user(db: Session, username: str, password: str, role: str = "user") -> User:
    user = User(
        username=username,
        hashed_password=hash_password(password),
        role=role,
        created_at=datetime.now(timezone.utc),
    )
    db.add(user)
    db.flush()
    return user


def list_users(db: Session, limit: int = 20, offset: int = 0) -> tuple[list[User], int]:
    total = db.scalar(select(func.count()).select_from(User))
    users = list(
        db.scalars(select(User).order_by(User.id).limit(limit).offset(offset)).all()
    )
    return users, total


def delete_user(db: Session, user_id: int) -> bool:
    user = db.get(User, user_id)
    if user is None:
        return False
    db.delete(user)
    return True
