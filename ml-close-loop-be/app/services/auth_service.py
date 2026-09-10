from datetime import datetime, timedelta, timezone
import uuid

from jose import JWTError, jwt
from passlib.context import CryptContext
from sqlalchemy import func, select
from sqlalchemy.orm import Session

from app.config import settings
from app.models.revoked_token import RevokedToken
from app.models.user import User

pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")


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


def decode_token(token: str, db: Session | None = None) -> dict | None:
    try:
        payload = jwt.decode(
            token, settings.jwt_secret, algorithms=[settings.jwt_algorithm]
        )
        if db is not None and payload.get("jti"):
            revoked = db.scalar(
                select(RevokedToken).where(RevokedToken.jti == payload["jti"])
            )
            if revoked is not None:
                return None
        return payload
    except JWTError:
        return None


def revoke_token(token: str, db: Session) -> None:
    """Persist token revocation in DB (P2-1 durable revocation)."""
    try:
        payload = jwt.decode(
            token, settings.jwt_secret, algorithms=[settings.jwt_algorithm]
        )
        if payload and "jti" in payload:
            exp_ts = payload.get("exp")
            expires_at = (
                datetime.fromtimestamp(exp_ts, tz=timezone.utc)
                if exp_ts
                else datetime.now(timezone.utc) + timedelta(hours=1)
            )
            existing = db.scalar(
                select(RevokedToken).where(RevokedToken.jti == payload["jti"])
            )
            if existing is None:
                db.add(
                    RevokedToken(
                        jti=payload["jti"],
                        revoked_at=datetime.now(timezone.utc),
                        expires_at=expires_at,
                    )
                )
                db.flush()
    except JWTError:
        pass


def is_token_revoked(token: str, db: Session) -> bool:
    """Check if a token has been revoked via DB (P2-1)."""
    try:
        payload = jwt.decode(
            token, settings.jwt_secret, algorithms=[settings.jwt_algorithm]
        )
        if payload and "jti" in payload:
            return (
                db.scalar(
                    select(RevokedToken).where(RevokedToken.jti == payload["jti"])
                )
                is not None
            )
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
