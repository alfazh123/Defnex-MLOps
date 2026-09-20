from datetime import datetime, timedelta, timezone
import secrets
import uuid

from jose import JWTError, jwt
from passlib.context import CryptContext
from sqlalchemy import func, select
from sqlalchemy.orm import Session
import structlog

from app.config import settings
from app.models.password_reset import PasswordResetToken
from app.models.revoked_refresh_token import RevokedRefreshToken
from app.models.user import User

PASSWORD_RESET_TOKEN_TTL_MINUTES = 30

logger = structlog.get_logger(__name__)

pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")


def hash_password(password: str) -> str:
    return pwd_context.hash(password)


def verify_password(plain: str, hashed: str) -> bool:
    return pwd_context.verify(plain, hashed)


def create_access_token(data: dict, family_id: str | None = None) -> str:
    """Issue a short-lived access token (never checked against revocation, see decode_token)."""
    to_encode = data.copy()
    expire = datetime.now(timezone.utc) + timedelta(minutes=settings.jwt_expire_minutes)
    to_encode.update({"exp": expire, "type": "access", "jti": uuid.uuid4().hex})
    if family_id:
        # Carried only so /auth/logout can identify + revoke the refresh-token
        # family from the access token alone (issue #125).
        to_encode["family_id"] = family_id
    return jwt.encode(to_encode, settings.jwt_secret, algorithm=settings.jwt_algorithm)


def create_refresh_token(data: dict, family_id: str | None = None) -> str:
    """Issue a refresh token. `family_id` links it to prior/future rotations of
    the same login session, so reuse detection can revoke the whole chain."""
    to_encode = data.copy()
    expire = datetime.now(timezone.utc) + timedelta(
        minutes=settings.jwt_refresh_expire_minutes
    )
    to_encode.update(
        {
            "exp": expire,
            "type": "refresh",
            "jti": uuid.uuid4().hex,
            "family_id": family_id or uuid.uuid4().hex,
        }
    )
    return jwt.encode(to_encode, settings.jwt_secret, algorithm=settings.jwt_algorithm)


def issue_token_pair(user: User, family_id: str | None = None) -> tuple[str, str]:
    """Create a linked (access, refresh) pair for `user`, sharing one family_id
    across the whole session so rotation/logout/reuse-detection can find each other."""
    fam = family_id or uuid.uuid4().hex
    data = {"sub": str(user.id), "role": user.role}
    return create_access_token(data, family_id=fam), create_refresh_token(
        data, family_id=fam
    )


def decode_token(token: str, db: Session | None = None) -> dict | None:
    """Decode + verify a JWT's signature and expiry only.

    Access tokens are short-lived (settings.jwt_expire_minutes) and are
    deliberately NOT checked against any revocation store: natural expiry is
    fast enough that a DB round-trip on every authenticated request isn't
    worth it (issue #125). `db` is accepted (and ignored) only so existing
    callers - notably app/api/deps.py - don't need to change their call
    signature. Refresh-token revocation/rotation is handled separately by
    rotate_refresh_token() below.
    """
    try:
        return jwt.decode(
            token, settings.jwt_secret, algorithms=[settings.jwt_algorithm]
        )
    except JWTError:
        return None


def _family_marker(family_id: str) -> str:
    """Synthetic jti used to mark an entire refresh-token family as revoked."""
    return f"family-revoked:{family_id}"


def _is_refresh_jti_used(db: Session, jti: str) -> bool:
    return (
        db.scalar(select(RevokedRefreshToken).where(RevokedRefreshToken.jti == jti))
        is not None
    )


def _mark_refresh_jti_used(
    db: Session, jti: str, family_id: str, exp_ts: float | None
) -> None:
    expires_at = (
        datetime.fromtimestamp(exp_ts, tz=timezone.utc)
        if exp_ts
        else datetime.now(timezone.utc)
        + timedelta(minutes=settings.jwt_refresh_expire_minutes)
    )
    db.add(
        RevokedRefreshToken(
            jti=jti,
            family_id=family_id,
            revoked_at=datetime.now(timezone.utc),
            expires_at=expires_at,
        )
    )


def revoke_refresh_family(db: Session, family_id: str) -> None:
    """Revoke every refresh token belonging to `family_id`, including ones not
    individually recorded (rotation-mates that haven't been used yet)."""
    marker = _family_marker(family_id)
    if _is_refresh_jti_used(db, marker):
        return
    now = datetime.now(timezone.utc)
    db.add(
        RevokedRefreshToken(
            jti=marker,
            family_id=family_id,
            revoked_at=now,
            expires_at=now + timedelta(minutes=settings.jwt_refresh_expire_minutes),
        )
    )


def revoke_refresh_family_from_access_token(token: str, db: Session) -> None:
    """Used by /auth/logout: the client only has the access token, so pull the
    family_id out of it (best-effort - a malformed/expired token is a no-op)."""
    payload = decode_token(token)
    if payload is None:
        return
    family_id = payload.get("family_id")
    if family_id:
        revoke_refresh_family(db, family_id)


def rotate_refresh_token(token: str, db: Session) -> tuple[User, str, str] | None:
    """Validate a refresh token and, if it's still good, rotate it: the
    presented token is marked used/revoked and a brand-new (access, refresh)
    pair is issued in the same family.

    Returns None (and raises no exception) for any invalid case, including
    reuse of an already-rotated/revoked token - in that case the entire
    family is revoked as a theft indicator (issue #125 reuse detection).
    """
    try:
        payload = jwt.decode(
            token, settings.jwt_secret, algorithms=[settings.jwt_algorithm]
        )
    except JWTError:
        return None

    if payload.get("type") != "refresh":
        return None

    jti = payload.get("jti")
    user_id = payload.get("sub")
    if not jti or user_id is None:
        return None
    # Tokens minted before family_id existed fall back to a family-of-one.
    family_id = payload.get("family_id") or jti

    user = get_user_by_id(db, int(user_id))
    if user is None:
        return None

    if _is_refresh_jti_used(db, _family_marker(family_id)):
        # Family already revoked (prior reuse detection or logout).
        return None

    if _is_refresh_jti_used(db, jti):
        # This exact refresh token was already rotated/revoked and is being
        # presented again - treat as token theft and kill the whole family.
        revoke_refresh_family(db, family_id)
        logger.warning(
            "refresh_token_reuse_detected", user_id=user.id, family_id=family_id
        )
        return None

    _mark_refresh_jti_used(db, jti, family_id, payload.get("exp"))
    access_token, refresh_token = issue_token_pair(user, family_id=family_id)
    return user, access_token, refresh_token


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


def _utc_now_naive() -> datetime:
    """Naive-but-always-UTC "now" (matches idempotency_service._utc_now): SQLite's plain
    DateTime column type round-trips a tz-aware value unreliably, so both write and read
    sides of PasswordResetToken.expires_at must stay naive to compare correctly."""
    return datetime.now(timezone.utc).replace(tzinfo=None)


def create_password_reset_token(db: Session, user: User) -> str:
    """Issue #177: no email provider is wired in this codebase, so the caller (the router)
    logs the token via structlog instead of sending it - a placeholder channel, same
    mock-first pattern as MockTrainingRunner. Returns the raw token (only ever held in
    memory/logs, never re-derivable from the stored row)."""
    token = secrets.token_urlsafe(32)
    now = _utc_now_naive()
    db.add(
        PasswordResetToken(
            token=token,
            user_id=user.id,
            used=False,
            created_at=now,
            expires_at=now + timedelta(minutes=PASSWORD_RESET_TOKEN_TTL_MINUTES),
        )
    )
    db.flush()
    return token


def consume_password_reset_token(db: Session, token: str) -> User | None:
    """Validate and single-use-consume a password-reset token. Returns the target User on
    success, or None if the token is missing/expired/already used - the router maps any
    None into one generic 400 (never distinguishing which reason, to avoid giving an
    attacker a token-guessing oracle)."""
    row = db.get(PasswordResetToken, token)
    if row is None or row.used:
        return None
    if row.expires_at < _utc_now_naive():
        return None
    row.used = True
    db.flush()
    return get_user_by_id(db, row.user_id)
