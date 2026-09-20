import re
from datetime import datetime

from pydantic import BaseModel, field_validator


def _validate_password_strength(v: str) -> str:
    """Shared by UserCreate and PasswordResetConfirmRequest (issue #177) - a reset must not
    be allowed to set a weaker password than registration requires."""
    if len(v) < 8:
        raise ValueError("Password must be at least 8 characters")
    if not re.search(r"[A-Z]", v):
        raise ValueError("Password must contain at least one uppercase letter")
    if not re.search(r"\d", v):
        raise ValueError("Password must contain at least one digit")
    return v


class LoginRequest(BaseModel):
    username: str
    password: str


class TokenResponse(BaseModel):
    access_token: str
    refresh_token: str
    token_type: str = "bearer"
    user: "UserResponse"


class UserCreate(BaseModel):
    username: str
    password: str
    role: str = "user"

    @field_validator("password")
    @classmethod
    def validate_password(cls, v: str) -> str:
        return _validate_password_strength(v)


class PasswordResetRequest(BaseModel):
    """POST /auth/forgot-password (issue #177). Always responds the same way regardless of
    whether `username` exists, to avoid leaking which usernames are registered."""

    username: str


class PasswordResetConfirmRequest(BaseModel):
    """POST /auth/reset-password (issue #177)."""

    token: str
    new_password: str

    @field_validator("new_password")
    @classmethod
    def validate_new_password(cls, v: str) -> str:
        return _validate_password_strength(v)


class RefreshRequest(BaseModel):
    refresh_token: str


class UserResponse(BaseModel):
    id: int
    username: str
    role: str
    created_at: datetime


TokenResponse.model_rebuild()
