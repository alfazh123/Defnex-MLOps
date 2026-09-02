from typing import Generic, TypeVar

from pydantic import BaseModel

T = TypeVar("T")


class PaginatedResponse(BaseModel, Generic[T]):
    items: list[T]
    total: int
    page: int
    size: int
    pages: int


class ErrorDetail(BaseModel):
    """openapi.yaml Error.error (PRD §18 standard error envelope)."""

    code: str
    message: str


class ErrorResponse(BaseModel):
    """openapi.yaml Error — `{"error": {"code": ..., "message": ...}}`."""

    error: ErrorDetail
