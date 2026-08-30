from pydantic import BaseModel


class ErrorDetail(BaseModel):
    """openapi.yaml Error.error (PRD §18 standard error envelope)."""

    code: str
    message: str


class ErrorResponse(BaseModel):
    """openapi.yaml Error — `{"error": {"code": ..., "message": ...}}`."""

    error: ErrorDetail
