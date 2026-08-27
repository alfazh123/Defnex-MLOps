from fastapi import HTTPException


class APIError(HTTPException):
    """HTTPException whose body already matches openapi.yaml's Error envelope (PRD §18)."""

    def __init__(self, status_code: int, code: str, message: str):
        super().__init__(status_code=status_code, detail={"error": {"code": code, "message": message}})
