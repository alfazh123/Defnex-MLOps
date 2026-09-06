from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session

from app.api.deps import get_current_user, require_admin
from app.api.errors import APIError
from app.db.session import get_db
from app.models.user import User
from app.schemas.common import ErrorResponse
from app.schemas.eval_set import (
    EvalSetSummary,
    EvalSetVersion,
    EvalSetVersionCreateRequest,
)
from app.services import eval_set_service

router = APIRouter(tags=["Eval Sets"])


@router.post(
    "/eval-sets/{eval_set_id}/versions",
    response_model=EvalSetVersion,
    status_code=201,
    responses={409: {"model": ErrorResponse}},
)
def create_eval_set_version(
    eval_set_id: str,
    request: EvalSetVersionCreateRequest,
    db: Session = Depends(get_db),
    _admin: User = Depends(require_admin),
) -> EvalSetVersion:
    try:
        version = eval_set_service.create_eval_set_version(db, eval_set_id, request)
    except ValueError as exc:
        raise APIError(409, "EVAL_SET_OVERLAP", str(exc)) from exc
    db.commit()
    return eval_set_service.to_schema(version)


@router.get("/eval-sets", response_model=list[EvalSetSummary])
def list_eval_sets(
    db: Session = Depends(get_db),
    _user: User = Depends(get_current_user),
) -> list[EvalSetSummary]:
    return eval_set_service.list_eval_sets(db)


@router.get(
    "/eval-sets/{eval_set_id}/versions",
    response_model=list[EvalSetVersion],
    responses={404: {"model": ErrorResponse}},
)
def list_eval_set_versions(
    eval_set_id: str,
    db: Session = Depends(get_db),
    _user: User = Depends(get_current_user),
) -> list[EvalSetVersion]:
    versions = eval_set_service.list_eval_set_versions(db, eval_set_id)
    if not versions:
        raise APIError(
            404, "EVAL_SET_NOT_FOUND", f'eval_set_id "{eval_set_id}" not found'
        )
    return [eval_set_service.to_schema(v) for v in versions]


@router.get(
    "/eval-sets/{eval_set_id}/versions/{version}",
    response_model=EvalSetVersion,
    responses={404: {"model": ErrorResponse}},
)
def get_eval_set_version(
    eval_set_id: str,
    version: int,
    db: Session = Depends(get_db),
    _user: User = Depends(get_current_user),
) -> EvalSetVersion:
    result = eval_set_service.get_eval_set_version(db, eval_set_id, version)
    if result is None:
        raise APIError(
            404,
            "EVAL_SET_NOT_FOUND",
            f'eval_set_id "{eval_set_id}" version {version} not found',
        )
    return eval_set_service.to_schema(result)
