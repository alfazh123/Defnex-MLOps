from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session

from app.api.deps import (
    FilterParams,
    PaginationParams,
    get_current_user,
    get_filters,
    get_model_version_or_404,
    get_pagination,
    require_admin,
)
from app.api.errors import APIError
from app.db.session import get_db
from app.models.user import User
from app.schemas.common import ErrorResponse, PaginatedResponse
from app.schemas.feedback import (
    FeedbackCreateRequest,
    FeedbackCurateRequest,
    FeedbackRecord,
)
from app.services import feedback_service

router = APIRouter(tags=["Feedback"])


@router.post(
    "/feedback",
    response_model=FeedbackRecord,
    status_code=201,
    responses={404: {"model": ErrorResponse}},
)
def submit_feedback(
    request: FeedbackCreateRequest,
    db: Session = Depends(get_db),
    _user: User = Depends(get_current_user),
) -> FeedbackRecord:
    """Submit feedback on an inference response (issue #42). `model_id`/`version` must be an
    existing model version - `get_model_version_or_404` (issue #36's helper) resolves it the
    same way every other model-version-scoped endpoint does."""
    model_version = get_model_version_or_404(db, request.model_id, request.version)
    feedback = feedback_service.submit_feedback(db, model_version, request)
    db.commit()
    return feedback_service.to_schema(feedback)


@router.get("/feedback", response_model=PaginatedResponse[FeedbackRecord])
def list_feedback(
    db: Session = Depends(get_db),
    _user: User = Depends(get_current_user),
    pg: PaginationParams = Depends(get_pagination),
    fl: FilterParams = Depends(get_filters),
) -> PaginatedResponse[FeedbackRecord]:
    """`fl.status` filters on curation_status (PENDING/APPROVED/REJECTED); `fl.model` filters
    on the referenced model_id - same `FilterParams`/`PaginationParams` every other list
    endpoint uses, no new pagination/filter logic (issue #42's own explicit requirement)."""
    rows, total = feedback_service.list_feedback(
        db, limit=pg.limit, offset=pg.offset, status=fl.status, model=fl.model
    )
    return PaginatedResponse(
        items=[feedback_service.to_schema(row) for row in rows],
        total=total,
        page=pg.page,
        size=pg.size,
        pages=PaginationParams.pages_from(total, pg.size),
    )


@router.get("/feedback/candidates", response_model=list[dict])
def list_candidates(
    db: Session = Depends(get_db),
    _admin: User = Depends(require_admin),
) -> list[dict]:
    """Every APPROVED feedback row as a canonical record, ready to feed straight into
    `POST .../validate` or the from-feedback dataset-version endpoint (issue #42)."""
    return feedback_service.approved_candidates_as_records(db)


def _get_feedback_or_404(db: Session, feedback_id: str):
    feedback = feedback_service.get_feedback(db, feedback_id)
    if feedback is None:
        raise APIError(
            404, "FEEDBACK_NOT_FOUND", f'feedback_id "{feedback_id}" not found'
        )
    return feedback


@router.post(
    "/feedback/{feedback_id}/approve",
    response_model=FeedbackRecord,
    responses={404: {"model": ErrorResponse}, 409: {"model": ErrorResponse}},
)
def approve_feedback(
    feedback_id: str,
    request: FeedbackCurateRequest | None = None,
    db: Session = Depends(get_db),
    _admin: User = Depends(require_admin),
) -> FeedbackRecord:
    _get_feedback_or_404(db, feedback_id)
    try:
        feedback = feedback_service.approve_feedback(
            db, feedback_id, request.curated_by if request else None
        )
    except ValueError as exc:
        raise APIError(409, "FEEDBACK_NOT_PENDING", str(exc)) from exc
    db.commit()
    return feedback_service.to_schema(feedback)


@router.post(
    "/feedback/{feedback_id}/reject",
    response_model=FeedbackRecord,
    responses={404: {"model": ErrorResponse}, 409: {"model": ErrorResponse}},
)
def reject_feedback(
    feedback_id: str,
    request: FeedbackCurateRequest | None = None,
    db: Session = Depends(get_db),
    _admin: User = Depends(require_admin),
) -> FeedbackRecord:
    _get_feedback_or_404(db, feedback_id)
    try:
        feedback = feedback_service.reject_feedback(
            db, feedback_id, request.curated_by if request else None
        )
    except ValueError as exc:
        raise APIError(409, "FEEDBACK_NOT_PENDING", str(exc)) from exc
    db.commit()
    return feedback_service.to_schema(feedback)
