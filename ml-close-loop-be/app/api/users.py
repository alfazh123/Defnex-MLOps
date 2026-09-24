from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session

from app.api.deps import (
    FilterParams,
    PaginationParams,
    get_filters,
    get_pagination,
    require_admin,
)
from app.api.errors import APIError
from app.db.session import get_db
from app.models.user import User
from app.schemas.auth import UserResponse
from app.schemas.common import PaginatedResponse
from app.services import auth_service

router = APIRouter(tags=["Users"])


@router.get("/users", response_model=PaginatedResponse[UserResponse])
def list_users(
    db: Session = Depends(get_db),
    _admin: User = Depends(require_admin),
    pg: PaginationParams = Depends(get_pagination),
    fl: FilterParams = Depends(get_filters),
) -> PaginatedResponse[UserResponse]:
    users, total = auth_service.list_users(
        db, limit=pg.limit, offset=pg.offset, search=fl.search
    )
    return PaginatedResponse(
        items=[
            UserResponse(
                id=u.id, username=u.username, role=u.role, created_at=u.created_at
            )
            for u in users
        ],
        total=total,
        page=pg.page,
        size=pg.size,
        pages=PaginationParams.pages_from(total, pg.size),
    )


@router.delete("/users/{user_id}", status_code=204)
def delete_user(
    user_id: int, db: Session = Depends(get_db), _admin: User = Depends(require_admin)
):
    if not auth_service.delete_user(db, user_id):
        raise APIError(404, "USER_NOT_FOUND", f"user_id {user_id} not found")
    db.commit()
