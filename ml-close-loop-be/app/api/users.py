from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session

from app.api.deps import require_admin
from app.api.errors import APIError
from app.db.session import get_db
from app.models.user import User
from app.schemas.auth import UserResponse
from app.services import auth_service

router = APIRouter(tags=["Users"])


@router.get("/users", response_model=list[UserResponse])
def list_users(db: Session = Depends(get_db), _admin: User = Depends(require_admin)) -> list[UserResponse]:
    users = auth_service.list_users(db)
    return [UserResponse(id=u.id, username=u.username, role=u.role, created_at=u.created_at) for u in users]


@router.delete("/users/{user_id}", status_code=204)
def delete_user(user_id: int, db: Session = Depends(get_db), _admin: User = Depends(require_admin)):
    if not auth_service.delete_user(db, user_id):
        raise APIError(404, "USER_NOT_FOUND", f'user_id {user_id} not found')
    db.commit()
