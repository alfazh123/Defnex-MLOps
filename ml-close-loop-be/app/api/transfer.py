from fastapi import APIRouter, Depends
from sqlalchemy import func, select
from sqlalchemy.orm import Session

from app.api.deps import (
    PaginationParams,
    get_current_user,
    get_pagination,
    require_admin,
)
from app.api.errors import APIError
from app.db.session import get_db
from app.models.transfer import ArtifactTransfer
from app.models.user import User
from app.schemas.common import PaginatedResponse
from app.schemas.transfer import TransferInitiateRequest, TransferResponse
from app.services import artifact_transfer

router = APIRouter(tags=["Transfer"])


def _to_response(t: ArtifactTransfer) -> TransferResponse:
    return TransferResponse(
        transfer_id=t.transfer_id,
        artifact_uri=t.artifact_uri,
        source_host=t.source_host,
        target_host=t.target_host,
        checksum_before=t.checksum_before,
        checksum_after=t.checksum_after,
        status=t.status,  # type: ignore[arg-type]
        created_at=t.created_at,
        completed_at=t.completed_at,
        error_message=t.error_message,
    )


@router.post(
    "/transfers",
    response_model=TransferResponse,
    responses={409: {"model": dict}},
)
def initiate_transfer(
    body: TransferInitiateRequest,
    db: Session = Depends(get_db),
    _admin: User = Depends(require_admin),
) -> TransferResponse:
    """Initiate an artifact transfer (admin-only, PRD §20.3)."""
    transfer = artifact_transfer.initiate_transfer(
        db,
        artifact_uri=body.artifact_uri,
        source_host=body.source_host,
        target_host=body.target_host,
    )
    db.commit()
    return _to_response(transfer)


@router.get(
    "/transfers",
    response_model=PaginatedResponse[TransferResponse],
)
def list_transfers(
    db: Session = Depends(get_db),
    _user: User = Depends(get_current_user),
    pagination=Depends(get_pagination),
) -> PaginatedResponse[TransferResponse]:
    """List artifact transfers (paginated)."""
    total = db.scalar(select(func.count()).select_from(ArtifactTransfer)) or 0
    rows = (
        db.scalars(
            select(ArtifactTransfer)
            .order_by(ArtifactTransfer.created_at.desc())
            .offset(pagination.offset)
            .limit(pagination.limit)
        )
        .unique()
        .all()
    )
    return PaginatedResponse(
        items=[_to_response(r) for r in rows],
        total=total,
        page=pagination.page,
        size=pagination.size,
        pages=PaginationParams.pages_from(total, pagination.size),
    )


@router.get(
    "/transfers/{transfer_id}",
    response_model=TransferResponse,
    responses={404: {"model": dict}},
)
def get_transfer(
    transfer_id: str,
    db: Session = Depends(get_db),
    _user: User = Depends(get_current_user),
) -> TransferResponse:
    """Get a single transfer record."""
    transfer = db.get(ArtifactTransfer, transfer_id)
    if transfer is None:
        raise APIError(
            404, "TRANSFER_NOT_FOUND", f"transfer_id {transfer_id!r} not found"
        )
    return _to_response(transfer)
