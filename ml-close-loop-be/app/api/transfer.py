"""Artifact transfer API endpoints (issue #72, PRD §20.3)."""

from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session

from app.api.deps import (
    PaginationParams,
    get_current_user,
    get_pagination,
    require_admin,
)
from app.api.errors import APIError
from app.db.session import get_db
from app.models.user import User
from app.schemas.common import PaginatedResponse
from app.schemas.transfer import Transfer, TransferInitiateRequest
from app.services import artifact_transfer

router = APIRouter(tags=["Transfers"])


@router.post(
    "/transfers",
    response_model=Transfer,
    status_code=201,
)
def initiate_transfer(
    request: TransferInitiateRequest,
    db: Session = Depends(get_db),
    _admin: User = Depends(require_admin),
) -> Transfer:
    """Initiate an artifact transfer (PRD §20.3). Admin only."""
    transfer = artifact_transfer.initiate_transfer(
        db, request.artifact_uri, request.source_host, request.target_host
    )
    db.commit()
    return transfer


@router.get(
    "/transfers",
    response_model=PaginatedResponse[Transfer],
)
def list_transfers(
    db: Session = Depends(get_db),
    _user: User = Depends(get_current_user),
    pg=Depends(get_pagination),
) -> PaginatedResponse[Transfer]:
    """List transfers. Any authenticated user."""
    transfers, total = artifact_transfer.list_transfers(
        db, limit=pg.limit, offset=pg.offset
    )
    return PaginatedResponse(
        items=[Transfer.model_validate(t) for t in transfers],
        total=total,
        page=pg.page,
        size=pg.size,
        pages=PaginationParams.pages_from(total, pg.size),
    )


@router.get(
    "/transfers/{transfer_id}",
    response_model=Transfer,
    responses={404: {"model": dict}},
)
def get_transfer(
    transfer_id: str,
    db: Session = Depends(get_db),
    _user: User = Depends(get_current_user),
) -> Transfer:
    """Get a transfer by ID. Any authenticated user."""
    transfer = db.get(artifact_transfer.ArtifactTransfer, transfer_id)
    if transfer is None:
        raise APIError(404, "RESOURCE_NOT_FOUND", f"transfer {transfer_id} not found")
    return Transfer.model_validate(transfer)
