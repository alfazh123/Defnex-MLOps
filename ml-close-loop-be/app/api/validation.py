from fastapi import APIRouter, Depends
from pydantic import BaseModel
from sqlalchemy.orm import Session

from app.api.deps import get_current_user, require_admin
from app.api.errors import APIError
from app.db.session import get_db
from app.models.user import User
from app.schemas.common import ErrorResponse
from app.schemas.validation import ValidationReport
from app.services import dataset_service, validation_service

router = APIRouter(tags=["Validation"])


class ValidateDatasetVersionRequest(BaseModel):
    """Optional body for POST .../validate (openapi.yaml)."""

    rule_set_version: str | None = None


def _get_dataset_version_or_404(db: Session, dataset_id: str, version: int):
    result = dataset_service.get_dataset_version(db, dataset_id, version)
    if result is None:
        raise APIError(404, "DATASET_NOT_FOUND", f'dataset_id "{dataset_id}" version {version} not found')
    return result


@router.post(
    "/datasets/{dataset_id}/versions/{version}/validate",
    response_model=ValidationReport,
    status_code=201,
    responses={404: {"model": ErrorResponse}, 409: {"model": ErrorResponse}},
)
def validate_dataset_version(
    dataset_id: str,
    version: int,
    request: ValidateDatasetVersionRequest | None = None,
    db: Session = Depends(get_db),
    _admin: User = Depends(require_admin),
) -> ValidationReport:
    dataset_version = _get_dataset_version_or_404(db, dataset_id, version)
    if dataset_version.status != "PROCESSED":
        raise APIError(
            409, "VALIDATION_INCOMPLETE", "Dataset version has not completed processing yet."
        )

    kwargs = {}
    if request is not None and request.rule_set_version is not None:
        kwargs["rule_set_version"] = request.rule_set_version

    # No intake/normalization pipeline persists actual record content anywhere in this
    # codebase yet (see validation_service module docstring + progress.txt US-005/US-006
    # notes) - `records` is an empty list until a future story adds that storage.
    report = validation_service.validate_dataset_version(db, dataset_version, records=[], **kwargs)
    return validation_service.to_schema(report)


@router.get(
    "/datasets/{dataset_id}/versions/{version}/validation-reports",
    response_model=list[ValidationReport],
    responses={404: {"model": ErrorResponse}},
)
def list_validation_reports(
    dataset_id: str, version: int, db: Session = Depends(get_db), _user: User = Depends(get_current_user)
) -> list[ValidationReport]:
    dataset_version = _get_dataset_version_or_404(db, dataset_id, version)
    reports = validation_service.list_validation_reports(db, dataset_version)
    return [validation_service.to_schema(r) for r in reports]


@router.get(
    "/datasets/{dataset_id}/versions/{version}/validation-reports/latest",
    response_model=ValidationReport,
    responses={404: {"model": ErrorResponse}},
)
def get_latest_validation_report(
    dataset_id: str, version: int, db: Session = Depends(get_db), _user: User = Depends(get_current_user)
) -> ValidationReport:
    dataset_version = _get_dataset_version_or_404(db, dataset_id, version)
    report = validation_service.get_latest_validation_report(db, dataset_version)
    if report is None:
        raise APIError(
            404,
            "VALIDATION_REPORT_NOT_FOUND",
            f'No validation report has been run yet for dataset_id "{dataset_id}" version {version}',
        )
    return validation_service.to_schema(report)
