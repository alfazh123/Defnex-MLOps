from datetime import datetime, timezone

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.models.eval_set import EvalSet, EvalSetVersion as EvalSetVersionModel
from app.models.validation import ValidationReport
from app.schemas.eval_set import (
    EvalSetSummary,
    EvalSetVersion as EvalSetVersionSchema,
    EvalSetVersionCreateRequest,
)
from app.services.validation_service import _user_content


def register_eval_set(db: Session, eval_set_id: str) -> EvalSet:
    """Get-or-create the EvalSet row for `eval_set_id` (issue #43)."""

    eval_set = db.get(EvalSet, eval_set_id)
    if eval_set is None:
        eval_set = EvalSet(eval_set_id=eval_set_id)
        db.add(eval_set)
        db.flush()
    return eval_set


def _validated_user_contents(db: Session) -> set[str]:
    """Every normalized user-content string seen in any stored validation report.

    The reverse side of H8: a new eval-set record must not appear in data that has
    already been validated (and could be - or already is - trained on).
    """

    contents: set[str] = set()
    reports = db.scalars(select(ValidationReport)).all()
    for report in reports:
        for record in report.records or []:
            if (content := _user_content(record)) is not None:
                contents.add(content)
    return contents


def create_eval_set_version(
    db: Session, eval_set_id: str, request: EvalSetVersionCreateRequest
) -> EvalSetVersionModel:
    """Create the next version for `eval_set_id`, registering the eval set if needed.

    Refuses records whose normalized user content is already present in any validated
    training report (H8 leakage, validation-rules.md H8) - the eval set and the training
    data are kept disjoint in both directions, not only at validation time.
    """

    register_eval_set(db, eval_set_id)

    validated_contents = _validated_user_contents(db)
    overlaps = {
        content
        for record in request.records
        if (content := _user_content(record)) is not None
        and content in validated_contents
    }
    if overlaps:
        raise ValueError(
            f'eval set "{eval_set_id}" overlaps validated training data on '
            f"{len(overlaps)} user-content value(s); refusing to add it (H8 leakage)"
        )

    latest_version = db.scalar(
        select(EvalSetVersionModel.version)
        .where(EvalSetVersionModel.eval_set_id == eval_set_id)
        .order_by(EvalSetVersionModel.version.desc())
    )
    next_version = (latest_version or 0) + 1

    version = EvalSetVersionModel(
        eval_set_id=eval_set_id,
        version=next_version,
        records=request.records,
        created_at=datetime.now(timezone.utc),
        created_by=None,
    )
    db.add(version)
    db.flush()
    return version


def get_eval_set_version(
    db: Session, eval_set_id: str, version: int
) -> EvalSetVersionModel | None:
    return db.scalar(
        select(EvalSetVersionModel).where(
            EvalSetVersionModel.eval_set_id == eval_set_id,
            EvalSetVersionModel.version == version,
        )
    )


def get_latest_eval_set_version(
    db: Session, eval_set_id: str
) -> EvalSetVersionModel | None:
    return db.scalar(
        select(EvalSetVersionModel)
        .where(EvalSetVersionModel.eval_set_id == eval_set_id)
        .order_by(EvalSetVersionModel.version.desc())
    )


def list_eval_set_versions(db: Session, eval_set_id: str) -> list[EvalSetVersionModel]:
    return list(
        db.scalars(
            select(EvalSetVersionModel)
            .where(EvalSetVersionModel.eval_set_id == eval_set_id)
            .order_by(EvalSetVersionModel.version.desc())
        )
    )


def list_eval_sets(db: Session) -> list[EvalSetSummary]:
    """Every eval set with its latest version + version count, for GET /eval-sets."""

    eval_sets = db.scalars(select(EvalSet).order_by(EvalSet.eval_set_id)).all()
    return [
        EvalSetSummary(
            eval_set_id=eval_set.eval_set_id,
            latest_version=eval_set.versions[-1].version if eval_set.versions else None,
            version_count=len(eval_set.versions),
        )
        for eval_set in eval_sets
    ]


def to_schema(version: EvalSetVersionModel) -> EvalSetVersionSchema:
    return EvalSetVersionSchema(
        eval_set_id=version.eval_set_id,
        version=version.version,
        record_count=len(version.records),
        records=version.records,
        created_at=version.created_at,
        created_by=version.created_by,
    )
