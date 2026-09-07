import uuid
from datetime import datetime, timezone

from sqlalchemy import func, select, update
from sqlalchemy.orm import Session, selectinload

from app.models.feedback import Feedback
from app.models.model import ModelVersion
from app.schemas.feedback import FeedbackCreateRequest, FeedbackRecord


def submit_feedback(
    db: Session, model_version: ModelVersion, request: FeedbackCreateRequest
) -> Feedback:
    """Record feedback on an inference response, `PENDING` curation (issue #42)."""

    feedback = Feedback(
        feedback_id=f"feedback-{uuid.uuid4().hex[:6]}",
        model_version_id=model_version.id,
        prompt=request.prompt,
        response=request.response,
        rating=request.rating,
        correction=request.correction,
        curation_status="PENDING",
        submitted_at=datetime.now(timezone.utc),
    )
    db.add(feedback)
    db.flush()
    return feedback


def _curate(
    db: Session, feedback_id: str, new_status: str, curated_by: str | None
) -> Feedback:
    """Atomically transition `feedback_id` PENDING -> `new_status` (issue #42).

    Compare-and-set on `curation_status` (same pattern as `training_service.claim_training_run`,
    issue #33): only the caller that flips PENDING -> new_status at the SQL level wins. Two
    concurrent approve/reject calls on the same row can then not both succeed - the loser's
    `rowcount` is 0 and it raises, matching the issue's REQUIRED concurrency criterion. Works on
    SQLite (`SELECT ... FOR UPDATE` is a no-op there) and PostgreSQL.
    """

    result = db.execute(
        update(Feedback)
        .where(
            Feedback.feedback_id == feedback_id, Feedback.curation_status == "PENDING"
        )
        .values(
            curation_status=new_status,
            curated_by=curated_by,
            curated_at=datetime.now(timezone.utc),
        )
        .execution_options(synchronize_session=False)
    )
    if result.rowcount != 1:
        raise ValueError(
            f"feedback {feedback_id!r} is not PENDING (already curated, or lost a "
            "concurrent curation race)"
        )
    db.flush()
    feedback = db.get(Feedback, feedback_id)
    assert feedback is not None
    db.refresh(feedback)
    return feedback


def approve_feedback(db: Session, feedback_id: str, curated_by: str | None) -> Feedback:
    return _curate(db, feedback_id, "APPROVED", curated_by)


def reject_feedback(db: Session, feedback_id: str, curated_by: str | None) -> Feedback:
    return _curate(db, feedback_id, "REJECTED", curated_by)


def get_feedback(db: Session, feedback_id: str) -> Feedback | None:
    return db.get(Feedback, feedback_id)


def list_feedback(
    db: Session,
    limit: int = 20,
    offset: int = 0,
    status: str | None = None,
    model: str | None = None,
) -> tuple[list[Feedback], int]:
    base_filter = select(Feedback).options(selectinload(Feedback.model_version))

    if status is not None:
        base_filter = base_filter.where(Feedback.curation_status == status)
    if model is not None:
        base_filter = base_filter.join(ModelVersion).where(
            ModelVersion.model_id.ilike(f"%{model}%")
        )

    total = db.scalar(select(func.count()).select_from(base_filter.subquery()))
    rows = list(
        db.scalars(
            base_filter.order_by(Feedback.submitted_at.desc())
            .limit(limit)
            .offset(offset)
        ).all()
    )
    return rows, total


def approved_candidates_as_records(db: Session) -> list[dict]:
    """Every `APPROVED` feedback row as a canonical record, ready to feed straight into
    `POST .../validate` (issue #35) or `create_dataset_version_from_feedback` (issue #42).

    Uses `correction` as the assistant ground truth when a reviewer supplied one (a human
    fix to a flawed response) - it's a better training target than the original `response`;
    falls back to `response` when there is no correction.
    """

    rows = db.scalars(
        select(Feedback).where(Feedback.curation_status == "APPROVED")
    ).all()
    return [
        {
            "id": row.feedback_id,
            "messages": [
                {"role": "user", "content": row.prompt},
                {
                    "role": "assistant",
                    "content": row.correction if row.correction else row.response,
                },
            ],
            "metadata": {"source": "feedback", "rating": row.rating},
        }
        for row in rows
    ]


def to_schema(feedback: Feedback) -> FeedbackRecord:
    model_version = feedback.model_version
    return FeedbackRecord(
        feedback_id=feedback.feedback_id,
        model_id=model_version.model_id,
        version=model_version.version,
        prompt=feedback.prompt,
        response=feedback.response,
        rating=feedback.rating,
        correction=feedback.correction,
        curation_status=feedback.curation_status,
        submitted_by=feedback.submitted_by,
        submitted_at=feedback.submitted_at,
        curated_by=feedback.curated_by,
        curated_at=feedback.curated_at,
    )
