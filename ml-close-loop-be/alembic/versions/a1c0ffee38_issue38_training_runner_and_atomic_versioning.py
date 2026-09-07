"""issue #38: version name/metadata columns + training run wall-clock bounds

Adds the immutable per-version `name`, `git_commit`, and `training_started_at`/
`training_completed_at` to `model_versions`, and `started_at`/`finished_at` to
`training_runs` (single source of truth for the version metadata timestamps).

Revision ID: a1c0ffee38
Revises: 8b3c5144cda6
Create Date: 2026-09-06 12:00:00.000000

"""

from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = "a1c0ffee38"
down_revision: Union[str, Sequence[str], None] = "8b3c5144cda6"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Upgrade schema."""
    op.add_column("model_versions", sa.Column("name", sa.String(), nullable=True))
    op.add_column("model_versions", sa.Column("git_commit", sa.String(), nullable=True))
    op.add_column(
        "model_versions",
        sa.Column("training_started_at", sa.DateTime(), nullable=True),
    )
    op.add_column(
        "model_versions",
        sa.Column("training_completed_at", sa.DateTime(), nullable=True),
    )
    op.add_column(
        "training_runs", sa.Column("started_at", sa.DateTime(), nullable=True)
    )
    op.add_column(
        "training_runs", sa.Column("finished_at", sa.DateTime(), nullable=True)
    )


def downgrade() -> None:
    """Downgrade schema."""
    op.drop_column("training_runs", "finished_at")
    op.drop_column("training_runs", "started_at")
    op.drop_column("model_versions", "training_completed_at")
    op.drop_column("model_versions", "training_started_at")
    op.drop_column("model_versions", "git_commit")
    op.drop_column("model_versions", "name")
