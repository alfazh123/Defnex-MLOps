"""add retry_of to training_runs (issue #61)

Revision ID: 34cc778c7f78
Revises: 0f6d1c9e4a2f
Create Date: 2026-09-09 00:00:00.000000

"""

from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = "34cc778c7f78"
down_revision: Union[str, Sequence[str], None] = "0f6d1c9e4a2f"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Upgrade schema: retry lineage id back to the immutable original FAILED run.

    Nullable (the original run has no retry_of). The FK lives on the ORM model
    (`TrainingRun.retry_of`, honored by `create_all`); the migration adds the plain column
    because SQLite's Alembic dialect cannot ADD COLUMN with a FK constraint (its
    copy-and-move batch mode is not used anywhere else in this repo — every added column,
    e.g. heartbeat_at, follows the same plain add_column pattern).
    """
    op.add_column(
        "training_runs",
        sa.Column("retry_of", sa.String(), nullable=True),
    )
    op.create_index(
        op.f("ix_training_runs_retry_of"), "training_runs", ["retry_of"], unique=False
    )


def downgrade() -> None:
    """Downgrade schema."""
    op.drop_index(op.f("ix_training_runs_retry_of"), table_name="training_runs")
    op.drop_column("training_runs", "retry_of")
