"""add retry_count to training_runs (P2-6)

Revision ID: f1a2b3c4d5e6
Revises: e21e265f06fb
Create Date: 2026-09-10 00:00:00.000000

"""

from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = "f1a2b3c4d5e6"
down_revision: Union[str, Sequence[str], None] = "e21e265f06fb"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Upgrade schema: track how many times a STALE run has been reclaimed."""
    op.add_column(
        "training_runs",
        sa.Column("retry_count", sa.Integer(), nullable=False, server_default="0"),
    )


def downgrade() -> None:
    """Downgrade schema."""
    op.drop_column("training_runs", "retry_count")
