"""add heartbeat_at to training_runs (issue #60)

Revision ID: 0f6d1c9e4a2f
Revises: c4dcd1b7a1f8
Create Date: 2026-09-08 00:00:00.000000

"""

from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = "0f6d1c9e4a2f"
down_revision: Union[str, Sequence[str], None] = "c4dcd1b7a1f8"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Upgrade schema: worker liveness timestamp for job heartbeat/stale detection.

    PostgreSQL-compatible timestamp (sa.DateTime() => TIMESTAMP WITHOUT TIME ZONE); the
    app writes timezone-aware UTC values. Works identically on the SQLite dev database.
    """
    op.add_column(
        "training_runs",
        sa.Column("heartbeat_at", sa.DateTime(), nullable=True),
    )


def downgrade() -> None:
    """Downgrade schema."""
    op.drop_column("training_runs", "heartbeat_at")
