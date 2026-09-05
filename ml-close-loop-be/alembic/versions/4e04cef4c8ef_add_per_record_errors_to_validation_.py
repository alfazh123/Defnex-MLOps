"""add per_record_errors to validation reports

Revision ID: 4e04cef4c8ef
Revises: 190c9e1c557c
Create Date: 2026-09-05 22:30:59.074022

"""

from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = "4e04cef4c8ef"
down_revision: Union[str, Sequence[str], None] = "190c9e1c557c"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Upgrade schema."""
    op.add_column(
        "validation_reports",
        sa.Column("per_record_errors", sa.JSON(), nullable=False, server_default="[]"),
    )
    op.add_column(
        "validation_reports",
        sa.Column("content_hash", sa.String(), nullable=False, server_default=""),
    )


def downgrade() -> None:
    """Downgrade schema."""
    op.drop_column("validation_reports", "content_hash")
    op.drop_column("validation_reports", "per_record_errors")
