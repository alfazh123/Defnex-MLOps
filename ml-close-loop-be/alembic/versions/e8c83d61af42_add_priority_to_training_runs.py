"""add priority to training_runs (issue #135)

Revision ID: e8c83d61af42
Revises: 7c7c173a77fe
Create Date: 2026-09-11 00:05:00.000000

"""

from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = "e8c83d61af42"
down_revision: Union[str, Sequence[str], None] = "7c7c173a77fe"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Upgrade schema: fair-use claim-order priority ("low"/"normal"/"high")."""
    op.add_column(
        "training_runs",
        sa.Column("priority", sa.String(), nullable=False, server_default="normal"),
    )


def downgrade() -> None:
    """Downgrade schema."""
    op.drop_column("training_runs", "priority")
