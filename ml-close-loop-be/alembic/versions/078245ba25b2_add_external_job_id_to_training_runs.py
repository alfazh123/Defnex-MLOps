"""add external_job_id to training_runs

Revision ID: 078245ba25b2
Revises: aabb00c1c001
Create Date: 2026-09-05 00:00:00.000000

"""

from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = "078245ba25b2"
down_revision: Union[str, Sequence[str], None] = "aabb00c1c001"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Upgrade schema."""
    op.add_column(
        "training_runs", sa.Column("external_job_id", sa.String(), nullable=True)
    )


def downgrade() -> None:
    """Downgrade schema."""
    op.drop_column("training_runs", "external_job_id")
