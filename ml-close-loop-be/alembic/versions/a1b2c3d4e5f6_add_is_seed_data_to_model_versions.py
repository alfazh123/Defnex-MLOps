"""add is_seed_data to model_versions (issue #165)

Revision ID: a1b2c3d4e5f6
Revises: f8a1c9b0d3e2
Create Date: 2026-09-20 00:00:00.000000

"""

from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = "a1b2c3d4e5f6"
down_revision: Union[str, Sequence[str], None] = "f8a1c9b0d3e2"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.add_column(
        "model_versions",
        sa.Column(
            "is_seed_data",
            sa.Boolean(),
            nullable=False,
            server_default=sa.false(),
        ),
    )


def downgrade() -> None:
    op.drop_column("model_versions", "is_seed_data")
