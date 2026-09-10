"""add_notebook_url_to_compute_resources

Revision ID: f4a7c8e2d1b3
Revises: 213418ebfc30, e21e265f06fb
Create Date: 2026-09-10 00:00:00.000000

"""

from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = "f4a7c8e2d1b3"
down_revision: Union[str, Sequence[str], None] = ("213418ebfc30", "e21e265f06fb")
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.add_column(
        "compute_resources", sa.Column("notebook_url", sa.String(), nullable=True)
    )


def downgrade() -> None:
    op.drop_column("compute_resources", "notebook_url")
