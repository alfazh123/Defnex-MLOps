"""add evaluation_requested to model_versions (issue #128)

Revision ID: c1d2e3f4a5b6
Revises: f1a2b3c4d5e6
Create Date: 2026-09-10 00:00:01.000000

"""

from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = "c1d2e3f4a5b6"
down_revision: Union[str, Sequence[str], None] = "f1a2b3c4d5e6"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Upgrade schema: async evaluation queue flag (issue #128) - set by POST
    .../evaluation instead of writing caller-supplied signal numbers directly; cleared by
    the evaluation worker's compare-and-set claim."""
    op.add_column(
        "model_versions",
        sa.Column(
            "evaluation_requested",
            sa.Boolean(),
            nullable=False,
            server_default=sa.false(),
        ),
    )


def downgrade() -> None:
    """Downgrade schema."""
    op.drop_column("model_versions", "evaluation_requested")
