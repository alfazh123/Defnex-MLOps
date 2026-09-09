"""add environments table and seed default/staging/production (issue #67)

Revision ID: 5f3a2b1c0d4e
Revises: 0f6d1c9e4a2f
Create Date: 2026-09-09 00:00:00.000000

"""

from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = "5f3a2b1c0d4e"
down_revision: Union[str, Sequence[str], None] = "0f6d1c9e4a2f"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


environments_table = sa.table(
    "environments",
    sa.column("name", sa.String),
    sa.column("description", sa.String),
)

_SEED_ROWS = (
    {
        "name": "default",
        "description": "Backward-compat single-environment mode (pre-#67).",
    },
    {
        "name": "staging",
        "description": "Integration validation environment (PRD §16.1).",
    },
    {"name": "production", "description": "Live serving environment (PRD §16.1)."},
)


def upgrade() -> None:
    """Create the environments table and seed the environment ladder (PRD §16.1).

    The `deployments.environment` column already exists as a plain string (PRD §15); this seeds
    the environments those strings name so `staging`/`production` are first-class entities.
    """
    op.create_table(
        "environments",
        sa.Column("name", sa.String(), nullable=False),
        sa.Column("description", sa.String(), nullable=True),
        sa.PrimaryKeyConstraint("name"),
    )
    op.bulk_insert(environments_table, list(_SEED_ROWS))


def downgrade() -> None:
    """Downgrade schema."""
    op.drop_table("environments")
