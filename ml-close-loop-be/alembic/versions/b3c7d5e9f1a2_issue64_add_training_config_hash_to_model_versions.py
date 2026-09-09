"""add training_config_hash to model_versions (issue #64)

Revision ID: b3c7d5e9f1a2
Revises: 0f6d1c9e4a2f
Create Date: 2026-09-09 00:00:00.000000

"""

from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = "b3c7d5e9f1a2"
down_revision: Union[str, Sequence[str], None] = "0f6d1c9e4a2f"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Upgrade schema: deterministic id of the training config (sha256[0:16]) so the §37
    lineage chain can point at one config hash per version. Nullable — pre-existing rows
    predate the hash; new registrations always store it."""
    op.add_column(
        "model_versions",
        sa.Column("training_config_hash", sa.String(), nullable=True),
    )


def downgrade() -> None:
    """Downgrade schema."""
    op.drop_column("model_versions", "training_config_hash")
