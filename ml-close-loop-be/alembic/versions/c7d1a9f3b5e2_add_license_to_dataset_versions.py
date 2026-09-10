"""add license to dataset_versions (issue #134)

Revision ID: c7d1a9f3b5e2
Revises: f1a2b3c4d5e6
Create Date: 2026-09-10 00:00:00.000000

"""

from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = "c7d1a9f3b5e2"
down_revision: Union[str, Sequence[str], None] = "f1a2b3c4d5e6"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Upgrade schema: dataset version license (e.g. "cc-by-nc-4.0", "apache-2.0"), nullable
    for existing rows. The promotion flow surfaces this alongside the base model's license
    (app/config.py BASE_MODEL_LICENSES) and warns - never auto-blocks - on a non-commercial
    license heading toward production (issue #134)."""
    op.add_column(
        "dataset_versions",
        sa.Column("license", sa.String(), nullable=True),
    )


def downgrade() -> None:
    """Downgrade schema."""
    op.drop_column("dataset_versions", "license")
