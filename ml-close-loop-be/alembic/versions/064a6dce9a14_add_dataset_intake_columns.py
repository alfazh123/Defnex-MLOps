"""add_dataset_intake_columns

Revision ID: 064a6dce9a14
Revises: f4a7c8e2d1b3
Create Date: 2026-09-10 13:47:41.412652

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = '064a6dce9a14'
down_revision: Union[str, Sequence[str], None] = 'f4a7c8e2d1b3'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.add_column('datasets', sa.Column('display_name', sa.String(), nullable=True))
    op.add_column('datasets', sa.Column('description', sa.String(), nullable=True))
    op.add_column('dataset_versions', sa.Column('raw_file_uri', sa.String(), nullable=True))
    op.add_column('dataset_versions', sa.Column('canonical_file_uri', sa.String(), nullable=True))


def downgrade() -> None:
    op.drop_column('dataset_versions', 'canonical_file_uri')
    op.drop_column('dataset_versions', 'raw_file_uri')
    op.drop_column('datasets', 'description')
    op.drop_column('datasets', 'display_name')
