"""add eval set storage + eval set reference columns (issue #43)

Revision ID: 8b3c5144cda6
Revises: 4e04cef4c8ef
Create Date: 2026-09-06 00:00:00.000000

"""

from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = "8b3c5144cda6"
down_revision: Union[str, Sequence[str], None] = "4e04cef4c8ef"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Upgrade schema."""
    op.create_table(
        "eval_sets",
        sa.Column("eval_set_id", sa.String(), nullable=False),
        sa.PrimaryKeyConstraint("eval_set_id"),
    )
    op.create_table(
        "eval_set_versions",
        sa.Column("id", sa.Integer(), nullable=False),
        sa.Column("eval_set_id", sa.String(), nullable=False),
        sa.Column("version", sa.Integer(), nullable=False),
        sa.Column("records", sa.JSON(), nullable=False),
        sa.Column("created_at", sa.DateTime(), nullable=False),
        sa.Column("created_by", sa.String(), nullable=True),
        sa.ForeignKeyConstraint(["eval_set_id"], ["eval_sets.eval_set_id"]),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("eval_set_id", "version", name="uq_eval_set_version"),
    )
    op.add_column(
        "validation_reports",
        sa.Column("records", sa.JSON(), nullable=False, server_default="[]"),
    )
    op.add_column(
        "model_versions",
        sa.Column("eval_set_id", sa.String(), nullable=True),
    )
    op.add_column(
        "model_versions",
        sa.Column("eval_set_version", sa.Integer(), nullable=True),
    )
    op.add_column(
        "promotion_decisions",
        sa.Column("eval_set_id", sa.String(), nullable=True),
    )
    op.add_column(
        "promotion_decisions",
        sa.Column("eval_set_version", sa.Integer(), nullable=True),
    )


def downgrade() -> None:
    """Downgrade schema."""
    op.drop_column("promotion_decisions", "eval_set_version")
    op.drop_column("promotion_decisions", "eval_set_id")
    op.drop_column("model_versions", "eval_set_version")
    op.drop_column("model_versions", "eval_set_id")
    op.drop_column("validation_reports", "records")
    op.drop_table("eval_set_versions")
    op.drop_table("eval_sets")
