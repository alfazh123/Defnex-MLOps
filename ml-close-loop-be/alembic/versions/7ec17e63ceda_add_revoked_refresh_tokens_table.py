"""add revoked_refresh_tokens table

Revision ID: 7ec17e63ceda
Revises: 489a47ac4d8d
Create Date: 2026-09-10 20:54:18.365251

"""

from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = "7ec17e63ceda"
down_revision: Union[str, Sequence[str], None] = "489a47ac4d8d"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Create revoked_refresh_tokens table (issue #125).

    NOTE: `alembic revision --autogenerate` also picked up unrelated pending
    model changes (artifact_transfers table, dataset_versions/datasets columns,
    training_runs.retry_of FK) from other in-flight work on this repo. Those
    are out of scope for issue #125 and were manually stripped from this
    migration - only the revoked_refresh_tokens table belongs here.
    """
    op.create_table(
        "revoked_refresh_tokens",
        sa.Column("id", sa.Integer(), autoincrement=True, nullable=False),
        sa.Column("jti", sa.String(), nullable=False),
        sa.Column("family_id", sa.String(), nullable=False),
        sa.Column("revoked_at", sa.DateTime(), nullable=False),
        sa.Column("expires_at", sa.DateTime(), nullable=False),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("jti", name="uq_revoked_refresh_tokens_jti"),
    )
    op.create_index(
        op.f("ix_revoked_refresh_tokens_family_id"),
        "revoked_refresh_tokens",
        ["family_id"],
        unique=False,
    )
    op.create_index(
        op.f("ix_revoked_refresh_tokens_jti"),
        "revoked_refresh_tokens",
        ["jti"],
        unique=False,
    )


def downgrade() -> None:
    """Drop revoked_refresh_tokens table."""
    op.drop_index(
        op.f("ix_revoked_refresh_tokens_jti"), table_name="revoked_refresh_tokens"
    )
    op.drop_index(
        op.f("ix_revoked_refresh_tokens_family_id"), table_name="revoked_refresh_tokens"
    )
    op.drop_table("revoked_refresh_tokens")
