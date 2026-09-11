"""add audit_logs table

Revision ID: 3d4d0ab13cd5
Revises: 91e28c0a95ea
Create Date: 2026-09-11 08:58:12.470286

"""

from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa

# revision identifiers, used by Alembic.
revision: str = "3d4d0ab13cd5"
down_revision: Union[str, Sequence[str], None] = "91e28c0a95ea"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Upgrade schema.

    NOTE: autogenerate also detected unrelated pre-existing drift on main (a missing
    `artifact_transfers` table, missing dataset_versions/datasets columns, and a missing
    training_runs.retry_of FK) - none of that belongs in an issue #129 (audit log) migration,
    so it was stripped out here. See the PR description for that drift, reported separately
    per CLAUDE.md's no-silent-scope-change rule; it is not fixed by this migration.
    """
    op.create_table(
        "audit_logs",
        sa.Column("id", sa.Integer(), autoincrement=True, nullable=False),
        sa.Column("actor_id", sa.Integer(), nullable=True),
        sa.Column("action", sa.String(), nullable=False),
        sa.Column("resource_type", sa.String(), nullable=False),
        sa.Column("resource_id", sa.String(), nullable=False),
        sa.Column("before_json", sa.JSON(), nullable=True),
        sa.Column("after_json", sa.JSON(), nullable=True),
        sa.Column("result", sa.String(), nullable=False),
        sa.Column("reason", sa.String(), nullable=True),
        sa.Column("created_at", sa.DateTime(), nullable=False),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index(
        op.f("ix_audit_logs_action"), "audit_logs", ["action"], unique=False
    )
    op.create_index(
        op.f("ix_audit_logs_actor_id"), "audit_logs", ["actor_id"], unique=False
    )
    op.create_index(
        op.f("ix_audit_logs_created_at"), "audit_logs", ["created_at"], unique=False
    )
    op.create_index(
        op.f("ix_audit_logs_resource_id"), "audit_logs", ["resource_id"], unique=False
    )
    op.create_index(
        op.f("ix_audit_logs_resource_type"),
        "audit_logs",
        ["resource_type"],
        unique=False,
    )


def downgrade() -> None:
    """Downgrade schema."""
    op.drop_index(op.f("ix_audit_logs_resource_type"), table_name="audit_logs")
    op.drop_index(op.f("ix_audit_logs_resource_id"), table_name="audit_logs")
    op.drop_index(op.f("ix_audit_logs_created_at"), table_name="audit_logs")
    op.drop_index(op.f("ix_audit_logs_actor_id"), table_name="audit_logs")
    op.drop_index(op.f("ix_audit_logs_action"), table_name="audit_logs")
    op.drop_table("audit_logs")
