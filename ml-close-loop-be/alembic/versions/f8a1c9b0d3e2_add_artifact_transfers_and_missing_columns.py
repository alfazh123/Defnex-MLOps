"""add_artifact_transfers_table_and_missing_columns

Fixes schema drift where the ORM models declared columns/tables that no prior
migration ever created: `ArtifactTransfer` (issue #72) had no migration at all,
`Dataset.display_name`/`description` and `DatasetVersion.raw_file_uri`/
`canonical_file_uri` were added to the model without a matching migration, and
`TrainingRun.retry_of` (column already exists) was missing its self-referential
foreign key to `training_runs.training_run_id`. Found while seeding a fresh
local database after the Batch 7-10 merges surfaced it as a hard runtime
failure (`OperationalError: no such column: datasets.display_name`).

Revision ID: f8a1c9b0d3e2
Revises: 1ce554582542
Create Date: 2026-09-11 14:40:00.000000

"""

from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = "f8a1c9b0d3e2"
down_revision: Union[str, Sequence[str], None] = "1ce554582542"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Upgrade schema."""
    op.create_table(
        "artifact_transfers",
        sa.Column("transfer_id", sa.String(), nullable=False),
        sa.Column("artifact_uri", sa.String(), nullable=False),
        sa.Column("source_host", sa.String(), nullable=False),
        sa.Column("target_host", sa.String(), nullable=False),
        sa.Column("checksum_before", sa.String(), nullable=False),
        sa.Column("checksum_after", sa.String(), nullable=True),
        sa.Column("status", sa.String(), nullable=False),
        sa.Column("created_at", sa.DateTime(), nullable=False),
        sa.Column("completed_at", sa.DateTime(), nullable=True),
        sa.Column("error_message", sa.String(), nullable=True),
        sa.PrimaryKeyConstraint("transfer_id"),
    )
    op.add_column(
        "dataset_versions", sa.Column("raw_file_uri", sa.String(), nullable=True)
    )
    op.add_column(
        "dataset_versions", sa.Column("canonical_file_uri", sa.String(), nullable=True)
    )
    op.add_column("datasets", sa.Column("display_name", sa.String(), nullable=True))
    op.add_column("datasets", sa.Column("description", sa.String(), nullable=True))
    with op.batch_alter_table("training_runs") as batch_op:
        batch_op.create_foreign_key(
            "fk_training_runs_retry_of_training_run_id",
            "training_runs",
            ["retry_of"],
            ["training_run_id"],
        )


def downgrade() -> None:
    """Downgrade schema."""
    with op.batch_alter_table("training_runs") as batch_op:
        batch_op.drop_constraint(
            "fk_training_runs_retry_of_training_run_id", type_="foreignkey"
        )
    op.drop_column("datasets", "description")
    op.drop_column("datasets", "display_name")
    op.drop_column("dataset_versions", "canonical_file_uri")
    op.drop_column("dataset_versions", "raw_file_uri")
    op.drop_table("artifact_transfers")
