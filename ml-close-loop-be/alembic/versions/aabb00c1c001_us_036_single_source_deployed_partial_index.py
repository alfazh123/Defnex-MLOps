"""US-036 single-source deployed partial unique index

Revision ID: aabb00c1c001
Revises: 190c9e1c557c
Create Date: 2026-09-05 22:30:00.000000

"""

from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = "aabb00c1c001"
down_revision: Union[str, Sequence[str], None] = "190c9e1c557c"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None

_WHERE = "status = 'DEPLOYED'"


def upgrade() -> None:
    """Make two DEPLOYED versions for one model_id impossible at the DB level.

    The ModelVersion.status == "DEPLOYED" flag is the single source of truth for
    "which version is production"; this partial unique index guarantees at most
    one row per model_id can ever hold it (issue #36). SQLite (dev) and Postgres
    both support partial indexes, so the invariant holds on both backends.

    The index cannot be created if the bug this issue fixes already left >1 DEPLOYED
    version for a model_id, so duplicates are resolved first: for each model_id keep
    the version with the most recent `deployments.deployed_at` (falling back to the
    highest version when there is no history row) and retire the rest.
    """
    op.execute(
        sa.text(
            """
            UPDATE model_versions
            SET status = 'RETIRED'
            WHERE status = 'DEPLOYED'
              AND id NOT IN (
                SELECT keeper.id
                FROM (
                  SELECT v.id,
                         ROW_NUMBER() OVER (
                           PARTITION BY v.model_id
                           ORDER BY h.latest_deployed_at DESC NULLS LAST, v.version DESC
                         ) AS rn
                  FROM model_versions v
                  LEFT JOIN (
                    SELECT model_id, model_version, MAX(deployed_at) AS latest_deployed_at
                    FROM deployments
                    GROUP BY model_id, model_version
                  ) h ON h.model_id = v.model_id AND h.model_version = v.version
                  WHERE v.status = 'DEPLOYED'
                ) keeper
                WHERE keeper.rn = 1
              )
            """
        )
    )
    op.create_index(
        "uq_model_versions_one_deployed",
        "model_versions",
        ["model_id"],
        unique=True,
        sqlite_where=sa.text(_WHERE),
        postgresql_where=sa.text(_WHERE),
    )


def downgrade() -> None:
    op.drop_index(
        "uq_model_versions_one_deployed",
        table_name="model_versions",
    )
