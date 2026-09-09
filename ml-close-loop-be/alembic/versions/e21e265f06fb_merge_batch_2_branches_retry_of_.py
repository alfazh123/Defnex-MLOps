"""merge batch 2 branches (retry_of, environments, training_config_hash)

Revision ID: e21e265f06fb
Revises: 34cc778c7f78, 5f3a2b1c0d4e, b3c7d5e9f1a2
Create Date: 2026-09-09 12:56:32.939341

"""

from typing import Sequence, Union


# revision identifiers, used by Alembic.
revision: str = "e21e265f06fb"
down_revision: Union[str, Sequence[str], None] = (
    "34cc778c7f78",
    "5f3a2b1c0d4e",
    "b3c7d5e9f1a2",
)
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Upgrade schema."""
    pass


def downgrade() -> None:
    """Downgrade schema."""
    pass
