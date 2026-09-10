"""merge heads before revoked_refresh_tokens

Revision ID: 489a47ac4d8d
Revises: f1a2b3c4d5e6, f4a7c8e2d1b3
Create Date: 2026-09-10 20:53:56.278169

"""

from typing import Sequence, Union


# revision identifiers, used by Alembic.
revision: str = "489a47ac4d8d"
down_revision: Union[str, Sequence[str], None] = ("f1a2b3c4d5e6", "f4a7c8e2d1b3")
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Upgrade schema."""
    pass


def downgrade() -> None:
    """Downgrade schema."""
    pass
