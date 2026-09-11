"""merge heads before audit log

Revision ID: 91e28c0a95ea
Revises: 1a61b851415e, 7ec17e63ceda, c1d2e3f4a5b6, c7d1a9f3b5e2
Create Date: 2026-09-11 08:51:30.466338

"""

from typing import Sequence, Union


# revision identifiers, used by Alembic.
revision: str = "91e28c0a95ea"
down_revision: Union[str, Sequence[str], None] = (
    "1a61b851415e",
    "7ec17e63ceda",
    "c1d2e3f4a5b6",
    "c7d1a9f3b5e2",
)
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Upgrade schema."""
    pass


def downgrade() -> None:
    """Downgrade schema."""
    pass
