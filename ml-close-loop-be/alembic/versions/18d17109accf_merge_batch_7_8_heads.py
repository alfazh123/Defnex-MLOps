"""merge batch 7/8 heads

Revision ID: 18d17109accf
Revises: 1a61b851415e, c1d2e3f4a5b6, c7d1a9f3b5e2, 7ec17e63ceda
Create Date: 2026-09-10 22:01:42.529897

"""

from typing import Sequence, Union


# revision identifiers, used by Alembic.
revision: str = "18d17109accf"
down_revision: Union[str, Sequence[str], None] = (
    "1a61b851415e",
    "c1d2e3f4a5b6",
    "c7d1a9f3b5e2",
    "7ec17e63ceda",
)
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Upgrade schema."""
    pass


def downgrade() -> None:
    """Downgrade schema."""
    pass
