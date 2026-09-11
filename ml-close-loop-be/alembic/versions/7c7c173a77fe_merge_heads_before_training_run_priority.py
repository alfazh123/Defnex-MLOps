"""merge heads before training_run priority

Revision ID: 7c7c173a77fe
Revises: 1a61b851415e, 7ec17e63ceda, c1d2e3f4a5b6, c7d1a9f3b5e2
Create Date: 2026-09-11 00:00:00.000000

"""

from typing import Sequence, Union


# revision identifiers, used by Alembic.
revision: str = "7c7c173a77fe"
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
