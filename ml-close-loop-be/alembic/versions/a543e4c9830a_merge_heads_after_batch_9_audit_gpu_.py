"""merge heads after batch 9 (audit, gpu priority, notifications)

Revision ID: a543e4c9830a
Revises: 3d4d0ab13cd5, d32c491e39a6, e8c83d61af42
Create Date: 2026-09-11 10:27:50.555655

"""

from typing import Sequence, Union


# revision identifiers, used by Alembic.
revision: str = "a543e4c9830a"
down_revision: Union[str, Sequence[str], None] = (
    "3d4d0ab13cd5",
    "d32c491e39a6",
    "e8c83d61af42",
)
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Upgrade schema."""
    pass


def downgrade() -> None:
    """Downgrade schema."""
    pass
