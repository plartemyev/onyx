"""add tool_call response size and token stats

Revision ID: 690544b34376
Revises: a3f7c2d94e15
Create Date: 2026-10-28 19:50:38.302380

"""

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision = "690544b34376"
down_revision = "a3f7c2d94e15"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column("tool_call", sa.Column("response_bytes", sa.Integer(), nullable=True))
    op.add_column(
        "tool_call", sa.Column("response_tokens", sa.Integer(), nullable=True)
    )


def downgrade() -> None:
    op.drop_column("tool_call", "response_tokens")
    op.drop_column("tool_call", "response_bytes")
