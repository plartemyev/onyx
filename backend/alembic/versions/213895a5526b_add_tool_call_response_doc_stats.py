"""add tool_call response doc stats

Revision ID: 213895a5526b
Revises: 690544b34376
Create Date: 2026-11-01

"""

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql


# revision identifiers, used by Alembic.
revision = "213895a5526b"
down_revision = "690544b34376"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column(
        "tool_call",
        sa.Column("response_doc_stats", postgresql.JSONB(), nullable=True),
    )


def downgrade() -> None:
    op.drop_column("tool_call", "response_doc_stats")
