"""add image caption cache

Revision ID: 6b86ce1d1464
Revises: b7e4c9f2a6d3
Create Date: 2026-09-22 09:43:59.605524

"""

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision = "6b86ce1d1464"
down_revision = "b7e4c9f2a6d3"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "image_caption_cache",
        sa.Column("id", sa.Integer(), autoincrement=True, nullable=False),
        # sha256 hex of the raw image bytes
        sa.Column("content_hash", sa.String(64), nullable=False),
        sa.Column("model_name", sa.String(), nullable=False),
        # sha256 hex of the system prompt + question the caption answers
        sa.Column("prompt_hash", sa.String(64), nullable=False),
        sa.Column("caption", sa.Text(), nullable=False),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.func.now(),
            nullable=False,
        ),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint(
            "content_hash",
            "model_name",
            "prompt_hash",
            name="uq_image_caption_cache_key",
        ),
    )


def downgrade() -> None:
    op.drop_table("image_caption_cache")
