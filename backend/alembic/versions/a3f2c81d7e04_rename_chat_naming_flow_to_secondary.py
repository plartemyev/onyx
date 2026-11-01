"""rename chat_naming flow to secondary

The CHAT_NAMING pointer flow becomes the SECONDARY flow: the same
admin-designated model now serves every secondary LLM flow (query
rephrase/expansion, document selection, section-relevance classification,
time filter, search scope, session naming, memory update), not only chat
naming. The column stores plain strings (native_enum=False), so the rename
is a data update only.

Revision ID: a3f2c81d7e04
Revises: 213895a5526b
Create Date: 2026-11-02

"""

from alembic import op

# revision identifiers, used by Alembic.
revision = "a3f2c81d7e04"
down_revision = "213895a5526b"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.execute(
        "UPDATE llm_model_flow SET llm_model_flow_type = 'secondary' "
        "WHERE llm_model_flow_type = 'chat_naming'"
    )


def downgrade() -> None:
    op.execute(
        "UPDATE llm_model_flow SET llm_model_flow_type = 'chat_naming' "
        "WHERE llm_model_flow_type = 'secondary'"
    )
