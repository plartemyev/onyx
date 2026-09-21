"""add_analyze_image_tool

Revision ID: b7e4c9f2a6d3
Revises: c3f8d2a47e91
Create Date: 2026-09-21 12:00:00.000000

"""

from alembic import op
import sqlalchemy as sa

# revision identifiers, used by Alembic.
revision = "b7e4c9f2a6d3"
down_revision = "c3f8d2a47e91"
branch_labels = None
depends_on = None


ANALYZE_IMAGE_TOOL = {
    "name": "AnalyzeImageTool",
    "display_name": "Analyze Image",
    "description": (
        "The Analyze Image Action lets the agent fetch images from direct URLs "
        "and understand their content with the vision model. The image is "
        "attached to the chat and the agent receives a description, or the "
        "answer to a focused question about it."
    ),
    "in_code_tool_id": "AnalyzeImageTool",
    "enabled": True,
}


def upgrade() -> None:
    conn = op.get_bind()

    # Check if tool already exists
    existing = conn.execute(
        sa.text("SELECT id FROM tool WHERE in_code_tool_id = :in_code_tool_id"),
        {"in_code_tool_id": ANALYZE_IMAGE_TOOL["in_code_tool_id"]},
    ).fetchone()

    if existing:
        tool_id = existing[0]
        # Update existing tool
        conn.execute(
            sa.text("""
                UPDATE tool
                SET name = :name,
                    display_name = :display_name,
                    description = :description
                WHERE in_code_tool_id = :in_code_tool_id
                """),
            ANALYZE_IMAGE_TOOL,
        )
    else:
        # Insert new tool
        conn.execute(
            sa.text("""
                INSERT INTO tool (name, display_name, description, in_code_tool_id, enabled)
                VALUES (:name, :display_name, :description, :in_code_tool_id, :enabled)
                """),
            ANALYZE_IMAGE_TOOL,
        )
        # Get the newly inserted tool's id
        result = conn.execute(
            sa.text("SELECT id FROM tool WHERE in_code_tool_id = :in_code_tool_id"),
            {"in_code_tool_id": ANALYZE_IMAGE_TOOL["in_code_tool_id"]},
        ).fetchone()
        tool_id = result[0]  # ty: ignore[not-subscriptable]

    # Associate the tool with all existing personas
    persona_ids = conn.execute(sa.text("SELECT id FROM persona")).fetchall()

    for (persona_id,) in persona_ids:
        # Check if association already exists
        exists = conn.execute(
            sa.text("""
                SELECT 1 FROM persona__tool
                WHERE persona_id = :persona_id AND tool_id = :tool_id
                """),
            {"persona_id": persona_id, "tool_id": tool_id},
        ).fetchone()

        if not exists:
            conn.execute(
                sa.text("""
                    INSERT INTO persona__tool (persona_id, tool_id)
                    VALUES (:persona_id, :tool_id)
                    """),
                {"persona_id": persona_id, "tool_id": tool_id},
            )


def downgrade() -> None:
    # We don't remove the tool on downgrade since it's fine to have it around.
    # If we upgrade again, it will be a no-op.
    pass
