"""Unit tests for the in-turn search scope extension (llm_loop)."""

from onyx.chat.llm_loop import extend_search_scope_with_indexed_files
from onyx.context.search.models import PersonaSearchInfo
from onyx.tools.models import IndexedPageRef
from onyx.tools.tool_implementations.search.search_tool import SearchTool


def _bare_search_tool(chat_scoped: bool, attached: list[str]) -> SearchTool:
    # The class supports bare instances (via __new__) for tests.
    tool = SearchTool.__new__(SearchTool)
    tool.chat_scoped = chat_scoped
    tool.persona_search_info = PersonaSearchInfo(
        document_set_names=[],
        search_start_date=None,
        attached_document_ids=attached,
        hierarchy_node_ids=[],
    )
    return tool


def _ref(user_file_id: str) -> IndexedPageRef:
    return IndexedPageRef(
        url="https://example.com/page",
        user_file_id=user_file_id,
        file_id="file-id",
        name="Page.md",
    )


def test_extends_chat_scoped_tool_scope() -> None:
    tool = _bare_search_tool(chat_scoped=True, attached=["uf-1"])

    extend_search_scope_with_indexed_files([tool], [_ref("uf-2"), _ref("uf-3")])

    assert sorted(tool.persona_search_info.attached_document_ids) == [
        "uf-1",
        "uf-2",
        "uf-3",
    ]


def test_dedupes_known_ids() -> None:
    tool = _bare_search_tool(chat_scoped=True, attached=["uf-1"])

    extend_search_scope_with_indexed_files([tool], [_ref("uf-1")])

    assert tool.persona_search_info.attached_document_ids == ["uf-1"]


def test_ignores_unscoped_search_tool() -> None:
    """A non-chat-scoped internal search (Search UI, Slack) must not have its
    scope narrowed by chat turn artifacts."""
    tool = _bare_search_tool(chat_scoped=False, attached=[])

    extend_search_scope_with_indexed_files([tool], [_ref("uf-1")])

    assert tool.persona_search_info.attached_document_ids == []
