"""Secondary-LLM wiring in SearchTool: fallback, history trimming, and the
document-selection budget clamp."""

from unittest.mock import MagicMock

from onyx.configs.constants import MessageType
from onyx.tools.models import ChatMinimalTextMessage
from onyx.tools.tool_implementations.search.constants import (
    SELECTION_MIN_CONTENT_BUDGET_TOKENS,
    SELECTION_PROMPT_RESERVE_TOKENS,
)
from onyx.tools.tool_implementations.search.search_tool import (
    SearchTool,
    _selection_content_budget,
    _trim_history_for_expansion,
)

MODULE = "onyx.tools.tool_implementations.search.search_tool"


def _llm(max_input_tokens: int) -> MagicMock:
    return MagicMock(config=MagicMock(max_input_tokens=max_input_tokens))


def _make_tool(llm: MagicMock, secondary_llm: MagicMock | None) -> SearchTool:
    return SearchTool(
        tool_id=1,
        emitter=MagicMock(),
        user=MagicMock(is_anonymous=False),
        persona_search_info=MagicMock(document_set_names=[]),
        llm=llm,
        document_index=MagicMock(),
        user_selected_filters=None,
        project_id_filter=None,
        enable_slack_search=False,
        auto_detect_filters=False,
        secondary_llm=secondary_llm,
    )


def test_secondary_llm_falls_back_to_main() -> None:
    main = _llm(100_000)
    tool = _make_tool(main, None)
    assert tool.secondary_llm is main


def test_secondary_llm_used_when_designated() -> None:
    main = _llm(100_000)
    secondary = _llm(8_000)
    tool = _make_tool(main, secondary)
    assert tool.secondary_llm is secondary


def test_selection_budget_unclamped_for_large_window() -> None:
    # 25 chunks * 512 tokens * multiplier 2
    assert _selection_content_budget(_llm(200_000), None) == 25 * 512 * 2


def test_selection_budget_clamped_to_small_window() -> None:
    assert _selection_content_budget(_llm(8_000), None) == (
        8_000 - SELECTION_PROMPT_RESERVE_TOKENS
    )


def test_selection_budget_floors_for_tiny_window() -> None:
    assert _selection_content_budget(_llm(1_024), None) == (
        SELECTION_MIN_CONTENT_BUDGET_TOKENS
    )


def test_selection_budget_respects_override() -> None:
    # A smaller override wins over both the default chunk budget and the
    # window-derived budget.
    assert _selection_content_budget(_llm(200_000), 4) == 4 * 512 * 2


def _history(count: int, chars: int) -> list[ChatMinimalTextMessage]:
    return [
        ChatMinimalTextMessage(message="x" * chars, message_type=MessageType.USER)
        for _ in range(count)
    ]


def test_trim_history_keeps_newest_within_budget() -> None:
    history = _history(count=10, chars=3_000)  # ~758 tokens each
    llm = _llm(4_000)  # budget = 4000 - 2048 = 1952 -> two messages fit

    trimmed = _trim_history_for_expansion(history, llm)

    assert len(trimmed) == 2
    assert trimmed[-1] is history[-1]


def test_trim_history_always_keeps_last_message() -> None:
    history = _history(count=1, chars=400_000)
    llm = _llm(1_000)

    trimmed = _trim_history_for_expansion(history, llm)

    assert trimmed == history


def test_trim_history_noop_when_it_fits() -> None:
    history = _history(count=3, chars=100)
    llm = _llm(100_000)

    assert _trim_history_for_expansion(history, llm) == history
