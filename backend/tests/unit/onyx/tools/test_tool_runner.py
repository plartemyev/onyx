from typing import Any
from unittest.mock import Mock, patch

from onyx.chat.models import ChatMessageSimple
from onyx.configs.constants import MessageType
from onyx.prompts.tool_prompts import (
    TOOL_CALL_DROPPED_CONCURRENCY_PROMPT,
    TOOL_CALL_LOST_PROMPT,
    TOOL_CALL_MERGED_PROMPT,
)
from onyx.server.query_and_chat.placement import Placement
from onyx.tools.interface import Tool
from onyx.tools.models import ToolCallKickoff, ToolResponse
from onyx.tools.tool_runner import _merge_tool_calls, run_tool_calls


def _make_tool_call(
    tool_name: str,
    tool_args: dict,
    tool_call_id: str = "call_1",
    turn_index: int = 0,
    tab_index: int = 0,
) -> ToolCallKickoff:
    """Helper to create a ToolCallKickoff for testing."""
    return ToolCallKickoff(
        tool_call_id=tool_call_id,
        tool_name=tool_name,
        tool_args=tool_args,
        placement=Placement(turn_index=turn_index, tab_index=tab_index),
    )


class TestMergeToolCalls:
    """Tests for _merge_tool_calls function."""

    def test_empty_list(self) -> None:
        """Empty input returns empty output."""
        result = _merge_tool_calls([])
        assert result == []

    def test_single_search_tool_call_not_merged(self) -> None:
        """A single SearchTool call is returned as-is (no merging needed)."""
        call = _make_tool_call(
            tool_name="internal_search",
            tool_args={"queries": ["query1"]},
            tool_call_id="call_1",
        )
        result = _merge_tool_calls([call])

        assert len(result) == 1
        assert result[0].tool_name == "internal_search"
        assert result[0].tool_args == {"queries": ["query1"]}
        assert result[0].tool_call_id == "call_1"

    def test_single_web_search_tool_call_not_merged(self) -> None:
        """A single WebSearchTool call is returned as-is."""
        call = _make_tool_call(
            tool_name="web_search",
            tool_args={"queries": ["web query"]},
        )
        result = _merge_tool_calls([call])

        assert len(result) == 1
        assert result[0].tool_name == "web_search"
        assert result[0].tool_args == {"queries": ["web query"]}

    def test_single_open_url_tool_call_not_merged(self) -> None:
        """A single OpenURLTool call is returned as-is."""
        call = _make_tool_call(
            tool_name="open_url",
            tool_args={"urls": ["https://example.com"]},
        )
        result = _merge_tool_calls([call])

        assert len(result) == 1
        assert result[0].tool_name == "open_url"
        assert result[0].tool_args == {"urls": ["https://example.com"]}

    def test_multiple_search_tool_calls_merged(self) -> None:
        """Multiple SearchTool calls have their queries merged into one call."""
        calls = [
            _make_tool_call(
                tool_name="internal_search",
                tool_args={"queries": ["query1", "query2"]},
                tool_call_id="call_1",
            ),
            _make_tool_call(
                tool_name="internal_search",
                tool_args={"queries": ["query3"]},
                tool_call_id="call_2",
            ),
        ]
        result = _merge_tool_calls(calls)

        assert len(result) == 1
        assert result[0].tool_name == "internal_search"
        assert result[0].tool_args["queries"] == ["query1", "query2", "query3"]
        # Uses first call's ID
        assert result[0].tool_call_id == "call_1"

    def test_multiple_web_search_tool_calls_merged(self) -> None:
        """Multiple WebSearchTool calls have their queries merged."""
        calls = [
            _make_tool_call(
                tool_name="web_search",
                tool_args={"queries": ["web1"]},
                tool_call_id="call_1",
            ),
            _make_tool_call(
                tool_name="web_search",
                tool_args={"queries": ["web2", "web3"]},
                tool_call_id="call_2",
            ),
        ]
        result = _merge_tool_calls(calls)

        assert len(result) == 1
        assert result[0].tool_name == "web_search"
        assert result[0].tool_args["queries"] == ["web1", "web2", "web3"]

    def test_multiple_open_url_tool_calls_merged(self) -> None:
        """Multiple OpenURLTool calls have their urls merged."""
        calls = [
            _make_tool_call(
                tool_name="open_url",
                tool_args={"urls": ["https://a.com"]},
                tool_call_id="call_1",
            ),
            _make_tool_call(
                tool_name="open_url",
                tool_args={"urls": ["https://b.com", "https://c.com"]},
                tool_call_id="call_2",
            ),
        ]
        result = _merge_tool_calls(calls)

        assert len(result) == 1
        assert result[0].tool_name == "open_url"
        assert result[0].tool_args["urls"] == [
            "https://a.com",
            "https://b.com",
            "https://c.com",
        ]

    def test_non_mergeable_tool_not_merged(self) -> None:
        """Non-mergeable tools (e.g., python) are returned as separate calls."""
        calls = [
            _make_tool_call(
                tool_name="run_python",
                tool_args={"code": "print(1)"},
                tool_call_id="call_1",
            ),
            _make_tool_call(
                tool_name="run_python",
                tool_args={"code": "print(2)"},
                tool_call_id="call_2",
            ),
        ]
        result = _merge_tool_calls(calls)

        assert len(result) == 2
        assert result[0].tool_args["code"] == "print(1)"
        assert result[1].tool_args["code"] == "print(2)"

    def test_mixed_mergeable_and_non_mergeable(self) -> None:
        """Mix of mergeable and non-mergeable tools handles correctly."""
        calls = [
            _make_tool_call(
                tool_name="internal_search",
                tool_args={"queries": ["q1"]},
                tool_call_id="search_1",
            ),
            _make_tool_call(
                tool_name="run_python",
                tool_args={"code": "x = 1"},
                tool_call_id="python_1",
            ),
            _make_tool_call(
                tool_name="internal_search",
                tool_args={"queries": ["q2"]},
                tool_call_id="search_2",
            ),
        ]
        result = _merge_tool_calls(calls)

        # Should have 2 calls: merged search + python
        assert len(result) == 2

        tool_names = {r.tool_name for r in result}
        assert tool_names == {"internal_search", "run_python"}

        search_result = next(r for r in result if r.tool_name == "internal_search")
        assert search_result.tool_args["queries"] == ["q1", "q2"]

        python_result = next(r for r in result if r.tool_name == "run_python")
        assert python_result.tool_args["code"] == "x = 1"

    def test_multiple_different_mergeable_tools(self) -> None:
        """Multiple different mergeable tools each get merged separately."""
        calls = [
            _make_tool_call(
                tool_name="internal_search",
                tool_args={"queries": ["search1"]},
            ),
            _make_tool_call(
                tool_name="web_search",
                tool_args={"queries": ["web1"]},
            ),
            _make_tool_call(
                tool_name="internal_search",
                tool_args={"queries": ["search2"]},
            ),
            _make_tool_call(
                tool_name="web_search",
                tool_args={"queries": ["web2"]},
            ),
        ]
        result = _merge_tool_calls(calls)

        # Should have 2 merged calls
        assert len(result) == 2

        search_result = next(r for r in result if r.tool_name == "internal_search")
        assert search_result.tool_args["queries"] == ["search1", "search2"]

        web_result = next(r for r in result if r.tool_name == "web_search")
        assert web_result.tool_args["queries"] == ["web1", "web2"]

    def test_preserves_first_call_placement(self) -> None:
        """Merged call uses the placement from the first call."""
        calls = [
            _make_tool_call(
                tool_name="internal_search",
                tool_args={"queries": ["q1"]},
                turn_index=1,
                tab_index=2,
            ),
            _make_tool_call(
                tool_name="internal_search",
                tool_args={"queries": ["q2"]},
                turn_index=3,
                tab_index=4,
            ),
        ]
        result = _merge_tool_calls(calls)

        assert len(result) == 1
        assert result[0].placement.turn_index == 1
        assert result[0].placement.tab_index == 2

    def test_preserves_other_args_from_first_call(self) -> None:
        """Merged call preserves non-merge-field args from the first call."""
        calls = [
            _make_tool_call(
                tool_name="internal_search",
                tool_args={"queries": ["q1"], "other_param": "value1"},
            ),
            _make_tool_call(
                tool_name="internal_search",
                tool_args={"queries": ["q2"], "other_param": "value2"},
            ),
        ]
        result = _merge_tool_calls(calls)

        assert len(result) == 1
        assert result[0].tool_args["queries"] == ["q1", "q2"]
        # Other params from first call are preserved
        assert result[0].tool_args["other_param"] == "value1"

    def test_handles_empty_queries_list(self) -> None:
        """Handles calls with empty queries lists."""
        calls = [
            _make_tool_call(
                tool_name="internal_search",
                tool_args={"queries": []},
            ),
            _make_tool_call(
                tool_name="internal_search",
                tool_args={"queries": ["q1"]},
            ),
        ]
        result = _merge_tool_calls(calls)

        assert len(result) == 1
        assert result[0].tool_args["queries"] == ["q1"]

    def test_handles_missing_merge_field(self) -> None:
        """Handles calls where the merge field is missing entirely."""
        calls = [
            _make_tool_call(
                tool_name="internal_search",
                tool_args={},  # No queries field
            ),
            _make_tool_call(
                tool_name="internal_search",
                tool_args={"queries": ["q1"]},
            ),
        ]
        result = _merge_tool_calls(calls)

        assert len(result) == 1
        assert result[0].tool_args["queries"] == ["q1"]

    def test_handles_string_value_instead_of_list(self) -> None:
        """Handles edge case where merge field is a string instead of list."""
        calls = [
            _make_tool_call(
                tool_name="internal_search",
                tool_args={"queries": "single_query"},  # String instead of list
            ),
            _make_tool_call(
                tool_name="internal_search",
                tool_args={"queries": ["q2"]},
            ),
        ]
        result = _merge_tool_calls(calls)

        assert len(result) == 1
        # String should be converted to list item
        assert result[0].tool_args["queries"] == ["single_query", "q2"]


class _FakeTool(Tool):
    """Minimal Tool for run_tool_calls tests; records nothing, always succeeds."""

    def __init__(self, name: str, tool_id: int = 1) -> None:
        super().__init__(emitter=Mock())
        self._name = name
        self._id = tool_id

    @property
    def id(self) -> int:
        return self._id

    @property
    def name(self) -> str:
        return self._name

    @property
    def description(self) -> str:
        return "fake tool"

    @property
    def display_name(self) -> str:
        return "Fake"

    def tool_definition(self) -> dict:
        return {
            "type": "function",
            "function": {
                "name": self._name,
                "parameters": {"type": "object", "properties": {}},
            },
        }

    def emit_start(self, placement: Placement) -> None:  # noqa: ARG002
        return None

    def run(
        self,
        placement: Placement,  # noqa: ARG002
        override_kwargs: Any = None,  # noqa: ARG002
        **llm_kwargs: Any,  # noqa: ARG002
    ) -> ToolResponse:
        return ToolResponse(rich_response=None, llm_facing_response="executed")


def _kickoff(tool_call_id: str, tool_name: str) -> ToolCallKickoff:
    return ToolCallKickoff(
        tool_call_id=tool_call_id,
        tool_name=tool_name,
        tool_args={"queries": [tool_call_id]},
        placement=Placement(turn_index=0, tab_index=0),
    )


def _run(tool_calls: list[ToolCallKickoff], **kwargs: Any) -> list[ToolResponse]:
    history = [
        ChatMessageSimple(message="q", token_count=1, message_type=MessageType.USER)
    ]
    tools = [_FakeTool(tool_call.tool_name) for tool_call in tool_calls]
    return run_tool_calls(
        tool_calls=tool_calls,
        tools=tools,
        message_history=history,
        user_memory_context=None,
        user_info=None,
        citation_mapping={},
        next_citation_num=1,
        **kwargs,
    ).tool_responses


class TestRunToolCallsTombstones:
    """Dropped calls get explicit tombstone responses so every tool_call_id
    the model emitted is answered exactly once in history."""

    def test_merged_call_gets_explicit_tombstone(self) -> None:
        calls = [
            _kickoff("call_1", "internal_search"),
            _kickoff("call_2", "internal_search"),
        ]
        responses = {r.tool_call.tool_call_id: r for r in _run(calls)}

        assert set(responses) == {"call_1", "call_2"}
        assert responses["call_1"].llm_facing_response == "executed"
        assert responses["call_2"].llm_facing_response == TOOL_CALL_MERGED_PROMPT
        assert responses["call_2"].rich_response is None

    def test_over_cap_calls_get_dropped_tombstone(self) -> None:
        calls = [
            _kickoff("call_1", "fake_tool_a"),
            _kickoff("call_2", "fake_tool_b"),
            _kickoff("call_3", "fake_tool_c"),
        ]
        responses = {
            r.tool_call.tool_call_id: r for r in _run(calls, max_concurrent_tools=1)
        }

        assert responses["call_1"].llm_facing_response == "executed"
        assert (
            responses["call_2"].llm_facing_response
            == TOOL_CALL_DROPPED_CONCURRENCY_PROMPT
        )
        assert (
            responses["call_3"].llm_facing_response
            == TOOL_CALL_DROPPED_CONCURRENCY_PROMPT
        )

    def test_zero_cap_tombstones_every_call(self) -> None:
        calls = [_kickoff("call_1", "fake_tool_a")]
        responses = _run(calls, max_concurrent_tools=0)

        assert len(responses) == 1
        assert responses[0].llm_facing_response == TOOL_CALL_DROPPED_CONCURRENCY_PROMPT

    def test_single_call_produces_no_tombstone(self) -> None:
        calls = [_kickoff("call_1", "fake_tool_a")]
        responses = _run(calls)

        assert len(responses) == 1
        assert responses[0].llm_facing_response == "executed"

    def test_parallel_calls_of_non_mergeable_tool_are_not_tombstoned(self) -> None:
        calls = [
            _kickoff("call_1", "fake_tool_a"),
            _kickoff("call_2", "fake_tool_b"),
        ]
        responses = {r.tool_call.tool_call_id: r for r in _run(calls)}

        assert responses["call_1"].llm_facing_response == "executed"
        assert responses["call_2"].llm_facing_response == "executed"

    def test_lost_execution_gets_lost_tombstone(self) -> None:
        """A call the threadpool layer loses (result None) still answers with
        the lost-call tombstone instead of vanishing from history."""
        tool = _FakeTool("fake_tool_a")
        history = [
            ChatMessageSimple(message="q", token_count=1, message_type=MessageType.USER)
        ]
        calls = [_kickoff("call_1", "fake_tool_a")]

        with patch(
            "onyx.tools.tool_runner.run_functions_tuples_in_parallel",
            return_value=[None],
        ):
            responses = run_tool_calls(
                tool_calls=calls,
                tools=[tool],
                message_history=history,
                user_memory_context=None,
                user_info=None,
                citation_mapping={},
                next_citation_num=1,
            ).tool_responses

        assert len(responses) == 1
        assert responses[0].llm_facing_response == TOOL_CALL_LOST_PROMPT
