"""DR compaction wiring tests.

The DR orchestrator and final report build every request through
construct_message_history with token_counter + tool_response_digests +
llm (the same plumbing the regular chat loop uses). This guards the
contract those call sites depend on: when a DR-shaped history overflows,
compacted sub-agent reports keep a digest of their findings instead of
the bare compaction notice (the 48h-run amnesia failure mode).
"""

from onyx.chat.llm_loop import construct_message_history
from onyx.chat.models import ChatMessageSimple, ToolCallSimple
from onyx.configs.constants import MessageType


def _dr_orchestrator_history(
    report_chars: int, report_tokens: int, count: int
) -> list[ChatMessageSimple]:
    """The orchestrator's shape: the user query, then per-cycle
    assistant-with-tool_calls + sub-agent report responses."""
    history = [
        ChatMessageSimple(
            message="Research kernel releases.",
            token_count=10,
            message_type=MessageType.USER,
        )
    ]
    for i in range(count):
        call_id = f"call-{i}"
        history.append(
            ChatMessageSimple(
                message="",
                token_count=10,
                message_type=MessageType.ASSISTANT,
                tool_calls=[
                    ToolCallSimple(
                        tool_call_id=call_id,
                        tool_name="research_agent",
                        tool_arguments={"research_topic": f"topic {i}"},
                        token_count=10,
                    )
                ],
                image_files=None,
            )
        )
        history.append(
            ChatMessageSimple(
                message=f"Report {i}: " + "x" * report_chars,
                token_count=report_tokens,
                message_type=MessageType.TOOL_CALL_RESPONSE,
                tool_call_id=call_id,
                image_files=None,
            )
        )
    return history


def test_overflowing_dr_history_stubs_reports_with_digests() -> None:
    """DR passes token_counter + a digest cache; stubs must carry the
    extractive digest (tier 1), not the bare notice."""
    history = _dr_orchestrator_history(report_chars=12000, report_tokens=12000, count=6)
    digests: dict[str, str] = {}
    token_counter = len

    total = sum(m.token_count for m in history)
    built = construct_message_history(
        system_prompt=ChatMessageSimple(
            message="orchestrator", token_count=10, message_type=MessageType.SYSTEM
        ),
        custom_agent_prompt=None,
        simple_chat_history=history,
        reminder_message=None,
        context_files=None,
        # Fits ~2 raw exchanges, leaves room for 2 digest stubs, must drop
        # the rest — exercising the stub tier, not just drops.
        available_tokens=total // 3,
        token_counter=token_counter,
        tool_response_digests=digests,
        llm=None,
        available_tool_names=set(),
    )

    by_id = {
        m.tool_call_id: m
        for m in built
        if m.message_type == MessageType.TOOL_CALL_RESPONSE and m.tool_call_id
    }
    # call-5 stays raw; call-4..call-1 stub with digests; call-0 drops.
    assert len(by_id) == 5
    raw_ids = {cid for cid, m in by_id.items() if len(m.message) > 6000}
    stubbed_ids = set(by_id) - raw_ids
    assert raw_ids == {"call-5"}
    assert stubbed_ids == {"call-4", "call-3", "call-2", "call-1"}
    assert digests, "stubbed reports must derive digests into the cache"
    # The stubbing pass derives digests before the drop decision, so the
    # dropped call-0 may appear here too; every kept stub must have one.
    assert stubbed_ids <= set(digests)
    for call_id in stubbed_ids:
        stub = by_id[call_id]
        assert len(stub.message) < 12000, "stub must shrink the report"
        assert digests[call_id][:80] in stub.message, (
            f"digest for {call_id} missing from its stub"
        )


def test_fitting_dr_history_replays_verbatim() -> None:
    """No compaction when everything fits; the input messages are kept."""
    history = _dr_orchestrator_history(report_chars=50, report_tokens=20, count=2)
    total = sum(m.token_count for m in history)
    digests: dict[str, str] = {}
    built = construct_message_history(
        system_prompt=ChatMessageSimple(
            message="orchestrator", token_count=10, message_type=MessageType.SYSTEM
        ),
        custom_agent_prompt=None,
        simple_chat_history=history,
        reminder_message=None,
        context_files=None,
        available_tokens=total + 100,
        token_counter=len,
        tool_response_digests=digests,
        llm=None,
        available_tool_names=set(),
    )
    assert not digests
    responses = [
        m.message for m in built if m.message_type == MessageType.TOOL_CALL_RESPONSE
    ]
    assert any("Report 0:" in r for r in responses)
    assert any("Report 1:" in r for r in responses)
