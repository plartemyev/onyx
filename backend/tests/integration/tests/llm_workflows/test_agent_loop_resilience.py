"""Integration tests for agent-loop resilience in long tool turns.

A mock LLM repeats one tool call every cycle, so each cycle appends an
assistant tool-call message plus a large tool response to the in-turn tail.
With a model whose context window is small, the tail alone exceeds the
history budget mid-turn: the turn must compact and complete instead of
dying with the raw context-overflow error. A mid-turn stop must end the
stream promptly with the user-cancelled stop reason.
"""

import json
import threading
import time

from tests.integration.common_utils.constants import API_SERVER_URL
from tests.integration.common_utils.http_client import client
from tests.integration.common_utils.managers.chat import ChatSessionManager
from tests.integration.common_utils.managers.llm_provider import LLMProviderManager
from tests.integration.common_utils.managers.tool import ToolManager
from tests.integration.common_utils.test_models import DATestUser

_DUMMY_OPENAI_API_KEY = "sk-mock-llm-workflow-tests"

# ~20k characters of stdout per cycle (~5k tokens) — large enough that two
# cycles of un-compacted tail overflow a 30k-token window.
_TOOL_CODE = "print('x' * 20000)"
_MOCK_TOOL_CALL = json.dumps({"name": "run_python", "arguments": {"code": _TOOL_CODE}})


def _assert_integration_mode_enabled() -> None:
    from onyx.configs import app_configs

    assert app_configs.INTEGRATION_TESTS_MODE is True, (
        "Integration tests require INTEGRATION_TESTS_MODE=true."
    )


def _get_python_tool_id(admin_user: DATestUser) -> int:
    tool = ToolManager.get_by_in_code_id("PythonTool", admin_user)
    if tool is None or tool.id is None:
        raise AssertionError("PythonTool must exist for this test")
    return tool.id


def test_tool_turn_survives_context_overflow(
    admin_user: DATestUser,
) -> None:
    """The in-turn tail (assistant narration + tool responses) grows every
    cycle; when it alone exceeds the history budget the turn compacts it and
    completes instead of raising the turn-killing context-overflow error."""
    _assert_integration_mode_enabled()

    LLMProviderManager.create(
        user_performing_action=admin_user,
        api_key=_DUMMY_OPENAI_API_KEY,
        # Small window: system prompt + tools + user message fit, but the
        # repeated 20k-char tool responses overflow the tail after a few
        # cycles.
        max_input_tokens=30000,
    )
    chat_session = ChatSessionManager.create(user_performing_action=admin_user)
    python_tool_id = _get_python_tool_id(admin_user)

    response = ChatSessionManager.send_message(
        chat_session_id=chat_session.id,
        message="run the same code every turn",
        user_performing_action=admin_user,
        forced_tool_ids=[python_tool_id],
        mock_llm_response=_MOCK_TOOL_CALL,
    )

    assert response.error is None, f"Unexpected stream error: {response.error}"
    # The turn ran several tool cycles (proof it survived the overflow) and
    # still produced a streamed answer.
    assert len(response.tool_call_debug) >= 3
    assert response.full_message.strip(), "Expected a streamed answer"


def test_stop_mid_tool_turn_reports_user_cancelled(
    admin_user: DATestUser,
) -> None:
    """Pressing stop mid-turn ends the stream with the user-cancelled stop
    reason instead of letting the turn crawl to its own end."""
    _assert_integration_mode_enabled()

    LLMProviderManager.create(
        user_performing_action=admin_user,
        api_key=_DUMMY_OPENAI_API_KEY,
    )
    chat_session = ChatSessionManager.create(user_performing_action=admin_user)
    python_tool_id = _get_python_tool_id(admin_user)

    # Each cycle sleeps 5 seconds in the sandbox, so the turn is guaranteed
    # to still be running when stop is called.
    slow_tool_call = json.dumps(
        {
            "name": "run_python",
            "arguments": {"code": "import time\ntime.sleep(5)\nprint('x' * 1000)"},
        }
    )

    received: list[dict] = []
    done = threading.Event()

    def _consume() -> None:
        from onyx.server.query_and_chat.models import (
            AUTO_PLACE_AFTER_LATEST_MESSAGE,
            SendMessageRequest,
        )

        chat_message_req = SendMessageRequest(
            message="run the slow code every turn",
            chat_session_id=chat_session.id,
            parent_message_id=AUTO_PLACE_AFTER_LATEST_MESSAGE,
            file_descriptors=[],
            forced_tool_id=python_tool_id,
            mock_llm_response=slow_tool_call,
        )
        with client.stream(
            "POST",
            f"{API_SERVER_URL}/chat/send-chat-message",
            json=chat_message_req.model_dump(mode="json"),
            headers=admin_user.headers,
            cookies=admin_user.cookies,
        ) as response:
            received.extend(json.loads(line) for line in response.iter_lines() if line)
        done.set()

    thread = threading.Thread(target=_consume)
    thread.start()

    # Let at least one full tool cycle complete, then stop the session.
    time.sleep(15)
    stop_response = client.post(
        f"{API_SERVER_URL}/chat/stop-chat-session/{chat_session.id}",
        headers=admin_user.headers,
    )
    stop_response.raise_for_status()

    assert done.wait(timeout=120), "Stream did not end after stop"
    thread.join(timeout=10)

    stop_packets = [
        packet for packet in received if (packet.get("obj") or {}).get("type") == "stop"
    ]
    assert stop_packets, "Expected a stop packet"
    assert stop_packets[-1]["obj"].get("stop_reason") == "user_cancelled", (
        f"Unexpected stop reason: {stop_packets[-1]['obj']}"
    )

    errors = [
        packet
        for packet in received
        if (packet.get("obj") or {}).get("type") == "error" or packet.get("error")
    ]
    assert not errors, f"Unexpected stream errors: {errors}"

    # The partial turn is persisted and the session is browsable afterwards.
    history = ChatSessionManager.get_chat_history(
        chat_session=chat_session,
        user_performing_action=admin_user,
    )
    assert history, "Expected persisted messages after a stopped turn"
