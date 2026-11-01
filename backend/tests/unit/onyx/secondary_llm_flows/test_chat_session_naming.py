from typing import cast
from unittest.mock import MagicMock, patch
from uuid import uuid4

from onyx.chat.models import ChatMessageSimple
from onyx.configs.constants import MessageType
from onyx.db.models import ChatMessage
from onyx.error_handling.error_codes import OnyxErrorCode
from onyx.error_handling.exceptions import OnyxError
from onyx.secondary_llm_flows.chat_session_naming import (
    DEFAULT_CHAT_SESSION_NAME,
    generate_chat_session_name,
    get_fallback_chat_session_name,
)
from onyx.server.query_and_chat.chat_backend import (
    _generate_or_fallback_chat_session_name,
)


def _chat_message(message: str, message_type: MessageType) -> ChatMessage:
    chat_message = MagicMock()
    chat_message.message = message
    chat_message.message_type = message_type
    return cast(ChatMessage, chat_message)


def test_fallback_chat_session_name_uses_first_user_message() -> None:
    chat_history = [
        _chat_message("Assistant text", MessageType.ASSISTANT),
        _chat_message("A" * 50, MessageType.USER),
        _chat_message("Later user text", MessageType.USER),
    ]

    assert get_fallback_chat_session_name(chat_history) == f"{'A' * 40}..."


def test_fallback_chat_session_name_handles_empty_history() -> None:
    assert get_fallback_chat_session_name([]) == DEFAULT_CHAT_SESSION_NAME


def test_rate_limited_chat_naming_returns_fallback() -> None:
    first_user_message = "Explain cost-based usage metering"
    chat_history = [_chat_message(first_user_message, MessageType.USER)]
    request = MagicMock()
    user = MagicMock()
    user.id = uuid4()

    with (
        patch(
            "onyx.server.query_and_chat.chat_backend.check_token_rate_limits",
            side_effect=OnyxError(OnyxErrorCode.RATE_LIMITED),
        ),
        patch(
            "onyx.server.query_and_chat.chat_backend.get_llm_for_persona"
        ) as get_llm_for_persona,
    ):
        generated_name = _generate_or_fallback_chat_session_name(
            chat_history=chat_history,
            request=request,
            user=user,
            chat_session_id=uuid4(),
            persona=None,
            llm_override=None,
        )

    assert generated_name == first_user_message
    get_llm_for_persona.assert_not_called()


def _naming_llm(max_input_tokens: int) -> MagicMock:
    return MagicMock(
        config=MagicMock(
            max_input_tokens=max_input_tokens,
            model_name="test-model",
            model_provider="test-provider",
            deployment_name=None,
        )
    )


def test_generate_session_name_caps_history_to_model_window() -> None:
    llm = _naming_llm(8_000)
    # ~7000 tokens each: budget = 8000 - reserve -> exactly one fits.
    chat_history = [
        ChatMessageSimple(
            message=f"msg {i}",
            token_count=7_000,
            message_type=MessageType.USER,
        )
        for i in range(4)
    ]
    llm.invoke.return_value = MagicMock(
        choice=MagicMock(message=MagicMock(content=" Test Session "))
    )

    name = generate_chat_session_name(chat_history, llm)

    assert name == "Test Session"
    sent_history = llm.invoke.call_args.args[0]
    # System + reminder + at most one capped message.
    assert len(sent_history) <= 3
    # The newest message survives the cap.
    assert sent_history[-2].content == "msg 3"


def test_generate_session_name_keeps_full_history_when_it_fits() -> None:
    llm = _naming_llm(100_000)
    chat_history = [
        ChatMessageSimple(
            message=f"msg {i}",
            token_count=100,
            message_type=MessageType.USER,
        )
        for i in range(5)
    ]
    llm.invoke.return_value = MagicMock(
        choice=MagicMock(message=MagicMock(content="Test Session"))
    )

    generate_chat_session_name(chat_history, llm)

    sent_history = llm.invoke.call_args.args[0]
    # System + reminder + the full history, in order.
    assert len(sent_history) == len(chat_history) + 2
    assert sent_history[1].content == "msg 0"
    assert sent_history[-2].content == "msg 4"
