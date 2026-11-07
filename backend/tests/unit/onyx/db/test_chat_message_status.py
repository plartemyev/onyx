from onyx.configs.constants import ChatMessageStatus, MessageType
from onyx.db.chat import (
    TERMINATED_RESPONSE_PLACEHOLDER_MESSAGE,
    derive_chat_message_status,
)
from onyx.db.models import ChatMessage


def _assistant_message(
    message_id: int,
    parent_id: int | None,
    message: str,
    error: str | None = None,
) -> ChatMessage:
    return ChatMessage(
        id=message_id,
        parent_message_id=parent_id,
        message=message,
        error=error,
        message_type=MessageType.ASSISTANT,
    )


def test_non_assistant_messages_have_no_status() -> None:
    user_msg = ChatMessage(
        id=1, parent_message_id=None, message="hi", message_type=MessageType.USER
    )
    assert derive_chat_message_status(user_msg, None) is None
    assert derive_chat_message_status(user_msg, 1) is None


def test_finalized_message_is_completed() -> None:
    msg = _assistant_message(10, parent_id=9, message="the answer")
    assert derive_chat_message_status(msg, None) == ChatMessageStatus.COMPLETED
    # A finalized row that briefly still matches the live fence reads
    # COMPLETED, not PROCESSING: its text is no longer the placeholder.
    assert derive_chat_message_status(msg, 10) == ChatMessageStatus.COMPLETED


def test_placeholder_without_live_run_is_interrupted() -> None:
    msg = _assistant_message(
        10, parent_id=9, message=TERMINATED_RESPONSE_PLACEHOLDER_MESSAGE
    )
    assert derive_chat_message_status(msg, None) == ChatMessageStatus.INTERRUPTED
    # A live run that is not this message's run does not resurrect it.
    assert derive_chat_message_status(msg, 99) == ChatMessageStatus.INTERRUPTED


def test_placeholder_of_live_single_model_run_is_processing() -> None:
    # Single-model runs: the fence run id is the reserved assistant message id.
    msg = _assistant_message(
        10, parent_id=9, message=TERMINATED_RESPONSE_PLACEHOLDER_MESSAGE
    )
    assert derive_chat_message_status(msg, 10) == ChatMessageStatus.PROCESSING


def test_placeholder_of_live_multi_model_run_is_processing() -> None:
    # Multi-model runs: the fence run id is the user message id, and the
    # reserved placeholders are its children.
    msg = _assistant_message(
        11, parent_id=9, message=TERMINATED_RESPONSE_PLACEHOLDER_MESSAGE
    )
    assert derive_chat_message_status(msg, 9) == ChatMessageStatus.PROCESSING


def test_error_beats_processing() -> None:
    # The error column is written just before the fence clears; during that
    # window the message must read FAILED, not PROCESSING.
    msg = _assistant_message(
        10,
        parent_id=9,
        message=TERMINATED_RESPONSE_PLACEHOLDER_MESSAGE,
        error="Error from some-model: boom",
    )
    assert derive_chat_message_status(msg, 10) == ChatMessageStatus.FAILED
