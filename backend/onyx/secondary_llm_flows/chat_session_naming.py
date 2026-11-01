from collections.abc import Sequence

from onyx.chat.llm_step import translate_history_to_llm_format
from onyx.chat.models import ChatMessageSimple
from onyx.configs.chat_configs import SECONDARY_LLM_MAX_OUTPUT_TOKENS
from onyx.configs.constants import MessageType
from onyx.db.models import ChatMessage
from onyx.llm.interfaces import LLM
from onyx.llm.models import ReasoningEffort
from onyx.llm.utils import llm_response_to_string
from onyx.prompts.chat_prompts import CHAT_NAMING_REMINDER, CHAT_NAMING_SYSTEM_PROMPT
from onyx.tracing.flows import LLMFlow
from onyx.tracing.llm_utils import llm_generation_span, record_llm_response
from onyx.utils.logger import setup_logger

logger = setup_logger()

DEFAULT_CHAT_SESSION_NAME = "New Chat"
FALLBACK_CHAT_SESSION_NAME_LENGTH = 40

# Headroom in the naming model's window for the system prompt, the reminder,
# and the short generated name itself.
NAMING_HISTORY_RESERVE_TOKENS = 1024


def _cap_history_to_window(
    chat_history: list[ChatMessageSimple],
    llm: LLM,
) -> list[ChatMessageSimple]:
    """Keep the newest messages whose stored token counts fit the naming
    model's window. The full session can dwarf a small secondary model; the
    title only needs the recent conversation. The newest message is always
    kept."""
    budget = llm.config.max_input_tokens - NAMING_HISTORY_RESERVE_TOKENS
    if budget <= 0:
        return chat_history[-1:]

    total = 0
    cut = len(chat_history) - 1
    for idx in range(len(chat_history) - 1, -1, -1):
        total += max(chat_history[idx].token_count, 0)
        if total > budget:
            break
        cut = idx
    return chat_history[cut:]


def get_fallback_chat_session_name(chat_history: Sequence[ChatMessage]) -> str:
    user_message = next(
        (
            message.message.strip()
            for message in chat_history
            if message.message_type == MessageType.USER and message.message.strip()
        ),
        "",
    )
    if not user_message:
        return DEFAULT_CHAT_SESSION_NAME
    if len(user_message) <= FALLBACK_CHAT_SESSION_NAME_LENGTH:
        return user_message
    return user_message[:FALLBACK_CHAT_SESSION_NAME_LENGTH].rstrip() + "..."


def generate_chat_session_name(
    chat_history: list[ChatMessageSimple],
    llm: LLM,
) -> str:
    system_prompt = ChatMessageSimple(
        message=CHAT_NAMING_SYSTEM_PROMPT,
        token_count=100,
        message_type=MessageType.SYSTEM,
    )

    reminder_prompt = ChatMessageSimple(
        message=CHAT_NAMING_REMINDER,
        token_count=100,
        message_type=MessageType.USER_REMINDER,
    )

    complete_message_history = (
        [system_prompt] + _cap_history_to_window(chat_history, llm) + [reminder_prompt]
    )

    llm_facing_history = translate_history_to_llm_format(
        complete_message_history, llm.config
    )

    with llm_generation_span(
        llm=llm,
        flow=LLMFlow.CHAT_SESSION_NAMING,
        input_messages=llm_facing_history,
    ) as span_generation:
        response = llm.invoke(
            llm_facing_history,
            reasoning_effort=ReasoningEffort.OFF,
            max_tokens=SECONDARY_LLM_MAX_OUTPUT_TOKENS,
        )
        record_llm_response(span_generation, response)
        new_name_raw = llm_response_to_string(response)

    return new_name_raw.strip().strip('"')
