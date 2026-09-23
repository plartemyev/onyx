"""Tests for llm_loop.py, including history construction and empty-response paths."""

from contextlib import nullcontext
from typing import Any
from unittest.mock import Mock, patch

import pytest

from onyx.chat.llm_loop import (
    _COMPACTED_TOOL_RESPONSE_TOKEN_FALLBACK,
    _REFUSAL_FINISH_REASONS,
    ContextWindowExceededError,
    EmptyLLMResponseError,
    _build_empty_llm_response_error,
    _cycle_history_token_budget,
    _try_fallback_tool_extraction,
    construct_message_history,
    count_message_replay_tokens,
    run_llm_loop,
    select_reminder_text,
)
from onyx.chat.models import (
    ChatLoadedFile,
    ChatMessageSimple,
    ContextFileMetadata,
    ExtractedContextFiles,
    FileToolMetadata,
    LlmStepResult,
    ToolCallSimple,
)
from onyx.configs.constants import MessageType
from onyx.file_store.models import ChatFileType
from onyx.llm.interfaces import LLMConfig, ToolChoiceOptions
from onyx.prompts.chat_prompts import (
    IMAGE_GEN_REMINDER,
    OPEN_URL_REMINDER,
    TOOL_CALL_RESPONSE_COMPACTED,
)
from onyx.prompts.tool_prompts import TOOL_CALL_MERGED_PROMPT
from onyx.server.query_and_chat.placement import Placement
from onyx.tools.constants import FILE_READER_TOOL_NAME
from onyx.tools.interface import Tool
from onyx.tools.models import ToolCallKickoff, ToolResponse
from onyx.tools.tool_implementations.search.search_tool import SearchTool


def create_message(
    content: str, message_type: MessageType, token_count: int | None = None
) -> ChatMessageSimple:
    """Helper to create a ChatMessageSimple for testing."""
    if token_count is None:
        # Simple token estimation: ~1 token per 4 characters
        token_count = max(1, len(content) // 4)
    return ChatMessageSimple(
        message=content,
        token_count=token_count,
        message_type=message_type,
    )


def create_assistant_with_tool_call(
    tool_call_id: str, tool_name: str, token_count: int
) -> ChatMessageSimple:
    """Helper to create an ASSISTANT message with tool_calls for testing."""
    tool_call = ToolCallSimple(
        tool_call_id=tool_call_id,
        tool_name=tool_name,
        tool_arguments={},
        token_count=token_count,
    )
    return ChatMessageSimple(
        message="",
        token_count=token_count,
        message_type=MessageType.ASSISTANT,
        tool_calls=[tool_call],
    )


def create_tool_response(
    tool_call_id: str, content: str, token_count: int
) -> ChatMessageSimple:
    """Helper to create a TOOL_CALL_RESPONSE message for testing."""
    return ChatMessageSimple(
        message=content,
        token_count=token_count,
        message_type=MessageType.TOOL_CALL_RESPONSE,
        tool_call_id=tool_call_id,
    )


def create_context_files(
    num_files: int = 0, num_images: int = 0, tokens_per_file: int = 100
) -> ExtractedContextFiles:
    """Helper to create ExtractedContextFiles for testing."""
    file_texts = [f"Project file {i} content" for i in range(num_files)]
    file_metadata = [
        ContextFileMetadata(
            file_id=f"file_{i}",
            filename=f"file_{i}.txt",
            file_content=f"Project file {i} content",
        )
        for i in range(num_files)
    ]
    image_files = [
        ChatLoadedFile(
            file_id=f"image_{i}",
            content=b"",
            file_type=ChatFileType.IMAGE,
            filename=f"image_{i}.png",
            content_text=None,
            token_count=50,
        )
        for i in range(num_images)
    ]
    return ExtractedContextFiles(
        file_texts=file_texts,
        image_files=image_files,
        use_as_search_filter=False,
        total_token_count=num_files * tokens_per_file,
        file_metadata=file_metadata,
        uncapped_token_count=num_files * tokens_per_file,
    )


class TestConstructMessageHistory:
    """Tests for the construct_message_history function."""

    def test_basic_no_truncation(self) -> None:
        """Test basic functionality when all messages fit within token budget."""
        system_prompt = create_message(
            "You are a helpful assistant", MessageType.SYSTEM, 10
        )
        user_msg1 = create_message("Hello", MessageType.USER, 5)
        assistant_msg1 = create_message("Hi there!", MessageType.ASSISTANT, 5)
        user_msg2 = create_message("How are you?", MessageType.USER, 5)

        simple_chat_history = [user_msg1, assistant_msg1, user_msg2]
        context_files = create_context_files()

        result = construct_message_history(
            system_prompt=system_prompt,
            custom_agent_prompt=None,
            simple_chat_history=simple_chat_history,
            reminder_message=None,
            context_files=context_files,
            available_tokens=1000,
        )

        # Should have: system, user1, assistant1, user2
        assert len(result) == 4
        assert result[0] == system_prompt
        assert result[1] == user_msg1
        assert result[2] == assistant_msg1
        assert result[3] == user_msg2

    def test_with_custom_agent_prompt(self) -> None:
        """Test that custom agent prompt is inserted before the last user message."""
        system_prompt = create_message("System", MessageType.SYSTEM, 10)
        user_msg1 = create_message("First message", MessageType.USER, 5)
        assistant_msg1 = create_message("Response", MessageType.ASSISTANT, 5)
        user_msg2 = create_message("Second message", MessageType.USER, 5)
        custom_agent = create_message("Custom instructions", MessageType.USER, 10)

        simple_chat_history = [user_msg1, assistant_msg1, user_msg2]
        context_files = create_context_files()

        result = construct_message_history(
            system_prompt=system_prompt,
            custom_agent_prompt=custom_agent,
            simple_chat_history=simple_chat_history,
            reminder_message=None,
            context_files=context_files,
            available_tokens=1000,
        )

        # Should have: system, user1, assistant1, custom_agent, user2
        assert len(result) == 5
        assert result[0] == system_prompt
        assert result[1] == user_msg1
        assert result[2] == assistant_msg1
        assert result[3] == custom_agent  # Before last user message
        assert result[4] == user_msg2

    def test_with_context_files(self) -> None:
        """Test that project files are inserted before the last user message."""
        system_prompt = create_message("System", MessageType.SYSTEM, 10)
        user_msg1 = create_message("First message", MessageType.USER, 5)
        user_msg2 = create_message("Second message", MessageType.USER, 5)

        simple_chat_history = [user_msg1, user_msg2]
        context_files = create_context_files(num_files=2, tokens_per_file=50)

        result = construct_message_history(
            system_prompt=system_prompt,
            custom_agent_prompt=None,
            simple_chat_history=simple_chat_history,
            reminder_message=None,
            context_files=context_files,
            available_tokens=1000,
        )

        # Should have: system, user1, context_files_message, user2
        assert len(result) == 4
        assert result[0] == system_prompt
        assert result[1] == user_msg1
        assert (
            result[2].message_type == MessageType.USER
        )  # Project files as user message
        assert "documents" in result[2].message  # Should contain JSON structure
        assert result[3] == user_msg2

    def test_with_reminder_message(self) -> None:
        """Test that reminder message is added at the very end."""
        system_prompt = create_message("System", MessageType.SYSTEM, 10)
        user_msg = create_message("Hello", MessageType.USER, 5)
        reminder = create_message("Remember to cite sources", MessageType.USER, 10)

        simple_chat_history = [user_msg]
        context_files = create_context_files()

        result = construct_message_history(
            system_prompt=system_prompt,
            custom_agent_prompt=None,
            simple_chat_history=simple_chat_history,
            reminder_message=reminder,
            context_files=context_files,
            available_tokens=1000,
        )

        # Should have: system, user, reminder
        assert len(result) == 3
        assert result[0] == system_prompt
        assert result[1] == user_msg
        assert result[2] == reminder  # At the end

    def test_tool_calls_after_last_user_message(self) -> None:
        """Test that tool calls and responses after last user message are preserved."""
        system_prompt = create_message("System", MessageType.SYSTEM, 10)
        user_msg1 = create_message("First message", MessageType.USER, 5)
        assistant_msg1 = create_message("Response", MessageType.ASSISTANT, 5)
        user_msg2 = create_message("Search for X", MessageType.USER, 5)
        assistant_with_tool = create_assistant_with_tool_call("tc_1", "search", 5)
        tool_response = create_tool_response("tc_1", "Search results...", 10)

        simple_chat_history = [
            user_msg1,
            assistant_msg1,
            user_msg2,
            assistant_with_tool,
            tool_response,
        ]
        context_files = create_context_files()

        result = construct_message_history(
            system_prompt=system_prompt,
            custom_agent_prompt=None,
            simple_chat_history=simple_chat_history,
            reminder_message=None,
            context_files=context_files,
            available_tokens=1000,
        )

        # Should have: system, user1, assistant1, user2, assistant_with_tool, tool_response
        assert len(result) == 6
        assert result[0] == system_prompt
        assert result[1] == user_msg1
        assert result[2] == assistant_msg1
        assert result[3] == user_msg2
        assert result[4] == assistant_with_tool
        assert result[5] == tool_response

    def test_custom_agent_and_project_before_last_user_with_tools_after(self) -> None:
        """Test correct ordering with custom agent, project files, and tool calls."""
        system_prompt = create_message("System", MessageType.SYSTEM, 10)
        user_msg1 = create_message("First", MessageType.USER, 5)
        user_msg2 = create_message("Second", MessageType.USER, 5)
        assistant_with_tool = create_assistant_with_tool_call("tc_1", "tool", 5)
        custom_agent = create_message("Custom", MessageType.USER, 10)

        simple_chat_history = [user_msg1, user_msg2, assistant_with_tool]
        context_files = create_context_files(num_files=1, tokens_per_file=50)

        result = construct_message_history(
            system_prompt=system_prompt,
            custom_agent_prompt=custom_agent,
            simple_chat_history=simple_chat_history,
            reminder_message=None,
            context_files=context_files,
            available_tokens=1000,
        )

        # Should have: system, user1, custom_agent, context_files, user2, assistant_with_tool
        assert len(result) == 6
        assert result[0] == system_prompt
        assert result[1] == user_msg1
        assert result[2] == custom_agent  # Before last user message
        assert result[3].message_type == MessageType.USER  # Project files
        assert "documents" in result[3].message
        assert result[4] == user_msg2  # Last user message
        assert result[5] == assistant_with_tool  # After last user message

    def test_construct_message_history_does_not_duplicate_project_images(
        self,
    ) -> None:
        """Project images are attached upstream in convert_chat_history; this
        function must not re-attach them. Simulates the realistic state where
        the last user message in simple_chat_history already carries the
        project images, and asserts they appear exactly once."""
        system_prompt = create_message("System", MessageType.SYSTEM, 10)

        project_image = ChatLoadedFile(
            file_id="project_image",
            content=b"",
            file_type=ChatFileType.IMAGE,
            filename="project.png",
            content_text=None,
            token_count=50,
        )
        # Simulate convert_chat_history's output: the last user message already
        # has the project image attached.
        user_msg = ChatMessageSimple(
            message="What is in this image?",
            token_count=5,
            message_type=MessageType.USER,
            image_files=[project_image],
        )

        simple_chat_history = [user_msg]
        context_files = ExtractedContextFiles(
            file_texts=[],
            image_files=[project_image],
            use_as_search_filter=False,
            total_token_count=0,
            file_metadata=[],
            uncapped_token_count=0,
        )

        result = construct_message_history(
            system_prompt=system_prompt,
            custom_agent_prompt=None,
            simple_chat_history=simple_chat_history,
            reminder_message=None,
            context_files=context_files,
            available_tokens=1000,
        )

        last_message = result[-1]
        assert last_message.message == "What is in this image?"
        assert last_message.image_files is not None
        assert len(last_message.image_files) == 1
        assert last_message.image_files[0].file_id == "project_image"

    def test_truncation_from_top(self) -> None:
        """Test that history is truncated from the top when token budget is exceeded."""
        system_prompt = create_message("System", MessageType.SYSTEM, 10)
        user_msg1 = create_message("First", MessageType.USER, 20)
        assistant_msg1 = create_message("Response 1", MessageType.ASSISTANT, 20)
        user_msg2 = create_message("Second", MessageType.USER, 20)
        assistant_msg2 = create_message("Response 2", MessageType.ASSISTANT, 20)
        user_msg3 = create_message("Third", MessageType.USER, 20)

        simple_chat_history = [
            user_msg1,
            assistant_msg1,
            user_msg2,
            assistant_msg2,
            user_msg3,
        ]
        context_files = create_context_files()

        # Budget only allows last 3 messages + system (10 + 20 + 20 + 20 = 70 tokens)
        result = construct_message_history(
            system_prompt=system_prompt,
            custom_agent_prompt=None,
            simple_chat_history=simple_chat_history,
            reminder_message=None,
            context_files=context_files,
            available_tokens=80,
        )

        # Should have: system, user2, assistant2, user3
        # user1 and assistant1 should be truncated
        assert len(result) == 4
        assert result[0] == system_prompt
        assert result[1] == user_msg2  # user1 truncated
        assert result[2] == assistant_msg2
        assert result[3] == user_msg3

    def test_truncation_preserves_last_user_and_messages_after(self) -> None:
        """Test that truncation preserves the last user message and everything after it."""
        system_prompt = create_message("System", MessageType.SYSTEM, 10)
        user_msg1 = create_message("First", MessageType.USER, 30)
        user_msg2 = create_message("Second", MessageType.USER, 20)
        assistant_with_tool = create_assistant_with_tool_call("tc_1", "tool", 20)
        tool_response = create_tool_response("tc_1", "tool_response", 20)

        simple_chat_history = [user_msg1, user_msg2, assistant_with_tool, tool_response]
        context_files = create_context_files()

        # Budget only allows last user message and messages after + system
        # (10 + 20 + 20 + 20 = 70 tokens)
        result = construct_message_history(
            system_prompt=system_prompt,
            custom_agent_prompt=None,
            simple_chat_history=simple_chat_history,
            reminder_message=None,
            context_files=context_files,
            available_tokens=80,
        )

        # Should have: system, user2, assistant_with_tool, tool_response
        # user1 should be truncated, but user2 and everything after preserved
        assert len(result) == 4
        assert result[0] == system_prompt
        assert result[1] == user_msg2  # user1 truncated
        assert result[2] == assistant_with_tool
        assert result[3] == tool_response

    def test_truncation_drops_orphaned_tool_response(self) -> None:
        """If truncation drops an assistant tool call, its orphaned tool response is removed."""
        system_prompt = create_message("System", MessageType.SYSTEM, 10)
        user_msg1 = create_message("First", MessageType.USER, 10)
        assistant_with_tool = create_assistant_with_tool_call("tc_1", "tool", 25)
        tool_response = create_tool_response("tc_1", "tool_response", 5)
        assistant_msg1 = create_message("Used the tool above", MessageType.ASSISTANT, 5)
        user_msg2 = create_message("Latest question", MessageType.USER, 10)

        simple_chat_history = [
            user_msg1,
            assistant_with_tool,
            tool_response,
            assistant_msg1,
            user_msg2,
        ]
        context_files = create_context_files()

        # Remaining history budget is 10 tokens (30 total - 10 system - 10 last user):
        # keeps [tool_response, assistant_msg1] from history_before_last_user,
        # but drops assistant_with_tool, making tool_response orphaned.
        result = construct_message_history(
            system_prompt=system_prompt,
            custom_agent_prompt=None,
            simple_chat_history=simple_chat_history,
            reminder_message=None,
            context_files=context_files,
            available_tokens=30,
        )

        # Orphaned tool response should be removed from final history.
        assert len(result) == 3
        assert result[0] == system_prompt
        assert result[1] == assistant_msg1
        assert result[2] == user_msg2

    def test_preserves_non_orphaned_tool_response(self) -> None:
        """Tool responses remain when their assistant tool call is present."""
        system_prompt = create_message("System", MessageType.SYSTEM, 10)
        user_msg1 = create_message("First", MessageType.USER, 10)
        assistant_with_tool = create_assistant_with_tool_call("tc_1", "tool", 20)
        tool_response = create_tool_response("tc_1", "tool_response", 5)
        user_msg2 = create_message("Latest question", MessageType.USER, 10)

        simple_chat_history = [user_msg1, assistant_with_tool, tool_response, user_msg2]
        context_files = create_context_files()

        # Remaining history budget is 25 tokens (45 total - 10 system - 10 last user):
        # keeps both assistant_with_tool and tool_response in history_before_last_user.
        result = construct_message_history(
            system_prompt=system_prompt,
            custom_agent_prompt=None,
            simple_chat_history=simple_chat_history,
            reminder_message=None,
            context_files=context_files,
            available_tokens=45,
        )

        assert len(result) == 4
        assert result[0] == system_prompt
        assert result[1] == assistant_with_tool
        assert result[2] == tool_response
        assert result[3] == user_msg2

    def test_empty_history(self) -> None:
        """Test handling of empty chat history."""
        system_prompt = create_message("System", MessageType.SYSTEM, 10)
        custom_agent = create_message("Custom", MessageType.USER, 10)
        reminder = create_message("Reminder", MessageType.USER, 10)

        simple_chat_history: list[ChatMessageSimple] = []
        context_files = create_context_files(num_files=1, tokens_per_file=50)

        result = construct_message_history(
            system_prompt=system_prompt,
            custom_agent_prompt=custom_agent,
            simple_chat_history=simple_chat_history,
            reminder_message=reminder,
            context_files=context_files,
            available_tokens=1000,
        )

        # Should have: system, custom_agent, context_files, reminder
        assert len(result) == 4
        assert result[0] == system_prompt
        assert result[1] == custom_agent
        assert result[2].message_type == MessageType.USER  # Project files
        assert result[3] == reminder

    def test_no_user_message_raises_error(self) -> None:
        """Test that an error is raised when there's no user message in history."""
        system_prompt = create_message("System", MessageType.SYSTEM, 10)
        assistant_msg = create_message("Response", MessageType.ASSISTANT, 5)
        assistant_with_tool = create_assistant_with_tool_call("tc_1", "tool", 5)

        simple_chat_history = [assistant_msg, assistant_with_tool]
        context_files = create_context_files()

        with pytest.raises(ValueError, match="No user message found"):
            construct_message_history(
                system_prompt=system_prompt,
                custom_agent_prompt=None,
                simple_chat_history=simple_chat_history,
                reminder_message=None,
                context_files=context_files,
                available_tokens=1000,
            )

    def test_not_enough_tokens_for_required_elements(self) -> None:
        """Test error when there aren't enough tokens for required elements."""
        system_prompt = create_message("System", MessageType.SYSTEM, 50)
        user_msg = create_message("Message", MessageType.USER, 50)
        custom_agent = create_message("Custom", MessageType.USER, 50)

        simple_chat_history = [user_msg]
        context_files = create_context_files(num_files=1, tokens_per_file=100)

        # Total required: 50 (system) + 50 (custom) + 100 (project) + 50 (user) = 250
        # But only 200 available
        with pytest.raises(ContextWindowExceededError):
            construct_message_history(
                system_prompt=system_prompt,
                custom_agent_prompt=custom_agent,
                simple_chat_history=simple_chat_history,
                reminder_message=None,
                context_files=context_files,
                available_tokens=200,
            )

    def test_last_user_message_too_large_raises_context_error(self) -> None:
        """The last user message alone cannot fit: unrecoverable, raises the
        classified context-window error (compaction has nothing to trade)."""
        system_prompt = create_message("System", MessageType.SYSTEM, 10)
        user_msg = create_message("Very long user message", MessageType.USER, 100)
        assistant_with_tool = create_assistant_with_tool_call("tc_1", "tool", 5)
        tool_response = create_tool_response("tc_1", "Results", 5)

        simple_chat_history = [user_msg, assistant_with_tool, tool_response]
        context_files = create_context_files()

        # Budget after system prompt: 40; the user message alone is 100.
        with pytest.raises(ContextWindowExceededError) as exc_info:
            construct_message_history(
                system_prompt=system_prompt,
                custom_agent_prompt=None,
                simple_chat_history=simple_chat_history,
                reminder_message=None,
                context_files=context_files,
                available_tokens=50,
            )
        assert exc_info.value.error_code == "CONTEXT_WINDOW_EXCEEDED"
        assert exc_info.value.is_retryable is False
        # The user-facing message must not leak raw token math.
        assert "context" in exc_info.value.client_error_msg.lower()

    def test_tail_overflow_stubs_oldest_tool_response(self) -> None:
        """A tail that alone exceeds the budget is compacted, not fatal: the
        oldest tool response is stubbed and the turn completes."""
        system_prompt = create_message("System", MessageType.SYSTEM, 10)
        user_msg = create_message("User question", MessageType.USER, 10)
        assistant_with_tool = create_assistant_with_tool_call("tc_1", "tool", 5)
        big_tool_response = create_tool_response("tc_1", "Huge result", 60)

        simple_chat_history = [user_msg, assistant_with_tool, big_tool_response]
        context_files = create_context_files()

        # History budget: 60 - 10 (system) = 50; after the 10-token user
        # message, 40 remain. Tail = 5 + 60 = 65 does not fit, so the
        # response is stubbed to ~34 tokens (5 + 34 = 39 <= 40).
        result = construct_message_history(
            system_prompt=system_prompt,
            custom_agent_prompt=None,
            simple_chat_history=simple_chat_history,
            reminder_message=None,
            context_files=context_files,
            available_tokens=60,
        )

        assert len(result) == 4
        assert result[0] == system_prompt
        assert result[1] == user_msg
        # The tool-call carrier stays (arguments are information rich)...
        assert result[2] == assistant_with_tool
        # ...and the response content is replaced by the compaction notice.
        assert result[3].message == TOOL_CALL_RESPONSE_COMPACTED
        assert result[3].tool_call_id == "tc_1"
        assert result[3].token_count == _COMPACTED_TOOL_RESPONSE_TOKEN_FALLBACK

    def test_tail_overflow_stub_does_not_mutate_shared_messages(self) -> None:
        """Stubbing must copy: the same message objects feed later cycles."""
        system_prompt = create_message("System", MessageType.SYSTEM, 10)
        user_msg = create_message("User question", MessageType.USER, 10)
        assistant_with_tool = create_assistant_with_tool_call("tc_1", "tool", 5)
        big_tool_response = create_tool_response("tc_1", "Huge result", 60)

        simple_chat_history = [user_msg, assistant_with_tool, big_tool_response]
        construct_message_history(
            system_prompt=system_prompt,
            custom_agent_prompt=None,
            simple_chat_history=simple_chat_history,
            reminder_message=None,
            context_files=create_context_files(),
            available_tokens=60,
        )

        assert simple_chat_history[2].token_count == 60
        assert simple_chat_history[2].message_type == MessageType.TOOL_CALL_RESPONSE
        assert simple_chat_history[2].tool_call_id == "tc_1"

    def test_tail_overflow_drops_oldest_exchange_when_stubbing_not_enough(
        self,
    ) -> None:
        """When stubbed responses still do not fit, whole exchanges are
        dropped oldest-first and the newest ones are kept."""
        system_prompt = create_message("System", MessageType.SYSTEM, 10)
        user_msg = create_message("User question", MessageType.USER, 10)
        assistant_1 = create_assistant_with_tool_call("tc_1", "tool", 5)
        response_1 = create_tool_response("tc_1", "Result 1", 20)
        assistant_2 = create_assistant_with_tool_call("tc_2", "tool", 5)
        response_2 = create_tool_response("tc_2", "Result 2", 20)

        simple_chat_history = [
            user_msg,
            assistant_1,
            response_1,
            assistant_2,
            response_2,
        ]
        context_files = create_context_files()

        # History budget: 50; after the user message, 40 remain. Stubbed
        # tail = 2 * (5 + 34) = 78 > 40, so the oldest exchange is dropped
        # entirely; the newest (39) is kept.
        result = construct_message_history(
            system_prompt=system_prompt,
            custom_agent_prompt=None,
            simple_chat_history=simple_chat_history,
            reminder_message=None,
            context_files=context_files,
            available_tokens=60,
        )

        assert len(result) == 4
        assert result[1] == user_msg
        assert result[2] == assistant_2
        assert result[3].message == TOOL_CALL_RESPONSE_COMPACTED
        assert result[3].tool_call_id == "tc_2"
        # The oldest exchange is gone entirely.
        assert all(msg is not assistant_1 for msg in result)
        assert all(msg is not response_1 for msg in result)

    def test_tail_within_budget_is_not_compacted(self) -> None:
        """No compaction pressure: tool responses replay verbatim (existing
        in-turn behavior is unchanged)."""
        system_prompt = create_message("System", MessageType.SYSTEM, 10)
        user_msg = create_message("User question", MessageType.USER, 10)
        assistant_with_tool = create_assistant_with_tool_call("tc_1", "tool", 5)
        tool_response = create_tool_response("tc_1", "Result", 15)

        simple_chat_history = [user_msg, assistant_with_tool, tool_response]
        result = construct_message_history(
            system_prompt=system_prompt,
            custom_agent_prompt=None,
            simple_chat_history=simple_chat_history,
            reminder_message=None,
            context_files=create_context_files(),
            available_tokens=100,
        )

        assert result[-2] == assistant_with_tool
        assert result[-1] == tool_response
        assert tool_response.message == "Result"
        assert tool_response.token_count == 15

    def test_not_enough_tokens_for_last_user_and_messages_after(self) -> None:
        """A tail that exceeds the budget no longer kills the turn: it is
        compacted so the request can still be built."""
        system_prompt = create_message("System", MessageType.SYSTEM, 10)
        user_msg1 = create_message("First", MessageType.USER, 10)
        user_msg2 = create_message("Second", MessageType.USER, 30)
        assistant_with_tool = create_assistant_with_tool_call("tc_1", "tool", 30)

        simple_chat_history = [user_msg1, user_msg2, assistant_with_tool]
        context_files = create_context_files()

        # Budget: 50 tokens → 40 after the system prompt. Old behavior raised
        # here (30 user + 30 assistant > 40); now the turn survives. The tail
        # (the 30-token assistant carrier) does not fit the 10 tokens left
        # after user_msg2 and has no response content to stub, so stage 2
        # drops it and the model answers from the user message alone.
        result = construct_message_history(
            system_prompt=system_prompt,
            custom_agent_prompt=None,
            simple_chat_history=simple_chat_history,
            reminder_message=None,
            context_files=context_files,
            available_tokens=50,
        )

        assert len(result) == 3
        assert result[0] == system_prompt
        assert result[1] == user_msg1
        assert result[2] == user_msg2

    def test_complex_scenario_all_elements(self) -> None:
        """Test a complex scenario with all elements combined."""
        system_prompt = create_message("System", MessageType.SYSTEM, 10)
        user_msg1 = create_message("First", MessageType.USER, 10)
        assistant_msg1 = create_message("Response 1", MessageType.ASSISTANT, 10)
        user_msg2 = create_message("Second", MessageType.USER, 10)
        assistant_msg2 = create_message("Response 2", MessageType.ASSISTANT, 10)
        user_msg3 = create_message("Third", MessageType.USER, 10)
        assistant_with_tool = create_assistant_with_tool_call("tc_1", "search", 10)
        tool_response = create_tool_response("tc_1", "Results", 10)
        custom_agent = create_message("Custom instructions", MessageType.USER, 15)
        reminder = create_message("Cite sources", MessageType.USER, 10)

        simple_chat_history = [
            user_msg1,
            assistant_msg1,
            user_msg2,
            assistant_msg2,
            user_msg3,
            assistant_with_tool,
            tool_response,
        ]
        context_files = create_context_files(num_files=2, tokens_per_file=20)

        result = construct_message_history(
            system_prompt=system_prompt,
            custom_agent_prompt=custom_agent,
            simple_chat_history=simple_chat_history,
            reminder_message=reminder,
            context_files=context_files,
            available_tokens=1000,
        )

        # Expected order:
        # system, user1, assistant1, user2, assistant2,
        # custom_agent, context_files, user3, assistant_with_tool, tool_response, reminder
        assert len(result) == 11
        assert result[0] == system_prompt
        assert result[1] == user_msg1
        assert result[2] == assistant_msg1
        assert result[3] == user_msg2
        assert result[4] == assistant_msg2
        assert result[5] == custom_agent  # Before last user
        assert (
            result[6].message_type == MessageType.USER
        )  # Project files before last user
        assert "documents" in result[6].message
        assert result[7] == user_msg3  # Last user message
        assert result[8] == assistant_with_tool  # After last user
        assert result[9] == tool_response  # After last user
        assert result[10] == reminder  # At the very end

    def test_context_files_json_format(self) -> None:
        """Test that project files are formatted correctly as JSON."""
        system_prompt = create_message("System", MessageType.SYSTEM, 10)
        user_msg = create_message("Hello", MessageType.USER, 5)

        simple_chat_history = [user_msg]
        context_files = create_context_files(num_files=2, tokens_per_file=50)

        result = construct_message_history(
            system_prompt=system_prompt,
            custom_agent_prompt=None,
            simple_chat_history=simple_chat_history,
            reminder_message=None,
            context_files=context_files,
            available_tokens=1000,
        )

        # Find the project files message
        project_message = result[1]  # Should be between system and user

        # Verify it's formatted as JSON
        assert "Here are some documents provided for context" in project_message.message
        assert '"documents"' in project_message.message
        assert '"document": 1' in project_message.message
        assert '"document": 2' in project_message.message
        assert '"contents"' in project_message.message
        assert "Project file 0 content" in project_message.message
        assert "Project file 1 content" in project_message.message

    def test_file_metadata_for_tool_produces_message(self) -> None:
        """When context_files has file_metadata_for_tool, a metadata listing
        message should be injected into the history."""
        system_prompt = create_message("System", MessageType.SYSTEM, 10)
        user_msg = create_message("Analyze the spreadsheet", MessageType.USER, 5)

        context_files = ExtractedContextFiles(
            file_texts=[],
            image_files=[],
            use_as_search_filter=False,
            total_token_count=0,
            file_metadata=[],
            uncapped_token_count=0,
            file_metadata_for_tool=[
                FileToolMetadata(
                    file_id="xlsx-1",
                    filename="report.xlsx",
                    approx_char_count=100000,
                ),
            ],
        )

        result = construct_message_history(
            system_prompt=system_prompt,
            custom_agent_prompt=None,
            simple_chat_history=[user_msg],
            reminder_message=None,
            context_files=context_files,
            available_tokens=1000,
            token_counter=_simple_token_counter,
            available_tool_names={"read_file"},
        )

        # Should have: system, tool_metadata_message, user
        assert len(result) == 3
        metadata_msg = result[1]
        assert metadata_msg.message_type == MessageType.USER
        assert "report.xlsx" in metadata_msg.message
        # read_file is offered, so the listing carries the id it consumes.
        assert "xlsx-1" in metadata_msg.message

    def test_metadata_only_and_text_files_both_present(self) -> None:
        """When both text content and tool metadata are present, both messages
        should appear in the history."""
        system_prompt = create_message("System", MessageType.SYSTEM, 10)
        user_msg = create_message("Summarize everything", MessageType.USER, 5)

        context_files = ExtractedContextFiles(
            file_texts=["Text file content here"],
            image_files=[],
            use_as_search_filter=False,
            total_token_count=100,
            file_metadata=[
                ContextFileMetadata(
                    file_id="txt-1",
                    filename="notes.txt",
                    file_content="Text file content here",
                ),
            ],
            uncapped_token_count=100,
            file_metadata_for_tool=[
                FileToolMetadata(
                    file_id="xlsx-1",
                    filename="data.xlsx",
                    approx_char_count=50000,
                ),
            ],
        )

        result = construct_message_history(
            system_prompt=system_prompt,
            custom_agent_prompt=None,
            simple_chat_history=[user_msg],
            reminder_message=None,
            context_files=context_files,
            available_tokens=2000,
            token_counter=_simple_token_counter,
        )

        # Should have: system, context_files_message, tool_metadata_message, user
        assert len(result) == 4
        # Context files message (text content)
        assert "documents" in result[1].message
        assert "Text file content here" in result[1].message
        # Tool metadata message
        assert "data.xlsx" in result[2].message
        assert result[3] == user_msg


def _simple_token_counter(text: str) -> int:
    """Approximate token counter for tests (~4 chars per token)."""
    return max(1, len(text) // 4)


def _make_file_metadata(
    file_id: str,
    filename: str,
    approx_chars: int = 50_000,
    staged_for_tools: bool = True,
) -> FileToolMetadata:
    return FileToolMetadata(
        file_id=file_id,
        filename=filename,
        approx_char_count=approx_chars,
        staged_for_tools=staged_for_tools,
    )


class TestNonVisionImageBudgeting:
    """When a non-vision model replays history images as text markers, the
    truncation budget must charge the marker cost, not the stored image token
    cost — otherwise history that actually fits gets evicted."""

    @staticmethod
    def _image_user_msg() -> ChatMessageSimple:
        image = ChatLoadedFile(
            file_id="img0",
            content=b"",
            file_type=ChatFileType.IMAGE,
            filename="img0.png",
            content_text=None,
            token_count=500,
        )
        return ChatMessageSimple(
            message="look at this",
            token_count=505,
            message_type=MessageType.USER,
            image_files=[image],
            image_token_count=500,
        )

    def _construct(self, replay_as_markers: bool) -> list[ChatMessageSimple]:
        simple_chat_history = [
            self._image_user_msg(),
            create_message("Response", MessageType.ASSISTANT, 5),
            create_message("Follow-up", MessageType.USER, 5),
        ]
        return construct_message_history(
            system_prompt=None,
            custom_agent_prompt=None,
            simple_chat_history=simple_chat_history,
            reminder_message=None,
            context_files=create_context_files(),
            available_tokens=100,
            token_counter=lambda _: 10,
            image_files_replayed_as_markers=replay_as_markers,
        )

    def test_full_image_cost_evicts_the_image_message(self) -> None:
        result = self._construct(replay_as_markers=False)
        assert [m.message for m in result] == ["Response", "Follow-up"]

    def test_marker_cost_keeps_the_image_message(self) -> None:
        result = self._construct(replay_as_markers=True)
        assert [m.message for m in result] == [
            "look at this",
            "Response",
            "Follow-up",
        ]

    @pytest.mark.parametrize("stored_image_tokens", [0, 20000])
    @pytest.mark.parametrize("configured_input_limit", [8000, 24000])
    def test_output_allowance_uses_image_replay_cost(
        self,
        stored_image_tokens: int,
        configured_input_limit: int,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        image_msg = self._image_user_msg()
        image_msg.token_count = stored_image_tokens + 5
        image_msg.image_token_count = stored_image_tokens
        monkeypatch.setattr(
            "onyx.chat.token_budget.GEN_AI_INPUT_TOKEN_SAFETY_MARGIN", 0.05
        )
        llm = Mock()
        llm.config = LLMConfig(
            model_provider="openai",
            model_name="text-only-model",
            temperature=0,
            max_input_tokens=configured_input_limit,
        )
        older_user = create_message("Old input", MessageType.USER, 20000)
        older_answer = create_message("Old answer", MessageType.ASSISTANT, 5)
        with (
            patch("onyx.chat.llm_loop.trace", return_value=nullcontext()),
            patch("onyx.llm.litellm_singleton.config.initialize_litellm"),
            patch(
                "onyx.chat.llm_loop.get_session_with_current_tenant",
                return_value=nullcontext(),
            ),
            patch("onyx.chat.llm_loop.get_default_base_system_prompt", return_value=""),
            patch("onyx.chat.llm_loop.select_reminder_text", return_value=""),
            patch("onyx.chat.llm_loop.model_supports_image_input", return_value=False),
            patch(
                "onyx.chat.token_budget.get_model_map",
                return_value={
                    "openai/text-only-model": {
                        "max_input_tokens": 24000,
                        "max_output_tokens": 16000,
                    }
                },
            ),
            patch(
                "onyx.chat.llm_loop.run_llm_step",
                return_value=(
                    LlmStepResult(answer="Done", tool_calls=None, reasoning=None),
                    False,
                ),
            ) as step,
        ):
            run_llm_loop(
                emitter=Mock(),
                state_container=Mock(),
                simple_chat_history=[older_user, older_answer, image_msg],
                tools=[],
                custom_agent_prompt=None,
                context_files=create_context_files(),
                persona=None,
                user_memory_context=None,
                llm=llm,
                token_counter=lambda _: 10,
            )

        if configured_input_limit == 8000:
            assert step.call_args.kwargs["history"] == [older_answer, image_msg]
            assert step.call_args.kwargs["max_tokens"] == 16000
        else:
            assert step.call_args.kwargs["history"] == [
                older_user,
                older_answer,
                image_msg,
            ]
            assert step.call_args.kwargs["max_tokens"] == 2780
        assert (
            count_message_replay_tokens(
                image_msg,
                image_files_replayed_as_markers=True,
                token_counter=lambda _: 10,
            )
            == 15
        )
        assert image_msg.token_count == stored_image_tokens + 5

    def test_vision_output_budget_keeps_stored_image_cost(self) -> None:
        assert count_message_replay_tokens(self._image_user_msg()) == 505

    def test_image_marker_budget_without_tokenizer(self) -> None:
        assert (
            count_message_replay_tokens(
                self._image_user_msg(), image_files_replayed_as_markers=True
            )
            == 45
        )


class TestForgottenFileMetadata:
    """Tests for the forgotten-files mechanism in construct_message_history.

    These cover the scenario where a user attaches a large file to a chat
    message. On the first turn the file content message is in the context
    window. On subsequent turns, it may be truncated by either:
      a) context-window budget limits, or
      b) summary-based truncation removing the message before
         convert_chat_history ever runs — leaving an "orphaned" metadata
         entry with no corresponding file_id-tagged ChatMessageSimple.

    The forgotten-files mechanism must detect both cases and inject a
    lightweight metadata message pointing the LLM at whichever retrieval path
    the deployment actually offers (read_file or internal search).

    This class covers a request that was given the FileReaderTool.
    TestForgottenFilesWithoutFileReader covers one that was not.
    """

    def _build(
        self,
        simple_chat_history: list[ChatMessageSimple],
        available_tokens: int = 10_000,
        all_injected_file_metadata: dict[str, FileToolMetadata] | None = None,
    ) -> list[ChatMessageSimple]:
        """Shorthand wrapper around construct_message_history."""
        return construct_message_history(
            system_prompt=create_message("system", MessageType.SYSTEM, 5),
            custom_agent_prompt=None,
            simple_chat_history=simple_chat_history,
            reminder_message=None,
            context_files=create_context_files(),
            available_tokens=available_tokens,
            token_counter=_simple_token_counter,
            all_injected_file_metadata=all_injected_file_metadata,
            available_tool_names={FILE_READER_TOOL_NAME},
        )

    @staticmethod
    def _find_forgotten_message(
        result: list[ChatMessageSimple],
    ) -> ChatMessageSimple | None:
        """Find the forgotten-files metadata message in the result, if any.

        Matches the file listing rather than the header: the header names
        read_file or internal search depending on the deployment.
        """
        for msg in result:
            if 'filename="' in msg.message:
                return msg
        return None

    # ------------------------------------------------------------------
    # Case 1: file message is still in context — no forgotten-files needed
    # ------------------------------------------------------------------

    def test_file_message_present_no_forgotten_metadata(self) -> None:
        """When the file message fits in context, no forgotten-file message
        should be injected.
        """
        file_meta = _make_file_metadata("file-abc", "moby_dick.txt")
        file_msg = create_message("Contents of moby dick...", MessageType.USER, 50)
        file_msg.file_id = "file-abc"

        history = [
            file_msg,
            create_message("Summarize this", MessageType.ASSISTANT, 20),
            create_message("What's chapter 1?", MessageType.USER, 10),
        ]
        result = self._build(
            history,
            available_tokens=10_000,
            all_injected_file_metadata={"file-abc": file_meta},
        )

        forgotten = self._find_forgotten_message(result)
        assert forgotten is None, (
            "Should not inject forgotten-files when file is in context"
        )
        # The file message itself should still be present
        assert any(m.file_id == "file-abc" for m in result)

    # ------------------------------------------------------------------
    # Case 2: file message dropped by context-window truncation
    # ------------------------------------------------------------------

    def test_file_message_dropped_by_truncation_gets_forgotten_metadata(self) -> None:
        """When the context budget is too tight and the file message gets
        truncated, a forgotten-files metadata message must appear.
        """
        file_meta = _make_file_metadata("file-abc", "moby_dick.txt")
        file_msg = create_message("x" * 2000, MessageType.USER, 500)
        file_msg.file_id = "file-abc"

        history = [
            file_msg,
            create_message("Got it", MessageType.ASSISTANT, 10),
            create_message("Tell me about ch1", MessageType.USER, 10),
        ]

        # Budget is just enough for the system prompt + last messages but
        # NOT the 500-token file message.
        result = self._build(
            history,
            available_tokens=100,
            all_injected_file_metadata={"file-abc": file_meta},
        )

        forgotten = self._find_forgotten_message(result)
        assert forgotten is not None, "Forgotten-files message should be injected"
        assert "moby_dick.txt" in forgotten.message
        assert "file-abc" in forgotten.message

        # The original file message should NOT be in context
        assert not any(
            getattr(m, "file_id", None) == "file-abc"  # ods: ignore[getattr]
            and m.message_type == MessageType.USER
            for m in result
            if m is not forgotten
        )

    # ------------------------------------------------------------------
    # Case 3: file message removed by summary truncation ("orphaned" metadata)
    # ------------------------------------------------------------------

    def test_orphaned_metadata_triggers_forgotten_files(self) -> None:
        """Simulates the scenario where summary truncation in process_message
        removed the file's original message BEFORE convert_chat_history ran,
        so no ChatMessageSimple has the file_id tag. The metadata is still
        passed via all_injected_file_metadata and must be treated as dropped.
        """
        file_meta = _make_file_metadata("file-abc", "moby_dick.txt")

        # History has no file_id-tagged message — it was already removed by
        # summary truncation. Only later conversation remains.
        history = [
            create_message("Summary of earlier convo", MessageType.ASSISTANT, 20),
            create_message("Now tell me about chapter 2", MessageType.USER, 10),
        ]

        result = self._build(
            history,
            available_tokens=10_000,
            all_injected_file_metadata={"file-abc": file_meta},
        )

        forgotten = self._find_forgotten_message(result)
        assert forgotten is not None, (
            "Orphaned file metadata should trigger forgotten-files message"
        )
        assert "moby_dick.txt" in forgotten.message
        assert "file-abc" in forgotten.message

    # ------------------------------------------------------------------
    # Case 4: multiple files — one survives, one is dropped
    # ------------------------------------------------------------------

    def test_mixed_files_only_dropped_ones_appear_in_forgotten(self) -> None:
        """When two files exist but only one's message is truncated, only the
        truncated file should appear in the forgotten-files metadata.
        """
        meta_a = _make_file_metadata("file-a", "big_file.txt")
        meta_b = _make_file_metadata("file-b", "small_file.txt")

        # file-a has a huge message that will be dropped, file-b fits
        file_msg_a = create_message("x" * 2000, MessageType.USER, 500)
        file_msg_a.file_id = "file-a"
        file_msg_b = create_message("small content", MessageType.USER, 5)
        file_msg_b.file_id = "file-b"

        history = [
            file_msg_a,
            create_message("ok", MessageType.ASSISTANT, 3),
            file_msg_b,
            create_message("ok", MessageType.ASSISTANT, 3),
            create_message("Compare the two files", MessageType.USER, 10),
        ]

        # Tight budget: system(5) + last-user(10) = 15 min. Give ~50 so
        # file_msg_b(5)+assistant(3)+assistant(3) fit but file_msg_a(500) won't.
        result = self._build(
            history,
            available_tokens=80,
            all_injected_file_metadata={"file-a": meta_a, "file-b": meta_b},
        )

        forgotten = self._find_forgotten_message(result)
        assert forgotten is not None
        assert "big_file.txt" in forgotten.message
        assert "file-a" in forgotten.message
        # file-b should NOT be in the forgotten message — it's still in context
        assert "small_file.txt" not in forgotten.message

    # ------------------------------------------------------------------
    # Case 5: no metadata dict → no forgotten-files message even if dropped
    # ------------------------------------------------------------------

    def test_no_metadata_dict_means_no_forgotten_message(self) -> None:
        """If all_injected_file_metadata is None (FileReaderTool not enabled),
        no forgotten-files message should be emitted even if file messages
        are dropped by truncation.
        """
        file_msg = create_message("x" * 2000, MessageType.USER, 500)
        file_msg.file_id = "file-abc"

        history = [
            file_msg,
            create_message("Got it", MessageType.ASSISTANT, 10),
            create_message("Tell me more", MessageType.USER, 10),
        ]

        result = self._build(
            history,
            available_tokens=100,
            all_injected_file_metadata=None,
        )

        forgotten = self._find_forgotten_message(result)
        assert forgotten is None, (
            "No forgotten-files message when metadata dict is None"
        )

    # ------------------------------------------------------------------
    # Case 6: orphaned metadata with multiple files, all summarized away
    # ------------------------------------------------------------------

    def test_multiple_orphaned_files_all_appear_in_forgotten(self) -> None:
        """All files from summarized-away messages should be listed in the
        forgotten-files message.
        """
        meta_a = _make_file_metadata("file-a", "report.pdf")
        meta_b = _make_file_metadata("file-b", "data.csv")

        # Both original messages were removed by summary truncation;
        # only post-summary messages remain.
        history = [
            create_message("Earlier discussion summarized", MessageType.ASSISTANT, 15),
            create_message("What patterns do you see?", MessageType.USER, 10),
        ]

        result = self._build(
            history,
            available_tokens=10_000,
            all_injected_file_metadata={"file-a": meta_a, "file-b": meta_b},
        )

        forgotten = self._find_forgotten_message(result)
        assert forgotten is not None
        assert "report.pdf" in forgotten.message
        assert "data.csv" in forgotten.message

    # ------------------------------------------------------------------
    # Case 7: file metadata persists across many turns after truncation
    # ------------------------------------------------------------------

    def test_forgotten_metadata_persists_across_many_turns(self) -> None:
        """Simulates the real bug: after the file's original message is
        summarized away, every subsequent turn should still include the
        forgotten-files metadata — not just the first turn after truncation.
        """
        file_meta = _make_file_metadata("file-abc", "moby_dick.txt")

        # Build several turns AFTER the file was already summarized away.
        # Each turn, construct_message_history is called fresh with the
        # same all_injected_file_metadata.
        for turn in range(5):
            messages = [
                create_message("Summary", MessageType.ASSISTANT, 15),
            ]
            # Add some back-and-forth after the summary
            for i in range(turn):
                messages.append(create_message(f"Question {i}", MessageType.USER, 5))
                messages.append(create_message(f"Answer {i}", MessageType.ASSISTANT, 5))
            messages.append(
                create_message(f"Latest question (turn {turn})", MessageType.USER, 5)
            )

            result = self._build(
                messages,
                available_tokens=10_000,
                all_injected_file_metadata={"file-abc": file_meta},
            )

            forgotten = self._find_forgotten_message(result)
            assert forgotten is not None, (
                f"Turn {turn}: forgotten-files message must persist every turn"
            )
            assert "moby_dick.txt" in forgotten.message


def _notice_for_dropped_file(
    available_tool_names: set[str] | None = None,
    staged_for_tools: bool = True,
) -> ChatMessageSimple:
    """Truncate one oversized attachment out of context and return the notice."""
    file_meta = _make_file_metadata(
        "file-abc", "sustainability.pdf", staged_for_tools=staged_for_tools
    )
    file_msg = create_message("x" * 2000, MessageType.USER, 500)
    file_msg.file_id = "file-abc"

    result = construct_message_history(
        system_prompt=create_message("system", MessageType.SYSTEM, 5),
        custom_agent_prompt=None,
        simple_chat_history=[
            file_msg,
            create_message("Got it", MessageType.ASSISTANT, 10),
            create_message("Summarize it", MessageType.USER, 10),
        ],
        reminder_message=None,
        context_files=create_context_files(),
        # Too tight for the 500-token file message.
        available_tokens=100,
        token_counter=_simple_token_counter,
        all_injected_file_metadata={"file-abc": file_meta},
        available_tool_names=available_tool_names,
    )
    notice = next((m for m in result if 'filename="' in m.message), None)
    assert notice is not None, "dropped file should still produce a notice"
    return notice


class TestForgottenFilesWithoutFileReader:
    """The forgotten-files notice must not name read_file where the tool is absent.

    FileReaderTool is only attached when the vector DB is disabled (see
    ``FileReaderTool.is_available``), but the notice was emitted whenever the
    persona had the tool row attached. On a vector-DB deployment that told the
    model to call a tool it had never been given, so it reported read_file as
    unavailable and fell back to guessing or web-searching the document.
    """

    def _build_with_dropped_file(
        self, available_tool_names: set[str] | None = None
    ) -> ChatMessageSimple:
        return _notice_for_dropped_file(available_tool_names or {SearchTool.NAME})

    def test_notice_does_not_name_read_file(self) -> None:
        notice = self._build_with_dropped_file()
        assert "read_file" not in notice.message

    def test_notice_points_at_internal_search(self) -> None:
        notice = self._build_with_dropped_file()
        assert "internal search" in notice.message
        assert "sustainability.pdf" in notice.message

    def test_notice_forbids_guessing_and_web_search(self) -> None:
        """The failure this replaced was the model web-searching the document."""
        notice = self._build_with_dropped_file()
        assert "Do not guess" in notice.message
        assert "search the web" in notice.message

    def test_notice_omits_the_file_id(self) -> None:
        """The file_id only means something to read_file; internal search takes
        a query, so showing the UUID invites another dead end.
        """
        notice = self._build_with_dropped_file()
        assert "file-abc" not in notice.message


class TestForgottenFilesNoticeFollowsConstructedTools:
    """The notice names a tool only when this request actually received it.

    Deployment config alone is not enough: internal search can be missing on a
    vector-DB deployment when the persona omits it, ``allowed_tool_ids``
    excludes it, or the search usage setting disables it.
    """

    def test_names_read_file_when_the_request_has_it(self) -> None:
        notice = _notice_for_dropped_file({"read_file", "internal_search"})
        assert "read_file" in notice.message
        # read_file is the one consumer of the UUID, so it comes back with it.
        assert "file-abc" in notice.message

    def test_names_internal_search_when_only_search_is_offered(self) -> None:
        notice = _notice_for_dropped_file({"internal_search"})
        assert "internal search" in notice.message
        assert "read_file" not in notice.message

    def test_names_python_when_it_is_the_only_reader(self) -> None:
        """The python tool is handed the files themselves, so an evicted file is
        still readable there. Calling it unreadable makes the model refuse work
        it could actually do.
        """
        notice = _notice_for_dropped_file({"run_python"})
        assert "python tool" in notice.message
        assert "no tool here can read them" not in notice.message
        assert "sustainability.pdf" in notice.message

    def test_python_tier_omits_the_file_id(self) -> None:
        """The UUID is a read_file identifier; PythonTool never sees it."""
        notice = _notice_for_dropped_file({"run_python"})
        assert "file-abc" not in notice.message

    def test_python_tier_does_not_promise_an_exact_path(self) -> None:
        """PythonTool normalizes and de-duplicates names at staging time, so the
        notice cannot know the sandbox path. It must not assert one.
        """
        notice = _notice_for_dropped_file({"run_python"})
        assert "by filename" not in notice.message
        assert "listing the working directory" in notice.message

    def test_python_tier_skipped_for_summary_truncated_files(self) -> None:
        """Summary truncation filters the message out of chat_history before
        load_all_chat_files runs, so those bytes never reach the python tool.
        Advertising python for them points the model at nothing.
        """
        notice = _notice_for_dropped_file({"run_python"}, staged_for_tools=False)
        assert "python tool" not in notice.message
        assert "no tool here can read them" in notice.message
        assert "sustainability.pdf" in notice.message

    def test_search_wins_over_python_when_both_are_offered(self) -> None:
        """Indexed retrieval beats writing code to parse an oversized file."""
        notice = _notice_for_dropped_file({"internal_search", "run_python"})
        assert "internal search" in notice.message
        assert "python tool" not in notice.message

    def test_python_only_persona_reaches_the_python_tier(self) -> None:
        """Regression for a real deployment: read_file is attached to the
        persona but filtered out by availability, internal_search was never
        attached, and run_python is live.
        """
        notice = _notice_for_dropped_file(
            {"generate_image", "web_search", "run_python", "open_url"}
        )
        assert "python tool" in notice.message
        assert "read_file" not in notice.message
        assert "internal search" not in notice.message

    def test_names_no_tool_when_the_request_has_no_reader(self) -> None:
        """No read_file, no search, no python — do not promise anything."""
        notice = _notice_for_dropped_file({"generate_image", "web_search"})
        assert "read_file" not in notice.message
        assert "internal search" not in notice.message
        assert "python tool" not in notice.message
        assert "no tool here can read them" in notice.message

    def test_still_forbids_guessing_when_no_tool_is_offered(self) -> None:
        notice = _notice_for_dropped_file({"generate_image", "web_search"})
        assert "Do not guess" in notice.message
        assert "search the web" in notice.message
        assert "sustainability.pdf" in notice.message

    def test_python_tier_also_forbids_guessing_and_web_search(self) -> None:
        notice = _notice_for_dropped_file({"run_python"})
        assert "Do not guess" in notice.message
        assert "search the web" in notice.message


class TestFallbackToolExtraction:
    def _tool_defs(self) -> list[dict]:
        return [
            {
                "type": "function",
                "function": {
                    "name": "internal_search",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "queries": {
                                "type": "array",
                                "items": {"type": "string"},
                            }
                        },
                        "required": ["queries"],
                    },
                },
            }
        ]

    def test_extracts_text_format_tool_call_under_auto(self) -> None:
        """Imitated flattened-history lines are parsed and executed.

        The model may echo the former `[Tool Call]` history format as content
        under tool_choice=AUTO; that must trigger extraction like XML does.
        """
        llm_step_result = LlmStepResult(
            reasoning=None,
            answer=None,
            raw_answer=(
                "Let me search for that.\n"
                "[Tool Call] name=internal_search id=9a8b7c6d-5e4f-3a2b-1c0d-9e8f7a6b5c4d"
                ' args={"queries": ["alpha"]}\n'
                "[Tool Result] id=9a8b7c6d-5e4f-3a2b-1c0d-9e8f7a6b5c4d\n"
                "some fabricated result"
            ),
            tool_calls=None,
        )

        result, attempted = _try_fallback_tool_extraction(
            llm_step_result=llm_step_result,
            tool_choice=ToolChoiceOptions.AUTO,
            tool_defs=self._tool_defs(),
            turn_index=1,
        )

        assert attempted is True
        assert result.tool_calls is not None
        assert len(result.tool_calls) == 1
        assert result.tool_calls[0].tool_name == "internal_search"
        assert result.tool_calls[0].tool_args == {"queries": ["alpha"]}
        assert result.tool_calls[0].tool_call_id.startswith("extracted_")

    def test_does_not_trigger_when_native_tool_calls_exist(self) -> None:
        """Imitation text alongside native calls is left as prose."""
        llm_step_result = LlmStepResult(
            reasoning=None,
            answer=(
                '[Tool Call] name=internal_search id=fake args={"queries": ["alpha"]}'
            ),
            tool_calls=[
                ToolCallKickoff(
                    tool_call_id="native-1",
                    tool_name="internal_search",
                    tool_args={"queries": ["native"]},
                    placement=Placement(turn_index=0),
                )
            ],
        )

        result, attempted = _try_fallback_tool_extraction(
            llm_step_result=llm_step_result,
            tool_choice=ToolChoiceOptions.AUTO,
            tool_defs=self._tool_defs(),
            turn_index=0,
        )

        assert attempted is False
        assert result is llm_step_result

    def test_extracts_from_answer_when_required_and_no_tool_calls(self) -> None:
        llm_step_result = LlmStepResult(
            reasoning=None,
            answer='{"name":"internal_search","arguments":{"queries":["alpha"]}}',
            tool_calls=None,
        )

        result, attempted = _try_fallback_tool_extraction(
            llm_step_result=llm_step_result,
            tool_choice=ToolChoiceOptions.REQUIRED,
            tool_defs=self._tool_defs(),
            turn_index=3,
        )

        assert attempted is True
        assert result.tool_calls is not None
        assert len(result.tool_calls) == 1
        assert result.tool_calls[0].tool_name == "internal_search"
        assert result.tool_calls[0].tool_args == {"queries": ["alpha"]}
        assert result.tool_calls[0].placement == Placement(turn_index=3)

    def test_falls_back_to_reasoning_when_answer_has_no_tool_calls(self) -> None:
        llm_step_result = LlmStepResult(
            reasoning='{"name":"internal_search","arguments":{"queries":["beta"]}}',
            answer="I should search first.",
            tool_calls=None,
        )

        result, attempted = _try_fallback_tool_extraction(
            llm_step_result=llm_step_result,
            tool_choice=ToolChoiceOptions.REQUIRED,
            tool_defs=self._tool_defs(),
            turn_index=5,
        )

        assert attempted is True
        assert result.tool_calls is not None
        assert len(result.tool_calls) == 1
        assert result.tool_calls[0].tool_name == "internal_search"
        assert result.tool_calls[0].tool_args == {"queries": ["beta"]}
        assert result.tool_calls[0].placement == Placement(turn_index=5)

    def test_extracts_xml_style_invoke_from_answer_when_required(self) -> None:
        llm_step_result = LlmStepResult(
            reasoning=None,
            answer=(
                '<function_calls><invoke name="internal_search">'
                '<parameter name="queries" string="false">'
                '["Onyx documentation", "Onyx docs", "Onyx platform"]'
                "</parameter></invoke></function_calls>"
            ),
            tool_calls=None,
        )

        result, attempted = _try_fallback_tool_extraction(
            llm_step_result=llm_step_result,
            tool_choice=ToolChoiceOptions.REQUIRED,
            tool_defs=self._tool_defs(),
            turn_index=7,
        )

        assert attempted is True
        assert result.tool_calls is not None
        assert len(result.tool_calls) == 1
        assert result.tool_calls[0].tool_name == "internal_search"
        assert result.tool_calls[0].tool_args == {
            "queries": ["Onyx documentation", "Onyx docs", "Onyx platform"]
        }
        assert result.tool_calls[0].placement == Placement(turn_index=7)

    def test_extracts_xml_style_invoke_from_answer_when_auto(self) -> None:
        llm_step_result = LlmStepResult(
            reasoning=None,
            # Runtime-faithful shape: filtered answer is empty, raw answer has XML payload.
            answer=None,
            raw_answer=(
                '<function_calls><invoke name="internal_search">'
                '<parameter name="queries" string="false">'
                '["Onyx documentation", "Onyx docs", "Onyx internal docs"]'
                "</parameter></invoke></function_calls>"
            ),
            tool_calls=None,
        )

        result, attempted = _try_fallback_tool_extraction(
            llm_step_result=llm_step_result,
            tool_choice=ToolChoiceOptions.AUTO,
            tool_defs=self._tool_defs(),
            turn_index=9,
        )

        assert attempted is True
        assert result.tool_calls is not None
        assert len(result.tool_calls) == 1
        assert result.tool_calls[0].tool_name == "internal_search"
        assert result.tool_calls[0].tool_args == {
            "queries": ["Onyx documentation", "Onyx docs", "Onyx internal docs"]
        }
        assert result.tool_calls[0].placement == Placement(turn_index=9)

    def test_extracts_from_raw_answer_when_filtered_answer_has_no_xml(self) -> None:
        llm_step_result = LlmStepResult(
            reasoning=None,
            answer="",
            raw_answer=(
                '<function_calls><invoke name="internal_search">'
                '<parameter name="queries" string="false">'
                '["Onyx documentation", "Onyx docs"]'
                "</parameter></invoke></function_calls>"
            ),
            tool_calls=None,
        )

        result, attempted = _try_fallback_tool_extraction(
            llm_step_result=llm_step_result,
            tool_choice=ToolChoiceOptions.AUTO,
            tool_defs=self._tool_defs(),
            turn_index=10,
        )

        assert attempted is True
        assert result.tool_calls is not None
        assert len(result.tool_calls) == 1
        assert result.tool_calls[0].tool_name == "internal_search"
        assert result.tool_calls[0].tool_args == {
            "queries": ["Onyx documentation", "Onyx docs"]
        }
        assert result.tool_calls[0].placement == Placement(turn_index=10)

    def test_does_not_attempt_fallback_for_auto_without_tool_call_hints(self) -> None:
        llm_step_result = LlmStepResult(
            reasoning=None,
            answer="Here is a normal answer with no tool call payload.",
            tool_calls=None,
        )

        result, attempted = _try_fallback_tool_extraction(
            llm_step_result=llm_step_result,
            tool_choice=ToolChoiceOptions.AUTO,
            tool_defs=self._tool_defs(),
            turn_index=2,
        )

        assert result is llm_step_result
        assert attempted is False

    def test_returns_unchanged_when_required_but_nothing_extractable(self) -> None:
        llm_step_result = LlmStepResult(
            reasoning="Need more info.",
            answer="Let me think.",
            tool_calls=None,
        )

        result, attempted = _try_fallback_tool_extraction(
            llm_step_result=llm_step_result,
            tool_choice=ToolChoiceOptions.REQUIRED,
            tool_defs=self._tool_defs(),
            turn_index=1,
        )

        assert result is llm_step_result
        assert attempted is True
        assert result.tool_calls is None

    def test_noop_when_tool_calls_already_present(self) -> None:
        existing_call = ToolCallKickoff(
            tool_call_id="call_existing",
            tool_name="internal_search",
            tool_args={"queries": ["already-set"]},
            placement=Placement(turn_index=0),
        )
        llm_step_result = LlmStepResult(
            reasoning=None,
            answer='{"name":"internal_search","arguments":{"queries":["alpha"]}}',
            tool_calls=[existing_call],
        )

        result, attempted = _try_fallback_tool_extraction(
            llm_step_result=llm_step_result,
            tool_choice=ToolChoiceOptions.REQUIRED,
            tool_defs=self._tool_defs(),
            turn_index=0,
        )

        assert result is llm_step_result
        assert attempted is False


class TestEmptyLlmResponseClassification:
    def _make_llm(self, provider: str = "openai", model: str = "gpt-5.2") -> Mock:
        llm = Mock()
        llm.config = LLMConfig(
            model_provider=provider,
            model_name=model,
            temperature=0.0,
            max_input_tokens=4096,
        )
        return llm

    def test_openai_empty_stream_is_classified_as_budget_exceeded(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr("onyx.chat.llm_loop.is_true_openai_model", lambda *_: True)

        err = _build_empty_llm_response_error(
            llm=self._make_llm(),
            llm_step_result=LlmStepResult(
                reasoning=None,
                answer=None,
                tool_calls=None,
                raw_answer=None,
            ),
            tool_choice=ToolChoiceOptions.AUTO,
        )

        assert isinstance(err, EmptyLLMResponseError)
        assert err.error_code == "BUDGET_EXCEEDED"
        assert err.is_retryable is False
        assert "quota" in err.client_error_msg.lower()

    def test_reasoning_only_response_uses_generic_empty_response_error(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr("onyx.chat.llm_loop.is_true_openai_model", lambda *_: True)

        err = _build_empty_llm_response_error(
            llm=self._make_llm(),
            llm_step_result=LlmStepResult(
                reasoning="scratchpad only",
                answer=None,
                tool_calls=None,
                raw_answer=None,
            ),
            tool_choice=ToolChoiceOptions.AUTO,
        )

        assert isinstance(err, EmptyLLMResponseError)
        assert err.error_code == "EMPTY_LLM_RESPONSE"
        assert err.is_retryable is True
        assert "quota" not in err.client_error_msg.lower()

    def test_generic_empty_response_message_includes_finish_reason(self) -> None:
        """Mode-A triage: a reasoning-only `length` stop must be distinguishable
        from a silent stream cut via the surfaced message."""
        err = _build_empty_llm_response_error(
            llm=self._make_llm(provider="ollama_chat", model="ornith-1.5:9b"),
            llm_step_result=LlmStepResult(
                reasoning="scratchpad only",
                answer=None,
                tool_calls=None,
                raw_answer=None,
                finish_reason="length",
            ),
            tool_choice=ToolChoiceOptions.AUTO,
        )

        assert "finish_reason=length" in err.client_error_msg
        assert err.finish_reason == "length"

    def test_generic_empty_response_message_handles_missing_finish_reason(
        self,
    ) -> None:
        err = _build_empty_llm_response_error(
            llm=self._make_llm(provider="ollama_chat", model="ornith-1.5:9b"),
            llm_step_result=LlmStepResult(
                reasoning="scratchpad only",
                answer=None,
                tool_calls=None,
                raw_answer=None,
            ),
            tool_choice=ToolChoiceOptions.AUTO,
        )

        assert "finish_reason=unknown" in err.client_error_msg

    def test_refusal_finish_reason_is_classified_as_model_refusal(self) -> None:
        """Anthropic refusal: HTTP 200, stop_reason="refusal" (normalized by
        LiteLLM to "content_filter"), no text or tool calls. Must surface as a
        refusal, not a generic empty-stream error."""
        err = _build_empty_llm_response_error(
            llm=self._make_llm(provider="anthropic", model="claude-fable-5"),
            llm_step_result=LlmStepResult(
                reasoning=None,
                answer=None,
                tool_calls=None,
                raw_answer=None,
                finish_reason="content_filter",
            ),
            tool_choice=ToolChoiceOptions.AUTO,
        )

        assert isinstance(err, EmptyLLMResponseError)
        assert err.error_code == "MODEL_REFUSAL"
        assert err.is_retryable is False
        assert err.finish_reason == "content_filter"
        assert "declined" in err.client_error_msg.lower()
        # Anthropic-specific fallback suggestion from the issue.
        assert "Claude Opus 4.8" in err.client_error_msg

    @pytest.mark.parametrize("finish_reason", sorted(_REFUSAL_FINISH_REASONS))
    def test_refusal_finish_reasons_take_precedence_over_budget_heuristic(
        self, monkeypatch: pytest.MonkeyPatch, finish_reason: str
    ) -> None:
        """Native provider refusal reasons may pass through gateways unchanged."""
        monkeypatch.setattr("onyx.chat.llm_loop.is_true_openai_model", lambda *_: True)

        err = _build_empty_llm_response_error(
            llm=self._make_llm(),
            llm_step_result=LlmStepResult(
                reasoning=None,
                answer=None,
                tool_calls=None,
                raw_answer=None,
                finish_reason=finish_reason,
            ),
            tool_choice=ToolChoiceOptions.AUTO,
        )

        assert err.error_code == "MODEL_REFUSAL"
        assert err.is_retryable is False
        assert err.finish_reason == finish_reason
        assert "Claude Opus 4.8" not in err.client_error_msg


class TestSelectReminderText:
    """The open_url nudge must be suppressed when the open_url tool is disabled,
    otherwise the model is told to call a tool it doesn't have (confusing
    "open_url is not available" replies)."""

    def _select(self, **overrides: Any) -> str | None:
        kwargs: dict[str, Any] = {
            "ran_image_gen": False,
            "just_ran_web_search": False,
            "has_open_url_tool": True,
            "out_of_cycles": False,
            "persona_task_prompt": None,
            "include_citation_reminder": False,
            "include_file_reminder": False,
        }
        kwargs.update(overrides)
        return select_reminder_text(**kwargs)

    def test_open_url_reminder_when_tool_available(self) -> None:
        result = self._select(just_ran_web_search=True, has_open_url_tool=True)
        assert result == OPEN_URL_REMINDER

    def test_no_open_url_reminder_when_tool_disabled(self) -> None:
        """Web search ran but open_url is disabled -> fall back, never nudge open_url."""
        result = self._select(just_ran_web_search=True, has_open_url_tool=False)
        assert result != OPEN_URL_REMINDER
        assert result is None  # nothing else to remind about in this scenario

    def test_open_url_reminder_suppressed_on_last_cycle(self) -> None:
        result = self._select(
            just_ran_web_search=True, has_open_url_tool=True, out_of_cycles=True
        )
        assert result != OPEN_URL_REMINDER

    def test_image_gen_reminder_takes_precedence(self) -> None:
        result = self._select(
            ran_image_gen=True, just_ran_web_search=True, has_open_url_tool=True
        )
        assert result == IMAGE_GEN_REMINDER


class _FakeLoopTool(Tool):
    """Minimal Tool for run_llm_loop tests: always succeeds, never searches."""

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
                "parameters": {
                    "type": "object",
                    "properties": {"queries": {"type": "array"}},
                },
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
        return ToolResponse(rich_response=None, llm_facing_response="tool result")


def _kickoff(tool_call_id: str, tool_name: str) -> ToolCallKickoff:
    return ToolCallKickoff(
        tool_call_id=tool_call_id,
        tool_name=tool_name,
        tool_args={"queries": [tool_call_id]},
        placement=Placement(turn_index=0, tab_index=0),
    )


class TestCycleHistoryTokenBudget:
    """The per-cycle history budget reserves later cycles' worst-case output so
    mid-turn truncation does not slide the request prefix (Ollama/vLLM prefix
    cache friendliness)."""

    def _budget(
        self,
        simple_chat_history: list[ChatMessageSimple],
        available_tokens: int = 1000,
        tool_token_budget: int = 0,
        remaining_cycles: int = 5,
        worst_case_cycle_tokens: int = 100,
        extra_reserved_tokens: int = 10,
    ) -> int:
        return _cycle_history_token_budget(
            available_tokens=available_tokens,
            tool_token_budget=tool_token_budget,
            remaining_cycles=remaining_cycles,
            worst_case_cycle_tokens=worst_case_cycle_tokens,
            simple_chat_history=simple_chat_history,
            image_files_replayed_as_markers=False,
            token_counter=None,
            extra_reserved_tokens=extra_reserved_tokens,
        )

    def test_no_reserve_before_tool_history_exists(self) -> None:
        """A plain single-cycle turn keeps the full budget — no worst-case
        penalty for history the turn will never use."""
        history = [
            create_message("Hello", MessageType.USER, 10),
            create_message("Hi", MessageType.ASSISTANT, 10),
            create_message("Follow-up", MessageType.USER, 10),
        ]
        assert self._budget(history) == 1000

    def test_reserve_applied_once_tool_history_exists(self) -> None:
        history = [
            create_message("Hello", MessageType.USER, 10),
            create_assistant_with_tool_call("tc_1", "tool", 20),
            create_tool_response("tc_1", "result", 30),
        ]
        # Tail = 60 history tokens + 10 fixed prompt tokens = 70.
        # Headroom = 930; per-cycle share = 155; reserve = min(5*100, 5*155) = 500.
        assert self._budget(history) == 1000 - 500

    def test_budget_grows_monotonically_across_cycles(self) -> None:
        history = [
            create_message("Hello", MessageType.USER, 10),
            create_assistant_with_tool_call("tc_1", "tool", 20),
            create_tool_response("tc_1", "result", 30),
        ]
        budgets = [
            self._budget(history, remaining_cycles=remaining)
            for remaining in (5, 4, 3, 2, 1, 0)
        ]

        assert budgets == sorted(budgets)
        # The final cycle reserves nothing and offers the full budget.
        assert budgets[-1] == 1000

    def test_reserve_never_breaks_the_must_keep_tail(self) -> None:
        """A tool response larger than the budget leaves no headroom, so the
        reserve is zero and the must-keep tail still fits."""
        history = [
            create_message("Hello", MessageType.USER, 10),
            create_assistant_with_tool_call("tc_1", "tool", 20),
            create_tool_response("tc_1", "result", 970),
        ]
        # Tail = 1000 history + 10 fixed = 1010 > budget; headroom <= 0.
        assert self._budget(history) == 1000

    def test_zero_worst_case_output_disables_reserve(self) -> None:
        history = [
            create_message("Hello", MessageType.USER, 10),
            create_assistant_with_tool_call("tc_1", "tool", 20),
            create_tool_response("tc_1", "result", 30),
        ]
        assert self._budget(history, worst_case_cycle_tokens=0) == 1000

    def test_no_reserve_on_the_final_cycle(self) -> None:
        history = [
            create_message("Hello", MessageType.USER, 10),
            create_assistant_with_tool_call("tc_1", "tool", 20),
            create_tool_response("tc_1", "result", 30),
        ]
        assert self._budget(history, remaining_cycles=0) == 1000


class TestRunLlmLoopCycleAlignedHistory:
    """The in-turn history must be truthful: this cycle's answer text is
    replayed cycle-aligned on its own tool-call message, and dropped tool
    calls answer with tombstones instead of vanishing."""

    def _run_loop(
        self, steps: list[tuple[LlmStepResult, bool]], tools: list[Tool]
    ) -> Any:
        """Run run_llm_loop with two mocked LLM steps; returns the step mock."""
        llm = Mock()
        llm.config = LLMConfig(
            model_provider="openai",
            model_name="text-model",
            temperature=0,
            max_input_tokens=24000,
        )
        with (
            patch("onyx.chat.llm_loop.trace", return_value=nullcontext()),
            patch("onyx.llm.litellm_singleton.config.initialize_litellm"),
            patch(
                "onyx.chat.llm_loop.get_session_with_current_tenant",
                return_value=nullcontext(),
            ),
            patch("onyx.chat.llm_loop.get_default_base_system_prompt", return_value=""),
            patch("onyx.chat.llm_loop.select_reminder_text", return_value=""),
            patch("onyx.chat.llm_loop.run_llm_step", side_effect=steps) as step,
        ):
            run_llm_loop(
                emitter=Mock(),
                state_container=Mock(),
                simple_chat_history=[
                    create_message("Find the thing", MessageType.USER, 10)
                ],
                tools=tools,
                custom_agent_prompt=None,
                context_files=create_context_files(),
                persona=None,
                user_memory_context=None,
                llm=llm,
                token_counter=lambda s: max(1, len(s) // 4),
            )
        return step

    def test_cycle_text_rides_on_its_own_tool_call_message(self) -> None:
        steps = [
            (
                LlmStepResult(
                    reasoning=None,
                    answer="Let me search for that.",
                    tool_calls=[_kickoff("call_1", "fake_search")],
                ),
                False,
            ),
            (
                LlmStepResult(reasoning=None, answer="Final answer.", tool_calls=None),
                False,
            ),
        ]
        step = self._run_loop(steps, tools=[_FakeLoopTool("fake_search")])

        second_history = step.call_args_list[1].kwargs["history"]
        assistant_with_tools = [
            m
            for m in second_history
            if m.message_type == MessageType.ASSISTANT and m.tool_calls
        ]
        assert len(assistant_with_tools) == 1
        assert assistant_with_tools[0].message == "Let me search for that."
        assert assistant_with_tools[0].tool_calls is not None
        assert [tc.tool_call_id for tc in assistant_with_tools[0].tool_calls] == [
            "call_1"
        ]
        # The tool response pairs with the call.
        tool_responses = [
            m
            for m in second_history
            if m.message_type == MessageType.TOOL_CALL_RESPONSE
        ]
        assert [m.tool_call_id for m in tool_responses] == ["call_1"]
        assert tool_responses[0].message == "tool result"

    def test_merged_call_keeps_a_tombstone_response_in_history(self) -> None:
        """Two parallel calls to a mergeable tool collapse into one execution;
        the merged-away call stays in history with an explicit tombstone."""
        steps = [
            (
                LlmStepResult(
                    reasoning=None,
                    answer=None,
                    tool_calls=[
                        _kickoff("call_1", SearchTool.NAME),
                        _kickoff("call_2", SearchTool.NAME),
                    ],
                ),
                False,
            ),
            (
                LlmStepResult(reasoning=None, answer="Final answer.", tool_calls=None),
                False,
            ),
        ]
        step = self._run_loop(steps, tools=[_FakeLoopTool(SearchTool.NAME)])

        second_history = step.call_args_list[1].kwargs["history"]
        assistant_with_tools = next(
            m
            for m in second_history
            if m.message_type == MessageType.ASSISTANT and m.tool_calls
        )
        # Both emitted calls stay in the assistant tool_calls array.
        assert [tc.tool_call_id for tc in assistant_with_tools.tool_calls or []] == [
            "call_1",
            "call_2",
        ]
        tool_responses = {
            m.tool_call_id: m.message
            for m in second_history
            if m.message_type == MessageType.TOOL_CALL_RESPONSE
        }
        assert tool_responses["call_1"] == "tool result"
        assert tool_responses["call_2"] == TOOL_CALL_MERGED_PROMPT


class TestForcedFinalAnswerCycle:
    """The out-of-cycles / post-image-gen cycle answers without running tools,
    but it keeps the tool schemas in the request with tool_choice=none so the
    rendered prompt head stays byte-identical to the earlier cycles and the
    serving stack's prompt cache keeps the whole turn prefix."""

    def _run_single_forced_cycle(
        self,
        step_result: LlmStepResult,
        tool: Tool,
    ) -> Any:
        """Run run_llm_loop with MAX_LLM_CYCLES=1, so the only cycle is the
        forced final answer one; returns the run_llm_step mock."""
        llm = Mock()
        llm.config = LLMConfig(
            model_provider="openai",
            model_name="text-model",
            temperature=0,
            max_input_tokens=24000,
        )
        with (
            patch("onyx.chat.llm_loop.trace", return_value=nullcontext()),
            patch("onyx.llm.litellm_singleton.config.initialize_litellm"),
            patch(
                "onyx.chat.llm_loop.get_session_with_current_tenant",
                return_value=nullcontext(),
            ),
            patch("onyx.chat.llm_loop.get_default_base_system_prompt", return_value=""),
            patch("onyx.chat.llm_loop.select_reminder_text", return_value=""),
            patch("onyx.chat.llm_loop.MAX_LLM_CYCLES", 1),
            patch(
                "onyx.chat.llm_loop.run_llm_step", return_value=(step_result, False)
            ) as step,
        ):
            run_llm_loop(
                emitter=Mock(),
                state_container=Mock(),
                simple_chat_history=[
                    create_message("Find the thing", MessageType.USER, 10)
                ],
                tools=[tool],
                custom_agent_prompt=None,
                context_files=create_context_files(),
                persona=None,
                user_memory_context=None,
                llm=llm,
                token_counter=lambda s: max(1, len(s) // 4),
            )
        return step

    def test_forced_cycle_keeps_tool_schemas_with_none_choice(self) -> None:
        """The forced final request carries the full tool definitions with
        tool_choice=none — not an empty tools array, which would change the
        prompt head and invalidate the cached prefix."""
        tool = _FakeLoopTool("fake_search")
        step = self._run_single_forced_cycle(
            LlmStepResult(
                reasoning=None,
                answer="Here is the final answer.",
                tool_calls=None,
            ),
            tool,
        )

        assert step.call_args_list[0].kwargs["tool_choice"] == ToolChoiceOptions.NONE
        assert step.call_args_list[0].kwargs["tool_definitions"] == [
            tool.tool_definition()
        ]

    def test_forced_cycle_drops_emitted_tool_calls_without_running_them(self) -> None:
        """A model that still tries to call tools on the forced cycle gets its
        calls dropped, not executed, and its narration is kept as the answer."""

        class SpyTool(_FakeLoopTool):
            run_count = 0

            def run(
                self,
                placement: Placement,  # noqa: ARG002
                override_kwargs: Any = None,  # noqa: ARG002
                **llm_kwargs: Any,  # noqa: ARG002
            ) -> ToolResponse:
                type(self).run_count += 1
                return super().run(placement, override_kwargs, **llm_kwargs)

        spy = SpyTool("fake_search")
        # Must not raise: the cycle counts as answered.
        self._run_single_forced_cycle(
            LlmStepResult(
                reasoning=None,
                answer="Here is the final answer.",
                tool_calls=[_kickoff("call_1", "fake_search")],
            ),
            spy,
        )

        assert SpyTool.run_count == 0
