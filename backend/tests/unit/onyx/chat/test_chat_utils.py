"""Tests for chat_utils.py: get_custom_agent_prompt, the shared parallel
tool-call history builder, and convert_chat_history's result-retention policy."""

from io import BytesIO
from typing import cast
from unittest.mock import MagicMock, patch

import pytest

from onyx.chat.chat_utils import (
    _build_tool_call_response_history_message,
    _get_or_extract_plaintext,
    build_parallel_tool_call_messages,
    convert_chat_history,
    create_tool_call_failure_messages,
    get_custom_agent_prompt,
)
from onyx.chat.models import ChatLoadedFile, ToolCallSimple
from onyx.configs.constants import DEFAULT_PERSONA_ID, MessageType
from onyx.db.models import ChatMessage
from onyx.file_store.models import ChatFileType
from onyx.prompts.chat_prompts import TOOL_CALL_RESPONSE_CROSS_MESSAGE
from onyx.prompts.tool_prompts import TOOL_CALL_FAILURE_PROMPT
from onyx.server.query_and_chat.placement import Placement
from onyx.tools.models import ToolCallKickoff


class TestGetCustomAgentPrompt:
    """Tests for the get_custom_agent_prompt function."""

    def _create_mock_persona(
        self,
        persona_id: int = 1,
        system_prompt: str | None = None,
        replace_base_system_prompt: bool = False,
    ) -> MagicMock:
        """Create a mock Persona with the specified attributes."""
        persona = MagicMock()
        persona.id = persona_id
        persona.system_prompt = system_prompt
        persona.replace_base_system_prompt = replace_base_system_prompt
        return persona

    def _create_mock_chat_session(
        self,
        project: MagicMock | None = None,
    ) -> MagicMock:
        """Create a mock ChatSession with the specified attributes."""
        chat_session = MagicMock()
        chat_session.project = project
        return chat_session

    def _create_mock_project(
        self,
        instructions: str = "",
    ) -> MagicMock:
        """Create a mock UserProject with the specified attributes."""
        project = MagicMock()
        project.instructions = instructions
        return project

    def test_default_persona_no_project(self) -> None:
        """Test that default persona without a project returns None."""
        persona = self._create_mock_persona(persona_id=DEFAULT_PERSONA_ID)
        chat_session = self._create_mock_chat_session(project=None)

        result = get_custom_agent_prompt(persona, chat_session)

        assert result is None

    def test_default_persona_with_project_instructions(self) -> None:
        """Test that default persona in a project returns project instructions."""
        persona = self._create_mock_persona(persona_id=DEFAULT_PERSONA_ID)
        project = self._create_mock_project(instructions="Do X and Y")
        chat_session = self._create_mock_chat_session(project=project)

        result = get_custom_agent_prompt(persona, chat_session)

        assert result == "Do X and Y"

    def test_default_persona_with_empty_project_instructions(self) -> None:
        """Test that default persona in a project with empty instructions returns None."""
        persona = self._create_mock_persona(persona_id=DEFAULT_PERSONA_ID)
        project = self._create_mock_project(instructions="")
        chat_session = self._create_mock_chat_session(project=project)

        result = get_custom_agent_prompt(persona, chat_session)

        assert result is None

    def test_custom_persona_replace_base_prompt_true(self) -> None:
        """Test that custom persona with replace_base_system_prompt=True returns None."""
        persona = self._create_mock_persona(
            persona_id=1,
            system_prompt="Custom system prompt",
            replace_base_system_prompt=True,
        )
        chat_session = self._create_mock_chat_session(project=None)

        result = get_custom_agent_prompt(persona, chat_session)

        assert result is None

    def test_custom_persona_with_system_prompt(self) -> None:
        """Test that custom persona with system_prompt returns the system_prompt."""
        persona = self._create_mock_persona(
            persona_id=1,
            system_prompt="Custom system prompt",
            replace_base_system_prompt=False,
        )
        chat_session = self._create_mock_chat_session(project=None)

        result = get_custom_agent_prompt(persona, chat_session)

        assert result == "Custom system prompt"

    def test_custom_persona_empty_string_system_prompt(self) -> None:
        """Test that custom persona with empty string system_prompt returns None."""
        persona = self._create_mock_persona(
            persona_id=1,
            system_prompt="",
            replace_base_system_prompt=False,
        )
        chat_session = self._create_mock_chat_session(project=None)

        result = get_custom_agent_prompt(persona, chat_session)

        assert result is None

    def test_custom_persona_none_system_prompt(self) -> None:
        """Test that custom persona with None system_prompt returns None."""
        persona = self._create_mock_persona(
            persona_id=1,
            system_prompt=None,
            replace_base_system_prompt=False,
        )
        chat_session = self._create_mock_chat_session(project=None)

        result = get_custom_agent_prompt(persona, chat_session)

        assert result is None

    def test_custom_persona_in_project_uses_persona_prompt(self) -> None:
        """Test that custom persona in a project uses persona's system_prompt, not project instructions."""
        persona = self._create_mock_persona(
            persona_id=1,
            system_prompt="Custom system prompt",
            replace_base_system_prompt=False,
        )
        project = self._create_mock_project(instructions="Project instructions")
        chat_session = self._create_mock_chat_session(project=project)

        result = get_custom_agent_prompt(persona, chat_session)

        # Should use persona's system_prompt, NOT project instructions
        assert result == "Custom system prompt"

    def test_custom_persona_replace_base_in_project(self) -> None:
        """Test that custom persona with replace_base_system_prompt=True in a project still returns None."""
        persona = self._create_mock_persona(
            persona_id=1,
            system_prompt="Custom system prompt",
            replace_base_system_prompt=True,
        )
        project = self._create_mock_project(instructions="Project instructions")
        chat_session = self._create_mock_chat_session(project=project)

        result = get_custom_agent_prompt(persona, chat_session)

        # Should return None because replace_base_system_prompt=True
        assert result is None


class TestBuildToolCallResponseHistoryMessage:
    def test_image_tool_uses_generated_images(self) -> None:
        message = _build_tool_call_response_history_message(
            tool_name="generate_image",
            generated_images=[{"file_id": "img-1", "revised_prompt": "p1"}],
            tool_call_response=None,
        )
        assert message == '[{"file_id": "img-1", "revised_prompt": "p1"}]'

    def test_non_image_tool_uses_placeholder(self) -> None:
        message = _build_tool_call_response_history_message(
            tool_name="web_search",
            generated_images=None,
            tool_call_response='{"raw":"value"}',
        )
        assert message == TOOL_CALL_RESPONSE_CROSS_MESSAGE


class TestBuildParallelToolCallMessages:
    """The canonical OpenAI parallel tool calling builder shared by the
    in-turn and cross-turn history paths."""

    def _tool_call(self, tool_call_id: str, token_count: int) -> ToolCallSimple:
        return ToolCallSimple(
            tool_call_id=tool_call_id,
            tool_name="internal_search",
            tool_arguments={"queries": ["q"]},
            token_count=token_count,
        )

    def test_shape_one_assistant_plus_one_response_per_call(self) -> None:
        calls = [self._tool_call("call-1", 10), self._tool_call("call-2", 20)]
        messages = build_parallel_tool_call_messages(
            tool_calls=calls,
            response_texts=["result one", "result two"],
            token_counter=len,
        )

        assert len(messages) == 3
        assistant, response_1, response_2 = messages
        assert assistant.message_type == MessageType.ASSISTANT
        assert assistant.message == ""
        assert assistant.tool_calls == calls
        # Assistant tokens = tool call tokens only (no assistant text).
        assert assistant.token_count == 30
        assert response_1.message_type == MessageType.TOOL_CALL_RESPONSE
        assert response_1.tool_call_id == "call-1"
        assert response_1.message == "result one"
        assert response_1.token_count == len("result one")
        assert response_2.tool_call_id == "call-2"
        assert response_2.message == "result two"

    def test_assistant_message_rides_on_the_tool_call_message(self) -> None:
        """Per-cycle answer text is replayed cycle-aligned: it belongs on the
        same assistant message as that cycle's tool calls."""
        calls = [self._tool_call("call-1", 10)]
        messages = build_parallel_tool_call_messages(
            tool_calls=calls,
            response_texts=["result"],
            token_counter=len,
            assistant_message="Let me search for that.",
        )

        assistant = messages[0]
        assert assistant.message == "Let me search for that."
        assert assistant.tool_calls == calls
        assert assistant.token_count == 10 + len("Let me search for that.")

    def test_image_files_attach_to_their_own_response(self) -> None:
        image = ChatLoadedFile(
            file_id="img-1",
            content=b"",
            file_type=ChatFileType.IMAGE,
            filename="img.png",
            content_text=None,
            token_count=0,
        )
        calls = [self._tool_call("call-1", 5), self._tool_call("call-2", 5)]
        messages = build_parallel_tool_call_messages(
            tool_calls=calls,
            response_texts=["with image", "without"],
            token_counter=len,
            image_files_by_tool_call_id={"call-1": [image]},
        )

        assert messages[1].image_files == [image]
        assert messages[2].image_files is None

    def test_mismatched_calls_and_responses_raise(self) -> None:
        with pytest.raises(ValueError, match="exactly one response"):
            build_parallel_tool_call_messages(
                tool_calls=[self._tool_call("call-1", 5)],
                response_texts=[],
                token_counter=len,
            )


class TestCreateToolCallFailureMessages:
    def test_uses_the_shared_builder_shape(self) -> None:
        kickoffs = [
            ToolCallKickoff(
                tool_call_id="call-1",
                tool_name="internal_search",
                tool_args={"queries": ["q"]},
                placement=Placement(turn_index=0),
            )
        ]
        messages = create_tool_call_failure_messages(kickoffs, token_counter=len)

        assert len(messages) == 2
        assistant, response = messages
        assert assistant.message_type == MessageType.ASSISTANT
        assert assistant.tool_calls is not None
        assert assistant.tool_calls[0].tool_call_id == "call-1"
        assert response.message_type == MessageType.TOOL_CALL_RESPONSE
        assert response.tool_call_id == "call-1"
        assert response.message == TOOL_CALL_FAILURE_PROMPT
        assert response.token_count == len(TOOL_CALL_FAILURE_PROMPT)

    def test_empty_input(self) -> None:
        assert create_tool_call_failure_messages([], token_counter=len) == []


class TestGetOrExtractPlaintext:
    """Tests for the plaintext extraction cache used by chat file loading."""

    def test_cache_hit_skips_extraction(self) -> None:
        file_store = MagicMock()
        file_store.read_file.return_value = BytesIO(b"cached text")
        extract_fn = MagicMock(return_value="should not be called")

        with (
            patch(
                "onyx.chat.chat_utils.get_default_file_store",
                return_value=file_store,
            ),
            patch("onyx.chat.chat_utils.store_plaintext") as store_plaintext,
        ):
            result = _get_or_extract_plaintext("file-1", extract_fn)

        assert result == "cached text"
        extract_fn.assert_not_called()
        store_plaintext.assert_not_called()

    def test_cache_miss_stores_extracted_text(self) -> None:
        file_store = MagicMock()
        file_store.read_file.side_effect = RuntimeError("not in store")
        extract_fn = MagicMock(return_value="extracted text")

        with (
            patch(
                "onyx.chat.chat_utils.get_default_file_store",
                return_value=file_store,
            ),
            patch("onyx.chat.chat_utils.store_plaintext") as store_plaintext,
        ):
            result = _get_or_extract_plaintext("file-2", extract_fn)

        assert result == "extracted text"
        extract_fn.assert_called_once()
        store_plaintext.assert_called_once_with("file-2", "extracted text")

    def test_cache_miss_caches_empty_extraction(self) -> None:
        """Files extract_file_text cannot process (e.g. .zip) return "".
        We must still cache that result so we don't re-fetch the file from
        object storage on every subsequent chat turn.
        """
        file_store = MagicMock()
        file_store.read_file.side_effect = RuntimeError("not in store")
        extract_fn = MagicMock(return_value="")

        with (
            patch(
                "onyx.chat.chat_utils.get_default_file_store",
                return_value=file_store,
            ),
            patch("onyx.chat.chat_utils.store_plaintext") as store_plaintext,
        ):
            result = _get_or_extract_plaintext("file-3", extract_fn)

        assert result == ""
        extract_fn.assert_called_once()
        store_plaintext.assert_called_once_with("file-3", "")


class TestConvertChatHistory:
    """Tests for convert_chat_history.

    Regression coverage for the project-image duplication bug: project images
    (passed via ``context_image_files``) must attach to the last USER message
    exactly once, and only to the last USER message — even when earlier USER
    messages exist in the history. Also covers the cross-turn tool result
    retention policy: the most recent tool-using turn keeps its results in
    full (bounded by ``max_recent_tool_response_tokens``), older turns are
    tombstoned.
    """

    def _make_chat_message(
        self,
        message: str,
        message_type: MessageType,
        token_count: int = 5,
    ) -> MagicMock:
        msg = MagicMock()
        msg.message = message
        msg.message_type = message_type
        msg.token_count = token_count
        msg.files = None
        msg.tool_calls = None
        return msg

    def _make_tool_call(
        self,
        tool_id: int,
        tool_call_id: str,
        turn_number: int = 0,
        tool_call_response: str = "original tool output",
        generated_images: list[dict] | None = None,
        tool_call_tokens: int = 12,
    ) -> MagicMock:
        tool_call = MagicMock()
        tool_call.tool_id = tool_id
        tool_call.tool_call_id = tool_call_id
        tool_call.turn_number = turn_number
        tool_call.tool_call_arguments = {"queries": ["alpha"]}
        tool_call.tool_call_tokens = tool_call_tokens
        tool_call.tool_call_response = tool_call_response
        tool_call.generated_images = generated_images
        return tool_call

    def test_attaches_project_images_to_last_user_message_only_once(
        self,
    ) -> None:
        project_image = ChatLoadedFile(
            file_id="project_image",
            content=b"",
            file_type=ChatFileType.IMAGE,
            filename="project.png",
            content_text=None,
            token_count=50,
        )

        chat_history = [
            self._make_chat_message("First question", MessageType.USER),
            self._make_chat_message("First answer", MessageType.ASSISTANT),
            self._make_chat_message("Second question", MessageType.USER),
        ]

        result = convert_chat_history(
            chat_history=cast(list[ChatMessage], chat_history),
            files=[],
            context_image_files=[project_image],
            additional_context=None,
            token_counter=lambda s: len(s),
            tool_id_to_name_map={},
        )

        user_messages = [
            m for m in result.simple_messages if m.message_type == MessageType.USER
        ]
        assert len(user_messages) == 2

        # First USER message must NOT carry the project image.
        first_user = user_messages[0]
        assert first_user.message == "First question"
        assert first_user.image_files is None

        # Last USER message carries the project image exactly once.
        last_user = user_messages[-1]
        assert last_user.message == "Second question"
        assert last_user.image_files is not None
        assert len(last_user.image_files) == 1
        assert last_user.image_files[0].file_id == "project_image"

    def test_tool_response_placeholder_token_count_is_measured(self) -> None:
        """Cross-turn tool responses are replayed as a placeholder — its
        budgeted token count must come from the token counter, not a
        hardcoded constant. A zero retention budget tombstones everything,
        isolating the placeholder path."""
        tool_call = self._make_tool_call(
            tool_id=1,
            tool_call_id="call-1",
            tool_call_response="original tool output",
        )

        assistant_msg = self._make_chat_message("final answer", MessageType.ASSISTANT)
        assistant_msg.tool_calls = [tool_call]

        chat_history = [
            self._make_chat_message("A question", MessageType.USER),
            assistant_msg,
        ]

        result = convert_chat_history(
            chat_history=cast(list[ChatMessage], chat_history),
            files=[],
            context_image_files=[],
            additional_context=None,
            token_counter=lambda s: len(s),
            tool_id_to_name_map={1: "internal_search"},
            max_recent_tool_response_tokens=0,
        )

        tool_responses = [
            m
            for m in result.simple_messages
            if m.message_type == MessageType.TOOL_CALL_RESPONSE
        ]
        assert len(tool_responses) == 1
        assert tool_responses[0].message == TOOL_CALL_RESPONSE_CROSS_MESSAGE
        assert tool_responses[0].token_count == len(TOOL_CALL_RESPONSE_CROSS_MESSAGE)

    def test_recent_turn_results_retained_in_full(self) -> None:
        """The most recent tool-using turn replays its full results; older
        turns keep the tombstone."""
        older_call = self._make_tool_call(
            tool_id=1,
            tool_call_id="call-old",
            turn_number=0,
            tool_call_response="results from turn one",
        )
        older_msg = self._make_chat_message("turn one answer", MessageType.ASSISTANT)
        older_msg.tool_calls = [older_call]

        recent_call = self._make_tool_call(
            tool_id=1,
            tool_call_id="call-new",
            turn_number=0,
            tool_call_response="results from turn two",
        )
        recent_msg = self._make_chat_message("turn two answer", MessageType.ASSISTANT)
        recent_msg.tool_calls = [recent_call]

        chat_history = [
            self._make_chat_message("A question", MessageType.USER),
            older_msg,
            self._make_chat_message("Follow-up", MessageType.USER),
            recent_msg,
        ]

        result = convert_chat_history(
            chat_history=cast(list[ChatMessage], chat_history),
            files=[],
            context_image_files=[],
            additional_context=None,
            token_counter=lambda s: len(s),
            tool_id_to_name_map={1: "internal_search"},
        )

        tool_responses = {
            m.tool_call_id: m.message
            for m in result.simple_messages
            if m.message_type == MessageType.TOOL_CALL_RESPONSE
        }
        assert tool_responses["call-old"] == TOOL_CALL_RESPONSE_CROSS_MESSAGE
        assert tool_responses["call-new"] == "results from turn two"

    def test_retention_cap_tombsrones_older_results_of_the_recent_turn(self) -> None:
        """Within the retained turn, newest results claim the budget first;
        anything older that no longer fits is tombstoned."""
        # Listed chronologically: call-1 ran first, call-2 second.
        big_call = self._make_tool_call(
            tool_id=1,
            tool_call_id="call-1",
            turn_number=0,
            tool_call_response="x" * 100,
        )
        small_call = self._make_tool_call(
            tool_id=1,
            tool_call_id="call-2",
            turn_number=0,
            tool_call_response="y" * 20,
        )
        recent_msg = self._make_chat_message("turn answer", MessageType.ASSISTANT)
        recent_msg.tool_calls = [big_call, small_call]

        chat_history = [
            self._make_chat_message("A question", MessageType.USER),
            recent_msg,
        ]

        result = convert_chat_history(
            chat_history=cast(list[ChatMessage], chat_history),
            files=[],
            context_image_files=[],
            additional_context=None,
            token_counter=lambda s: len(s),
            tool_id_to_name_map={1: "internal_search"},
            max_recent_tool_response_tokens=50,
        )

        tool_responses = {
            m.tool_call_id: m.message
            for m in result.simple_messages
            if m.message_type == MessageType.TOOL_CALL_RESPONSE
        }
        # The newest (call-2) fits in the 50-token budget; the older one does not.
        assert tool_responses["call-2"] == "y" * 20
        assert tool_responses["call-1"] == TOOL_CALL_RESPONSE_CROSS_MESSAGE

    def test_retention_cap_zero_tombstones_everything(self) -> None:
        single_call = self._make_tool_call(
            tool_id=1,
            tool_call_id="call-1",
            tool_call_response="full result",
        )
        recent_msg = self._make_chat_message("turn answer", MessageType.ASSISTANT)
        recent_msg.tool_calls = [single_call]

        chat_history = [
            self._make_chat_message("A question", MessageType.USER),
            recent_msg,
        ]

        result = convert_chat_history(
            chat_history=cast(list[ChatMessage], chat_history),
            files=[],
            context_image_files=[],
            additional_context=None,
            token_counter=lambda s: len(s),
            tool_id_to_name_map={1: "internal_search"},
            max_recent_tool_response_tokens=0,
        )

        tool_responses = [
            m.message
            for m in result.simple_messages
            if m.message_type == MessageType.TOOL_CALL_RESPONSE
        ]
        assert tool_responses == [TOOL_CALL_RESPONSE_CROSS_MESSAGE]

    def test_image_generation_results_replay_at_any_age(self) -> None:
        """generate_image responses replay their image references regardless
        of the retention policy — the model needs the file ids to edit them."""
        image_call = self._make_tool_call(
            tool_id=2,
            tool_call_id="call-img",
            tool_call_response="image metadata json",
            generated_images=[{"file_id": "img-1", "revised_prompt": "p1"}],
        )
        image_msg = self._make_chat_message("older turn answer", MessageType.ASSISTANT)
        image_msg.tool_calls = [image_call]

        recent_call = self._make_tool_call(
            tool_id=1,
            tool_call_id="call-recent",
            tool_call_response="recent results",
        )
        recent_msg = self._make_chat_message(
            "recent turn answer", MessageType.ASSISTANT
        )
        recent_msg.tool_calls = [recent_call]

        chat_history = [
            self._make_chat_message("A question", MessageType.USER),
            image_msg,
            self._make_chat_message("Follow-up", MessageType.USER),
            recent_msg,
        ]

        result = convert_chat_history(
            chat_history=cast(list[ChatMessage], chat_history),
            files=[],
            context_image_files=[],
            additional_context=None,
            token_counter=lambda s: len(s),
            tool_id_to_name_map={1: "internal_search", 2: "generate_image"},
            max_recent_tool_response_tokens=0,
        )

        tool_responses = {
            m.tool_call_id: m.message
            for m in result.simple_messages
            if m.message_type == MessageType.TOOL_CALL_RESPONSE
        }
        assert (
            tool_responses["call-img"]
            == '[{"file_id": "img-1", "revised_prompt": "p1"}]'
        )
        assert tool_responses["call-recent"] == TOOL_CALL_RESPONSE_CROSS_MESSAGE
