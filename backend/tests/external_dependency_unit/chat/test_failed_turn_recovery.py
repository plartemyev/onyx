"""Failed-turn recovery contract: save_failed_turn keeps the turn's work.

The incident (2026-11-06): a 4h21m turn failed on its final cycle and the
error path wrote ONLY an error string — eight cycles of tool calls, results
and reasoning were discarded. save_failed_turn must persist the partial
answer with an honest note, the reasoning, and the tool calls, so the
timeline shows the work and the next turn's prompt replays it for a nudge.
"""

from collections.abc import Generator
from uuid import UUID

import pytest
from sqlalchemy.orm import Session

from onyx.chat.chat_state import ChatStateContainer
from onyx.chat.process_message import save_failed_turn
from onyx.configs.constants import DocumentSource
from onyx.context.search.models import SearchDoc
from onyx.db.chat import (
    create_chat_session,
    get_or_create_root_message,
    reserve_message_id,
)
from onyx.db.models import ChatMessage, ChatSession, ToolCall, User
from onyx.tools.models import ToolCallInfo
from tests.external_dependency_unit.conftest import create_test_user, delete_test_user


@pytest.fixture
def owner(db_session: Session) -> Generator[User, None, None]:
    user = create_test_user(db_session, "failed-turn-recovery")
    yield user
    db_session.rollback()
    db_session.query(ChatMessage).filter(
        ChatMessage.chat_session_id.in_(
            db_session.query(ChatSession.id).filter(ChatSession.user_id == user.id)
        )
    ).delete(synchronize_session=False)
    db_session.query(ChatSession).filter(ChatSession.user_id == user.id).delete()
    delete_test_user(db_session, user)
    db_session.commit()


def _reserve_assistant(db_session: Session, session_id: UUID) -> ChatMessage:
    root = get_or_create_root_message(chat_session_id=session_id, db_session=db_session)
    return reserve_message_id(
        db_session=db_session,
        chat_session_id=session_id,
        parent_message=root.id,
    )


def _container_with_work() -> ChatStateContainer:
    doc = SearchDoc(
        document_id="doc-1",
        chunk_ind=0,
        semantic_identifier="spec sheet",
        blurb="an excerpt",
        source_type=DocumentSource.WEB,
        boost=0,
        hidden=False,
        metadata={},
        match_highlights=[],
    )
    container = ChatStateContainer()
    container.set_answer_tokens("The workbook was built with 23 sheets.")
    container.set_reasoning_tokens("planning the sheet structure")
    container.add_tool_call(
        ToolCallInfo(
            parent_tool_call_id=None,
            turn_index=0,
            tab_index=0,
            tool_name="run_python",
            tool_call_id="call-1",
            tool_id=1,
            reasoning_tokens=None,
            tool_call_arguments={"code": "import openpyxl"},
            tool_call_response="saved workbook_partial.xlsx",
            search_docs=[doc],
        )
    )
    container.add_search_docs([doc])
    return container


def test_save_failed_turn_keeps_partial_work(db_session: Session, owner: User) -> None:
    session = create_chat_session(
        db_session=db_session,
        description="recovery",
        user_id=owner.id,
        persona_id=None,
    )
    assistant = _reserve_assistant(db_session, session.id)
    assert assistant.message.startswith("Response was terminated")

    container = _container_with_work()
    error_text = "Error from test-model: The selected model returned no final answer."

    save_failed_turn(
        assistant_message_id=assistant.id,
        container=container,
        error_text=error_text,
        keeps_content=True,
    )

    # save_failed_turn commits on its own session; drop cached state.
    db_session.expire_all()
    stored = db_session.get(ChatMessage, assistant.id)
    assert stored is not None
    assert stored.message is not None
    assert stored.message.startswith("The workbook was built with 23 sheets.")
    assert "failed before it could finish" in stored.message
    assert error_text in stored.message
    assert stored.error == error_text
    assert stored.reasoning_tokens == "planning the sheet structure"

    tool_calls = (
        db_session.query(ToolCall)
        .filter(ToolCall.parent_chat_message_id == assistant.id)
        .all()
    )
    assert len(tool_calls) == 1
    assert tool_calls[0].tool_call_response == "saved workbook_partial.xlsx"


def test_save_failed_turn_without_work_falls_back_to_error_only(
    db_session: Session, owner: User
) -> None:
    session = create_chat_session(
        db_session=db_session,
        description="recovery-empty",
        user_id=owner.id,
        persona_id=None,
    )
    assistant = _reserve_assistant(db_session, session.id)

    save_failed_turn(
        assistant_message_id=assistant.id,
        container=ChatStateContainer(),
        error_text="Error from test-model: connection failed before any output.",
        keeps_content=True,
    )

    # save_failed_turn commits on its own session; drop cached state.
    db_session.expire_all()
    stored = db_session.get(ChatMessage, assistant.id)
    assert stored is not None
    assert (
        stored.message == "Error from test-model: connection failed before any output."
    )
    assert stored.error is not None
    assert stored.token_count == 0


def test_save_failed_turn_incognito_keeps_no_content(
    db_session: Session, owner: User
) -> None:
    session = create_chat_session(
        db_session=db_session,
        description="recovery-incognito",
        user_id=owner.id,
        persona_id=None,
    )
    assistant = _reserve_assistant(db_session, session.id)

    save_failed_turn(
        assistant_message_id=assistant.id,
        container=_container_with_work(),
        error_text="Error from test-model: boom.",
        keeps_content=False,
    )

    # save_failed_turn commits on its own session; drop cached state.
    db_session.expire_all()
    stored = db_session.get(ChatMessage, assistant.id)
    assert stored is not None
    # Generic error only; the partial answer must not land in the durable row.
    assert stored.message == "Error from test-model: boom."
    assert stored.reasoning_tokens is None
    tool_calls = (
        db_session.query(ToolCall)
        .filter(ToolCall.parent_chat_message_id == assistant.id)
        .all()
    )
    assert tool_calls == []
