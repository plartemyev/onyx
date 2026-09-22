"""Tests for extract_generated_file_descriptors in session_loading.py.

Covers recovery of code interpreter artifacts from stored tool call
responses, including latest-save dedupe and non-python tool filtering.
"""

import json
from unittest.mock import MagicMock

from pytest import MonkeyPatch
from sqlalchemy.orm import Session

from onyx.db.models import Tool as DbTool
from onyx.db.models import ToolCall as DbToolCall
from onyx.file_store.models import ChatFileType
from onyx.server.query_and_chat.session_loading import (
    extract_generated_file_descriptors,
)


def _make_db_tool_call(
    tool: DbTool,
    generated_files: list[dict[str, str]] | None,
) -> DbToolCall:
    response = ""
    if generated_files is not None:
        response = json.dumps({"generated_files": generated_files})
    return DbToolCall(
        tool_id=tool.id,
        chat_session_id=None,
        turn_number=0,
        tool_call_arguments={"code": "print('hi')"},
        tool_call_response=response,
    )


def _patch_get_tool(monkeypatch: MonkeyPatch, tool_by_id: dict[int, DbTool]) -> None:
    def _get_tool_by_id(tool_id: int, _db_session: Session) -> DbTool:
        return tool_by_id[tool_id]

    monkeypatch.setattr(
        "onyx.server.query_and_chat.session_loading.get_tool_by_id",
        _get_tool_by_id,
    )


def test_extracts_generated_files_from_python_tool_calls(
    monkeypatch: MonkeyPatch,
) -> None:
    python_tool = DbTool(id=1, name="run_python", in_code_tool_id="PythonTool")
    _patch_get_tool(monkeypatch, {1: python_tool})
    tool_call = _make_db_tool_call(
        python_tool,
        [
            {
                "filename": "annotated_micro_usb.png",
                "file_link": "http://localhost/api/chat/file/12f0e88f-7ff4",
            }
        ],
    )

    result = extract_generated_file_descriptors([tool_call], MagicMock())

    assert len(result) == 1
    assert result[0]["id"] == "12f0e88f-7ff4"
    assert result[0]["name"] == "annotated_micro_usb.png"
    assert result[0]["type"] == ChatFileType.IMAGE


def test_dedupes_repeated_saves_to_latest(monkeypatch: MonkeyPatch) -> None:
    python_tool = DbTool(id=1, name="run_python", in_code_tool_id="PythonTool")
    _patch_get_tool(monkeypatch, {1: python_tool})
    tool_call = _make_db_tool_call(
        python_tool,
        [
            {"filename": "out.png", "file_link": "http://x/api/chat/file/v1"},
            {"filename": "out.png", "file_link": "http://x/api/chat/file/v2"},
            {"filename": "out.png", "file_link": "http://x/api/chat/file/v3"},
        ],
    )

    result = extract_generated_file_descriptors([tool_call], MagicMock())

    assert [(d["name"], d["id"]) for d in result] == [("out.png", "v3")]


def test_ignores_non_python_tools(monkeypatch: MonkeyPatch) -> None:
    web_search_tool = DbTool(id=2, name="web_search", in_code_tool_id="WebSearchTool")
    _patch_get_tool(monkeypatch, {2: web_search_tool})
    tool_call = _make_db_tool_call(
        web_search_tool,
        [{"filename": "out.png", "file_link": "http://x/api/chat/file/v1"}],
    )

    assert extract_generated_file_descriptors([tool_call], MagicMock()) == []


def test_ignores_malformed_and_empty_responses(
    monkeypatch: MonkeyPatch,
) -> None:
    python_tool = DbTool(id=1, name="run_python", in_code_tool_id="PythonTool")
    _patch_get_tool(monkeypatch, {1: python_tool})
    malformed = DbToolCall(
        tool_id=1,
        chat_session_id=None,
        turn_number=0,
        tool_call_arguments={},
        tool_call_response="not json",
    )
    empty = _make_db_tool_call(python_tool, None)
    no_link = _make_db_tool_call(python_tool, [{"filename": "out.png"}])

    result = extract_generated_file_descriptors(
        [malformed, empty, no_link], MagicMock()
    )
    assert result == []


def test_recovers_session_from_the_incident(monkeypatch: MonkeyPatch) -> None:
    """The failing session saved annotated_micro_usb.png four times; recovery
    surfaces exactly the newest version."""
    python_tool = DbTool(id=1, name="run_python", in_code_tool_id="PythonTool")
    _patch_get_tool(monkeypatch, {1: python_tool})
    saved_ids = ["6a229524", "28f9261f", "87b0e295", "12f0e88f"]
    tool_calls = [
        _make_db_tool_call(python_tool, [])  # failed executions save nothing
    ] * 4 + [
        _make_db_tool_call(
            python_tool,
            [
                {
                    "filename": "annotated_micro_usb.png",
                    "file_link": f"http://x/api/chat/file/{file_id}",
                }
            ],
        )
        for file_id in saved_ids
    ]

    result = extract_generated_file_descriptors(tool_calls, MagicMock())

    assert len(result) == 1
    assert result[0]["id"] == "12f0e88f"
