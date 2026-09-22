"""Tests for save_chat.py.

Covers _extract_referenced_file_descriptors, placeholder link rewriting, and
sanitization in save_chat_turn.
"""

from unittest.mock import MagicMock

from pytest import MonkeyPatch

from onyx.chat import save_chat
from onyx.chat.save_chat import (
    _extract_referenced_file_descriptors,
    _rewrite_placeholder_file_links,
)
from onyx.file_store.models import ChatFileType
from onyx.tools.models import PythonExecutionFile, ToolCallInfo


def _make_tool_call_info(
    generated_files: list[PythonExecutionFile] | None = None,
    tool_name: str = "run_python",
) -> ToolCallInfo:
    return ToolCallInfo(
        parent_tool_call_id=None,
        turn_index=0,
        tab_index=0,
        tool_name=tool_name,
        tool_call_id="tc_1",
        tool_id=1,
        reasoning_tokens=None,
        tool_call_arguments={"code": "print('hi')"},
        tool_call_response="{}",
        generated_files=generated_files,
    )


# ---- _extract_referenced_file_descriptors tests ----


def test_returns_empty_when_no_generated_files() -> None:
    tool_call = _make_tool_call_info(generated_files=None)
    result = _extract_referenced_file_descriptors([tool_call], "some message")
    assert result == []


def test_attaches_unreferenced_files_as_fallback() -> None:
    """When the message references nothing (e.g. the model mangled the link),
    the generated files are still attached so artifacts stay downloadable."""
    files = [
        PythonExecutionFile(
            filename="chart.png",
            file_link="http://localhost/api/chat/file/abc-123",
        )
    ]
    tool_call = _make_tool_call_info(generated_files=files)
    result = _extract_referenced_file_descriptors([tool_call], "Here is your answer.")
    assert len(result) == 1
    assert result[0]["id"] == "abc-123"
    assert result[0]["name"] == "chart.png"


def test_filename_mention_attaches_file() -> None:
    """A filename mention without a working link (e.g. ![name](file_link))
    attaches the file."""
    file_id = "abc-123-def"
    files = [
        PythonExecutionFile(
            filename="chart.png",
            file_link=f"http://localhost/api/chat/file/{file_id}",
        )
    ]
    tool_call = _make_tool_call_info(generated_files=files)
    message = "Here is the annotated image: ![chart.png](file_link)"

    result = _extract_referenced_file_descriptors([tool_call], message)

    assert len(result) == 1
    assert result[0]["id"] == file_id
    assert result[0]["type"] == ChatFileType.IMAGE
    assert result[0]["name"] == "chart.png"


def test_filename_mention_collapses_to_latest_save() -> None:
    """Repeated saves of one filename attach only the newest file id."""
    files = [
        PythonExecutionFile(
            filename="chart.png", file_link="http://localhost/api/chat/file/v1"
        ),
        PythonExecutionFile(
            filename="chart.png", file_link="http://localhost/api/chat/file/v2"
        ),
        PythonExecutionFile(
            filename="chart.png", file_link="http://localhost/api/chat/file/v3"
        ),
    ]
    tool_call = _make_tool_call_info(generated_files=files)
    message = "Saved chart.png."

    result = _extract_referenced_file_descriptors([tool_call], message)

    assert len(result) == 1
    assert result[0]["id"] == "v3"


def test_exact_file_id_reference_wins_over_newer_save() -> None:
    """A link to an earlier version keeps pointing at that version and does
    not also pull in the newest save of the same filename."""
    files = [
        PythonExecutionFile(
            filename="chart.png", file_link="http://localhost/api/chat/file/old-1"
        ),
        PythonExecutionFile(
            filename="chart.png", file_link="http://localhost/api/chat/file/new-2"
        ),
    ]
    tool_call = _make_tool_call_info(generated_files=files)
    message = "[chart.png](http://localhost/api/chat/file/old-1)"

    result = _extract_referenced_file_descriptors([tool_call], message)

    assert len(result) == 1
    assert result[0]["id"] == "old-1"


def test_fallback_dedupes_and_keeps_order() -> None:
    """Fallback attachment dedupes by filename and keeps first-appearance
    order."""
    files = [
        PythonExecutionFile(
            filename="data.csv", file_link="http://localhost/api/chat/file/csv-1"
        ),
        PythonExecutionFile(
            filename="plot.png", file_link="http://localhost/api/chat/file/plot-1"
        ),
        PythonExecutionFile(
            filename="plot.png", file_link="http://localhost/api/chat/file/plot-2"
        ),
    ]
    tool_call = _make_tool_call_info(generated_files=files)

    result = _extract_referenced_file_descriptors([tool_call], "Done!")

    assert [(d["name"], d["id"]) for d in result] == [
        ("data.csv", "csv-1"),
        ("plot.png", "plot-2"),
    ]


def test_placeholder_link_rewritten_by_name() -> None:
    files = [
        PythonExecutionFile(
            filename="chart.png",
            file_link="http://localhost/api/chat/file/abc-123",
        )
    ]
    message = "Here: ![chart.png](file_link)"

    result = _rewrite_placeholder_file_links(message, files)

    assert result == "Here: ![chart.png](http://localhost/api/chat/file/abc-123)"


def test_placeholder_link_rewritten_in_anchor_form() -> None:
    files = [
        PythonExecutionFile(
            filename="data.csv",
            file_link="http://localhost/api/chat/file/csv-1",
        )
    ]
    message = "Download: [data.csv](file_link)"

    result = _rewrite_placeholder_file_links(message, files)

    assert result == "Download: [data.csv](http://localhost/api/chat/file/csv-1)"


def test_placeholder_link_single_file_resolves_without_name() -> None:
    files = [
        PythonExecutionFile(
            filename="chart.png",
            file_link="http://localhost/api/chat/file/abc-123",
        )
    ]

    result = _rewrite_placeholder_file_links("![img](file_link)", files)

    assert result == "![img](http://localhost/api/chat/file/abc-123)"


def test_placeholder_link_ambiguous_name_left_unchanged() -> None:
    files = [
        PythonExecutionFile(
            filename="a.png", file_link="http://localhost/api/chat/file/a-1"
        ),
        PythonExecutionFile(
            filename="b.png", file_link="http://localhost/api/chat/file/b-1"
        ),
    ]
    message = "![unknown.png](file_link)"

    result = _rewrite_placeholder_file_links(message, files)

    assert result == message


def test_placeholder_link_resolves_ambiguous_by_name() -> None:
    files = [
        PythonExecutionFile(
            filename="a.png", file_link="http://localhost/api/chat/file/a-1"
        ),
        PythonExecutionFile(
            filename="b.png", file_link="http://localhost/api/chat/file/b-1"
        ),
    ]
    message = "![b.png](file_link)"

    result = _rewrite_placeholder_file_links(message, files)

    assert result == "![b.png](http://localhost/api/chat/file/b-1)"


def test_placeholder_rewrite_ignores_prose_mention() -> None:
    """The word file_link in prose is not rewritten; only link targets are."""
    files = [
        PythonExecutionFile(
            filename="chart.png",
            file_link="http://localhost/api/chat/file/abc-123",
        )
    ]
    message = "The file_link field points to chart.png."

    result = _rewrite_placeholder_file_links(message, files)

    assert result == message


def test_extracts_referenced_file() -> None:
    file_id = "abc-123-def"
    files = [
        PythonExecutionFile(
            filename="chart.png",
            file_link=f"http://localhost/api/chat/file/{file_id}",
        )
    ]
    tool_call = _make_tool_call_info(generated_files=files)
    message = (
        f"Here is the chart: [chart.png](http://localhost/api/chat/file/{file_id})"
    )

    result = _extract_referenced_file_descriptors([tool_call], message)

    assert len(result) == 1
    assert result[0]["id"] == file_id
    assert result[0]["type"] == ChatFileType.IMAGE
    assert result[0]["name"] == "chart.png"


def test_filters_unreferenced_files() -> None:
    referenced_id = "ref-111"
    unreferenced_id = "unref-222"
    files = [
        PythonExecutionFile(
            filename="chart.png",
            file_link=f"http://localhost/api/chat/file/{referenced_id}",
        ),
        PythonExecutionFile(
            filename="data.csv",
            file_link=f"http://localhost/api/chat/file/{unreferenced_id}",
        ),
    ]
    tool_call = _make_tool_call_info(generated_files=files)
    message = f"Here is the chart: [chart.png](http://localhost/api/chat/file/{referenced_id})"

    result = _extract_referenced_file_descriptors([tool_call], message)

    assert len(result) == 1
    assert result[0]["id"] == referenced_id
    assert result[0]["name"] == "chart.png"


def test_extracts_from_multiple_tool_calls() -> None:
    id_1 = "file-aaa"
    id_2 = "file-bbb"
    tc1 = _make_tool_call_info(
        generated_files=[
            PythonExecutionFile(
                filename="plot.png",
                file_link=f"http://localhost/api/chat/file/{id_1}",
            )
        ]
    )
    tc2 = _make_tool_call_info(
        generated_files=[
            PythonExecutionFile(
                filename="report.csv",
                file_link=f"http://localhost/api/chat/file/{id_2}",
            )
        ]
    )
    message = f"[plot.png](http://localhost/api/chat/file/{id_1}) and [report.csv](http://localhost/api/chat/file/{id_2})"

    result = _extract_referenced_file_descriptors([tc1, tc2], message)

    assert len(result) == 2
    ids = {d["id"] for d in result}
    assert ids == {id_1, id_2}


def test_csv_file_type() -> None:
    file_id = "csv-123"
    files = [
        PythonExecutionFile(
            filename="data.csv",
            file_link=f"http://localhost/api/chat/file/{file_id}",
        )
    ]
    tool_call = _make_tool_call_info(generated_files=files)
    message = f"[data.csv](http://localhost/api/chat/file/{file_id})"

    result = _extract_referenced_file_descriptors([tool_call], message)

    assert len(result) == 1
    assert result[0]["type"] == ChatFileType.TABULAR


def test_unknown_extension_defaults_to_plain_text() -> None:
    file_id = "bin-456"
    files = [
        PythonExecutionFile(
            filename="output.xyz",
            file_link=f"http://localhost/api/chat/file/{file_id}",
        )
    ]
    tool_call = _make_tool_call_info(generated_files=files)
    message = f"[output.xyz](http://localhost/api/chat/file/{file_id})"

    result = _extract_referenced_file_descriptors([tool_call], message)

    assert len(result) == 1
    assert result[0]["type"] == ChatFileType.PLAIN_TEXT


def test_skips_tool_calls_without_generated_files() -> None:
    file_id = "img-789"
    tc_no_files = _make_tool_call_info(generated_files=None)
    tc_empty = _make_tool_call_info(generated_files=[])
    tc_with_files = _make_tool_call_info(
        generated_files=[
            PythonExecutionFile(
                filename="result.png",
                file_link=f"http://localhost/api/chat/file/{file_id}",
            )
        ]
    )
    message = f"[result.png](http://localhost/api/chat/file/{file_id})"

    result = _extract_referenced_file_descriptors(
        [tc_no_files, tc_empty, tc_with_files], message
    )

    assert len(result) == 1
    assert result[0]["id"] == file_id


# ---- save_chat_turn sanitization test ----


def test_save_chat_turn_sanitizes_message_and_reasoning(
    monkeypatch: MonkeyPatch,
) -> None:
    mock_tokenizer = MagicMock()
    mock_tokenizer.encode.return_value = [1, 2, 3]
    monkeypatch.setattr(save_chat, "get_tokenizer", lambda *_a, **_kw: mock_tokenizer)

    mock_msg = MagicMock()
    mock_msg.id = 1
    mock_msg.chat_session_id = "test"
    mock_msg.files = None

    mock_session = MagicMock()

    save_chat.save_chat_turn(
        message_text="hello\x00world\ud800",
        reasoning_tokens="think\x00ing\udfff",
        tool_calls=[],
        citation_to_doc={},
        all_search_docs={},
        db_session=mock_session,
        assistant_message=mock_msg,
    )

    assert mock_msg.message == "helloworld"
    assert mock_msg.reasoning_tokens == "thinking"


def test_save_chat_turn_rewrites_placeholder_and_attaches_file(
    monkeypatch: MonkeyPatch,
) -> None:
    mock_tokenizer = MagicMock()
    mock_tokenizer.encode.return_value = [1, 2, 3]
    monkeypatch.setattr(save_chat, "get_tokenizer", lambda *_a, **_kw: mock_tokenizer)

    mock_msg = MagicMock()
    mock_msg.id = 1
    mock_msg.chat_session_id = "test"
    mock_msg.files = None

    mock_session = MagicMock()

    tool_call = _make_tool_call_info(
        generated_files=[
            PythonExecutionFile(
                filename="chart.png",
                file_link="http://localhost/api/chat/file/abc-123",
            )
        ]
    )

    save_chat.save_chat_turn(
        message_text="Here is the chart: ![chart.png](file_link)",
        reasoning_tokens=None,
        tool_calls=[tool_call],
        citation_to_doc={},
        all_search_docs={},
        db_session=mock_session,
        assistant_message=mock_msg,
    )

    assert (
        mock_msg.message
        == "Here is the chart: ![chart.png](http://localhost/api/chat/file/abc-123)"
    )
    assert mock_msg.files is not None
    assert len(mock_msg.files) == 1
    assert mock_msg.files[0]["id"] == "abc-123"
    assert mock_msg.files[0]["name"] == "chart.png"
