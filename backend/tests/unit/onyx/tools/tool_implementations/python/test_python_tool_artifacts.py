"""Unit tests for PythonTool artifact persistence.

The sandbox is wiped between executions, so PythonTool keeps the bytes of
generated files and re-stages them as inputs on later run() calls. These tests
verify that re-staging, newest-wins dedupe by filename, and that artifacts
override user files with the same name.
"""

import json
from unittest.mock import MagicMock, patch

import pytest

from onyx.tools.models import ChatFile, PythonToolOverrideKwargs, PythonToolRichResponse
from onyx.tools.tool_implementations.python.code_interpreter_client import (
    StreamResultEvent,
    WorkspaceFile,
)
from onyx.tools.tool_implementations.python.python_tool import (
    PythonTool,
    _combine_staging_inputs,
)

TOOL_MODULE = "onyx.tools.tool_implementations.python.python_tool"


@pytest.fixture(autouse=True)
def _no_vision_model(monkeypatch: pytest.MonkeyPatch) -> None:
    """Image annotation needs a vision model and the settings store; keep
    these tests hermetic by configuring no vision model."""
    monkeypatch.setattr(
        f"{TOOL_MODULE}.get_tool_vision_llm", lambda: None, raising=True
    )


def _stream_result_with_file(path: str, file_id: str) -> StreamResultEvent:
    return StreamResultEvent(
        exit_code=0,
        timed_out=False,
        duration_ms=10,
        files=[WorkspaceFile(path=path, kind="file", file_id=file_id)],
    )


def _make_tool() -> PythonTool:
    emitter = MagicMock()
    return PythonTool(tool_id=1, emitter=emitter)


def _make_client_ctx(client: MagicMock) -> MagicMock:
    ctx = MagicMock()
    ctx.__enter__ = MagicMock(return_value=client)
    ctx.__exit__ = MagicMock(return_value=False)
    return ctx


def _run_tool(
    tool: PythonTool,
    client: MagicMock,
    files: list[ChatFile],
    code: str = "print('hi')",
) -> None:
    """Call tool.run() with mocked CodeInterpreterClient and file store."""
    from onyx.server.query_and_chat.placement import Placement

    file_store = MagicMock()
    file_store.save_file.return_value = "onyx-file-id"

    with (
        patch(
            f"{TOOL_MODULE}.CodeInterpreterClient",
            return_value=_make_client_ctx(client),
        ),
        patch(f"{TOOL_MODULE}.get_default_file_store", return_value=file_store),
    ):
        tool.run(
            placement=Placement(turn_index=0, tab_index=0),
            override_kwargs=PythonToolOverrideKwargs(chat_files=files),
            code=code,
        )


def _staged_paths(client: MagicMock) -> list[str]:
    _, kwargs = client.execute_streaming.call_args
    return [f["path"] for f in kwargs.get("files") or []]


# ---------------------------------------------------------------------------
# Pure helper: user files plus artifacts, artifacts win on name collision
# ---------------------------------------------------------------------------


def test_combine_staging_inputs_appends_artifacts() -> None:
    inputs = _combine_staging_inputs(
        [ChatFile(filename="in.csv", content=b"a")],
        {"chart.png": b"img"},
    )

    assert [(f.filename, f.content) for f in inputs] == [
        ("in.csv", b"a"),
        ("chart.png", b"img"),
    ]


def test_combine_staging_inputs_artifact_overrides_user_file() -> None:
    inputs = _combine_staging_inputs(
        [ChatFile(filename="data.csv", content=b"user-bytes")],
        {"data.csv": b"generated-bytes"},
    )

    assert len(inputs) == 1
    assert inputs[0].content == b"generated-bytes"


# ---------------------------------------------------------------------------
# End to end through run(): artifacts from call N stage into call N+1
# ---------------------------------------------------------------------------


@patch(f"{TOOL_MODULE}.CODE_INTERPRETER_BASE_URL", "http://fake:8000")
def test_generated_artifact_restaged_on_next_run() -> None:
    tool = _make_tool()
    client = MagicMock()
    client.upload_file.return_value = "ci-upload-id"
    client.download_file.return_value = b"png-bytes"
    client.execute_streaming.return_value = iter(
        [_stream_result_with_file("chart.png", "ci-generated-id")]
    )

    _run_tool(tool, client, [])

    # Second run with no chat files: the artifact from run 1 must be staged.
    client.execute_streaming.return_value = iter(
        [_stream_result_with_file("other.png", "ci-2")]
    )
    _run_tool(tool, client, [])

    assert _staged_paths(client) == ["chart.png"]


@patch(f"{TOOL_MODULE}.CODE_INTERPRETER_BASE_URL", "http://fake:8000")
def test_newest_artifact_wins_on_same_filename() -> None:
    tool = _make_tool()
    client = MagicMock()
    client.download_file.return_value = b"v1"
    client.upload_file.side_effect = ["id-v1", "id-v2"]

    client.execute_streaming.return_value = iter(
        [_stream_result_with_file("chart.png", "ci-1")]
    )
    _run_tool(tool, client, [])

    client.download_file.return_value = b"v2"
    client.execute_streaming.return_value = iter(
        [_stream_result_with_file("chart.png", "ci-2")]
    )
    _run_tool(tool, client, [])

    # Third run: only one chart.png staged, backed by the newest upload.
    client.execute_streaming.return_value = iter(
        [_stream_result_with_file("chart.png", "ci-3")]
    )
    _run_tool(tool, client, [])

    assert _staged_paths(client) == ["chart.png"]
    _, kwargs = client.execute_streaming.call_args
    assert kwargs["files"][0]["file_id"] == "id-v2"


@patch(f"{TOOL_MODULE}.CODE_INTERPRETER_BASE_URL", "http://fake:8000")
def test_artifact_with_user_file_name_replaces_it_in_staging() -> None:
    tool = _make_tool()
    client = MagicMock()
    client.download_file.return_value = b"generated"
    client.upload_file.side_effect = ["id-user", "id-generated"]

    client.execute_streaming.return_value = iter(
        [_stream_result_with_file("data.csv", "ci-1")]
    )
    _run_tool(tool, client, [ChatFile(filename="data.csv", content=b"user")])

    # Second run: the user file is replaced by the artifact of the same name.
    client.execute_streaming.return_value = iter(
        [_stream_result_with_file("data.csv", "ci-2")]
    )
    _run_tool(tool, client, [ChatFile(filename="data.csv", content=b"user")])

    assert _staged_paths(client) == ["data.csv"]
    _, kwargs = client.execute_streaming.call_args
    assert kwargs["files"][0]["file_id"] == "id-generated"


@patch(f"{TOOL_MODULE}.CODE_INTERPRETER_BASE_URL", "http://fake:8000")
def test_files_notice_set_when_files_generated() -> None:
    tool = _make_tool()
    client = MagicMock()
    client.download_file.return_value = b"png-bytes"
    client.execute_streaming.return_value = iter(
        [_stream_result_with_file("chart.png", "ci-1")]
    )

    from onyx.server.query_and_chat.placement import Placement

    file_store = MagicMock()
    file_store.save_file.return_value = "onyx-file-id"

    with (
        patch(
            f"{TOOL_MODULE}.CodeInterpreterClient",
            return_value=_make_client_ctx(client),
        ),
        patch(f"{TOOL_MODULE}.get_default_file_store", return_value=file_store),
    ):
        response = tool.run(
            placement=Placement(turn_index=0, tab_index=0),
            override_kwargs=PythonToolOverrideKwargs(chat_files=[]),
            code="print('hi')",
        )

    result = json.loads(response.llm_facing_response)
    assert result["files_notice"] is not None
    assert "chart" not in result["files_notice"]  # static note, not per-file


@patch(f"{TOOL_MODULE}.CODE_INTERPRETER_BASE_URL", "http://fake:8000")
def test_files_notice_absent_without_generated_files() -> None:
    tool = _make_tool()
    client = MagicMock()
    client.execute_streaming.return_value = iter(
        [StreamResultEvent(exit_code=0, timed_out=False, duration_ms=10, files=[])]
    )

    from onyx.server.query_and_chat.placement import Placement

    with patch(
        f"{TOOL_MODULE}.CodeInterpreterClient", return_value=_make_client_ctx(client)
    ):
        response = tool.run(
            placement=Placement(turn_index=0, tab_index=0),
            override_kwargs=PythonToolOverrideKwargs(chat_files=[]),
            code="print('hi')",
        )

    result = json.loads(response.llm_facing_response)
    assert result["files_notice"] is None


# ---------------------------------------------------------------------------
# Image annotation: generated images get captions + replay bytes
# ---------------------------------------------------------------------------


@patch(f"{TOOL_MODULE}.CODE_INTERPRETER_BASE_URL", "http://fake:8000")
def test_generated_image_is_annotated_and_replayed() -> None:
    tool = _make_tool()
    client = MagicMock()
    client.download_file.return_value = b"png-bytes"
    client.execute_streaming.return_value = iter(
        [
            StreamResultEvent(
                exit_code=0,
                timed_out=False,
                duration_ms=10,
                files=[
                    WorkspaceFile(path="chart.png", kind="file", file_id="ci-1"),
                    WorkspaceFile(path="out.csv", kind="file", file_id="ci-2"),
                ],
            )
        ]
    )

    from onyx.server.query_and_chat.placement import Placement

    file_store = MagicMock()
    file_store.save_file.return_value = "onyx-file-id"

    with (
        patch(
            f"{TOOL_MODULE}.CodeInterpreterClient",
            return_value=_make_client_ctx(client),
        ),
        patch(f"{TOOL_MODULE}.get_default_file_store", return_value=file_store),
        patch(
            f"{TOOL_MODULE}.get_tool_vision_llm", return_value=MagicMock()
        ) as mock_vision_llm,
        patch(
            f"{TOOL_MODULE}.annotate_images_in_parallel",
            return_value=["a bar chart of revenue"],
        ) as mock_annotate,
    ):
        response = tool.run(
            placement=Placement(turn_index=0, tab_index=0),
            override_kwargs=PythonToolOverrideKwargs(chat_files=[]),
            code="print('hi')",
        )

    # Only the image was annotated
    assert mock_vision_llm.called
    annotated_names = [name for name, _ in mock_annotate.call_args[0][1]]
    assert annotated_names == ["chart.png"]

    result = json.loads(response.llm_facing_response)
    captions = {f["filename"]: f["image_caption"] for f in result["generated_files"]}
    assert captions == {"chart.png": "a bar chart of revenue", "out.csv": None}

    # Replay bytes are carried for vision-capable chat models, image only
    rich = response.rich_response
    assert isinstance(rich, PythonToolRichResponse)
    assert len(rich.tool_images) == 1
    assert rich.tool_images[0].filename == "chart.png"
    assert rich.tool_images[0].file_id == "onyx-file-id"
    assert rich.tool_images[0].content == b"png-bytes"


@patch(f"{TOOL_MODULE}.CODE_INTERPRETER_BASE_URL", "http://fake:8000")
def test_no_vision_model_skips_annotation_but_keeps_replay_bytes() -> None:
    tool = _make_tool()
    client = MagicMock()
    client.download_file.return_value = b"png-bytes"
    client.execute_streaming.return_value = iter(
        [_stream_result_with_file("chart.png", "ci-1")]
    )

    from onyx.server.query_and_chat.placement import Placement

    file_store = MagicMock()
    file_store.save_file.return_value = "onyx-file-id"

    with (
        patch(
            f"{TOOL_MODULE}.CodeInterpreterClient",
            return_value=_make_client_ctx(client),
        ),
        patch(f"{TOOL_MODULE}.get_default_file_store", return_value=file_store),
    ):
        response = tool.run(
            placement=Placement(turn_index=0, tab_index=0),
            override_kwargs=PythonToolOverrideKwargs(chat_files=[]),
            code="print('hi')",
        )

    result = json.loads(response.llm_facing_response)
    assert result["generated_files"][0]["image_caption"] is None

    rich = response.rich_response
    assert isinstance(rich, PythonToolRichResponse)
    assert rich.tool_images[0].content == b"png-bytes"
