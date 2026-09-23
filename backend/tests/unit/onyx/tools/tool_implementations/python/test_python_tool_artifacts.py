"""Unit tests for PythonTool artifact persistence.

The sandbox is wiped between executions, so PythonTool keeps the bytes of
generated files and re-stages them as inputs on later run() calls. These tests
verify that re-staging, newest-wins dedupe by filename, and that artifacts
override user files with the same name.
"""

import json
from unittest.mock import MagicMock, patch

import pytest

from onyx.tools.models import (
    ChatFile,
    PythonToolOverrideKwargs,
    PythonToolRichResponse,
    ToolResponse,
)
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
) -> ToolResponse:
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
        return tool.run(
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


# ---------------------------------------------------------------------------
# Legacy artifact cache is bounded (oldest evicted first)
# ---------------------------------------------------------------------------


def test_artifacts_cache_records_and_overwrites() -> None:
    tool = _make_tool()
    tool._record_generated_artifact("a.csv", b"1")
    tool._record_generated_artifact("b.csv", b"2")
    # Same filename: newest content wins.
    tool._record_generated_artifact("a.csv", b"1-updated")

    assert tool._generated_artifacts == {"a.csv": b"1-updated", "b.csv": b"2"}


def test_artifacts_cache_evicts_oldest_first() -> None:
    from onyx.configs.app_configs import CODE_INTERPRETER_MAX_GENERATED_ARTIFACTS

    tool = _make_tool()
    cap = CODE_INTERPRETER_MAX_GENERATED_ARTIFACTS
    for i in range(cap + 5):
        tool._record_generated_artifact(f"file_{i}.bin", bytes([i % 256]))

    assert len(tool._generated_artifacts) == cap
    # Oldest entries (file_0..file_4) are gone; the newest survive.
    assert "file_0.bin" not in tool._generated_artifacts
    assert "file_4.bin" not in tool._generated_artifacts
    assert f"file_{cap + 4}.bin" in tool._generated_artifacts


def test_artifacts_cache_never_exceeds_cap_on_overwrite() -> None:
    from onyx.configs.app_configs import CODE_INTERPRETER_MAX_GENERATED_ARTIFACTS

    tool = _make_tool()
    cap = CODE_INTERPRETER_MAX_GENERATED_ARTIFACTS
    for i in range(cap):
        tool._record_generated_artifact(f"file_{i}.bin", b"x")
    # Re-recording an existing name must not grow the cache.
    tool._record_generated_artifact("file_0.bin", b"y")

    assert len(tool._generated_artifacts) == cap


# ---------------------------------------------------------------------------
# Generated-file cap: bulk extraction must not flood links or the LLM response
# ---------------------------------------------------------------------------


@patch(f"{TOOL_MODULE}.CODE_INTERPRETER_BASE_URL", "http://fake:8000")
def test_generated_files_capped_in_llm_response() -> None:
    from onyx.tools.tool_implementations.python.python_tool import (
        MAX_GENERATED_FILES_REGISTERED,
    )

    tool = _make_tool()
    client = MagicMock()
    client.download_file.return_value = b"bytes"
    total = MAX_GENERATED_FILES_REGISTERED + 50
    client.execute_streaming.return_value = iter(
        [
            StreamResultEvent(
                exit_code=0,
                timed_out=False,
                duration_ms=10,
                files=[
                    WorkspaceFile(path=f"extracted_{i}.h", kind="file", file_id=f"f{i}")
                    for i in range(total)
                ],
            )
        ]
    )

    response = _run_tool(tool, client, [])

    result = json.loads(response.llm_facing_response)
    # The LLM-facing list is capped and a summary notice names the rest.
    assert len(result["generated_files"]) == MAX_GENERATED_FILES_REGISTERED
    assert "50 further file(s)" in (result["files_notice"] or "")
    # Only the registered files are downloaded and saved.
    assert client.download_file.call_count == MAX_GENERATED_FILES_REGISTERED


@patch(f"{TOOL_MODULE}.CODE_INTERPRETER_BASE_URL", "http://fake:8000")
def test_generated_images_survive_the_cap() -> None:
    from onyx.tools.tool_implementations.python.python_tool import (
        MAX_GENERATED_FILES_REGISTERED,
    )

    tool = _make_tool()
    client = MagicMock()
    client.download_file.return_value = b"bytes"
    # A plot plus a bulk extraction, in extraction order.
    files = [
        WorkspaceFile(path=f"extracted_{i}.h", kind="file", file_id=f"f{i}")
        for i in range(MAX_GENERATED_FILES_REGISTERED + 20)
    ]
    files.append(WorkspaceFile(path="plot.png", kind="file", file_id="plot-1"))
    client.execute_streaming.return_value = iter(
        [StreamResultEvent(exit_code=0, timed_out=False, duration_ms=10, files=files)]
    )

    response = _run_tool(tool, client, [])

    result = json.loads(response.llm_facing_response)
    filenames = [f["filename"] for f in result["generated_files"]]
    assert len(filenames) == MAX_GENERATED_FILES_REGISTERED
    # The image is registered even though it is listed last in the workspace.
    assert "plot.png" in filenames
    assert "extracted_0.h" in filenames


@patch(f"{TOOL_MODULE}.CODE_INTERPRETER_BASE_URL", "http://fake:8000")
def test_small_file_lists_are_not_capped_or_reordered() -> None:
    tool = _make_tool()
    client = MagicMock()
    client.download_file.return_value = b"bytes"
    client.execute_streaming.return_value = iter(
        [
            StreamResultEvent(
                exit_code=0,
                timed_out=False,
                duration_ms=10,
                files=[
                    WorkspaceFile(path="b.csv", kind="file", file_id="f1"),
                    WorkspaceFile(path="a.png", kind="file", file_id="f2"),
                ],
            )
        ]
    )

    response = _run_tool(tool, client, [])

    result = json.loads(response.llm_facing_response)
    assert [f["filename"] for f in result["generated_files"]] == ["b.csv", "a.png"]
    assert result["files_notice"] is not None
    assert "further file(s)" not in result["files_notice"]
