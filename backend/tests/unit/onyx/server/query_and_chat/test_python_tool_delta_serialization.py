"""Unit tests for Python tool packet serialization.

Covers the PythonToolDelta `files` field: new packets carry filenames so the
UI can render images inline, and packets persisted before the field existed
(replayed on page reload) must still validate.
"""

from onyx.server.query_and_chat.placement import Placement
from onyx.server.query_and_chat.session_loading import create_python_tool_packets
from onyx.server.query_and_chat.streaming_models import (
    Packet,
    PythonToolDelta,
    PythonToolGeneratedFile,
)


def test_python_tool_delta_defaults_to_empty_files() -> None:
    delta = PythonToolDelta(stdout="out", stderr="", file_ids=["f1"])

    assert delta.files == []


def test_python_tool_delta_round_trips_files() -> None:
    delta = PythonToolDelta(
        stdout="out",
        stderr="",
        file_ids=["f1"],
        files=[PythonToolGeneratedFile(filename="chart.png", file_id="f1")],
    )

    parsed = PythonToolDelta.model_validate(delta.model_dump())
    assert parsed.files == [PythonToolGeneratedFile(filename="chart.png", file_id="f1")]


def test_create_python_tool_packets_without_files_still_valid() -> None:
    # Simulates replaying a stored response that predates the files field.
    packets = create_python_tool_packets(
        code="print('hi')",
        stdout="hi",
        stderr="",
        file_ids=["f1"],
        turn_index=0,
    )

    assert len(packets) == 3
    delta = packets[1].obj
    assert isinstance(delta, PythonToolDelta)
    assert delta.files == []
    assert delta.file_ids == ["f1"]


def test_create_python_tool_packets_with_files() -> None:
    packets = create_python_tool_packets(
        code="print('hi')",
        stdout="hi",
        stderr="",
        file_ids=["f1"],
        turn_index=1,
        tab_index=2,
        files=[PythonToolGeneratedFile(filename="chart.png", file_id="f1")],
    )

    delta = packets[1].obj
    assert isinstance(delta, PythonToolDelta)
    assert delta.files == [PythonToolGeneratedFile(filename="chart.png", file_id="f1")]
    assert packets[1].placement == Placement(turn_index=1, tab_index=2)


def test_packet_serialization_keeps_files_field() -> None:
    packet = Packet(
        placement=Placement(turn_index=0, tab_index=0),
        obj=PythonToolDelta(
            stdout="",
            stderr="",
            file_ids=["f1"],
            files=[PythonToolGeneratedFile(filename="a.png", file_id="f1")],
        ),
    )

    parsed = Packet.model_validate(packet.model_dump())
    obj = parsed.obj
    assert isinstance(obj, PythonToolDelta)
    assert obj.files[0].filename == "a.png"
