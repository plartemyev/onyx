"""Tests for the turn-start workspace-state note.

With a persistent sandbox, files from earlier turns survive in the workspace
but leave no trace in the conversation — a later turn re-downloads what is
already there. The note lists the top-level entries once per turn.
"""

from unittest.mock import MagicMock

import pytest

from onyx.tools.tool_implementations.python.python_tool import (
    fetch_workspace_state_note,
)

TOOL_MODULE = "onyx.tools.tool_implementations.python.python_tool"


def _with_session(
    monkeypatch: pytest.MonkeyPatch,
    paths: list[str],
) -> None:
    monkeypatch.setattr(f"{TOOL_MODULE}.fetch_ci_session_id", lambda _id: "ci-1")
    client = MagicMock()
    client.list_session_files.return_value = paths
    ctx = MagicMock()
    ctx.__enter__ = MagicMock(return_value=client)
    ctx.__exit__ = MagicMock(return_value=False)
    monkeypatch.setattr(f"{TOOL_MODULE}.CodeInterpreterClient", lambda: ctx)


def test_note_lists_top_level_entries(monkeypatch: pytest.MonkeyPatch) -> None:
    _with_session(
        monkeypatch,
        ["jdk-11.tar.gz", "work/jdk-11/bin/java", "work/jdk-11/release", "out.png"],
    )
    note = fetch_workspace_state_note("chat-1")
    assert note is not None
    # Top-level only: the extracted tree collapses to its directory.
    assert "jdk-11.tar.gz" in note
    assert "work" in note
    assert "out.png" in note
    assert "bin/java" not in note
    assert "Reuse these files" in note


def test_note_hidden_entries_are_skipped(monkeypatch: pytest.MonkeyPatch) -> None:
    _with_session(monkeypatch, [".venv/pyvenv.cfg", "data.csv"])
    note = fetch_workspace_state_note("chat-1")
    assert note is not None
    assert ".venv" not in note
    assert "data.csv" in note


def test_note_capped_entries(monkeypatch: pytest.MonkeyPatch) -> None:
    _with_session(monkeypatch, [f"file_{i}.txt" for i in range(30)])
    note = fetch_workspace_state_note("chat-1", max_entries=15)
    assert note is not None
    assert "and 15 more" in note


def test_none_without_session_mapping(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(f"{TOOL_MODULE}.fetch_ci_session_id", lambda _id: None)
    assert fetch_workspace_state_note("chat-1") is None
    assert fetch_workspace_state_note(None) is None


def test_none_when_listing_fails(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(f"{TOOL_MODULE}.fetch_ci_session_id", lambda _id: "ci-1")

    def _boom() -> None:
        raise RuntimeError("session expired")

    ctx = MagicMock()
    ctx.__enter__ = MagicMock(side_effect=_boom)
    ctx.__exit__ = MagicMock(return_value=False)
    monkeypatch.setattr(f"{TOOL_MODULE}.CodeInterpreterClient", lambda: ctx)

    assert fetch_workspace_state_note("chat-1") is None
