"""External-dependency tests for PythonTool persistent sandbox sessions.

Runs against a live code-interpreter service (>= 0.5.0) and Redis:
- workspace files written in one run_python call are readable in the next;
- packages installed (pip) in one call are importable in the next;
- the chat -> sandbox session mapping survives in Redis across tool
  instances (as it must across chat turns);
- the session is recreated when the mapped one disappears.

This is the multi-step workflow that failed in chat session
4dc4f5a9-b740-42ae-b85f-d1ae11672e0b (file written in one call was gone in
the next).
"""

from queue import Queue
from uuid import uuid4

import pytest

from onyx.chat.emitter import Emitter
from onyx.server.query_and_chat.placement import Placement
from onyx.tools.models import (
    LlmPythonExecutionResult,
    PythonToolOverrideKwargs,
    ToolResponse,
)
from onyx.tools.tool_implementations.python.python_tool import PythonTool
from onyx.tools.tool_implementations.python.session_store import (
    delete_ci_session_id,
    fetch_ci_session_id,
)


class _StubFileStore:
    """In-memory stand-in for the Onyx file store.

    The session tests focus on sandbox session behavior (persistence, dedup
    baseline, mapping), not object storage; the real store would drag in
    Postgres/MinIO setup that is irrelevant here.
    """

    def save_file(self, content: object, **kwargs: object) -> str:  # noqa: ARG002
        return f"stub-file-{uuid4()}"


@pytest.fixture(autouse=True)
def _stub_file_store(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        "onyx.tools.tool_implementations.python.python_tool.get_default_file_store",
        lambda: _StubFileStore(),
    )


@pytest.fixture
def placement() -> Placement:
    return Placement(turn_index=0)


def _make_tool() -> PythonTool:
    """A fresh tool instance, as each chat turn constructs one."""
    return PythonTool(
        tool_id=1,
        emitter=Emitter(merged_queue=Queue()),
        chat_session_id=f"test-ci-session-{uuid4()}",
    )


def _run_tool(
    tool: PythonTool, placement: Placement, code: str
) -> LlmPythonExecutionResult:
    response: ToolResponse = tool.run(
        placement=placement,
        override_kwargs=PythonToolOverrideKwargs(chat_files=[]),
        code=code,
    )
    return LlmPythonExecutionResult.model_validate_json(response.llm_facing_response)


def _skip_if_no_sessions() -> None:
    from onyx.tools.tool_implementations.python.code_interpreter_client import (
        CodeInterpreterClient,
    )

    with CodeInterpreterClient() as client:
        if not client.supports(client.execute_python_in_session_streaming):
            pytest.skip("code-interpreter service does not support sessions (>= 0.5.0)")


def _teardown_sandbox(chat_session_id: str) -> None:
    """Best-effort sandbox deletion so tests do not wait out the TTL."""
    from onyx.tools.tool_implementations.python.code_interpreter_client import (
        CodeInterpreterClient,
    )

    ci_session_id = fetch_ci_session_id(chat_session_id)
    if ci_session_id is None:
        return
    try:
        with CodeInterpreterClient() as client:
            client.delete_session(ci_session_id)
    except Exception:
        pass
    delete_ci_session_id(chat_session_id)


def test_session_state_persists_across_calls_and_turns(placement: Placement) -> None:
    """Call 1 writes a file and pip-installs a package; call 2 (a fresh
    PythonTool instance, as a new chat turn would construct) reads both."""
    chat_session_id = f"test-ci-session-{uuid4()}"
    tool_one = PythonTool(
        tool_id=1,
        emitter=Emitter(merged_queue=Queue()),
        chat_session_id=chat_session_id,
    )
    _skip_if_no_sessions()

    result_one = _run_tool(
        tool_one,
        placement,
        "\n".join(
            [
                "from pathlib import Path",
                "Path('research_state.json').write_text('{\"step\": 1}')",
                "import subprocess, sys",
                "r = subprocess.run([sys.executable, '-m', 'pip', 'install', "
                "'--quiet', 'six'], capture_output=True, text=True)",
                "print('pip_rc', r.returncode)",
                "print('wrote state file')",
            ]
        ),
    )
    assert result_one.exit_code == 0, result_one.stderr
    assert "pip_rc 0" in result_one.stdout
    assert result_one.files_notice is not None

    # Session id is mapped in Redis for the chat...
    ci_session_id = fetch_ci_session_id(chat_session_id)
    assert ci_session_id is not None

    # ...and a second tool instance (next chat turn) reuses the same session.
    tool_two = PythonTool(
        tool_id=1,
        emitter=Emitter(merged_queue=Queue()),
        chat_session_id=chat_session_id,
    )
    result_two = _run_tool(
        tool_two,
        placement,
        "\n".join(
            [
                "import json",
                "from pathlib import Path",
                "state = json.loads(Path('research_state.json').read_text())",
                "print('state step', state['step'])",
                "import six",
                "print('six ok', six.__version__)",
            ]
        ),
    )
    assert result_two.exit_code == 0, result_two.stderr
    assert "state step 1" in result_two.stdout
    assert "six ok" in result_two.stdout
    assert result_two.files_notice is not None

    _teardown_sandbox(chat_session_id)


def test_session_recreated_when_mapped_session_vanishes(placement: Placement) -> None:
    """Deleting the sandbox session out from under the mapping must yield a
    fresh working session on the next call, not a failed tool call."""
    from onyx.tools.tool_implementations.python.code_interpreter_client import (
        CodeInterpreterClient,
    )

    tool = _make_tool()
    assert tool._chat_session_id is not None
    chat_session_id: str = tool._chat_session_id
    _skip_if_no_sessions()

    result_one = _run_tool(
        tool,
        placement,
        "from pathlib import Path; Path('a.txt').write_text('one'); print('ok')",
    )
    assert result_one.exit_code == 0, result_one.stderr

    ci_session_id = fetch_ci_session_id(chat_session_id)
    assert ci_session_id is not None
    with CodeInterpreterClient() as client:
        client.delete_session(ci_session_id)

    result_two = _run_tool(
        tool,
        placement,
        "from pathlib import Path; print(Path('b.txt').write_text('two'))",
    )
    assert result_two.exit_code == 0, result_two.stderr
    assert fetch_ci_session_id(chat_session_id) != ci_session_id

    _teardown_sandbox(chat_session_id)


def test_session_dedup_baseline_limits_reported_files(placement: Placement) -> None:
    """A workspace file already reported in an earlier call is not reported
    again while unchanged — the dedup baseline keeps large artifacts from
    being re-downloaded every call."""
    chat_session_id = f"test-ci-session-{uuid4()}"
    tool = PythonTool(
        tool_id=1,
        emitter=Emitter(merged_queue=Queue()),
        chat_session_id=chat_session_id,
    )
    _skip_if_no_sessions()

    result_one = _run_tool(
        tool,
        placement,
        "from pathlib import Path; Path('big.bin').write_bytes(b'x' * 1024); print('done')",
    )
    assert result_one.exit_code == 0, result_one.stderr
    reported_once = [f.filename for f in result_one.generated_files]
    assert "big.bin" in reported_once

    result_two = _run_tool(
        tool,
        placement,
        "from pathlib import Path; Path('other.bin').write_bytes(b'y'); print('done')",
    )
    assert result_two.exit_code == 0, result_two.stderr
    reported_twice = [f.filename for f in result_two.generated_files]
    assert "other.bin" in reported_twice
    assert "big.bin" not in reported_twice, (
        "unchanged workspace file should be excluded by the dedup baseline"
    )

    _teardown_sandbox(chat_session_id)
