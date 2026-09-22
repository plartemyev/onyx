"""Route-level tests for the session API using a stub executor.

The routes are thin adapters over the executor; these tests verify HTTP
behavior (status codes, error mapping, request validation, workspace file
reporting) without spawning real containers. Docker-backed lifecycle
coverage lives in test_executor_docker_sessions.py.
"""

from collections.abc import Iterator

import pytest
from fastapi.testclient import TestClient

from app.main import app
from app.services import executor_base
from app.services.executor_base import (
    EntryKind,
    ExecutionResult,
    SessionInfo,
    SessionNotFoundError,
    StreamChunk,
    StreamResult,
    WorkspaceEntry,
)


class _StubExecutor(executor_base.BaseExecutor):
    """Executor double that records calls and returns canned results."""

    def __init__(self) -> None:
        self.created: list[dict] = []
        self.kept_alive: list[tuple[str, int]] = []
        self.executed: list[tuple[str, str]] = []
        self.staged: list[tuple[str, list]] = []
        self.deleted: list[str] = []
        self.fail_next_python: Exception | None = None

    def execute_python(self, **kwargs: object) -> ExecutionResult:
        raise NotImplementedError

    def create_session(self, **kwargs: object) -> SessionInfo:
        self.created.append(kwargs)
        return SessionInfo(session_id="code-session-stub", expires_at=1234.5)

    def delete_session(self, session_id: str) -> bool:
        self.deleted.append(session_id)
        return True

    def execute_bash_in_session(self, session_id: str, **kwargs: object) -> ExecutionResult:
        raise NotImplementedError

    def execute_python_in_session(
        self, session_id: str, *, code: str, **kwargs: object
    ) -> ExecutionResult:
        if self.fail_next_python is not None:
            exc, self.fail_next_python = self.fail_next_python, None
            raise exc
        self.executed.append((session_id, code))
        return ExecutionResult(
            stdout="hello",
            stderr="",
            exit_code=0,
            timed_out=False,
            duration_ms=5,
            files=(WorkspaceEntry(path="out.bin", kind=EntryKind.FILE, content=b"data"),),
        )

    def execute_python_in_session_streaming(
        self, session_id: str, *, code: str, **kwargs: object
    ) -> Iterator[executor_base.StreamEvent]:
        yield StreamChunk(stream="stdout", data="hello")
        yield StreamResult(
            exit_code=0,
            timed_out=False,
            duration_ms=5,
            files=(WorkspaceEntry(path="out.bin", kind=EntryKind.FILE, content=b"data"),),
        )

    def keepalive_session(self, session_id: str, *, ttl_seconds: int) -> SessionInfo:
        self.kept_alive.append((session_id, ttl_seconds))
        return SessionInfo(session_id=session_id, expires_at=9999.0)

    def stage_files_in_session(self, session_id: str, files: list) -> None:
        self.staged.append((session_id, files))

    def list_session_files(self, session_id: str) -> tuple[WorkspaceEntry, ...]:
        return (
            WorkspaceEntry(path="out.bin", kind=EntryKind.FILE, content=None),
            WorkspaceEntry(path=".venv", kind=EntryKind.DIRECTORY, content=None),
        )

    def read_session_file(self, session_id: str, path: str) -> bytes:
        if path == "missing.txt":
            raise LookupError(f"File not found in session workspace: {path}")
        return b"data"


@pytest.fixture
def client(monkeypatch: pytest.MonkeyPatch) -> TestClient:
    stub = _StubExecutor()
    monkeypatch.setattr("app.api.routes.get_executor", lambda: stub)
    # Expose the stub for assertions.
    app.state.stub_executor = stub
    return TestClient(app)


def test_create_session_passes_new_options(client: TestClient) -> None:
    response = client.post(
        "/v1/sessions",
        json={"ttl_seconds": 600, "network_enabled": False, "install_venv": False},
    )
    assert response.status_code == 201
    body = response.json()
    assert body["session_id"] == "code-session-stub"

    stub: _StubExecutor = app.state.stub_executor
    assert stub.created[0]["network_enabled"] is False
    assert stub.created[0]["install_venv"] is False


def test_session_python_round_trip(client: TestClient) -> None:
    response = client.post(
        "/v1/sessions/code-session-stub/python",
        json={"code": "print('hi')", "timeout_ms": 1000},
    )
    assert response.status_code == 200
    body = response.json()
    assert body["stdout"] == "hello"
    assert body["exit_code"] == 0
    assert body["files"][0]["path"] == "out.bin"


def test_session_python_stream_emits_sse(client: TestClient) -> None:
    with client.stream(
        "POST",
        "/v1/sessions/code-session-stub/python/stream",
        json={"code": "print('hi')", "timeout_ms": 1000},
    ) as response:
        assert response.status_code == 200
        body = b"".join(response.iter_raw()).decode()
    assert "event: output" in body
    assert "event: result" in body


def test_session_python_missing_session_maps_to_404(client: TestClient) -> None:
    stub: _StubExecutor = app.state.stub_executor
    stub.fail_next_python = SessionNotFoundError("code-session-gone")
    response = client.post(
        "/v1/sessions/code-session-gone/python",
        json={"code": "print('hi')"},
    )
    assert response.status_code == 404


def test_session_python_timeout_validation(client: TestClient) -> None:
    response = client.post(
        "/v1/sessions/code-session-stub/python",
        json={"code": "print('hi')", "timeout_ms": 10_000_000},
    )
    assert response.status_code == 422


def test_keepalive(client: TestClient) -> None:
    response = client.post(
        "/v1/sessions/code-session-stub/keepalive",
        json={"ttl_seconds": 1200},
    )
    assert response.status_code == 200
    assert response.json()["expires_at"] == 9999.0
    stub: _StubExecutor = app.state.stub_executor
    assert stub.kept_alive == [("code-session-stub", 1200)]


def test_list_files_excludes_nothing_at_route_level(client: TestClient) -> None:
    """The route reports whatever the executor returns; venv exclusion is the
    executor's job (verified against real tar output in the docker tests)."""
    response = client.get("/v1/sessions/code-session-stub/files")
    assert response.status_code == 200
    paths = [entry["path"] for entry in response.json()["files"]]
    assert "out.bin" in paths


def test_download_file_and_404(client: TestClient) -> None:
    ok = client.get("/v1/sessions/code-session-stub/files/out.bin")
    assert ok.status_code == 200
    assert ok.content == b"data"

    missing = client.get("/v1/sessions/code-session-stub/files/missing.txt")
    assert missing.status_code == 404


def test_stage_file_into_session(client: TestClient) -> None:
    response = client.post(
        "/v1/sessions/code-session-stub/files",
        params={"path": "input.csv"},
        files={"file": ("input.csv", b"a,b\n1,2")},
    )
    assert response.status_code == 204
    stub: _StubExecutor = app.state.stub_executor
    assert stub.staged[0][0] == "code-session-stub"
    assert stub.staged[0][1][0][0] == "input.csv"


def test_delete_session(client: TestClient) -> None:
    response = client.delete("/v1/sessions/code-session-stub")
    assert response.status_code == 204
    stub: _StubExecutor = app.state.stub_executor
    assert stub.deleted == ["code-session-stub"]
