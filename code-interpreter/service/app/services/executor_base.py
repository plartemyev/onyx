from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Generator, Sequence
from dataclasses import dataclass
from enum import StrEnum
from typing import Literal, Protocol


def wrap_last_line_interactive(code: str) -> str:
    """
    Wrap user code to execute in last-line-interactive mode.

    This uses Python's 'single' compilation mode for the last expression only,
    which automatically prints the value to stdout, mimicking Jupyter notebook behavior.
    Only the last line is affected; earlier expressions are not printed.

    Args:
        code: The Python code to wrap

    Returns:
        Wrapped Python code that will print the last expression's value if it's a bare expression
    """
    # Escape the code string for embedding in Python source
    code_escaped = code.replace("\\", "\\\\").replace("'", "\\'")

    wrapper = f"""import ast
import sys

# User code
code = '''{code_escaped}'''

# Parse the code
tree = ast.parse(code)

# Execute all statements except the last one normally
if len(tree.body) > 0:
    for node in tree.body[:-1]:
        code_obj = compile(ast.Module(body=[node], type_ignores=[]), '<stdin>', 'exec')
        exec(code_obj)

    # For the last statement, check if it's an expression
    last_node = tree.body[-1]
    if isinstance(last_node, ast.Expr):
        # Execute in 'single' mode to print the result
        interactive = ast.Interactive(body=[last_node])
        ast.fix_missing_locations(interactive)
        code_obj = compile(interactive, '<stdin>', 'single')
        exec(code_obj)
    else:
        # Not an expression, execute normally
        code_obj = compile(ast.Module(body=[last_node], type_ignores=[]), '<stdin>', 'exec')
        exec(code_obj)
"""
    return wrapper


@dataclass(frozen=True, slots=True)
class ExecutionResult:
    stdout: str
    stderr: str
    exit_code: int | None
    timed_out: bool
    duration_ms: int
    files: tuple[WorkspaceEntry, ...]


class EntryKind(StrEnum):
    FILE = "file"
    DIRECTORY = "directory"


@dataclass(frozen=True, slots=True)
class WorkspaceEntry:
    path: str
    kind: EntryKind
    content: bytes | None = None


@dataclass(frozen=True, slots=True)
class StreamChunk:
    """A chunk of output from the execution."""

    stream: Literal["stdout", "stderr"]
    data: str


@dataclass(frozen=True, slots=True)
class StreamResult:
    """Final execution result emitted at end of stream."""

    exit_code: int | None
    timed_out: bool
    duration_ms: int
    files: tuple[WorkspaceEntry, ...]


StreamEvent = StreamChunk | StreamResult


@dataclass(frozen=True, slots=True)
class HealthCheck:
    """Result of an executor health check."""

    status: Literal["ok", "error"]
    message: str | None = None


@dataclass(frozen=True, slots=True)
class SessionInfo:
    """Identifying information for a long-lived session."""

    session_id: str
    expires_at: float


SESSION_NAME_PREFIX = "code-session-"
SESSION_APP_LABEL = "code-interpreter"
SESSION_COMPONENT_LABEL = "session"
SESSION_EXPIRES_AT_KEY = "code-interpreter.expires-at"
# Written inside the session workspace; keepalives update it so expiry
# survives service restarts (docker labels are immutable at create time).
SESSION_EXPIRY_FILE = ".onyx-session-expires"
# Per-session virtualenv created in the workspace. Excluded from snapshots.
SESSION_VENV_DIR = ".venv"
# Java's per-user fontconfig cache, created under the workspace home the
# first time a JVM tool (e.g. PlantUML) runs. Control data, not an artifact,
# and its cache files are mode-600, which would fail a root-run snapshot tar.
JAVA_CONTROL_DIR = ".java"
# Control files that must never surface as workspace artifacts.
SESSION_CONTROL_EXCLUDES = (
    SESSION_EXPIRY_FILE,
    SESSION_VENV_DIR,
    "__main__.py",
    JAVA_CONTROL_DIR,
)


class SessionNotFoundError(LookupError):
    """Raised when a session ID does not refer to an existing session."""

    def __init__(self, session_id: str) -> None:
        super().__init__(f"Session '{session_id}' not found")
        self.session_id = session_id


class ExecutorProtocol(Protocol):
    def execute_python(
        self,
        *,
        code: str,
        stdin: str | None,
        timeout_ms: int,
        max_output_bytes: int,
        cpu_time_limit_sec: int | None = None,
        memory_limit_mb: int | None = None,
        files: Sequence[tuple[str, bytes]] | None = None,
        last_line_interactive: bool = True,
    ) -> ExecutionResult: ...


class BaseExecutor(ABC):
    def check_health(self) -> HealthCheck:
        """Check if the executor backend is operational.

        Default implementation returns ok. Override for backend-specific checks.
        """
        return HealthCheck(status="ok")

    @abstractmethod
    def execute_python(
        self,
        *,
        code: str,
        stdin: str | None,
        timeout_ms: int,
        max_output_bytes: int,
        cpu_time_limit_sec: int | None = None,
        memory_limit_mb: int | None = None,
        files: Sequence[tuple[str, bytes]] | None = None,
        last_line_interactive: bool = True,
    ) -> ExecutionResult:
        """Execute Python code in an isolated environment.

        Args:
            last_line_interactive: If True, the last line will print its value to stdout
                                   if it's a bare expression (only the last line is affected).
        """

    def execute_python_streaming(
        self,
        *,
        code: str,
        stdin: str | None,
        timeout_ms: int,
        max_output_bytes: int,
        cpu_time_limit_sec: int | None = None,
        memory_limit_mb: int | None = None,
        files: Sequence[tuple[str, bytes]] | None = None,
        last_line_interactive: bool = True,
    ) -> Generator[StreamEvent, None, None]:
        """Execute Python code and yield output chunks as they arrive.

        Yields StreamChunk events during execution, then a single StreamResult
        at the end. Default implementation raises NotImplementedError.
        """
        raise NotImplementedError(f"{type(self).__name__} does not support streaming execution")

    def create_session(
        self,
        *,
        ttl_seconds: int,
        files: Sequence[tuple[str, bytes]] | None = None,
        cpu_time_limit_sec: int | None = None,
        memory_limit_mb: int | None = None,
        network_enabled: bool | None = None,
        install_venv: bool = True,
    ) -> SessionInfo:
        """Create a long-lived execution environment.

        Returns identifying information for the session. The session is
        guaranteed to be torn down at or before ``expires_at`` even if this
        process crashes. ``network_enabled`` explicitly requests (True) or
        refuses (False) network access; None defers to the deployment
        default. ``install_venv`` requests a per-session virtualenv when the
        backend supports one.
        """
        raise NotImplementedError(f"{type(self).__name__} does not support sessions")

    def delete_session(self, session_id: str) -> bool:
        """Tear down a session by ID. Returns True if found and deleted."""
        raise NotImplementedError(f"{type(self).__name__} does not support sessions")

    def reap_expired_sessions(self) -> int:
        """Delete sessions whose TTL has elapsed. Returns number reaped."""
        return 0

    def execute_bash_in_session(
        self,
        session_id: str,
        *,
        cmd: str,
        timeout_ms: int,
        max_output_bytes: int,
    ) -> ExecutionResult:
        """Run a bash command inside an existing session.

        Raises ``SessionNotFoundError`` when the session does not exist.
        Network restrictions established at session creation remain in force.
        """
        raise NotImplementedError(f"{type(self).__name__} does not support sessions")

    def execute_python_in_session(
        self,
        session_id: str,
        *,
        code: str,
        stdin: str | None,
        timeout_ms: int,
        max_output_bytes: int,
        last_line_interactive: bool = True,
        files: Sequence[tuple[str, bytes]] | None = None,
    ) -> ExecutionResult:
        """Execute Python code inside an existing long-lived session.

        The session workspace persists across calls: files written by earlier
        calls (and packages installed into the session venv) stay available.
        Returns the usual result plus new/modified workspace files.

        Raises ``SessionNotFoundError`` when the session does not exist.
        """
        raise NotImplementedError(
            f"{type(self).__name__} does not support Python execution in sessions"
        )

    def execute_python_in_session_streaming(
        self,
        session_id: str,
        *,
        code: str,
        stdin: str | None,
        timeout_ms: int,
        max_output_bytes: int,
        last_line_interactive: bool = True,
        files: Sequence[tuple[str, bytes]] | None = None,
    ) -> Generator[StreamEvent, None, None]:
        """Streaming variant of :meth:`execute_python_in_session`."""
        raise NotImplementedError(
            f"{type(self).__name__} does not support streaming Python execution in sessions"
        )

    def keepalive_session(self, session_id: str, *, ttl_seconds: int) -> SessionInfo:
        """Extend a session's expiry to now + ``ttl_seconds``.

        Raises ``SessionNotFoundError`` when the session does not exist.
        """
        raise NotImplementedError(f"{type(self).__name__} does not support sessions")

    def stage_files_in_session(self, session_id: str, files: Sequence[tuple[str, bytes]]) -> None:
        """Stage additional files into an existing session's workspace.

        Raises ``SessionNotFoundError`` when the session does not exist.
        """
        raise NotImplementedError(f"{type(self).__name__} does not support sessions")

    def list_session_files(self, session_id: str) -> tuple[WorkspaceEntry, ...]:
        """List workspace entries of a session (metadata only, no content).

        Venv and control files are excluded. Raises
        ``SessionNotFoundError`` when the session does not exist.
        """
        raise NotImplementedError(f"{type(self).__name__} does not support sessions")

    def read_session_file(self, session_id: str, path: str) -> bytes:
        """Read a single file from a session's workspace.

        Raises ``SessionNotFoundError`` when the session does not exist and
        ``LookupError`` when the path does not exist in the workspace.
        """
        raise NotImplementedError(f"{type(self).__name__} does not support sessions")

    @staticmethod
    def truncate_output(stream: bytes, max_bytes: int) -> str:
        if len(stream) <= max_bytes:
            return stream.decode("utf-8", errors="replace")
        head = stream[: max(0, max_bytes - 32)]
        suffix = b"\n...[truncated]"
        return (head + suffix).decode("utf-8", errors="replace")
