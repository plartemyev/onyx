"""Docker-backed integration tests for persistent session execution.

These spawn real session containers on the host Docker daemon (same
sibling-container topology as the deployed service). They verify the
acceptance criteria for multi-step research:

- a file written in one in-session execution is readable in the next;
- a package installed into the session venv in one call is importable in
  the next (venv lives on the session workspace volume);
- keepalive extends expiry; file listing/read work; teardown removes the
  session and its workspace volume.
"""

import subprocess
from collections.abc import Generator

import pytest

from app.services.executor_base import SessionNotFoundError
from app.services.executor_docker import DockerExecutor

pytestmark = pytest.mark.integration

SESSION_ID_PREFIX = "code-session-"


def _docker_available() -> bool:
    try:
        result = subprocess.run(
            ["docker", "version", "--format", "{{.Server.Version}}"],
            capture_output=True,
            timeout=5,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired):
        return False
    return result.returncode == 0


@pytest.fixture
def executor() -> Generator[DockerExecutor, None, None]:
    if not _docker_available():
        pytest.skip("docker daemon not reachable from test environment")
    # Force the research executor image for venv-enabled sessions when the
    # default image does not provide uv.
    import os

    previous = os.environ.get("PYTHON_EXECUTOR_DOCKER_IMAGE")
    if previous:
        os.environ["PYTHON_EXECUTOR_DOCKER_IMAGE"] = previous
    yield DockerExecutor()


@pytest.fixture
def session(executor: DockerExecutor) -> Generator[str, None, None]:
    info = executor.create_session(ttl_seconds=900, install_venv=True)
    yield info.session_id
    executor.delete_session(info.session_id)


@pytest.mark.integration
class TestSessionLifecycle:
    def test_workspace_and_installs_persist_across_executions(
        self, executor: DockerExecutor, session: str
    ) -> None:
        result_one = executor.execute_python_in_session(
            session,
            code=(
                "from pathlib import Path\n"
                "Path('state.json').write_text('{\"step\": 1}')\n"
                "import subprocess, sys\n"
                "r = subprocess.run([sys.executable, '-m', 'pip', 'install', "
                "'--quiet', 'six'], capture_output=True, text=True)\n"
                "print('pip_rc', r.returncode)\n"
            ),
            stdin=None,
            timeout_ms=240_000,
            max_output_bytes=100_000,
        )
        assert result_one.exit_code == 0, result_one.stderr
        assert "pip_rc 0" in result_one.stdout
        # The written file is reported as a new workspace file...
        assert any(entry.path == "state.json" for entry in result_one.files)

        result_two = executor.execute_python_in_session(
            session,
            code=(
                "import json\n"
                "from pathlib import Path\n"
                "print('state', json.loads(Path('state.json').read_text())['step'])\n"
                "import six\n"
                "print('six import ok')\n"
            ),
            stdin=None,
            timeout_ms=60_000,
            max_output_bytes=100_000,
        )
        assert result_two.exit_code == 0, result_two.stderr
        assert "state 1" in result_two.stdout
        assert "six import ok" in result_two.stdout

    def test_exec_script_is_not_reported_as_workspace_file(
        self, executor: DockerExecutor, session: str
    ) -> None:
        result = executor.execute_python_in_session(
            session,
            code="print('plain')",
            stdin=None,
            timeout_ms=30_000,
            max_output_bytes=10_000,
        )
        assert result.exit_code == 0
        assert all(not entry.path.startswith(".onyx-exec-") for entry in result.files)
        assert all(entry.path != "__main__.py" for entry in result.files)

    def test_keepalive_extends_expiry(self, executor: DockerExecutor, session: str) -> None:
        info_before = executor._read_session_expiry(session, "0")
        info = executor.keepalive_session(session, ttl_seconds=1800)
        assert info.expires_at > info_before > 0

    def test_list_and_read_session_files(self, executor: DockerExecutor, session: str) -> None:
        executor.execute_python_in_session(
            session,
            code="from pathlib import Path; Path('artifact.txt').write_text('payload')",
            stdin=None,
            timeout_ms=30_000,
            max_output_bytes=10_000,
        )
        listing = executor.list_session_files(session)
        paths = [entry.path for entry in listing]
        assert "artifact.txt" in paths
        # Venv and control files never leak into listings.
        assert ".venv" not in paths
        assert ".onyx-session-expires" not in paths

        assert executor.read_session_file(session, "artifact.txt") == b"payload"

    def test_delete_removes_container_and_volume(self, executor: DockerExecutor) -> None:
        info = executor.create_session(ttl_seconds=900, install_venv=False)
        session_id = info.session_id
        volume = f"{session_id}-ws"
        assert executor.delete_session(session_id) is True

        # The container is gone (docker rm -f is idempotent for --rm
        # containers, so only inspect the actual state) and the workspace
        # volume was removed with it.
        inspect = subprocess.run(
            ["docker", "inspect", session_id], capture_output=True, check=False
        )
        assert inspect.returncode != 0
        volume_ls = subprocess.run(
            ["docker", "volume", "ls", "--format", "{{.Name}}"],
            capture_output=True,
            text=True,
            check=False,
        )
        assert volume not in volume_ls.stdout.splitlines()

    def test_unknown_session_raises(self, executor: DockerExecutor) -> None:
        with pytest.raises(SessionNotFoundError):
            executor.execute_python_in_session(
                f"{SESSION_ID_PREFIX}doesnotexist0000",
                code="print('nope')",
                stdin=None,
                timeout_ms=5_000,
                max_output_bytes=1_000,
            )
