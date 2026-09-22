import codecs
import io
import logging
import os
import selectors
import shlex
import subprocess
import tarfile
import time
import uuid
from collections.abc import Generator, Mapping, Sequence
from contextlib import contextmanager, suppress
from dataclasses import dataclass
from pathlib import Path
from shutil import which
from typing import Literal

from app.app_configs import (
    PYTHON_EXECUTOR_DOCKER_BIN,
    PYTHON_EXECUTOR_DOCKER_IMAGE,
    PYTHON_EXECUTOR_DOCKER_NETWORK,
    PYTHON_EXECUTOR_DOCKER_RUN_ARGS,
    SESSION_EXECUTOR_IMAGE,
    SESSION_MAX_LIFETIME_SEC,
    SESSION_NETWORK_MODE,
    SESSION_VENV_ENABLED,
)
from app.image_ref import normalize_image_ref
from app.services.executor_base import (
    SESSION_APP_LABEL,
    SESSION_COMPONENT_LABEL,
    SESSION_CONTROL_EXCLUDES,
    SESSION_EXPIRES_AT_KEY,
    SESSION_EXPIRY_FILE,
    SESSION_NAME_PREFIX,
    BaseExecutor,
    EntryKind,
    ExecutionResult,
    HealthCheck,
    SessionInfo,
    SessionNotFoundError,
    StreamChunk,
    StreamEvent,
    StreamResult,
    WorkspaceEntry,
    wrap_last_line_interactive,
)

logger = logging.getLogger(__name__)

EXEC_USER = "65532:65532"


def _looks_like_missing_container(stderr: bytes) -> bool:
    """Heuristic: ``docker exec`` writes these to stderr when the target is gone."""
    text = stderr.decode("utf-8", errors="replace").lower()
    return "no such container" in text or "is not running" in text


@dataclass
class _ExecContext:
    """Holds the live container and process for the duration of an execution."""

    container_name: str
    proc: subprocess.Popen[bytes]
    start: float


class DockerExecutor(BaseExecutor):
    def __init__(self) -> None:
        self.docker_binary = self._resolve_docker_binary()
        self.image = PYTHON_EXECUTOR_DOCKER_IMAGE
        self.run_args = PYTHON_EXECUTOR_DOCKER_RUN_ARGS
        # Session id -> expiry (secondary to the on-disk expiry file; survives
        # only within this service process).
        self._session_expiry: dict[str, float] = {}

    def check_health(self) -> HealthCheck:
        """Verify Docker daemon is reachable and the executor image is available."""
        # Check Docker daemon connectivity
        try:
            result = subprocess.run(
                [self.docker_binary, "version", "--format", "{{.Server.Version}}"],
                capture_output=True,
                timeout=5,
                check=False,
            )
        except FileNotFoundError:
            return HealthCheck(status="error", message="Docker binary not found")
        except subprocess.TimeoutExpired:
            return HealthCheck(status="error", message="Docker daemon not responding")

        if result.returncode != 0:
            stderr = result.stderr.decode("utf-8", errors="replace").strip()
            return HealthCheck(
                status="error",
                message=f"Docker daemon not reachable: {stderr}",
            )

        # Check executor image is available locally
        image_with_tag = normalize_image_ref(self.image)
        try:
            img_result = subprocess.run(
                [self.docker_binary, "image", "inspect", image_with_tag],
                capture_output=True,
                timeout=5,
                check=False,
            )
        except subprocess.TimeoutExpired:
            return HealthCheck(
                status="error",
                message=f"Timeout checking image {image_with_tag}",
            )

        if img_result.returncode != 0:
            return HealthCheck(
                status="error",
                message=f"Executor image {image_with_tag} not available locally",
            )

        return HealthCheck(status="ok")

    def _resolve_docker_binary(self) -> str:
        candidate = PYTHON_EXECUTOR_DOCKER_BIN
        docker_path = which(candidate)
        if docker_path is None:
            raise RuntimeError(
                "Docker CLI not found. Set PYTHON_EXECUTOR_DOCKER_BIN to the docker binary if it is"
                " installed in a non-standard location."
            )
        return docker_path

    def _kill_container(self, name: str) -> None:
        with suppress(Exception):
            subprocess.run(
                [self.docker_binary, "kill", name],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                check=False,
            )

    def _validate_relative_path(self, path_str: str) -> Path:
        path = Path(path_str)
        if path.is_absolute():
            raise ValueError("File paths must be relative.")

        sanitized_parts = []
        for part in path.parts:
            if part in ("", "."):
                continue
            if part == "..":
                raise ValueError("File paths must not contain '..'.")
            sanitized_parts.append(part)

        if not sanitized_parts:
            raise ValueError("File path must not be empty.")

        return Path(*sanitized_parts)

    def _create_tar_archive(
        self,
        code: str | None = None,
        files: Sequence[tuple[str, bytes]] | None = None,
        last_line_interactive: bool = True,
    ) -> bytes:
        """Create a tar archive optionally containing an entrypoint and files.

        Args:
            code: If provided, written as ``__main__.py`` at the archive root.
            last_line_interactive: If True and code is provided, wrap the code so
                the last line prints its value if it's a bare expression.
        """
        tar_buffer = io.BytesIO()
        with tarfile.open(fileobj=tar_buffer, mode="w") as tar:
            if code is not None:
                code_to_execute = (
                    wrap_last_line_interactive(code) if last_line_interactive else code
                )
                code_bytes = code_to_execute.encode("utf-8")
                code_info = tarfile.TarInfo(name="__main__.py")
                code_info.size = len(code_bytes)
                code_info.mode = 0o644
                tar.addfile(code_info, io.BytesIO(code_bytes))

            created_dirs: set[str] = set()
            for file_path, content in files or ():
                validated_path = self._validate_relative_path(file_path)
                if code is not None and validated_path == Path("__main__.py"):
                    raise ValueError(
                        "File path '__main__.py' is reserved for the execution entrypoint."
                    )

                parent_parts = validated_path.parts[:-1]
                for i in range(len(parent_parts)):
                    dir_path = "/".join(parent_parts[: i + 1])
                    if dir_path not in created_dirs:
                        dir_info = tarfile.TarInfo(name=dir_path + "/")
                        dir_info.type = tarfile.DIRTYPE
                        dir_info.mode = 0o755
                        tar.addfile(dir_info)
                        created_dirs.add(dir_path)

                file_info = tarfile.TarInfo(name=validated_path.as_posix())
                file_info.size = len(content)
                file_info.mode = 0o644
                tar.addfile(file_info, io.BytesIO(content))

        return tar_buffer.getvalue()

    def _extract_workspace_snapshot(
        self,
        container_name: str,
        extra_excludes: Sequence[str] | None = None,
    ) -> tuple[WorkspaceEntry, ...]:
        """Extract files from the container workspace after execution using tar."""
        try:
            excludes = [*SESSION_CONTROL_EXCLUDES, *(extra_excludes or ())]
            # Use tar to get all files from workspace (excluding control paths)
            tar_cmd = [self.docker_binary, "exec", container_name, "tar", "-c"]
            for exclude in excludes:
                tar_cmd.extend(["--exclude", exclude])
            tar_cmd.extend(["-C", "/workspace", "."])
            tar_result = subprocess.run(tar_cmd, capture_output=True, timeout=60)

            if tar_result.returncode != 0:
                return tuple()

            entries = []

            # Extract files from tar archive
            with tarfile.open(fileobj=io.BytesIO(tar_result.stdout), mode="r") as tar:
                for member in tar.getmembers():
                    # Skip the root directory
                    if member.name == ".":
                        continue

                    # Clean up the path (remove leading ./)
                    clean_path = member.name.lstrip("./")

                    if member.isdir():
                        entries.append(
                            WorkspaceEntry(path=clean_path, kind=EntryKind.DIRECTORY, content=None)
                        )
                    elif member.isfile():
                        # Extract file content
                        file_obj = tar.extractfile(member)
                        if file_obj:
                            content = file_obj.read()
                            entries.append(
                                WorkspaceEntry(
                                    path=clean_path, kind=EntryKind.FILE, content=content
                                )
                            )

            return tuple(entries)
        except (subprocess.TimeoutExpired, Exception):
            return tuple()

    def _build_run_command(
        self,
        container_name: str,
        cpu_time_limit_sec: int | None,
        memory_limit_mb: int | None,
        sleep_seconds: int,
        labels: Mapping[str, str] | None = None,
        network: str | None = None,
        volumes: Sequence[str] | None = None,
    ) -> list[str]:
        """Build a detached ``docker run`` command.

        ``sleep_seconds`` controls how long the container's idle ``sleep`` lasts;
        callers must ensure it exceeds their work duration. ``labels`` are
        attached for later filtering (e.g. by the session reaper). ``network``
        defaults to the deployment's executor network; pass ``"none"`` to
        isolate the container. ``volumes`` entries take the standard
        ``name:container-path`` form.
        """
        # We need CAP_CHOWN to set up the workspace, but drop privileges for execution
        cmd: list[str] = [
            self.docker_binary,
            "run",
            "-d",  # detached mode
            "--rm",
            "--pull",
            "never",
            "--network",
            network or PYTHON_EXECUTOR_DOCKER_NETWORK,
            "--name",
            container_name,
            "--cgroupns",
            "host",  # Use host cgroup namespace to avoid cgroup v2 issues in DinD
            "--pids-limit",
            "64",
            "--security-opt",
            "no-new-privileges",
            "--cap-drop",
            "ALL",
            "--cap-add",
            "CHOWN",
            "--workdir",
            "/workspace",
            "--tmpfs",
            "/tmp:rw,size=64m",  # noqa: S108 - intentionally constrain container tmpfs
            "--env",
            "PYTHONUNBUFFERED=1",
            "--env",
            "PYTHONDONTWRITEBYTECODE=1",
            "--env",
            "PYTHONIOENCODING=utf-8",
            "--env",
            "MPLCONFIGDIR=/tmp/matplotlib",
            "--env",
            "PIP_CACHE_DIR=/tmp/pip-cache",
        ]

        # The ephemeral workspace is a tmpfs; sessions instead pass a volume so
        # the workspace outlives individual executions.
        if not volumes:
            cmd.extend(
                [
                    "--tmpfs",
                    "/workspace:rw,uid=65532,gid=65532",
                ]
            )
        else:
            cmd.extend(f"--volume={volume}" for volume in volumes)

        for key, value in (labels or {}).items():
            cmd.extend(["--label", f"{key}={value}"])

        if cpu_time_limit_sec is not None:
            cpu_limit = max(cpu_time_limit_sec, 1)
            cmd.extend(["--ulimit", f"cpu={cpu_limit}:{cpu_limit}"])

        if memory_limit_mb is not None:
            memory_limit = max(memory_limit_mb, 16)
            mem_flag = f"{memory_limit}m"
            cmd.extend(["--memory", mem_flag, "--memory-swap", mem_flag])

        if self.run_args:
            cmd.extend(shlex.split(self.run_args))

        cmd.extend([self.image, "sleep", str(sleep_seconds)])
        return cmd

    def _upload_tar_to_container(self, container_name: str, tar_archive: bytes) -> None:
        """Stream a tar archive into the container workspace."""
        tar_cmd = [
            self.docker_binary,
            "exec",
            "-u",
            "65532:65532",
            "-i",
            container_name,
            "tar",
            "-x",
            "-C",
            "/workspace",
        ]
        tar_proc = subprocess.run(tar_cmd, input=tar_archive, capture_output=True)  # nosec B603
        if tar_proc.returncode != 0:
            raise RuntimeError(
                f"Failed to extract files: {tar_proc.stderr.decode('utf-8', errors='replace')}"
            )

    def _stage_files_in_container(
        self,
        container_name: str,
        code: str,
        files: Sequence[tuple[str, bytes]] | None,
        last_line_interactive: bool,
    ) -> None:
        """Create a tar archive and stream it into the container workspace."""
        tar_archive = self._create_tar_archive(code, files, last_line_interactive)
        self._upload_tar_to_container(container_name, tar_archive)

    @contextmanager
    def _run_in_container(
        self,
        *,
        code: str,
        cpu_time_limit_sec: int | None,
        memory_limit_mb: int | None,
        timeout_ms: int,
        files: Sequence[tuple[str, bytes]] | None,
        last_line_interactive: bool,
    ) -> Generator[_ExecContext, None, None]:
        """Create a container, stage files, start the Python process, and clean up.

        Yields an ``_ExecContext`` whose ``proc`` is ready for I/O (stdin is
        still open).  The container is killed in the ``finally`` block
        regardless of how the caller exits.
        """
        container_name = f"code-exec-{uuid.uuid4().hex}"

        cmd = self._build_run_command(
            container_name=container_name,
            cpu_time_limit_sec=cpu_time_limit_sec,
            memory_limit_mb=memory_limit_mb,
            sleep_seconds=(timeout_ms * 1000) + 10,
        )
        start_proc = subprocess.run(cmd, capture_output=True, text=True)  # nosec B603
        if start_proc.returncode != 0:
            raise RuntimeError(f"Failed to start container: {start_proc.stderr}")

        try:
            self._stage_files_in_container(container_name, code, files, last_line_interactive)

            start = time.perf_counter()
            exec_cmd = [
                self.docker_binary,
                "exec",
                "-u",
                "65532:65532",
                "-i",
                container_name,
                "python",
                "/workspace/__main__.py",
            ]

            proc = subprocess.Popen(  # nosec B603
                exec_cmd,
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=False,
            )

            yield _ExecContext(
                container_name=container_name,
                proc=proc,
                start=start,
            )
        finally:
            self._kill_container(container_name)

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
        """Create a long-lived session container with a volume-backed workspace.

        ``ttl_seconds`` sets the (extendable) expiry; the container's idle
        ``sleep`` runs for ``SESSION_MAX_LIFETIME_SEC`` so teardown is always
        bounded. ``network_enabled=None`` uses the deployment's
        ``SESSION_NETWORK_MODE`` posture; explicit True/False overrides it.
        """
        container_name = f"{SESSION_NAME_PREFIX}{uuid.uuid4().hex}"
        expires_at = time.time() + ttl_seconds
        volume_name = self._session_volume_name(container_name)
        image = SESSION_EXECUTOR_IMAGE or self.image
        network = self._resolve_session_network(network_enabled)

        with suppress(Exception):
            subprocess.run(  # nosec B603
                [self.docker_binary, "volume", "create", volume_name],
                capture_output=True,
                text=True,
                timeout=10,
                check=True,
            )

        cmd = self._build_run_command(
            container_name=container_name,
            cpu_time_limit_sec=cpu_time_limit_sec,
            memory_limit_mb=memory_limit_mb,
            sleep_seconds=max(SESSION_MAX_LIFETIME_SEC, ttl_seconds),
            labels={
                "app": SESSION_APP_LABEL,
                "component": SESSION_COMPONENT_LABEL,
                SESSION_EXPIRES_AT_KEY: str(expires_at),
            },
            network=network,
            volumes=[f"{volume_name}:/workspace"],
        )
        start_proc = subprocess.run(cmd, capture_output=True, text=True)  # nosec B603
        if start_proc.returncode != 0:
            self._remove_session_volume(volume_name)
            raise RuntimeError(f"Failed to start session container: {start_proc.stderr}")

        try:
            # The volume root is root-owned; hand it to the execution user.
            subprocess.run(  # nosec B603
                [
                    self.docker_binary,
                    "exec",
                    "-u",
                    "0:0",
                    container_name,
                    "chown",
                    EXEC_USER,
                    "/workspace",
                ],
                capture_output=True,
                text=True,
                timeout=30,
                check=True,
            )

            if install_venv and SESSION_VENV_ENABLED:
                self._install_session_venv(container_name)

            # Persist the expiry inside the workspace so it survives service
            # restarts (docker labels are immutable after create).
            self._write_session_expiry(container_name, expires_at)
            self._session_expiry[container_name] = expires_at

            if files:
                tar_archive = self._create_tar_archive(files=files)
                self._upload_tar_to_container(container_name, tar_archive)
        except Exception:
            self._kill_container(container_name)
            self._remove_session_volume(volume_name)
            self._session_expiry.pop(container_name, None)
            raise

        logger.info(
            "Created session container %s (image=%s network=%s expires at %s)",
            container_name,
            image,
            network,
            expires_at,
        )
        return SessionInfo(session_id=container_name, expires_at=expires_at)

    def _session_volume_name(self, session_id: str) -> str:
        return f"{session_id}-ws"

    def _remove_session_volume(self, volume_name: str) -> None:
        with suppress(Exception):
            subprocess.run(  # nosec B603
                [self.docker_binary, "volume", "rm", "-f", volume_name],
                capture_output=True,
                timeout=15,
                check=False,
            )

    def _resolve_session_network(self, network_enabled: bool | None) -> str:
        if network_enabled is True:
            return PYTHON_EXECUTOR_DOCKER_NETWORK
        if network_enabled is False:
            return "none"
        if SESSION_NETWORK_MODE == "none":
            return "none"
        return PYTHON_EXECUTOR_DOCKER_NETWORK

    def _install_session_venv(self, container_name: str) -> None:
        """Create /workspace/.venv (system-site-packages) and bootstrap pip.

        Best-effort: a failure (e.g. an executor image without the venv
        module) degrades to system-python execution and is logged.
        """
        venv_cmd = (
            "python3 -m venv --system-site-packages /workspace/.venv"
            " && /workspace/.venv/bin/python -m ensurepip --upgrade"
        )
        result = subprocess.run(  # nosec B603
            [
                self.docker_binary,
                "exec",
                "-u",
                EXEC_USER,
                "-w",
                "/workspace",
                container_name,
                "bash",
                "-c",
                venv_cmd,
            ],
            capture_output=True,
            text=True,
            timeout=120,
            check=False,
        )
        if result.returncode != 0:
            logger.warning(
                "Session venv creation failed (continuing with system python): %s",
                (result.stderr or "").strip()[-500:],
            )
            return
        # ensurepip is unavailable in some images (e.g. Arch venv without pip
        # seeded); try uv as a fallback bootstrap.
        pip_check = subprocess.run(  # nosec B603
            [
                self.docker_binary,
                "exec",
                "-u",
                EXEC_USER,
                container_name,
                "bash",
                "-c",
                "if ! test -x /workspace/.venv/bin/pip; then "
                "command -v uv >/dev/null 2>&1 "
                "&& uv pip install --python /workspace/.venv/bin/python pip; fi",
            ],
            capture_output=True,
            text=True,
            timeout=120,
            check=False,
        )
        if pip_check.returncode != 0:
            logger.warning(
                "Session venv pip bootstrap failed (pip unavailable in venv): %s",
                (pip_check.stderr or "").strip()[-500:],
            )

    def _write_session_expiry(self, container_name: str, expires_at: float) -> None:
        subprocess.run(  # nosec B603
            [
                self.docker_binary,
                "exec",
                "-u",
                EXEC_USER,
                container_name,
                "sh",
                "-c",
                f"echo {expires_at} > /workspace/{SESSION_EXPIRY_FILE}",
            ],
            capture_output=True,
            timeout=10,
            check=False,
        )

    def _read_session_expiry(self, session_id: str, label_value: str) -> float:
        """Best-effort expiry lookup: memory map, then expiry file, then label."""
        mapped = self._session_expiry.get(session_id)
        if mapped is not None:
            return mapped
        result = subprocess.run(  # nosec B603
            [
                self.docker_binary,
                "exec",
                session_id,
                "cat",
                f"/workspace/{SESSION_EXPIRY_FILE}",
            ],
            capture_output=True,
            text=True,
            timeout=5,
            check=False,
        )
        if result.returncode == 0:
            try:
                return float(result.stdout.strip())
            except ValueError:
                pass
        try:
            return float(label_value)
        except ValueError:
            return 0.0

    def _assert_session_running(self, session_id: str) -> None:
        if not session_id.startswith(SESSION_NAME_PREFIX):
            raise SessionNotFoundError(session_id)
        result = subprocess.run(  # nosec B603
            [
                self.docker_binary,
                "inspect",
                "-f",
                "{{.State.Running}}",
                session_id,
            ],
            capture_output=True,
            text=True,
            timeout=5,
            check=False,
        )
        if result.returncode != 0 or result.stdout.strip() != "true":
            raise SessionNotFoundError(session_id)

    def delete_session(self, session_id: str) -> bool:
        if not session_id.startswith(SESSION_NAME_PREFIX):
            return False
        result = subprocess.run(  # nosec B603
            [self.docker_binary, "rm", "-f", session_id],
            capture_output=True,
            text=True,
        )
        if result.returncode == 0:
            self._session_expiry.pop(session_id, None)
            self._remove_session_volume(self._session_volume_name(session_id))
            return True
        # docker rm -f exits non-zero only when the container does not exist.
        stderr = (result.stderr or "").lower()
        if "no such container" in stderr or "not found" in stderr:
            return False
        raise RuntimeError(f"Failed to delete session {session_id}: {result.stderr}")

    def reap_expired_sessions(self) -> int:
        list_cmd = [
            self.docker_binary,
            "ps",
            "-a",
            "--filter",
            f"label=app={SESSION_APP_LABEL}",
            "--filter",
            f"label=component={SESSION_COMPONENT_LABEL}",
            "--format",
            f'{{{{.Names}}}}\t{{{{.Label "{SESSION_EXPIRES_AT_KEY}"}}}}',
        ]
        try:
            list_result = subprocess.run(  # nosec B603
                list_cmd, capture_output=True, text=True, timeout=10
            )
        except subprocess.TimeoutExpired:
            logger.warning("Timed out listing session containers for reap")
            return 0

        if list_result.returncode != 0:
            logger.warning("Failed to list session containers: %s", list_result.stderr)
            return 0

        now = time.time()
        reaped = 0
        for line in list_result.stdout.splitlines():
            name, _, expires_str = line.partition("\t")
            name = name.strip()
            expires_str = expires_str.strip()
            if not name:
                continue
            expires_at = self._read_session_expiry(name, expires_str)
            if expires_at >= now:
                continue
            rm_result = subprocess.run(  # nosec B603
                [self.docker_binary, "rm", "-f", name],
                capture_output=True,
                text=True,
            )
            if rm_result.returncode == 0:
                self._session_expiry.pop(name, None)
                self._remove_session_volume(self._session_volume_name(name))
                reaped += 1
                logger.info("Reaped expired session container %s", name)
            else:
                logger.warning("Failed to reap session container %s: %s", name, rm_result.stderr)
        return reaped

    def execute_bash_in_session(
        self,
        session_id: str,
        *,
        cmd: str,
        timeout_ms: int,
        max_output_bytes: int,
    ) -> ExecutionResult:
        """Run a bash command inside an existing session container.

        The container was created with ``--network none`` at session-create time
        and that network namespace is what the exec inherits — no additional
        flags are needed (or accepted) for ``docker exec``.
        """
        if not session_id.startswith(SESSION_NAME_PREFIX):
            raise SessionNotFoundError(session_id)

        exec_cmd = [
            self.docker_binary,
            "exec",
            "-u",
            "65532:65532",
            session_id,
            "bash",
            "-c",
            cmd,
        ]

        start = time.perf_counter()
        proc = subprocess.Popen(  # nosec B603
            exec_cmd,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )

        try:
            stdout_bytes, stderr_bytes = proc.communicate(timeout=timeout_ms / 1000.0)
            timed_out = False
        except subprocess.TimeoutExpired:
            timed_out = True
            # Kill bash inside the container; pkill matches all bash procs in the
            # container — acceptable since the agent runs commands sequentially.
            subprocess.run(  # nosec B603
                [self.docker_binary, "exec", session_id, "pkill", "-9", "bash"],
                capture_output=True,
            )
            proc.kill()
            stdout_bytes, stderr_bytes = proc.communicate()

        duration_ms = int((time.perf_counter() - start) * 1000)
        exit_code = None if timed_out else proc.returncode

        if not timed_out and proc.returncode != 0 and _looks_like_missing_container(stderr_bytes):
            raise SessionNotFoundError(session_id)

        return ExecutionResult(
            stdout=self.truncate_output(stdout_bytes or b"", max_output_bytes),
            stderr=self.truncate_output(stderr_bytes or b"", max_output_bytes),
            exit_code=exit_code,
            timed_out=timed_out,
            duration_ms=duration_ms,
            files=tuple(),
        )

    def _session_python_cmd(self, session_id: str, script_path: str) -> tuple[list[str], str]:
        """Build the docker exec command for running a script in a session.

        Prefers the per-session venv python when present and puts it first on
        PATH, so plain `pip install` inside session code lands in the session
        venv (and thus persists on the workspace volume). Returns
        (cmd, script_name).
        """
        script_name = script_path.rsplit("/", 1)[-1]
        py_check = subprocess.run(  # nosec B603
            [
                self.docker_binary,
                "exec",
                "-u",
                EXEC_USER,
                session_id,
                "bash",
                "-c",
                "test -x /workspace/.venv/bin/python && echo venv || echo system",
            ],
            capture_output=True,
            text=True,
            timeout=10,
            check=False,
        )
        use_venv = py_check.returncode == 0 and py_check.stdout.strip() == "venv"
        python_bin = "/workspace/.venv/bin/python" if use_venv else "python3"
        # exec replaces the shell with the interpreter, so pipes/signals and
        # the pkill-by-script-name timeout kill still target the right process.
        bash_cmd = (
            "export PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1 "
            "PYTHONIOENCODING=utf-8 MPLCONFIGDIR=/tmp/matplotlib "  # noqa: S108
            "PIP_CACHE_DIR=/tmp/pip-cache;"
            + (" export PATH=/workspace/.venv/bin:$PATH;" if use_venv else "")
            + f" exec {python_bin} {script_path}"
        )
        cmd = [
            self.docker_binary,
            "exec",
            "-u",
            EXEC_USER,
            "-i",
            "-w",
            "/workspace",
            session_id,
            "bash",
            "-c",
            bash_cmd,
        ]
        return cmd, script_name

    def _stage_session_script(
        self,
        session_id: str,
        code: str,
        last_line_interactive: bool,
        files: Sequence[tuple[str, bytes]] | None,
    ) -> str:
        """Stage the exec script (and any extra files) into the session."""
        script_path = f"/workspace/.onyx-exec-{uuid.uuid4().hex}.py"
        script_name = script_path.rsplit("/", 1)[-1]
        code_to_execute = wrap_last_line_interactive(code) if last_line_interactive else code
        tar_archive = self._create_tar_archive(
            code=None,
            files=[(script_name, code_to_execute.encode("utf-8")), *(files or ())],
        )
        self._upload_tar_to_container(session_id, tar_archive)
        return script_path

    def _remove_session_script(self, session_id: str, script_path: str) -> None:
        with suppress(Exception):
            subprocess.run(  # nosec B603
                [
                    self.docker_binary,
                    "exec",
                    "-u",
                    EXEC_USER,
                    session_id,
                    "rm",
                    "-f",
                    script_path,
                ],
                capture_output=True,
                timeout=10,
                check=False,
            )

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
        """Execute Python inside a session; the workspace persists across calls."""
        self._assert_session_running(session_id)
        script_path = self._stage_session_script(session_id, code, last_line_interactive, files)
        exec_cmd, script_name = self._session_python_cmd(session_id, script_path)

        start = time.perf_counter()
        proc = subprocess.Popen(  # nosec B603
            exec_cmd,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=False,
        )
        try:
            input_bytes = stdin.encode("utf-8") if stdin is not None else None
            try:
                stdout_bytes, stderr_bytes = proc.communicate(
                    input=input_bytes,
                    timeout=timeout_ms / 1000.0,
                )
                timed_out = False
            except subprocess.TimeoutExpired:
                timed_out = True
                subprocess.run(  # nosec B603
                    [
                        self.docker_binary,
                        "exec",
                        session_id,
                        "pkill",
                        "-9",
                        "-f",
                        script_name,
                    ],
                    capture_output=True,
                )
                proc.kill()
                stdout_bytes, stderr_bytes = proc.communicate()
        finally:
            self._remove_session_script(session_id, script_path)

        workspace_snapshot = self._extract_workspace_snapshot(
            session_id, extra_excludes=[script_name]
        )
        duration_ms = int((time.perf_counter() - start) * 1000)
        exit_code = None if timed_out else proc.returncode

        return ExecutionResult(
            stdout=self.truncate_output(stdout_bytes or b"", max_output_bytes),
            stderr=self.truncate_output(stderr_bytes or b"", max_output_bytes),
            exit_code=exit_code,
            timed_out=timed_out,
            duration_ms=duration_ms,
            files=workspace_snapshot,
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
        """Streaming execution inside a session (SSE-friendly)."""
        self._assert_session_running(session_id)
        script_path = self._stage_session_script(session_id, code, last_line_interactive, files)
        exec_cmd, script_name = self._session_python_cmd(session_id, script_path)

        start = time.perf_counter()
        proc = subprocess.Popen(  # nosec B603
            exec_cmd,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=False,
        )
        try:
            _write_stdin(proc, stdin)
            deadline = time.monotonic() + (timeout_ms / 1000.0)
            timed_out = yield from _stream_process_output(proc, deadline, max_output_bytes)

            if timed_out:
                subprocess.run(  # nosec B603
                    [
                        self.docker_binary,
                        "exec",
                        session_id,
                        "pkill",
                        "-9",
                        "-f",
                        script_name,
                    ],
                    capture_output=True,
                )
                proc.kill()
            proc.wait()
        finally:
            self._remove_session_script(session_id, script_path)

        workspace_snapshot = self._extract_workspace_snapshot(
            session_id, extra_excludes=[script_name]
        )
        duration_ms = int((time.perf_counter() - start) * 1000)
        exit_code = None if timed_out else proc.returncode

        yield StreamResult(
            exit_code=exit_code,
            timed_out=timed_out,
            duration_ms=duration_ms,
            files=workspace_snapshot,
        )

    def keepalive_session(self, session_id: str, *, ttl_seconds: int) -> SessionInfo:
        """Extend the session expiry to now + ttl_seconds."""
        self._assert_session_running(session_id)
        expires_at = time.time() + ttl_seconds
        self._write_session_expiry(session_id, expires_at)
        self._session_expiry[session_id] = expires_at
        logger.info("Extended session %s expiry to %s", session_id, expires_at)
        return SessionInfo(session_id=session_id, expires_at=expires_at)

    def stage_files_in_session(self, session_id: str, files: Sequence[tuple[str, bytes]]) -> None:
        """Stage extra files into an existing session's workspace."""
        self._assert_session_running(session_id)
        if not files:
            return
        tar_archive = self._create_tar_archive(files=files)
        self._upload_tar_to_container(session_id, tar_archive)

    def list_session_files(self, session_id: str) -> tuple[WorkspaceEntry, ...]:
        """List workspace entries (metadata only). Venv/control files excluded."""
        self._assert_session_running(session_id)
        try:
            tar_cmd = [self.docker_binary, "exec", session_id, "tar", "-c"]
            for exclude in SESSION_CONTROL_EXCLUDES:
                tar_cmd.extend(["--exclude", exclude])
            tar_cmd.extend(["-C", "/workspace", "."])
            tar_result = subprocess.run(tar_cmd, capture_output=True, timeout=120)  # nosec B603
            if tar_result.returncode != 0:
                return tuple()

            entries: list[WorkspaceEntry] = []
            with tarfile.open(fileobj=io.BytesIO(tar_result.stdout), mode="r") as tar:
                for member in tar.getmembers():
                    if member.name == ".":
                        continue
                    clean_path = member.name.lstrip("./")
                    if not clean_path:
                        continue
                    if member.isdir():
                        entries.append(
                            WorkspaceEntry(path=clean_path, kind=EntryKind.DIRECTORY, content=None)
                        )
                    elif member.isfile():
                        entries.append(
                            WorkspaceEntry(
                                path=clean_path,
                                kind=EntryKind.FILE,
                                content=None,
                            )
                        )
            return tuple(entries)
        except Exception:
            return tuple()

    def read_session_file(self, session_id: str, path: str) -> bytes:
        """Read a single file from the session workspace."""
        self._assert_session_running(session_id)
        validated = self._validate_relative_path(path)
        target = validated.as_posix()
        tar_cmd = [
            self.docker_binary,
            "exec",
            session_id,
            "tar",
            "-c",
            "-C",
            "/workspace",
            target,
        ]
        tar_result = subprocess.run(tar_cmd, capture_output=True, timeout=60)  # nosec B603
        if tar_result.returncode != 0 or not tar_result.stdout:
            raise LookupError(f"File not found in session workspace: {path}")
        with tarfile.open(fileobj=io.BytesIO(tar_result.stdout), mode="r") as tar:
            for member in tar.getmembers():
                if member.isfile():
                    file_obj = tar.extractfile(member)
                    if file_obj is not None:
                        return file_obj.read()
        raise LookupError(f"File not found in session workspace: {path}")

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
        """Execute Python code inside an ephemeral Docker container with no network.

        Args:
            last_line_interactive: If True, the last line will print its value to stdout
                                   if it's a bare expression (only the last line is affected).
        """
        with self._run_in_container(
            code=code,
            cpu_time_limit_sec=cpu_time_limit_sec,
            memory_limit_mb=memory_limit_mb,
            timeout_ms=timeout_ms,
            files=files,
            last_line_interactive=last_line_interactive,
        ) as ctx:
            logger.debug(f"Executing code: {code}")

            try:
                input_bytes = stdin.encode("utf-8") if stdin is not None else None
                stdout_bytes, stderr_bytes = ctx.proc.communicate(
                    input=input_bytes,
                    timeout=timeout_ms / 1000.0,
                )
                timed_out = False
            except subprocess.TimeoutExpired:
                timed_out = True
                # Kill the Python process in the container (as root to ensure we can kill it)
                subprocess.run(
                    [
                        self.docker_binary,
                        "exec",
                        ctx.container_name,
                        "pkill",
                        "-9",
                        "python",
                    ],
                    capture_output=True,
                )
                ctx.proc.kill()
                stdout_bytes, stderr_bytes = ctx.proc.communicate()

            # Extract workspace snapshot
            workspace_snapshot = self._extract_workspace_snapshot(ctx.container_name)

        duration_ms = int((time.perf_counter() - ctx.start) * 1000)

        stdout = self.truncate_output(stdout_bytes or b"", max_output_bytes)
        logger.debug(f"stdout: {stdout}")
        stderr = self.truncate_output(stderr_bytes or b"", max_output_bytes)
        logger.debug(f"stderr: {stderr}")
        exit_code = None if timed_out else ctx.proc.returncode

        return ExecutionResult(
            stdout=stdout,
            stderr=stderr,
            exit_code=exit_code,
            timed_out=timed_out,
            duration_ms=duration_ms,
            files=workspace_snapshot,
        )

    def _terminate_process(self, ctx: _ExecContext, timed_out: bool) -> None:
        """Kill the process on timeout or wait for normal exit."""
        if timed_out:
            subprocess.run(
                [
                    self.docker_binary,
                    "exec",
                    ctx.container_name,
                    "pkill",
                    "-9",
                    "python",
                ],
                capture_output=True,
            )
            ctx.proc.kill()
        ctx.proc.wait()

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
        """Execute Python code and yield output chunks as they arrive via SSE.

        Yields StreamChunk events during execution, then a single StreamResult
        at the end containing exit_code, timing, and workspace files.
        """
        with self._run_in_container(
            code=code,
            cpu_time_limit_sec=cpu_time_limit_sec,
            memory_limit_mb=memory_limit_mb,
            timeout_ms=timeout_ms,
            files=files,
            last_line_interactive=last_line_interactive,
        ) as ctx:
            _write_stdin(ctx.proc, stdin)

            deadline = time.monotonic() + (timeout_ms / 1000.0)
            timed_out = yield from _stream_process_output(ctx.proc, deadline, max_output_bytes)

            self._terminate_process(ctx, timed_out)
            workspace_snapshot = self._extract_workspace_snapshot(ctx.container_name)

        duration_ms = int((time.perf_counter() - ctx.start) * 1000)
        exit_code = None if timed_out else ctx.proc.returncode

        yield StreamResult(
            exit_code=exit_code,
            timed_out=timed_out,
            duration_ms=duration_ms,
            files=workspace_snapshot,
        )


def _write_stdin(proc: subprocess.Popen[bytes], stdin: str | None) -> None:
    """Write optional stdin data and close the pipe."""
    if proc.stdin is None:
        raise RuntimeError("Failed to open subprocess stdin pipe")
    if stdin is not None:
        proc.stdin.write(stdin.encode("utf-8"))
    proc.stdin.close()


class _StreamTracker:
    """Per-stream state for incremental decoding with truncation."""

    __slots__ = ("stream", "decoder", "bytes_sent", "max_bytes")

    def __init__(self, stream: Literal["stdout", "stderr"], max_bytes: int) -> None:
        self.stream = stream
        self.decoder = codecs.getincrementaldecoder("utf-8")("replace")
        self.bytes_sent = 0
        self.max_bytes = max_bytes

    def decode_chunk(self, data: bytes) -> StreamChunk | None:
        """Decode a raw chunk and return a ``StreamChunk`` if within limits."""
        chunk: StreamChunk | None = None
        if self.bytes_sent < self.max_bytes:
            allowed = self.max_bytes - self.bytes_sent
            text = self.decoder.decode(data[:allowed], False)
            if text:
                chunk = StreamChunk(stream=self.stream, data=text)
        self.bytes_sent += len(data)
        return chunk

    def flush(self) -> StreamChunk | None:
        """Flush the decoder and return a final chunk if any bytes remain."""
        text = self.decoder.decode(b"", True)
        if text:
            return StreamChunk(stream=self.stream, data=text)
        return None


def _stream_process_output(
    proc: subprocess.Popen[bytes],
    deadline: float,
    max_output_bytes: int,
) -> Generator[StreamChunk, None, bool]:
    """Read stdout/stderr incrementally and yield ``StreamChunk`` events.

    Returns ``True`` if the process timed out, ``False`` otherwise.
    """
    if proc.stdout is None or proc.stderr is None:
        raise RuntimeError("Failed to open subprocess output pipes")

    sel = selectors.DefaultSelector()
    sel.register(proc.stdout, selectors.EVENT_READ, "stdout")
    sel.register(proc.stderr, selectors.EVENT_READ, "stderr")

    trackers: dict[str, _StreamTracker] = {
        "stdout": _StreamTracker("stdout", max_output_bytes),
        "stderr": _StreamTracker("stderr", max_output_bytes),
    }
    fds: dict[str, int] = {
        "stdout": proc.stdout.fileno(),
        "stderr": proc.stderr.fileno(),
    }
    timed_out = False
    chunk_size = 4096

    try:
        while sel.get_map():
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                timed_out = True
                break

            events = sel.select(timeout=min(remaining, 5.0))

            for key, _ in events:
                stream_name: str = key.data
                data = os.read(fds[stream_name], chunk_size)
                if not data:
                    sel.unregister(key.fileobj)
                    continue

                chunk = trackers[stream_name].decode_chunk(data)
                if chunk is not None:
                    yield chunk
    finally:
        sel.close()

    for tracker in trackers.values():
        chunk = tracker.flush()
        if chunk is not None:
            yield chunk

    return timed_out
