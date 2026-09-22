import hashlib
import mimetypes
import os
import re
from io import BytesIO
from typing import Any, cast

import requests as http_requests
from pydantic import BaseModel, TypeAdapter
from sqlalchemy.orm import Session
from typing_extensions import override

from onyx.chat.emitter import Emitter
from onyx.configs.app_configs import (
    CODE_INTERPRETER_BASE_URL,
    CODE_INTERPRETER_DEFAULT_TIMEOUT_MS,
    CODE_INTERPRETER_MAX_GENERATED_ARTIFACTS,
    CODE_INTERPRETER_MAX_OUTPUT_LENGTH,
    CODE_INTERPRETER_MAX_STAGED_BYTES,
    CODE_INTERPRETER_MAX_STAGED_FILES,
    CODE_INTERPRETER_SESSION_TTL_SECONDS,
    CODE_INTERPRETER_SESSIONS_ENABLED,
    CODE_INTERPRETER_STAGING_CONCURRENCY,
)
from onyx.configs.constants import FileOrigin
from onyx.db.code_interpreter import fetch_code_interpreter_server
from onyx.file_store.utils import build_full_frontend_file_url, get_default_file_store
from onyx.server.query_and_chat.placement import Placement
from onyx.server.query_and_chat.streaming_models import (
    Packet,
    PythonToolDelta,
    PythonToolGeneratedFile,
    PythonToolStart,
)
from onyx.tools.interface import Tool
from onyx.tools.models import (
    ChatFile,
    LlmPythonExecutionResult,
    PythonExecutionFile,
    PythonToolOverrideKwargs,
    PythonToolRichResponse,
    ToolCallException,
    ToolResponse,
    ToolResponseImage,
)
from onyx.tools.tool_implementations.image_analysis.shared import (
    annotate_images_in_parallel,
    get_tool_vision_llm,
)
from onyx.tools.tool_implementations.python.code_interpreter_client import (
    CodeInterpreterClient,
    FileInput,
    StreamErrorEvent,
    StreamOutputEvent,
    StreamResultEvent,
)
from onyx.tools.tool_implementations.python.session_store import (
    fetch_ci_session_id,
    fetch_staged_file_keys,
    fetch_workspace_file_ids,
    forget_staged_file_keys,
    forget_workspace_file_ids,
    refresh_session_ttls,
    store_ci_session_id,
    store_staged_file_keys,
    store_workspace_file_ids,
)
from onyx.tools.tool_implementations.utils import truncate_output as _truncate_output
from onyx.utils.logger import setup_logger
from onyx.utils.threadpool_concurrency import run_functions_tuples_in_parallel

logger = setup_logger()

CODE_FIELD = "code"
CODE_INTERPRETER_DEFAULT_FILENAME = "file"
CODE_INTERPRETER_FILENAME_MAX_LENGTH = 200
CODE_INTERPRETER_UNSAFE_FILENAME_CHARS = re.compile(r"[\x00-\x1f/\\:\*\?\"<>\|]+")
# Shown to the LLM when files were generated
FILES_NOTICE_TEMPLATE = (
    "Generated files are saved and stay available by filename in later "
    "executions of this session. Image files are displayed to the user in chat; "
    "to show one, embed its exact file_link URL from generated_files as the "
    "markdown image target, copied verbatim. Never write a placeholder in "
    "place of the URL."
)
# Shown to the LLM when execution ran in the persistent session sandbox
SESSION_NOTICE_TEMPLATE = (
    "This execution ran in your persistent session sandbox: files in the "
    "working directory and packages installed with pip/uv stay available in "
    "later calls of this chat."
)


class _SessionUnavailable(Exception):
    """Session mode could not be set up; fall back to ephemeral execution."""


def _safe_code_interpreter_filename(filename: str) -> str:
    sanitized = CODE_INTERPRETER_UNSAFE_FILENAME_CHARS.sub("_", filename)
    sanitized = sanitized.strip().strip(".")
    if not sanitized:
        return CODE_INTERPRETER_DEFAULT_FILENAME

    base, ext = os.path.splitext(sanitized)
    if not base:
        base = CODE_INTERPRETER_DEFAULT_FILENAME

    max_base_len = max(1, CODE_INTERPRETER_FILENAME_MAX_LENGTH - len(ext))
    return f"{base[:max_base_len]}{ext}"


def _dedupe_code_interpreter_filename(
    filename: str,
    seen_filenames: set[str],
    fallback_id: str,
) -> str:
    safe_filename = _safe_code_interpreter_filename(filename)
    if safe_filename not in seen_filenames:
        seen_filenames.add(safe_filename)
        return safe_filename

    base, ext = os.path.splitext(safe_filename)
    suffix = f"_{fallback_id}{ext}"
    max_base_len = max(1, CODE_INTERPRETER_FILENAME_MAX_LENGTH - len(suffix))
    deduped_filename = f"{base[:max_base_len]}{suffix}"
    seen_filenames.add(deduped_filename)
    return deduped_filename


class _StagePlan(BaseModel):
    file_name: str
    content: bytes
    cache_key: tuple[str, str]  # (file_name, content_hash)


def _code_references_file(filename: str, code: str) -> bool:
    """Heuristic: the code wants a file if its name appears literally in the
    code. Checked on the raw filename and its sanitized (sandbox) form, since
    the model may write either. Filename-only — no object-store read."""
    if not filename:
        return False
    return filename in code or _safe_code_interpreter_filename(filename) in code


def _read_chat_file_content(chat_file: ChatFile) -> bytes:
    """Materialize a (possibly lazy) chat file's bytes; the object-store read
    happens here. Extracted so reads can be batched in parallel."""
    return chat_file.content


def _staging_priority(chat_files: list[ChatFile], code: str) -> list[int]:
    """``chat_files`` indices in staging-priority order: files the code names
    first, then the rest, newest-first within each group."""
    referenced: list[int] = []
    unreferenced: list[int] = []
    for idx, chat_file in enumerate(chat_files):
        bucket = (
            referenced
            if _code_references_file(chat_file.filename, code)
            else unreferenced
        )
        bucket.append(idx)
    return list(reversed(referenced)) + list(reversed(unreferenced))


class _StagingSelection(BaseModel):
    # Selected (file, content) pairs in chronological order.
    files: list[tuple[ChatFile, bytes]]
    # Files excluded by the count/byte caps (not counting read failures).
    dropped_by_caps: int
    # Filenames whose bytes could not be read from the object store.
    read_failures: list[str]


def _select_files_for_staging(
    chat_files: list[ChatFile],
    code: str,
    *,
    max_files: int,
    max_bytes: int,
    read_concurrency: int,
) -> _StagingSelection:
    """Choose which session files to stage, reading bytes only for the chosen.

    Files the ``code`` names explicitly are staged first; the remaining
    count/byte budget is backfilled with the most recent of the rest. Candidates
    are read in bounded parallel batches — concurrency hides read latency while
    only one batch is held in memory at a time.
    """
    # Count cap: only the top candidates can ever be staged, so never read more.
    candidates = _staging_priority(chat_files, code)[:max_files]

    selected: dict[int, bytes] = {}
    read_failures: list[str] = []
    staged_bytes = 0
    for start in range(0, len(candidates), read_concurrency):
        batch = candidates[start : start + read_concurrency]
        contents = run_functions_tuples_in_parallel(
            [(_read_chat_file_content, (chat_files[idx],)) for idx in batch],
            allow_failures=True,
            max_workers=read_concurrency,
        )

        over_budget = False
        for idx, content in zip(batch, contents, strict=True):
            if content is None:
                logger.warning(
                    "Failed to read file for Python execution: %s",
                    chat_files[idx].filename,
                )
                read_failures.append(chat_files[idx].filename)
                continue
            # Always stage at least one file; otherwise stop at the byte budget.
            if selected and staged_bytes + len(content) > max_bytes:
                over_budget = True
                break
            selected[idx] = content
            staged_bytes += len(content)
        if over_budget:
            break

    chronological = [(chat_files[idx], selected[idx]) for idx in sorted(selected)]
    return _StagingSelection(
        files=chronological,
        dropped_by_caps=len(chat_files) - len(selected) - len(read_failures),
        read_failures=read_failures,
    )


def _build_staging_notice(
    dropped_count: int,
    total_files: int,
    failed_files: list[str],
) -> str | None:
    """LLM-facing note when some session files are absent — dropped by the caps
    or failed to stage (read/upload error) — so the model doesn't assume they're
    available. ``None`` when everything staged."""
    parts: list[str] = []
    if dropped_count > 0:
        parts.append(
            f"{dropped_count} of {total_files} session files were not staged due "
            f"to per-execution limits ({CODE_INTERPRETER_MAX_STAGED_FILES} files / "
            f"{CODE_INTERPRETER_MAX_STAGED_BYTES} bytes); files referenced in the "
            f"code were prioritized."
        )
    if failed_files:
        parts.append(
            f"Failed to stage {len(failed_files)} file(s): {', '.join(failed_files)}."
        )
    return " ".join(parts) if parts else None


def _combine_staging_inputs(
    chat_files: list[ChatFile],
    generated_artifacts: dict[str, bytes],
) -> list[ChatFile]:
    """User files plus previously generated artifacts, as sandbox inputs.

    An artifact with the same filename as a user file replaces it: later code
    that reads that name expects the generated content."""
    inputs = [
        chat_file
        for chat_file in chat_files
        if chat_file.filename not in generated_artifacts
    ]
    inputs.extend(
        ChatFile(filename=name, content=content)
        for name, content in generated_artifacts.items()
    )
    return inputs


class PythonTool(Tool[PythonToolOverrideKwargs]):
    """
    Python code execution tool using an external Code Interpreter service.

    This tool allows executing Python code in a secure, isolated sandbox environment.
    It supports uploading files from the chat session and downloading generated files.

    When the code-interpreter service supports sessions (>= 0.5.0) and sessions
    are enabled, all executions of a chat run inside one persistent sandbox
    session: workspace files, installed packages, and other state survive across
    tool calls and chat turns. Otherwise each execution runs in a fresh sandbox
    and generated artifacts are re-staged as inputs (legacy behavior).
    """

    # OpenAI reserves the function name "python" for its own harness and
    # rejects requests that define a tool with that name (400 invalid_request_error,
    # enforced server-side since 2026-07-21) — never rename this back to "python".
    NAME = "run_python"
    DISPLAY_NAME = "Code Interpreter"
    DESCRIPTION = "Execute Python code in an isolated sandbox environment."

    def __init__(
        self,
        tool_id: int,
        emitter: Emitter,
        chat_session_id: str | None = None,
    ) -> None:
        super().__init__(emitter=emitter)
        self._id = tool_id
        self._chat_session_id = chat_session_id
        # Resolved lazily from Redis (the mapping survives across chat turns).
        self._ci_session_id: str | None = None
        # Cache of (filename, content_hash) -> ci_file_id to avoid re-uploading
        # the same file on every tool call iteration within the same agent session.
        # Filename is included in the key so two files with identical bytes but
        # different names each get their own upload slot.
        # TTL assumption: code-interpreter file TTLs (typically hours) greatly
        # exceed the lifetime of a single agent session (at most MAX_LLM_CYCLES
        # iterations, typically a few minutes), so stale-ID eviction is not needed.
        # Legacy path only.
        self._uploaded_file_cache: dict[tuple[str, str], str] = {}
        # Generated artifacts from earlier executions of this message run, keyed
        # by filename (newest wins). Legacy path only: the sandbox is wiped per
        # call there, so artifacts are re-staged as inputs to let later calls
        # build on them. Bounded by CODE_INTERPRETER_MAX_GENERATED_ARTIFACTS
        # (oldest evicted first).
        self._generated_artifacts: dict[str, bytes] = {}

    @property
    def id(self) -> int:
        return self._id

    @property
    def name(self) -> str:
        return self.NAME

    @property
    def description(self) -> str:
        return self.DESCRIPTION

    @property
    def display_name(self) -> str:
        return self.DISPLAY_NAME

    @override
    @classmethod
    def is_available(cls, db_session: Session) -> bool:
        if not CODE_INTERPRETER_BASE_URL:
            return False
        server = fetch_code_interpreter_server(db_session)
        if not server.server_enabled:
            return False

        with CodeInterpreterClient() as client:
            return client.health(use_cache=True).healthy

    def tool_definition(self) -> dict:
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": {
                    "type": "object",
                    "properties": {
                        CODE_FIELD: {
                            "type": "string",
                            "description": "Python source code to execute",
                        },
                    },
                    "required": [CODE_FIELD],
                },
            },
        }

    def emit_start(self, placement: Placement) -> None:
        """Emit start packet for this tool. Code will be emitted in run() method."""
        # Note: PythonToolStart requires code, but we don't have it in emit_start
        # The code is available in run() method via llm_kwargs
        # We'll emit the start packet in run() instead

    def _upload_and_stage(
        self,
        client: CodeInterpreterClient,
        selected: list[tuple[ChatFile, bytes]],
    ) -> tuple[list[FileInput], list[str]]:
        """Upload the selected files, returning stage specs and the names of any
        that failed to upload.

        Cache misses upload concurrently (bounded); the cache is written
        single-threaded afterward so it is never mutated from worker threads.
        Files are staged in chronological order for deterministic dedup naming.
        """
        seen_filenames: set[str] = set()
        plans: list[_StagePlan] = []
        for ind, (chat_file, content) in enumerate(selected):
            file_name = _dedupe_code_interpreter_filename(
                chat_file.filename, seen_filenames, str(ind)
            )
            content_hash = hashlib.sha256(content).hexdigest()
            plans.append(
                _StagePlan(
                    file_name=file_name,
                    content=content,
                    cache_key=(file_name, content_hash),
                )
            )

        # allow_failures keeps one bad upload from sinking the batch; each upload
        # is individually bounded by the client's per-request timeout.
        misses = [p for p in plans if p.cache_key not in self._uploaded_file_cache]
        if misses:
            upload_results = run_functions_tuples_in_parallel(
                [(client.upload_file, (p.content, p.file_name)) for p in misses],
                allow_failures=True,
                max_workers=CODE_INTERPRETER_STAGING_CONCURRENCY,
            )
            for plan, ci_file_id in zip(misses, upload_results, strict=True):
                if ci_file_id is None:
                    logger.warning(
                        "Failed to upload file for Python execution: %s", plan.file_name
                    )
                    continue
                self._uploaded_file_cache[plan.cache_key] = ci_file_id

        files_to_stage: list[FileInput] = []
        failed_uploads: list[str] = []
        for plan in plans:
            ci_file_id = self._uploaded_file_cache.get(plan.cache_key)
            if ci_file_id is None:
                failed_uploads.append(plan.file_name)
                continue
            files_to_stage.append({"path": plan.file_name, "file_id": ci_file_id})
            logger.info("Staged file for Python execution: %s", plan.file_name)
        return files_to_stage, failed_uploads

    def run(
        self,
        placement: Placement,
        override_kwargs: PythonToolOverrideKwargs,
        **llm_kwargs: Any,
    ) -> ToolResponse:
        """
        Execute Python code in the Code Interpreter service.

        Args:
            placement: The placement info (turn_index and tab_index) for this tool call.
            override_kwargs: Contains chat_files to stage for execution
            **llm_kwargs: Contains 'code' parameter from LLM

        Returns:
            ToolResponse with execution results
        """
        if CODE_FIELD not in llm_kwargs:
            raise ToolCallException(
                message=f"Missing required '{CODE_FIELD}' parameter in python tool call",
                llm_facing_message=(
                    f"The python tool requires a '{CODE_FIELD}' parameter containing "
                    f"the Python code to execute. Please provide like: "
                    f'{{"code": "print(\'Hello, world!\')"}}'
                ),
            )
        code = cast(str, llm_kwargs[CODE_FIELD])
        chat_files = override_kwargs.chat_files if override_kwargs else []

        # Emit start event with the code
        self.emitter.emit(
            Packet(
                placement=placement,
                obj=PythonToolStart(code=code),
            )
        )

        # Create Code Interpreter client — context manager ensures
        # session.close() is called on every exit path.
        with CodeInterpreterClient() as client:
            if self._use_sessions(client):
                # One recreate attempt when a mapped session has vanished
                # (expired past its TTL, reaped, or the service restarted).
                for allow_recreate in (True, False):
                    try:
                        return self._run_in_session(client, placement, code, chat_files)
                    except http_requests.HTTPError as e:
                        status_code = (
                            e.response.status_code if e.response is not None else None
                        )
                        if allow_recreate and status_code == 404:
                            logger.warning(
                                "Code-interpreter session vanished (HTTP 404); recreating"
                            )
                            self._reset_session()
                            continue
                        raise
                    except _SessionUnavailable as e:
                        logger.warning(
                            "Session mode unavailable (%s); falling back to "
                            "ephemeral execution",
                            e,
                        )
                        break

            # Legacy sessionless path: the sandbox is wiped per call, so
            # artifacts from earlier executions are re-staged as inputs.
            staging_inputs = _combine_staging_inputs(
                chat_files, self._generated_artifacts
            )
            selection = _select_files_for_staging(
                staging_inputs,
                code,
                max_files=CODE_INTERPRETER_MAX_STAGED_FILES,
                max_bytes=CODE_INTERPRETER_MAX_STAGED_BYTES,
                read_concurrency=CODE_INTERPRETER_STAGING_CONCURRENCY,
            )
            files_to_stage, upload_failures = self._upload_and_stage(
                client, selection.files
            )

            staging_notice = _build_staging_notice(
                selection.dropped_by_caps,
                len(staging_inputs),
                selection.read_failures + upload_failures,
            )
            if staging_notice:
                logger.warning(staging_notice)

            return self._execute_stream(
                client,
                placement,
                code,
                ci_session_id=None,
                chat_session_id=None,
                files=files_to_stage or None,
                staging_notice=staging_notice,
            )

    def _record_generated_artifact(self, filename: str, content: bytes) -> None:
        """Keep a generated artifact for re-staging in later (legacy) executions.

        Newest wins per filename; the cache is bounded by
        CODE_INTERPRETER_MAX_GENERATED_ARTIFACTS with oldest-first eviction.
        """
        self._generated_artifacts[filename] = content
        while len(self._generated_artifacts) > CODE_INTERPRETER_MAX_GENERATED_ARTIFACTS:
            self._generated_artifacts.pop(next(iter(self._generated_artifacts)))

    def _use_sessions(self, client: CodeInterpreterClient) -> bool:
        """True when persistent sessions are enabled and the service supports them."""
        return (
            CODE_INTERPRETER_SESSIONS_ENABLED
            and bool(self._chat_session_id)
            and client.supports(
                client.execute_python_in_session_streaming,
                client.keepalive_session,
                client.stage_session_file,
            )
        )

    def _reset_session(self) -> None:
        """Drop the local session id and stale server-side tracking."""
        self._ci_session_id = None
        if self._chat_session_id:
            forget_staged_file_keys(self._chat_session_id)
            forget_workspace_file_ids(self._chat_session_id)

    def _ensure_session(self, client: CodeInterpreterClient) -> str:
        """Return a live session id: keepalive the mapped one or create fresh."""
        if not self._chat_session_id:
            raise _SessionUnavailable("no chat session id")
        chat_session_id = self._chat_session_id

        if self._ci_session_id is None:
            self._ci_session_id = fetch_ci_session_id(chat_session_id)

        if self._ci_session_id:
            try:
                client.keepalive_session(
                    self._ci_session_id,
                    ttl_seconds=CODE_INTERPRETER_SESSION_TTL_SECONDS,
                )
                refresh_session_ttls(
                    chat_session_id, CODE_INTERPRETER_SESSION_TTL_SECONDS
                )
                return self._ci_session_id
            except http_requests.HTTPError as e:
                status_code = e.response.status_code if e.response is not None else None
                if status_code == 404:
                    logger.info(
                        "Mapped code-interpreter session no longer exists; recreating"
                    )
                    self._reset_session()
                else:
                    raise _SessionUnavailable(f"keepalive failed: {e}") from e
            except Exception as e:
                raise _SessionUnavailable(f"keepalive failed: {e}") from e

        try:
            created = client.create_session(
                ttl_seconds=CODE_INTERPRETER_SESSION_TTL_SECONDS,
                install_venv=True,
            )
        except Exception as e:
            raise _SessionUnavailable(f"session creation failed: {e}") from e

        self._ci_session_id = created.session_id
        store_ci_session_id(
            chat_session_id,
            created.session_id,
            CODE_INTERPRETER_SESSION_TTL_SECONDS,
        )
        forget_staged_file_keys(chat_session_id)
        forget_workspace_file_ids(chat_session_id)
        return created.session_id

    def _stage_chat_files_into_session(
        self,
        client: CodeInterpreterClient,
        ci_session_id: str,
        chat_session_id: str,
        chat_files: list[ChatFile],
        code: str,
    ) -> str | None:
        """Upload chat files the session does not have yet; return a staging notice.

        Session files persist in the workspace, so each file is uploaded at
        most once per session (tracked in Redis by filename + content hash).
        """
        selection = _select_files_for_staging(
            chat_files,
            code,
            max_files=CODE_INTERPRETER_MAX_STAGED_FILES,
            max_bytes=CODE_INTERPRETER_MAX_STAGED_BYTES,
            read_concurrency=CODE_INTERPRETER_STAGING_CONCURRENCY,
        )
        staged_keys = fetch_staged_file_keys(chat_session_id)
        new_keys: set[str] = set()
        failed_uploads: list[str] = []
        for chat_file, content in selection.files:
            key = f"{chat_file.filename}:{hashlib.sha256(content).hexdigest()}"
            if key in staged_keys:
                continue
            try:
                client.stage_session_file(
                    ci_session_id,
                    content,
                    _safe_code_interpreter_filename(chat_file.filename),
                )
                new_keys.add(key)
            except Exception:
                logger.warning(
                    "Failed to stage file into code-interpreter session: %s",
                    chat_file.filename,
                )
                failed_uploads.append(chat_file.filename)

        if new_keys:
            staged_keys.update(new_keys)
            store_staged_file_keys(
                chat_session_id,
                staged_keys,
                CODE_INTERPRETER_SESSION_TTL_SECONDS,
            )

        return _build_staging_notice(
            selection.dropped_by_caps,
            len(chat_files),
            selection.read_failures + failed_uploads,
        )

    def _run_in_session(
        self,
        client: CodeInterpreterClient,
        placement: Placement,
        code: str,
        chat_files: list[ChatFile],
    ) -> ToolResponse:
        """Execute in the chat's persistent sandbox session."""
        if not self._chat_session_id:
            raise _SessionUnavailable("no chat session id")
        chat_session_id = self._chat_session_id

        ci_session_id = self._ensure_session(client)
        staging_notice = self._stage_chat_files_into_session(
            client, ci_session_id, chat_session_id, chat_files, code
        )
        # Known workspace files act as the dedup baseline: the server skips
        # them when reporting new/modified files (missing ids are skipped too).
        baseline = fetch_workspace_file_ids(chat_session_id)
        baseline_files: list[FileInput] = [
            {"path": path, "file_id": file_id} for path, file_id in baseline.items()
        ]
        return self._execute_stream(
            client,
            placement,
            code,
            ci_session_id=ci_session_id,
            chat_session_id=chat_session_id,
            files=baseline_files or None,
            staging_notice=staging_notice,
        )

    def _execute_stream(
        self,
        client: CodeInterpreterClient,
        placement: Placement,
        code: str,
        *,
        ci_session_id: str | None,
        chat_session_id: str | None,
        files: list[FileInput] | None,
        staging_notice: str | None,
    ) -> ToolResponse:
        """Run one execution and process its streamed result into a ToolResponse."""
        try:
            logger.debug("Executing code: %s", code)

            # Execute code with streaming (falls back to batch if unavailable)
            stdout_parts: list[str] = []
            stderr_parts: list[str] = []
            result_event: StreamResultEvent | None = None

            if ci_session_id is not None:
                event_stream = client.execute_python_in_session_streaming(
                    ci_session_id,
                    code,
                    timeout_ms=CODE_INTERPRETER_DEFAULT_TIMEOUT_MS,
                    files=files,
                )
            else:
                event_stream = client.execute_streaming(
                    code=code,
                    timeout_ms=CODE_INTERPRETER_DEFAULT_TIMEOUT_MS,
                    files=files,
                )

            for event in event_stream:
                if isinstance(event, StreamOutputEvent):
                    if event.stream == "stdout":
                        stdout_parts.append(event.data)
                    else:
                        stderr_parts.append(event.data)
                    # Emit incremental delta to frontend
                    self.emitter.emit(
                        Packet(
                            placement=placement,
                            obj=PythonToolDelta(
                                stdout=(event.data if event.stream == "stdout" else ""),
                                stderr=(event.data if event.stream == "stderr" else ""),
                            ),
                        )
                    )
                elif isinstance(event, StreamResultEvent):
                    result_event = event
                elif isinstance(event, StreamErrorEvent):
                    raise RuntimeError(f"Code interpreter error: {event.message}")

            if result_event is None:
                raise RuntimeError(
                    "Code interpreter stream ended without a result event"
                )

            return self._process_result(
                client,
                placement,
                result_event,
                stdout_parts,
                stderr_parts,
                ci_session_id=ci_session_id,
                chat_session_id=chat_session_id,
                staging_notice=staging_notice,
            )

        except Exception as e:
            logger.error("Python execution failed: %s", e)
            error_msg = str(e)

            # Emit error delta
            self.emitter.emit(
                Packet(
                    placement=placement,
                    obj=PythonToolDelta(
                        stdout="",
                        stderr=error_msg,
                        file_ids=[],
                    ),
                )
            )

            # Return error result
            result = LlmPythonExecutionResult(
                stdout="",
                stderr=error_msg,
                exit_code=-1,
                timed_out=False,
                generated_files=[],
                error=error_msg,
                staging_notice=staging_notice,
            )

            adapter = TypeAdapter(LlmPythonExecutionResult)
            llm_response = adapter.dump_json(result).decode()

            return ToolResponse(
                rich_response=None,
                llm_facing_response=llm_response,
            )

    def _process_result(
        self,
        client: CodeInterpreterClient,
        placement: Placement,
        result_event: StreamResultEvent,
        stdout_parts: list[str],
        stderr_parts: list[str],
        *,
        ci_session_id: str | None,
        chat_session_id: str | None,
        staging_notice: str | None,
    ) -> ToolResponse:
        """Record reported workspace files and build the ToolResponse."""
        full_stdout = "".join(stdout_parts)
        full_stderr = "".join(stderr_parts)

        # Truncate output for LLM consumption
        truncated_stdout = _truncate_output(
            full_stdout, CODE_INTERPRETER_MAX_OUTPUT_LENGTH, "stdout"
        )
        truncated_stderr = _truncate_output(
            full_stderr, CODE_INTERPRETER_MAX_OUTPUT_LENGTH, "stderr"
        )

        # Track reported files for the session dedup baseline: on the next
        # session execution the server only returns new/modified files.
        reported_workspace_files: dict[str, str] = {}

        # Handle generated files
        generated_files: list[PythonExecutionFile] = []
        generated_file_ids: list[str] = []
        file_ids_to_cleanup: list[str] = []
        file_store = get_default_file_store()
        # Generated images, as (filename, bytes, Onyx file id,
        # PythonExecutionFile) — captions attach to the file entry and
        # the bytes replay to vision-capable chat models.
        images_to_annotate: list[tuple[str, bytes, str, PythonExecutionFile]] = []

        for workspace_file in result_event.files:
            if workspace_file.kind != "file" or not workspace_file.file_id:
                continue

            try:
                # Download file from Code Interpreter
                file_content = client.download_file(workspace_file.file_id)

                # Determine MIME type from file extension
                filename = workspace_file.path.split("/")[-1]
                mime_type, _ = mimetypes.guess_type(filename)
                # Default to binary if we can't determine the type
                mime_type = mime_type or "application/octet-stream"

                # Save to Onyx file store
                onyx_file_id = file_store.save_file(
                    content=BytesIO(file_content),
                    display_name=filename,
                    file_origin=FileOrigin.CHAT_IMAGE_GEN,
                    file_type=mime_type,
                )

                if ci_session_id is not None:
                    reported_workspace_files[workspace_file.path] = (
                        workspace_file.file_id
                    )
                else:
                    # Legacy path only: the sandbox is wiped per call, so keep
                    # artifacts for re-staging in later executions (newest wins,
                    # bounded — oldest evicted first).
                    self._record_generated_artifact(filename, file_content)

                generated_file = PythonExecutionFile(
                    filename=filename,
                    file_link=build_full_frontend_file_url(onyx_file_id),
                )
                generated_files.append(generated_file)
                generated_file_ids.append(onyx_file_id)

                if mime_type.startswith("image/"):
                    images_to_annotate.append(
                        (filename, file_content, onyx_file_id, generated_file)
                    )

                # Mark for cleanup
                file_ids_to_cleanup.append(workspace_file.file_id)

            except Exception as e:
                logger.error(
                    "Failed to handle generated file %s: %s",
                    workspace_file.path,
                    e,
                )

        if ci_session_id is not None and chat_session_id:
            store_workspace_file_ids(
                chat_session_id,
                reported_workspace_files,
                CODE_INTERPRETER_SESSION_TTL_SECONDS,
            )

        # Cleanup Code Interpreter files (generated snapshot copies) — legacy
        # path only. In session mode the copies back the dedup baseline for
        # later calls (they expire on the service's own file TTL), and the
        # session workspace keeps the file itself.
        if ci_session_id is None:
            for ci_file_id in file_ids_to_cleanup:
                try:
                    client.delete_file(ci_file_id)
                except Exception as e:
                    logger.error(
                        "Failed to delete Code Interpreter generated file %s: %s",
                        ci_file_id,
                        e,
                    )

        # Note: staged input files are intentionally not deleted here because
        # _uploaded_file_cache reuses their file_ids across iterations. They are
        # orphaned when the session ends, but the code interpreter cleans up
        # stale files on its own TTL.

        # Describe generated images with the configured captioning model
        # so the agent learns what they contain. Degrades to no caption
        # when no vision model is available.
        annotations: list[str | None] = [None] * len(images_to_annotate)
        if images_to_annotate:
            vision_llm = get_tool_vision_llm()
            if vision_llm is not None:
                annotations = annotate_images_in_parallel(
                    vision_llm,
                    [
                        (filename, content)
                        for filename, content, _, _ in images_to_annotate
                    ],
                )
            for (_, _, _, generated_file), annotation in zip(
                images_to_annotate, annotations, strict=True
            ):
                generated_file.image_caption = annotation

        # Emit file_ids once files are processed
        if generated_file_ids:
            self.emitter.emit(
                Packet(
                    placement=placement,
                    obj=PythonToolDelta(
                        file_ids=generated_file_ids,
                        files=[
                            PythonToolGeneratedFile(
                                filename=generated_file.filename,
                                file_id=generated_file_id,
                            )
                            for generated_file, generated_file_id in zip(
                                generated_files, generated_file_ids, strict=True
                            )
                        ],
                    ),
                )
            )

        # Build result
        files_notice = FILES_NOTICE_TEMPLATE if generated_files else None
        if ci_session_id is not None:
            files_notice = (
                f"{files_notice} {SESSION_NOTICE_TEMPLATE}"
                if files_notice
                else SESSION_NOTICE_TEMPLATE
            )

        result = LlmPythonExecutionResult(
            stdout=truncated_stdout,
            stderr=truncated_stderr,
            exit_code=result_event.exit_code,
            timed_out=result_event.timed_out,
            generated_files=generated_files,
            error=(None if result_event.exit_code == 0 else truncated_stderr),
            staging_notice=staging_notice,
            files_notice=files_notice,
        )

        # Serialize result for LLM
        adapter = TypeAdapter(LlmPythonExecutionResult)
        llm_response = adapter.dump_json(result).decode()

        return ToolResponse(
            rich_response=PythonToolRichResponse(
                generated_files=generated_files,
                tool_images=[
                    ToolResponseImage(
                        filename=generated_file.filename,
                        file_id=onyx_file_id,
                        content=content,
                    )
                    for _, content, onyx_file_id, generated_file in images_to_annotate
                ],
            ),
            llm_facing_response=llm_response,
        )

    @classmethod
    @override
    def should_emit_argument_deltas(cls) -> bool:
        return True
