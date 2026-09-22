from __future__ import annotations

import logging
from collections.abc import Iterator
from pathlib import Path

from fastapi import APIRouter, File, HTTPException, UploadFile, status
from fastapi.responses import Response, StreamingResponse

from app.app_configs import get_settings
from app.models.schemas import (
    BashExecRequest,
    BashExecResponse,
    CreateSessionRequest,
    CreateSessionResponse,
    ExecuteFile,
    ExecuteRequest,
    ExecuteResponse,
    FileMetadataResponse,
    KeepaliveRequest,
    KeepaliveResponse,
    ListFilesResponse,
    SessionFileEntry,
    SessionFilesResponse,
    SessionPythonRequest,
    StreamErrorEvent,
    StreamOutputEvent,
    StreamResultEvent,
    UploadFileResponse,
    WorkspaceFile,
)
from app.services.executor_base import (
    EntryKind,
    SessionNotFoundError,
    StreamChunk,
    StreamResult,
    WorkspaceEntry,
)
from app.services.executor_factory import execute_python, execute_python_streaming, get_executor
from app.services.file_storage import FileStorageService

router = APIRouter()

logger = logging.getLogger(__name__)

# Initialize file storage service
_file_storage: FileStorageService | None = None


def get_file_storage() -> FileStorageService:
    """Get or create the global FileStorageService instance."""
    global _file_storage
    if _file_storage is None:
        settings = get_settings()
        _file_storage = FileStorageService(Path(settings.file_storage_dir))
    return _file_storage


def _validate_timeout(req: ExecuteRequest) -> None:
    settings = get_settings()
    if req.timeout_ms > settings.max_exec_timeout_ms:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
            detail=f"timeout_ms exceeds maximum of {settings.max_exec_timeout_ms} ms",
        )


def _resolve_uploaded_files(
    files: list[ExecuteFile],
    storage: FileStorageService,
    required: bool = True,
) -> tuple[list[tuple[str, bytes]], dict[str, bytes]]:
    """Resolve uploaded file IDs into content for the executor.

    Returns (staged_files, input_files_map). With ``required=False`` (session
    dedup baselines), missing IDs are skipped instead of failing the request —
    the service's file TTL may have reclaimed old snapshot copies.
    """
    staged_files: list[tuple[str, bytes]] = []
    input_files_map: dict[str, bytes] = {}
    for file in files:
        try:
            content, _ = storage.get_file(file.file_id)
        except FileNotFoundError as exc:
            if not required:
                logger.info(
                    "Skipping missing file %s (id %s) in session dedup baseline",
                    file.path,
                    file.file_id,
                )
                continue
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"File with ID '{file.file_id}' not found for path '{file.path}'.",
            ) from exc
        staged_files.append((file.path, content))
        input_files_map[file.path] = content
    return staged_files, input_files_map


def _stage_request_files(
    req: ExecuteRequest,
    storage: FileStorageService,
) -> tuple[list[tuple[str, bytes]], dict[str, bytes]]:
    """Resolve uploaded file IDs into content for the executor.

    Returns (staged_files, input_files_map).
    """
    return _resolve_uploaded_files(req.files, storage)


def _save_workspace_files(
    entries: tuple[WorkspaceEntry, ...],
    input_files_map: dict[str, bytes],
    storage: FileStorageService,
) -> list[WorkspaceFile]:
    """Filter and save new/modified workspace files to storage."""
    workspace_files: list[WorkspaceFile] = []
    for entry in entries:
        if entry.kind == EntryKind.DIRECTORY:
            continue
        if entry.kind == EntryKind.FILE and entry.content is not None:
            if entry.path in input_files_map and entry.content == input_files_map[entry.path]:
                continue
            file_id = storage.save_file(entry.content, entry.path)
            workspace_files.append(WorkspaceFile(path=entry.path, kind=entry.kind, file_id=file_id))
    return workspace_files


@router.post("/execute", response_model=ExecuteResponse, status_code=status.HTTP_200_OK)
def execute(req: ExecuteRequest) -> ExecuteResponse:
    """Execute provided Python code synchronously within an isolated Docker container."""
    _validate_timeout(req)
    settings = get_settings()
    storage = get_file_storage()
    staged_files, input_files_map = _stage_request_files(req, storage)

    try:
        result = execute_python(
            code=req.code,
            stdin=req.stdin,
            timeout_ms=req.timeout_ms,
            max_output_bytes=settings.max_output_bytes,
            cpu_time_limit_sec=settings.cpu_time_limit_sec,
            memory_limit_mb=settings.memory_limit_mb,
            files=staged_files,
            last_line_interactive=req.last_line_interactive,
        )
    except ValueError as exc:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
            detail=str(exc),
        ) from exc

    return ExecuteResponse(
        stdout=result.stdout,
        stderr=result.stderr,
        exit_code=result.exit_code,
        timed_out=result.timed_out,
        duration_ms=result.duration_ms,
        files=_save_workspace_files(result.files, input_files_map, storage),
    )


@router.post("/execute/stream")
def execute_stream(req: ExecuteRequest) -> StreamingResponse:
    """Execute Python code with streaming output via Server-Sent Events."""
    _validate_timeout(req)
    settings = get_settings()
    storage = get_file_storage()
    staged_files, input_files_map = _stage_request_files(req, storage)

    def generate() -> Iterator[str]:
        try:
            for event in execute_python_streaming(
                code=req.code,
                stdin=req.stdin,
                timeout_ms=req.timeout_ms,
                max_output_bytes=settings.max_output_bytes,
                cpu_time_limit_sec=settings.cpu_time_limit_sec,
                memory_limit_mb=settings.memory_limit_mb,
                files=staged_files,
                last_line_interactive=req.last_line_interactive,
            ):
                if isinstance(event, StreamChunk):
                    yield StreamOutputEvent(stream=event.stream, data=event.data).to_sse()

                elif isinstance(event, StreamResult):
                    yield StreamResultEvent(
                        exit_code=event.exit_code,
                        timed_out=event.timed_out,
                        duration_ms=event.duration_ms,
                        files=_save_workspace_files(event.files, input_files_map, storage),
                    ).to_sse()

        except Exception as exc:
            yield StreamErrorEvent(message=str(exc)).to_sse()

    return StreamingResponse(
        generate(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


@router.post("/files", response_model=UploadFileResponse, status_code=status.HTTP_201_CREATED)
async def upload_file(file: UploadFile = File(...)) -> UploadFileResponse:  # noqa: B008
    """Upload a file for later use in code execution."""
    settings = get_settings()
    storage = get_file_storage()

    # Read file content
    content = await file.read()

    # Validate file size
    max_size_bytes = settings.max_file_size_mb * 1024 * 1024
    if len(content) > max_size_bytes:
        raise HTTPException(
            status_code=status.HTTP_413_CONTENT_TOO_LARGE,
            detail=f"File size exceeds maximum of {settings.max_file_size_mb} MB",
        )

    # Save file and get ID
    filename = file.filename or "unnamed"
    file_id = storage.save_file(content, filename)

    return UploadFileResponse(
        file_id=file_id,
        filename=filename,
        size_bytes=len(content),
    )


@router.get("/files/{file_id}")
async def download_file(file_id: str) -> Response:
    """Download a previously uploaded file by its ID."""
    storage = get_file_storage()

    try:
        content, metadata = storage.get_file(file_id)
    except FileNotFoundError as exc:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"File with ID '{file_id}' not found",
        ) from exc

    return Response(
        content=content,
        media_type="application/octet-stream",
        headers={
            "Content-Disposition": f'attachment; filename="{metadata.filename}"',
        },
    )


@router.get("/files", response_model=ListFilesResponse, status_code=status.HTTP_200_OK)
def list_files() -> ListFilesResponse:
    """List all uploaded files with their metadata."""
    storage = get_file_storage()
    files = storage.list_files()

    return ListFilesResponse(
        files=[
            FileMetadataResponse(
                file_id=f.file_id,
                filename=f.filename,
                size_bytes=f.size_bytes,
                upload_time=f.upload_time,
            )
            for f in files
        ]
    )


@router.delete("/files/{file_id}")
def delete_file(file_id: str) -> Response:
    """Delete a previously uploaded file by its ID."""
    storage = get_file_storage()

    if not storage.delete_file(file_id):
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"File with ID '{file_id}' not found",
        )

    return Response(status_code=status.HTTP_204_NO_CONTENT)


@router.post(
    "/sessions",
    response_model=CreateSessionResponse,
    status_code=status.HTTP_201_CREATED,
)
def create_session(req: CreateSessionRequest) -> CreateSessionResponse:
    """Create a long-lived code-executor session with the given TTL.

    The session is guaranteed to be torn down at or before its (possibly
    extended) TTL, even if the API service crashes and restarts. The
    workspace is volume-backed, so files persist across calls; an optional
    per-session venv lets package installs persist too.
    """
    settings = get_settings()
    storage = get_file_storage()
    staged_files, _ = _resolve_uploaded_files(req.files, storage)

    try:
        info = get_executor().create_session(
            ttl_seconds=req.ttl_seconds,
            files=staged_files,
            cpu_time_limit_sec=settings.cpu_time_limit_sec,
            memory_limit_mb=settings.memory_limit_mb,
            network_enabled=req.network_enabled,
            install_venv=req.install_venv,
        )
    except NotImplementedError as exc:
        raise HTTPException(
            status_code=status.HTTP_501_NOT_IMPLEMENTED,
            detail=str(exc),
        ) from exc
    except ValueError as exc:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
            detail=str(exc),
        ) from exc

    return CreateSessionResponse(
        session_id=info.session_id,
        expires_at=info.expires_at,
    )


@router.delete("/sessions/{session_id}", status_code=status.HTTP_204_NO_CONTENT)
def delete_session(session_id: str) -> Response:
    """Tear down a session pod by ID."""
    try:
        deleted = get_executor().delete_session(session_id)
    except NotImplementedError as exc:
        raise HTTPException(
            status_code=status.HTTP_501_NOT_IMPLEMENTED,
            detail=str(exc),
        ) from exc

    if not deleted:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Session '{session_id}' not found",
        )

    return Response(status_code=status.HTTP_204_NO_CONTENT)


@router.post(
    "/sessions/{session_id}/bash",
    response_model=BashExecResponse,
    status_code=status.HTTP_200_OK,
)
def session_exec_bash(session_id: str, req: BashExecRequest) -> BashExecResponse:
    """Run a bash command inside an existing session.

    The session pod has no network access (enforced at session creation), and
    that restriction continues to apply for every command run via this route.
    """
    settings = get_settings()
    if req.timeout_ms > settings.max_exec_timeout_ms:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
            detail=f"timeout_ms exceeds maximum of {settings.max_exec_timeout_ms} ms",
        )

    try:
        result = get_executor().execute_bash_in_session(
            session_id,
            cmd=req.cmd,
            timeout_ms=req.timeout_ms,
            max_output_bytes=settings.max_output_bytes,
        )
    except SessionNotFoundError as exc:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=str(exc),
        ) from exc
    except NotImplementedError as exc:
        raise HTTPException(
            status_code=status.HTTP_501_NOT_IMPLEMENTED,
            detail=str(exc),
        ) from exc

    return BashExecResponse(
        stdout=result.stdout,
        stderr=result.stderr,
        exit_code=result.exit_code,
        timed_out=result.timed_out,
        duration_ms=result.duration_ms,
    )


def _session_python_impl(session_id: str, req: SessionPythonRequest) -> ExecuteResponse:
    """Run Python inside a session (shared by the sync and streaming routes).

    ``req.files`` is a dedup baseline of files the caller already has, not a
    staging list: the session workspace persists, so new files are staged via
    the session files route and only genuinely new/modified files are returned.
    """
    settings = get_settings()
    if req.timeout_ms > settings.max_exec_timeout_ms:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
            detail=f"timeout_ms exceeds maximum of {settings.max_exec_timeout_ms} ms",
        )
    storage = get_file_storage()
    _, input_files_map = _resolve_uploaded_files(req.files, storage, required=False)

    try:
        result = get_executor().execute_python_in_session(
            session_id,
            code=req.code,
            stdin=req.stdin,
            timeout_ms=req.timeout_ms,
            max_output_bytes=settings.max_output_bytes,
            last_line_interactive=req.last_line_interactive,
        )
    except SessionNotFoundError as exc:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=str(exc),
        ) from exc
    except NotImplementedError as exc:
        raise HTTPException(
            status_code=status.HTTP_501_NOT_IMPLEMENTED,
            detail=str(exc),
        ) from exc
    except ValueError as exc:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
            detail=str(exc),
        ) from exc

    return ExecuteResponse(
        stdout=result.stdout,
        stderr=result.stderr,
        exit_code=result.exit_code,
        timed_out=result.timed_out,
        duration_ms=result.duration_ms,
        files=_save_workspace_files(result.files, input_files_map, storage),
    )


@router.post(
    "/sessions/{session_id}/python",
    response_model=ExecuteResponse,
    status_code=status.HTTP_200_OK,
)
def session_exec_python(session_id: str, req: SessionPythonRequest) -> ExecuteResponse:
    """Execute Python code inside an existing session.

    The workspace persists between calls: files written by earlier calls and
    packages installed into the session venv stay available. Returns new or
    modified workspace files alongside the execution output.
    """
    return _session_python_impl(session_id, req)


@router.post("/sessions/{session_id}/python/stream")
def session_exec_python_stream(session_id: str, req: SessionPythonRequest) -> StreamingResponse:
    """Execute Python code inside an existing session with SSE streaming output."""
    settings = get_settings()
    if req.timeout_ms > settings.max_exec_timeout_ms:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
            detail=f"timeout_ms exceeds maximum of {settings.max_exec_timeout_ms} ms",
        )
    storage = get_file_storage()
    _, input_files_map = _resolve_uploaded_files(req.files, storage, required=False)

    def generate() -> Iterator[str]:
        try:
            for event in get_executor().execute_python_in_session_streaming(
                session_id,
                code=req.code,
                stdin=req.stdin,
                timeout_ms=req.timeout_ms,
                max_output_bytes=settings.max_output_bytes,
                last_line_interactive=req.last_line_interactive,
            ):
                if isinstance(event, StreamChunk):
                    yield StreamOutputEvent(stream=event.stream, data=event.data).to_sse()
                elif isinstance(event, StreamResult):
                    yield StreamResultEvent(
                        exit_code=event.exit_code,
                        timed_out=event.timed_out,
                        duration_ms=event.duration_ms,
                        files=_save_workspace_files(event.files, input_files_map, storage),
                    ).to_sse()
        except SessionNotFoundError as exc:
            yield StreamErrorEvent(message=str(exc)).to_sse()
        except Exception as exc:  # noqa: BLE001
            yield StreamErrorEvent(message=str(exc)).to_sse()

    return StreamingResponse(
        generate(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


@router.post(
    "/sessions/{session_id}/keepalive",
    response_model=KeepaliveResponse,
    status_code=status.HTTP_200_OK,
)
def session_keepalive(session_id: str, req: KeepaliveRequest) -> KeepaliveResponse:
    """Extend a session's expiry to now + ttl_seconds."""
    try:
        info = get_executor().keepalive_session(session_id, ttl_seconds=req.ttl_seconds)
    except SessionNotFoundError as exc:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=str(exc),
        ) from exc
    except NotImplementedError as exc:
        raise HTTPException(
            status_code=status.HTTP_501_NOT_IMPLEMENTED,
            detail=str(exc),
        ) from exc

    return KeepaliveResponse(session_id=info.session_id, expires_at=info.expires_at)


@router.get(
    "/sessions/{session_id}/files",
    response_model=SessionFilesResponse,
    status_code=status.HTTP_200_OK,
)
def session_list_files(session_id: str) -> SessionFilesResponse:
    """List workspace entries of a session (venv and control files excluded)."""
    try:
        entries = get_executor().list_session_files(session_id)
    except SessionNotFoundError as exc:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=str(exc),
        ) from exc
    except NotImplementedError as exc:
        raise HTTPException(
            status_code=status.HTTP_501_NOT_IMPLEMENTED,
            detail=str(exc),
        ) from exc

    return SessionFilesResponse(
        files=[SessionFileEntry(path=entry.path, kind=entry.kind) for entry in entries]
    )


@router.get("/sessions/{session_id}/files/{file_path:path}")
def session_download_file(session_id: str, file_path: str) -> Response:
    """Read a single file from a session's workspace."""
    try:
        content = get_executor().read_session_file(session_id, file_path)
    except SessionNotFoundError as exc:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=str(exc),
        ) from exc
    except LookupError as exc:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=str(exc),
        ) from exc
    except NotImplementedError as exc:
        raise HTTPException(
            status_code=status.HTTP_501_NOT_IMPLEMENTED,
            detail=str(exc),
        ) from exc

    filename = file_path.rsplit("/", 1)[-1]
    return Response(
        content=content,
        media_type="application/octet-stream",
        headers={
            "Content-Disposition": f'attachment; filename="{filename}"',
        },
    )


@router.post("/sessions/{session_id}/files", status_code=status.HTTP_204_NO_CONTENT)
async def session_stage_file(
    session_id: str,
    file: UploadFile = File(...),  # noqa: B008
    path: str = "",
) -> Response:
    """Stage an additional file into a session's workspace.

    The destination path (relative to the workspace root) is given via the
    ``path`` query parameter; it defaults to the uploaded filename.
    """
    settings = get_settings()
    content = await file.read()
    if len(content) > settings.max_file_size_mb * 1024 * 1024:
        raise HTTPException(
            status_code=status.HTTP_413_CONTENT_TOO_LARGE,
            detail=(f"File size exceeds maximum of {settings.max_file_size_mb} MB"),
        )

    destination = path or file.filename or "unnamed"
    try:
        get_executor().stage_files_in_session(session_id, [(destination, content)])
    except SessionNotFoundError as exc:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=str(exc),
        ) from exc
    except ValueError as exc:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
            detail=str(exc),
        ) from exc
    except NotImplementedError as exc:
        raise HTTPException(
            status_code=status.HTTP_501_NOT_IMPLEMENTED,
            detail=str(exc),
        ) from exc

    return Response(status_code=status.HTTP_204_NO_CONTENT)
