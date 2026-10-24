import datetime
import uuid
from typing import List
from uuid import UUID

from fastapi import HTTPException, UploadFile
from pydantic import BaseModel, ConfigDict, Field
from sqlalchemy import func
from sqlalchemy.orm import Session
from starlette.background import BackgroundTasks

from onyx.configs.app_configs import DISABLE_VECTOR_DB
from onyx.configs.constants import (
    CELERY_USER_FILE_PROCESSING_TASK_EXPIRES,
    FileOrigin,
    OnyxCeleryPriority,
    OnyxCeleryQueues,
    OnyxCeleryTask,
)
from onyx.db.enums import UserFileStatus
from onyx.db.models import Project__UserFile, User, UserFile, UserProject
from onyx.server.documents.connector import upload_files
from onyx.server.features.projects.projects_file_utils import (
    RejectedFile,
    categorize_uploaded_files,
)
from onyx.utils.logger import setup_logger
from shared_configs.contextvars import get_current_tenant_id

logger = setup_logger()


class CategorizedFilesResult(BaseModel):
    user_files: list[UserFile]
    rejected_files: list[RejectedFile]
    id_to_temp_id: dict[str, str]
    # Filenames that should be stored but not indexed.
    skip_indexing_filenames: set[str] = Field(default_factory=set)
    # Allow SQLAlchemy ORM models inside this result container
    model_config = ConfigDict(arbitrary_types_allowed=True)

    @property
    def indexable_files(self) -> list[UserFile]:
        return [
            uf
            for uf in self.user_files
            if (uf.name or "") not in self.skip_indexing_filenames
        ]


def build_hashed_file_key(file: UploadFile) -> str:
    name_prefix = (file.filename or "")[:50]
    return f"{file.size}|{name_prefix}"


def create_user_files(
    files: List[UploadFile],
    project_id: int | None,
    user: User,
    db_session: Session,
    link_url: str | None = None,
    temp_id_map: dict[str, str] | None = None,
    incognito_session_id: UUID | None = None,
) -> CategorizedFilesResult:
    # Chat attachments (no project) defer all validation and processing until
    # a sent message references the file. Project uploads keep the full
    # validation and start indexing immediately: project files are context for
    # every message in the project and are not tied to a single send.
    categorized_files = categorize_uploaded_files(
        files, db_session, defer_validation=project_id is None
    )
    # NOTE: At the moment, zip metadata is not used for user files.
    # Should revisit to decide whether this should be a feature.
    upload_response = upload_files(categorized_files.acceptable, FileOrigin.USER_FILE)
    user_files = []
    rejected_files = categorized_files.rejected
    id_to_temp_id: dict[str, str] = {}
    # Pair returned storage paths with the same set of acceptable files we uploaded
    for file_path, file in zip(
        upload_response.file_paths, categorized_files.acceptable, strict=True
    ):
        new_id = uuid.uuid4()
        new_temp_id = (
            temp_id_map.get(build_hashed_file_key(file)) if temp_id_map else None
        )
        if new_temp_id is not None:
            id_to_temp_id[str(new_id)] = new_temp_id
        should_skip = (file.filename or "") in categorized_files.skip_indexing
        if should_skip:
            status = UserFileStatus.SKIPPED
        elif project_id is None:
            status = UserFileStatus.UPLOADED
        else:
            status = UserFileStatus.PROCESSING
        new_file = UserFile(
            id=new_id,
            user_id=user.id,
            file_id=file_path,
            name=file.filename,
            token_count=categorized_files.acceptable_file_to_token_count.get(
                file.filename or ""
            ),
            link_url=link_url,
            content_type=file.content_type,
            file_type=file.content_type,
            status=status,
            incognito=incognito_session_id is not None,
            incognito_session_id=incognito_session_id,
            last_accessed_at=datetime.datetime.now(datetime.timezone.utc),
        )
        # Persist the UserFile first to satisfy FK constraints for association table
        db_session.add(new_file)
        db_session.flush()
        # Incognito uploads may use a project as context but never join it.
        if project_id and incognito_session_id is None:
            project_to_user_file = Project__UserFile(
                project_id=project_id,
                user_file_id=new_file.id,
            )
            db_session.add(project_to_user_file)
        user_files.append(new_file)
    db_session.commit()
    return CategorizedFilesResult(
        user_files=user_files,
        rejected_files=rejected_files,
        id_to_temp_id=id_to_temp_id,
        skip_indexing_filenames=categorized_files.skip_indexing,
    )


def upload_files_to_user_files(
    files: List[UploadFile],
    project_id: int | None,
    user: User,
    temp_id_map: dict[str, str] | None,
    db_session: Session,
    background_tasks: BackgroundTasks | None = None,
    incognito_session_id: UUID | None = None,
) -> CategorizedFilesResult:
    if project_id is not None and user is not None:
        if not check_project_ownership(project_id, user.id, db_session):
            raise HTTPException(status_code=404, detail="Project not found")

    categorized_files_result = create_user_files(
        files,
        project_id,
        user,
        db_session,
        temp_id_map=temp_id_map,
        incognito_session_id=incognito_session_id,
    )
    user_files = categorized_files_result.user_files
    rejected_files = categorized_files_result.rejected_files
    id_to_temp_id = categorized_files_result.id_to_temp_id
    # Deferred (chat-attached) files stay UPLOADED here — processing starts
    # when a sent message references them (trigger_user_file_processing).
    files_to_process = [
        uf
        for uf in categorized_files_result.indexable_files
        if uf.status == UserFileStatus.PROCESSING
    ]
    # Trigger per-file processing immediately for the current tenant
    tenant_id = get_current_tenant_id()
    for rejected_file in rejected_files:
        logger.warning(
            "File %s rejected for %s", rejected_file.filename, rejected_file.reason
        )

    if DISABLE_VECTOR_DB and background_tasks is not None:
        from onyx.background.task_utils import drain_processing_loop

        background_tasks.add_task(drain_processing_loop, tenant_id)
        for user_file in files_to_process:
            logger.info(
                "Queued in-process processing for user_file_id=%s", user_file.id
            )
    else:
        from onyx.background.celery.versioned_apps.client import app as client_app

        for user_file in files_to_process:
            task = client_app.send_task(
                OnyxCeleryTask.PROCESS_SINGLE_USER_FILE,
                kwargs={"user_file_id": user_file.id, "tenant_id": tenant_id},
                queue=OnyxCeleryQueues.USER_FILE_PROCESSING,
                priority=OnyxCeleryPriority.HIGH,
                expires=CELERY_USER_FILE_PROCESSING_TASK_EXPIRES,
            )
            logger.info(
                "Triggered indexing for user_file_id=%s with task_id=%s",
                user_file.id,
                task.id,
            )

    return CategorizedFilesResult(
        user_files=user_files,
        rejected_files=rejected_files,
        id_to_temp_id=id_to_temp_id,
        skip_indexing_filenames=categorized_files_result.skip_indexing_filenames,
    )


def trigger_user_file_processing(
    db_session: Session,
    user_file_ids: list[UUID],
    tenant_id: str,
) -> int:
    """Start processing for deferred (UPLOADED) user files.

    Called when a sent message references files. Flips UPLOADED → PROCESSING
    (so the beat safety net can also recover these) and hands the files to the
    processing queue. Fire-and-forget: callers must not block on completion.

    Returns the number of files transitioned. Already-processing/completed
    files are ignored, so repeated sends and resends are no-ops.
    """
    if not user_file_ids:
        return 0

    deferred_files = (
        db_session.query(UserFile)
        .filter(
            UserFile.id.in_(user_file_ids),
            UserFile.status == UserFileStatus.UPLOADED,
        )
        .all()
    )
    if not deferred_files:
        return 0

    for user_file in deferred_files:
        user_file.status = UserFileStatus.PROCESSING
    db_session.commit()

    if DISABLE_VECTOR_DB:
        # Lite deployments have no user-file Celery consumer; the request path
        # cannot use BackgroundTasks here, so drain in a daemon thread.
        import threading

        from onyx.background.task_utils import drain_processing_loop

        threading.Thread(
            target=drain_processing_loop,
            kwargs={"tenant_id": tenant_id},
            name="user-file-processing-trigger",
            daemon=True,
        ).start()
    else:
        from onyx.background.celery.versioned_apps.client import app as client_app

        for user_file in deferred_files:
            task = client_app.send_task(
                OnyxCeleryTask.PROCESS_SINGLE_USER_FILE,
                kwargs={"user_file_id": user_file.id, "tenant_id": tenant_id},
                queue=OnyxCeleryQueues.USER_FILE_PROCESSING,
                priority=OnyxCeleryPriority.HIGH,
                expires=CELERY_USER_FILE_PROCESSING_TASK_EXPIRES,
            )
            logger.info(
                "Triggered deferred processing for user_file_id=%s with task_id=%s",
                user_file.id,
                task.id,
            )

    return len(deferred_files)


def check_project_ownership(
    project_id: int, user_id: UUID | None, db_session: Session
) -> bool:
    # In no-auth mode, all projects are accessible
    if user_id is None:
        # Verify project exists
        return (
            db_session.query(UserProject).filter(UserProject.id == project_id).first()
            is not None
        )

    return (
        db_session.query(UserProject)
        .filter(UserProject.id == project_id, UserProject.user_id == user_id)
        .first()
        is not None
    )


def get_user_files_from_project(
    project_id: int, user_id: UUID | None, db_session: Session
) -> list[UserFile]:
    # First check if the user owns the project
    if not check_project_ownership(project_id, user_id, db_session):
        return []

    return (
        db_session.query(UserFile)
        .join(Project__UserFile)
        .filter(Project__UserFile.project_id == project_id)
        .all()
    )


def get_project_instructions(db_session: Session, project_id: int | None) -> str | None:
    """Return the project's instruction text from the project, else None.

    Safe helper that swallows DB errors and returns None on any failure.
    """
    if not project_id:
        return None
    try:
        project = (
            db_session.query(UserProject)
            .filter(UserProject.id == project_id)
            .one_or_none()
        )
        if not project or not project.instructions:
            return None
        instructions = project.instructions.strip()
        return instructions or None
    except Exception:
        return None


def get_project_token_count(
    project_id: int | None,
    user_id: UUID | None,
    db_session: Session,
) -> int:
    """Return sum of token_count for all user files in the given project.

    If project_id is None, returns 0.
    """
    if project_id is None:
        return 0

    total_tokens = (
        db_session.query(func.coalesce(func.sum(UserFile.token_count), 0))
        .filter(
            UserFile.user_id == user_id,
            UserFile.projects.any(id=project_id),
        )
        .scalar()
        or 0
    )

    return int(total_tokens)
