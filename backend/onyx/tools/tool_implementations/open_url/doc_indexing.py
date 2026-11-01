"""Index crawled pages as chat-searchable user files.

A large open_url fetch is saved verbatim (the cleaned markdown the tool
already produces) as a UserFile and queued for the standard user-file
indexing pipeline. The page becomes reachable through internal search and
read_file in later turns of the chat, so compaction can stub the raw
payload with a digest plus a pointer instead of losing the text.

Fire-and-forget by design: indexing runs on the user-file pipeline, and
any failure here only costs the searchable copy — the tool response still
carries the fetched content. Incognito turns index nothing: the session
must not leave searchable artifacts behind.
"""

from __future__ import annotations

import logging
from io import BytesIO
from typing import Callable
from uuid import uuid4

from sqlalchemy.orm import Session

from onyx.configs.app_configs import OPEN_URL_DOC_INDEXING, OPEN_URL_INDEX_MIN_CHARS
from onyx.configs.constants import (
    CELERY_USER_FILE_PROCESSING_TASK_EXPIRES,
    FileOrigin,
    OnyxCeleryPriority,
    OnyxCeleryQueues,
    OnyxCeleryTask,
)
from onyx.context.search.models import InferenceSection
from onyx.db.enums import UserFileStatus
from onyx.db.models import User, UserFile
from onyx.file_store.utils import get_default_file_store
from onyx.tools.models import IndexedPageRef
from onyx.utils.url import normalize_url
from shared_configs.contextvars import (
    get_current_incognito_record_mode,
    get_current_tenant_id,
)

logger = logging.getLogger(__name__)


def section_link(section: InferenceSection) -> str | None:
    if section.center_chunk.source_links:
        return next(iter(section.center_chunk.source_links.values()), None)
    return None


def _existing_user_file_for_link(
    db_session: Session, user: User, link: str
) -> UserFile | None:
    """The user's most recent non-terminal file for this URL, if any.

    Re-fetching a URL reuses the existing document instead of indexing a
    second copy of the same text.
    """
    return (
        db_session.query(UserFile)
        .filter(
            UserFile.user_id == user.id,
            UserFile.link_url == link,
            UserFile.status.notin_([UserFileStatus.DELETING, UserFileStatus.FAILED]),
        )
        .order_by(UserFile.created_at.desc())
        .first()
    )


def _enqueue_user_file_processing(user_file_id: str) -> None:
    """Hand the file to the user-file processing pipeline."""
    tenant_id = get_current_tenant_id()

    from onyx.configs.app_configs import DISABLE_VECTOR_DB

    if DISABLE_VECTOR_DB:
        # Lite deployments have no user-file Celery consumer; drain in a
        # daemon thread (same fallback as trigger_user_file_processing).
        import threading

        from onyx.background.task_utils import drain_processing_loop

        threading.Thread(
            target=drain_processing_loop,
            kwargs={"tenant_id": tenant_id},
            name="open-url-doc-indexing",
            daemon=True,
        ).start()
        return

    from onyx.background.celery.versioned_apps.client import app as client_app

    client_app.send_task(
        OnyxCeleryTask.PROCESS_SINGLE_USER_FILE,
        kwargs={"user_file_id": user_file_id, "tenant_id": tenant_id},
        queue=OnyxCeleryQueues.USER_FILE_PROCESSING,
        priority=OnyxCeleryPriority.HIGH,
        expires=CELERY_USER_FILE_PROCESSING_TASK_EXPIRES,
    )


def index_crawled_pages(
    sections: list[InferenceSection],
    user: User,
    db_session: Session,
    token_counter: Callable[[str], int] | None,
) -> list[IndexedPageRef]:
    """Save each large crawled section as an indexed user file.

    Skipped silently (with a log line) for incognito turns, small pages,
    and anything that fails — the tool response carries the content either
    way. Commits its own rows.
    """
    if not OPEN_URL_DOC_INDEXING:
        return []
    if get_current_incognito_record_mode() is not None:
        return []

    refs: list[IndexedPageRef] = []
    for section in sections:
        try:
            content = section.combined_content or ""
            if len(content) < OPEN_URL_INDEX_MIN_CHARS:
                continue
            link = section_link(section)
            if not link:
                continue
            normalized_link = normalize_url(link)

            existing = _existing_user_file_for_link(db_session, user, normalized_link)
            if existing is not None:
                refs.append(
                    IndexedPageRef(
                        url=normalized_link,
                        user_file_id=str(existing.id),
                        file_id=existing.file_id,
                        name=existing.name,
                    )
                )
                continue

            title = (section.center_chunk.semantic_identifier or link).strip()
            name = f"{title[:100]}.md" if title else f"{normalized_link}.md"
            file_id = get_default_file_store().save_file(
                content=BytesIO(content.encode("utf-8")),
                display_name=name,
                file_origin=FileOrigin.USER_FILE,
                file_type="text/markdown",
            )
            token_count = token_counter(content) if token_counter else len(content) // 4
            user_file = UserFile(
                id=uuid4(),
                user_id=user.id,
                file_id=file_id,
                name=name,
                token_count=token_count,
                link_url=normalized_link,
                content_type="text/markdown",
                file_type="text/markdown",
                status=UserFileStatus.PROCESSING,
            )
            db_session.add(user_file)
            db_session.commit()

            _enqueue_user_file_processing(str(user_file.id))
            logger.info(
                "Indexed crawled page %s as user_file_id=%s (%d chars)",
                normalized_link,
                user_file.id,
                len(content),
            )
            refs.append(
                IndexedPageRef(
                    url=normalized_link,
                    user_file_id=str(user_file.id),
                    file_id=file_id,
                    name=name,
                )
            )
        except Exception:
            logger.exception(
                "Failed to index a crawled page; continuing without the searchable copy"
            )
    return refs
