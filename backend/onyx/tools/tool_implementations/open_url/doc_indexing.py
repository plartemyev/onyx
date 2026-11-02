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

import time
from io import BytesIO
from typing import Callable
from uuid import UUID, uuid4

from sqlalchemy.orm import Session

from onyx.configs.app_configs import (
    OPEN_URL_DOC_INDEXING,
    OPEN_URL_INDEX_MIN_CHARS,
    OPEN_URL_INDEX_WAIT_SECONDS,
)
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
from onyx.utils.logger import setup_logger
from onyx.utils.url import normalize_url
from shared_configs.contextvars import (
    get_current_incognito_record_mode,
    get_current_tenant_id,
)

logger = setup_logger()

# How often the hold loop re-reads user_file statuses.
_OPEN_URL_INDEX_POLL_SECONDS = 2.0
# Terminal statuses the wait stops waiting on (a failed file never becomes
# searchable; the response falls back to the not-ready note for it).
_SEARCH_TERMINAL = (
    UserFileStatus.COMPLETED,
    UserFileStatus.FAILED,
    UserFileStatus.DELETING,
)


def _wait_until_searchable(
    db_session: Session, user_file_ids: list[UUID], budget_seconds: float
) -> None:
    """Hold until every user file reaches a terminal indexing state (or the
    budget runs out). Bounded, so a busy queue can stall the tool response by
    at most `budget_seconds`."""
    if budget_seconds <= 0:
        return
    deadline = time.monotonic() + budget_seconds
    pending = set(user_file_ids)
    while pending:
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            return
        time.sleep(min(_OPEN_URL_INDEX_POLL_SECONDS, remaining))
        db_session.expire_all()
        statuses = dict(
            db_session.query(UserFile.id, UserFile.status)
            .filter(UserFile.id.in_(pending))
            .all()
        )
        pending = {
            user_file_id
            for user_file_id, status in statuses.items()
            if status not in _SEARCH_TERMINAL
        }


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
                        search_ready=existing.status == UserFileStatus.COMPLETED,
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
                    # Freshly queued: the searchable copy does not exist yet.
                    search_ready=False,
                )
            )
        except Exception:
            logger.exception(
                "Failed to index a crawled page; continuing without the searchable copy"
            )

    # Hold the tool answer until the saved pages are actually searchable, so
    # the response's pointer note is true and internal_search works in the
    # same turn. Bounded by OPEN_URL_INDEX_WAIT_SECONDS.
    ref_ids = [UUID(ref.user_file_id) for ref in refs]
    _wait_until_searchable(db_session, ref_ids, OPEN_URL_INDEX_WAIT_SECONDS)
    db_session.expire_all()
    final_statuses = (
        dict(
            db_session.query(UserFile.id, UserFile.status)
            .filter(UserFile.id.in_(ref_ids))
            .all()
        )
        if ref_ids
        else {}
    )
    for ref in refs:
        ref.search_ready = (
            final_statuses.get(UUID(ref.user_file_id)) == UserFileStatus.COMPLETED
        )
    return refs
