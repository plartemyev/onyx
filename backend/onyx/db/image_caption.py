"""Content-addressed cache for vision captions produced by tool image annotation.

The same image bytes reappear across agent cycles, retries, and turns of a
chat; re-annotating them pays a vision call for a caption that is already
stored. Rows are keyed by (image content, model, prompt), so a caption is only
reused when the call would produce the same answer anyway."""

import hashlib

from sqlalchemy import select
from sqlalchemy.dialects.postgresql import insert as pg_insert
from sqlalchemy.orm import Session

from onyx.db.engine.sql_engine import get_session_with_current_tenant_if_none
from onyx.db.models import ImageCaptionCache


def image_caption_content_hash(image_data: bytes) -> str:
    """Stable key for the image itself: sha256 of the raw bytes."""
    return hashlib.sha256(image_data).hexdigest()


def image_caption_prompt_hash(system_prompt: str, question: str | None) -> str:
    """Stable key for what the caption is asked to do."""
    material = f"{system_prompt}\x00{question or ''}"
    return hashlib.sha256(material.encode("utf-8")).hexdigest()


def get_cached_image_caption(
    content_hash: str,
    model_name: str,
    prompt_hash: str,
    db_session: Session | None = None,
) -> str | None:
    """Return the stored caption for this (bytes, model, prompt) triple, or
    None when it was never annotated (or was annotated differently)."""
    with get_session_with_current_tenant_if_none(db_session) as session:
        caption = session.scalars(
            select(ImageCaptionCache.caption)
            .where(
                ImageCaptionCache.content_hash == content_hash,
                ImageCaptionCache.model_name == model_name,
                ImageCaptionCache.prompt_hash == prompt_hash,
            )
            .limit(1)
        ).first()
    return caption


def store_image_caption(
    content_hash: str,
    model_name: str,
    prompt_hash: str,
    caption: str,
    db_session: Session | None = None,
) -> None:
    """Store a caption. Concurrent annotators of the same key are fine: the
    first insert wins, the rest do nothing."""
    with get_session_with_current_tenant_if_none(db_session) as session:
        stmt = pg_insert(ImageCaptionCache).values(
            content_hash=content_hash,
            model_name=model_name,
            prompt_hash=prompt_hash,
            caption=caption,
        )
        session.execute(
            stmt.on_conflict_do_nothing(
                constraint="uq_image_caption_cache_key",
            )
        )
        session.commit()
