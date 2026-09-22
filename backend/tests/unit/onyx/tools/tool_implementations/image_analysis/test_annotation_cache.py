"""Tests for the caption cache in the shared image-annotation helpers.

Identical (bytes, model, prompt) triples must reuse the stored caption
instead of paying another vision call, and cache problems must never fail
the annotation."""

from typing import Any
from unittest.mock import MagicMock

import pytest

from onyx.db.image_caption import image_caption_content_hash, image_caption_prompt_hash
from onyx.tools.tool_implementations.image_analysis import shared

PNG_BYTES = b"\x89PNG\r\n\x1a\npngdata"
JPEG_BYTES = b"\xff\xd8\xff\xe0jpegdata"


def _llm(model_name: str = "ornith-1.5:9b") -> Any:
    llm = MagicMock()
    llm.config.model_name = model_name
    return llm


@pytest.fixture()
def cache_state(monkeypatch: pytest.MonkeyPatch) -> dict[str, Any]:
    """In-memory stand-in for the caption cache, wired into shared.py."""
    state: dict[str, Any] = {
        "rows": {},
        "lookups": 0,
        "stores": 0,
        "lookup_error": None,
        "store_error": None,
    }

    def _lookup(content_hash: str, model_name: str, prompt_hash: str) -> str | None:
        state["lookups"] += 1
        if state["lookup_error"] is not None:
            raise state["lookup_error"]
        return state["rows"].get((content_hash, model_name, prompt_hash))

    def _store(
        content_hash: str, model_name: str, prompt_hash: str, caption: str
    ) -> None:
        state["stores"] += 1
        if state["store_error"] is not None:
            raise state["store_error"]
        state["rows"][(content_hash, model_name, prompt_hash)] = caption

    monkeypatch.setattr(shared, "get_cached_image_caption", _lookup)
    monkeypatch.setattr(shared, "store_image_caption", _store)
    return state


def test_cache_hit_skips_vision_call(
    monkeypatch: pytest.MonkeyPatch, cache_state: dict[str, Any]
) -> None:
    llm = _llm()
    key = (
        image_caption_content_hash(PNG_BYTES),
        "ornith-1.5:9b",
        image_caption_prompt_hash(shared.AGENT_IMAGE_ANNOTATION_SYSTEM_PROMPT, None),
    )
    cache_state["rows"][key] = "A rocket parked on the launch pad."

    summarize = MagicMock()
    monkeypatch.setattr(shared, "summarize_image_pipeline", summarize)

    caption = shared.annotate_image(llm, PNG_BYTES, "meme.png")

    assert caption == "A rocket parked on the launch pad."
    summarize.assert_not_called()
    assert cache_state["stores"] == 0


def test_cache_miss_annotates_and_stores(
    monkeypatch: pytest.MonkeyPatch, cache_state: dict[str, Any]
) -> None:
    llm = _llm()
    monkeypatch.setattr(
        shared, "summarize_image_pipeline", MagicMock(return_value="fresh caption")
    )

    caption = shared.annotate_image(llm, PNG_BYTES, "meme.png")

    assert caption == "fresh caption"
    assert cache_state["stores"] == 1
    assert (
        cache_state["rows"][
            (
                image_caption_content_hash(PNG_BYTES),
                "ornith-1.5:9b",
                image_caption_prompt_hash(
                    shared.AGENT_IMAGE_ANNOTATION_SYSTEM_PROMPT, None
                ),
            )
        ]
        == "fresh caption"
    )


def test_different_question_is_a_different_cache_key(
    monkeypatch: pytest.MonkeyPatch, cache_state: dict[str, Any]
) -> None:
    llm = _llm()
    monkeypatch.setattr(
        shared, "summarize_image_pipeline", MagicMock(return_value="answer")
    )

    shared.annotate_image(llm, PNG_BYTES, "meme.png", question="what is funny?")
    shared.annotate_image(llm, PNG_BYTES, "meme.png", question="what text is shown?")

    assert cache_state["lookups"] == 2
    assert cache_state["stores"] == 2


def test_unsupported_format_is_not_stored(
    monkeypatch: pytest.MonkeyPatch, cache_state: dict[str, Any]
) -> None:
    from onyx.file_processing.image_summarization import UnsupportedImageFormatError

    llm = _llm()

    def _raise(*args: Any, **kwargs: Any) -> str:  # noqa: ARG001
        raise UnsupportedImageFormatError("unsupported")

    monkeypatch.setattr(shared, "summarize_image_pipeline", _raise)

    assert shared.annotate_image(llm, PNG_BYTES, "meme.png") is None
    assert cache_state["stores"] == 0


def test_annotation_failure_is_not_stored(
    monkeypatch: pytest.MonkeyPatch, cache_state: dict[str, Any]
) -> None:
    llm = _llm()

    def _raise(*args: Any, **kwargs: Any) -> str:  # noqa: ARG001
        raise ValueError("vision call failed")

    monkeypatch.setattr(shared, "summarize_image_pipeline", _raise)

    assert shared.annotate_image(llm, PNG_BYTES, "meme.png") is None
    assert cache_state["stores"] == 0


def test_cache_errors_do_not_fail_annotation(
    monkeypatch: pytest.MonkeyPatch, cache_state: dict[str, Any]
) -> None:
    llm = _llm()
    cache_state["lookup_error"] = RuntimeError("db down")
    cache_state["store_error"] = RuntimeError("db down")
    monkeypatch.setattr(
        shared, "summarize_image_pipeline", MagicMock(return_value="fresh caption")
    )

    caption = shared.annotate_image(llm, PNG_BYTES, "meme.png")

    assert caption == "fresh caption"


def test_caption_keys_are_stable() -> None:
    assert image_caption_content_hash(PNG_BYTES) == image_caption_content_hash(
        PNG_BYTES
    )
    assert image_caption_content_hash(PNG_BYTES) != image_caption_content_hash(
        JPEG_BYTES
    )
    base = shared.AGENT_IMAGE_ANNOTATION_SYSTEM_PROMPT
    assert image_caption_prompt_hash(base, None) == image_caption_prompt_hash(base, "")
    assert image_caption_prompt_hash(base, "a") != image_caption_prompt_hash(base, "b")
