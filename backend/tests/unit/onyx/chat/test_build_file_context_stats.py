"""build_file_context_stats measures the per-document stats for the timeline.

It must keep only the current message's text documents, measure the extracted
text (bytes + tokens via the provided counter), and survive a failing
tokenizer by falling back to the loaded file's stored count.
"""

from onyx.chat.chat_utils import build_file_context_stats
from onyx.chat.models import ChatLoadedFile
from onyx.file_store.models import ChatFileType


def make_loaded_file(
    file_id: str,
    file_type: ChatFileType,
    content_text: str | None,
    token_count: int = 0,
) -> ChatLoadedFile:
    return ChatLoadedFile.lazy_loaded(
        file_id=file_id,
        file_type=file_type,
        filename=f"{file_id}.txt",
        content_text=content_text,
        token_count=token_count,
        loader=lambda: b"",
    )


def test_keeps_only_current_message_text_files() -> None:
    files = [
        make_loaded_file("doc-1", ChatFileType.DOC, "hello"),
        make_loaded_file("doc-2", ChatFileType.DOC, "world"),
        make_loaded_file("img-1", ChatFileType.IMAGE, None),
        make_loaded_file("gen-1", ChatFileType.PLAIN_TEXT, "generated"),
    ]
    stats = build_file_context_stats(
        loaded_files=files,
        file_ids={"doc-1", "img-1"},
        token_counter=lambda text: len(text),
    )
    assert [stat.file_id for stat in stats] == ["doc-1"]
    assert stats[0].content_bytes == len("hello")
    assert stats[0].token_count == len("hello")


def test_missing_content_yields_none_stats() -> None:
    files = [make_loaded_file("doc-1", ChatFileType.DOC, None)]
    stats = build_file_context_stats(
        loaded_files=files,
        file_ids={"doc-1"},
        token_counter=lambda text: len(text),
    )
    assert len(stats) == 1
    assert stats[0].content_bytes is None
    assert stats[0].token_count is None


def test_tokenizer_failure_falls_back_to_stored_count() -> None:
    def failing_counter(_text: str) -> int:
        raise RuntimeError("engine down")

    files = [make_loaded_file("doc-1", ChatFileType.DOC, "hello", token_count=7)]
    stats = build_file_context_stats(
        loaded_files=files,
        file_ids={"doc-1"},
        token_counter=failing_counter,
    )
    assert stats[0].token_count == 7
    assert stats[0].content_bytes == len("hello")


def test_upload_order_is_preserved() -> None:
    files = [
        make_loaded_file("b", ChatFileType.PLAIN_TEXT, "b"),
        make_loaded_file("a", ChatFileType.PLAIN_TEXT, "a"),
    ]
    stats = build_file_context_stats(
        loaded_files=files,
        file_ids={"a", "b"},
        token_counter=lambda text: len(text),
    )
    assert [stat.file_id for stat in stats] == ["b", "a"]
