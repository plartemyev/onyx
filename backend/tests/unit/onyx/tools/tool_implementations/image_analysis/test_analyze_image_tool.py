import json
from typing import Any
from unittest.mock import MagicMock

import pytest

from onyx.file_store.models import ChatFileType, InMemoryChatFile
from onyx.server.query_and_chat.placement import Placement
from onyx.tools.models import ToolCallException
from onyx.tools.tool_implementations.image_analysis import analyze_image_tool
from onyx.tools.tool_implementations.image_analysis.analyze_image_tool import (
    ANALYZE_MAX_IMAGES,
    AnalyzeImageTool,
)
from onyx.tools.tool_implementations.open_url.models import FailedFetch
from onyx.tools.tool_implementations.open_url.onyx_web_crawler import FetchedFile

PNG_MAGIC = b"\x89PNG\r\n\x1a\n"
PNG_BYTES = PNG_MAGIC + b"pngdata"
JPEG_BYTES = b"\xff\xd8\xff\xe0jpegdata"


def _fetched(content: bytes, content_type: str | None) -> FetchedFile:
    return FetchedFile(content=content, content_type=content_type)


def _stored_image(
    file_id: str = "stored-1",
    content: bytes = PNG_BYTES,
    file_type: ChatFileType = ChatFileType.IMAGE,
    filename: str | None = "stored.png",
) -> InMemoryChatFile:
    return InMemoryChatFile(
        file_id=file_id,
        content=content,
        file_type=file_type,
        filename=filename,
    )


def _make_tool(
    results: dict[str, Any],
    monkeypatch: pytest.MonkeyPatch,
    vision_llm: Any = None,
    annotations: list[str | None] | None = None,
    stored_files: dict[str, InMemoryChatFile | Exception] | None = None,
) -> tuple[AnalyzeImageTool, MagicMock]:
    """Build an AnalyzeImageTool whose crawler returns the given per-URL
    results and whose file store assigns sequential file ids.

    `stored_files` maps file_ids to what `load_chat_file_by_id` returns for
    them; a missing entry raises, like an unknown file_id would."""
    tool = AnalyzeImageTool(tool_id=1, emitter=MagicMock())

    def _download(
        url: str,
        **kwargs: Any,  # noqa: ARG001
    ) -> FetchedFile | FailedFetch:
        return results[url]

    monkeypatch.setattr(tool._crawler, "download_file_bytes", _download)  # noqa: SLF001

    saved_ids = [f"file-{index}" for index in range(1, 100)]
    file_store = MagicMock()
    file_store.save_file.side_effect = saved_ids
    monkeypatch.setattr(
        analyze_image_tool, "get_default_file_store", lambda: file_store
    )
    monkeypatch.setattr(
        analyze_image_tool,
        "build_full_frontend_file_url",
        lambda file_id: f"https://frontend.test/files/{file_id}",
    )
    monkeypatch.setattr(analyze_image_tool, "get_tool_vision_llm", lambda: vision_llm)

    stored = stored_files or {}

    def _load(file_id: str) -> InMemoryChatFile:
        entry = stored.get(file_id)
        if isinstance(entry, Exception):
            raise entry
        if entry is None:
            raise KeyError(f"no stored file {file_id}")
        return entry

    monkeypatch.setattr(analyze_image_tool, "load_chat_file_by_id", _load)

    if annotations is not None:
        monkeypatch.setattr(
            analyze_image_tool,
            "annotate_images_in_parallel",
            lambda _llm, _images, **_kwargs: annotations,
        )
    return tool, file_store


def _run(
    tool: AnalyzeImageTool,
    urls: list[str] | None = None,
    question: str | None = None,
    file_ids: list[str] | None = None,
) -> Any:
    placement = Placement(turn_index=0, tab_index=0)
    kwargs: dict[str, Any] = {}
    if urls is not None:
        kwargs["urls"] = urls
    if file_ids is not None:
        kwargs["file_ids"] = file_ids
    if question is not None:
        kwargs["question"] = question
    return tool.run(placement=placement, override_kwargs=None, **kwargs)


class TestAnalyzeImageToolRun:
    def test_annotates_images_and_builds_responses(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        vision_llm = MagicMock()
        tool, file_store = _make_tool(
            {
                "https://example.com/chart.png": _fetched(PNG_BYTES, "image/png"),
            },
            monkeypatch,
            vision_llm=vision_llm,
            annotations=["A bar chart of 2024 revenue"],
        )

        # The agent's question must reach the vision-model call
        seen_questions: list[str | None] = []

        def _fake_annotate(
            _llm: Any,
            _images: list[tuple[str, bytes]],
            question: str | None = None,
        ) -> list[str | None]:
            seen_questions.append(question)
            return ["A bar chart of 2024 revenue"]

        monkeypatch.setattr(
            analyze_image_tool, "annotate_images_in_parallel", _fake_annotate
        )

        response = _run(
            tool,
            ["https://example.com/chart.png"],
            question="What does the chart show?",
        )

        assert seen_questions == ["What does the chart show?"]
        assert file_store.save_file.call_count == 1
        rich = response.rich_response
        assert [f.filename for f in rich.files] == ["chart.png"]
        assert rich.files[0].annotation == "A bar chart of 2024 revenue"
        assert rich.failures == []
        assert rich.tool_images[0].content == PNG_BYTES
        assert rich.tool_images[0].file_id == "file-1"

        llm_payload = json.loads(response.llm_facing_response)
        assert llm_payload["images"][0]["annotation"] == "A bar chart of 2024 revenue"
        assert "notice" not in llm_payload

    def test_sniffs_mime_type_when_content_type_missing(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        tool, _ = _make_tool(
            {
                "https://example.com/pic": _fetched(JPEG_BYTES, None),
            },
            monkeypatch,
            annotations=["a photo"],
        )

        response = _run(tool, ["https://example.com/pic"])

        assert [f.mime_type for f in response.rich_response.files] == ["image/jpeg"]

    def test_non_image_url_fails(self, monkeypatch: pytest.MonkeyPatch) -> None:
        tool, _ = _make_tool(
            {
                "https://example.com/page": _fetched(b"<html></html>", "text/html"),
            },
            monkeypatch,
        )

        response = _run(tool, ["https://example.com/page"])

        assert response.rich_response is None
        assert "does not point at an image" in response.llm_facing_response

    def test_crawler_failures_are_reported_not_raised(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        tool, _ = _make_tool(
            {
                "https://example.com/blocked.png": FailedFetch(
                    url="https://example.com/blocked.png",
                    failure_reason="blocked by a Cloudflare bot challenge",
                ),
                "https://example.com/ok.png": _fetched(PNG_BYTES, "image/png"),
            },
            monkeypatch,
            annotations=["ok image"],
        )

        response = _run(
            tool,
            ["https://example.com/blocked.png", "https://example.com/ok.png"],
        )

        rich = response.rich_response
        assert len(rich.files) == 1
        assert rich.failures[0].url == "https://example.com/blocked.png"
        llm_payload = json.loads(response.llm_facing_response)
        assert llm_payload["failures"][0]["failure_reason"] == (
            "blocked by a Cloudflare bot challenge"
        )

    def test_no_vision_model_still_attaches_images(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        tool, _ = _make_tool(
            {
                "https://example.com/chart.png": _fetched(PNG_BYTES, "image/png"),
            },
            monkeypatch,
            vision_llm=None,
        )

        response = _run(tool, ["https://example.com/chart.png"])

        rich = response.rich_response
        assert rich.files[0].annotation is None
        assert rich.tool_images[0].content == PNG_BYTES

        llm_payload = json.loads(response.llm_facing_response)
        assert "No vision model is configured" in llm_payload["notice"]

    def test_missing_params_raises_tool_call_exception(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        tool, _ = _make_tool({}, monkeypatch)
        placement = Placement(turn_index=0, tab_index=0)

        with pytest.raises(ToolCallException):
            tool.run(placement=placement, override_kwargs=None)

    def test_url_cap_applied(self, monkeypatch: pytest.MonkeyPatch) -> None:
        results = {
            f"https://example.com/{index}.png": _fetched(PNG_BYTES, "image/png")
            for index in range(ANALYZE_MAX_IMAGES + 3)
        }
        tool, file_store = _make_tool(results, monkeypatch)

        urls = [f"https://example.com/{index}.png" for index in range(len(results))]
        _run(tool, urls)

        assert file_store.save_file.call_count == ANALYZE_MAX_IMAGES

    def test_url_cap_reported_in_success_response(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """URLs dropped by the per-call cap are called out to the LLM."""
        results = {
            f"https://example.com/{index}.png": _fetched(PNG_BYTES, "image/png")
            for index in range(ANALYZE_MAX_IMAGES + 3)
        }
        tool, _ = _make_tool(results, monkeypatch)

        urls = [f"https://example.com/{index}.png" for index in range(len(results))]
        response = _run(tool, urls)

        llm_payload = json.loads(response.llm_facing_response)
        assert (
            f"{ANALYZE_MAX_IMAGES} images were attempted" in llm_payload["note"]
            and "3 were skipped" in llm_payload["note"]
        )

    def test_url_cap_reported_when_all_fail(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        results = {
            f"https://example.com/{index}.png": FailedFetch(
                url=f"https://example.com/{index}.png", failure_reason="HTTP 404"
            )
            for index in range(ANALYZE_MAX_IMAGES + 2)
        }
        tool, _ = _make_tool(results, monkeypatch)

        urls = [f"https://example.com/{index}.png" for index in range(len(results))]
        response = _run(tool, urls)

        assert response.rich_response is None
        assert "2 were skipped" in response.llm_facing_response


class TestAnalyzeImageFileIds:
    def test_analyzes_stored_file_by_id(self, monkeypatch: pytest.MonkeyPatch) -> None:
        tool, file_store = _make_tool(
            {},
            monkeypatch,
            vision_llm=MagicMock(),
            annotations=["A screenshot of the login page"],
            stored_files={"stored-1": _stored_image()},
        )

        response = _run(tool, file_ids=["stored-1"])

        # Already in the file store — nothing new saved
        assert file_store.save_file.call_count == 0
        rich = response.rich_response
        assert [f.filename for f in rich.files] == ["stored.png"]
        assert rich.files[0].file_id == "stored-1"
        assert rich.files[0].annotation == "A screenshot of the login page"
        # `url` points at the stored copy so the LLM can embed it
        assert rich.files[0].url == rich.files[0].file_url
        assert rich.failures == []
        assert rich.tool_images[0].content == PNG_BYTES

        llm_payload = json.loads(response.llm_facing_response)
        assert llm_payload["images"][0]["file_id"] == "stored-1"
        assert llm_payload["failures"] == []

    def test_mixed_urls_and_file_ids(self, monkeypatch: pytest.MonkeyPatch) -> None:
        tool, file_store = _make_tool(
            {
                "https://example.com/chart.png": _fetched(JPEG_BYTES, "image/jpeg"),
            },
            monkeypatch,
            vision_llm=MagicMock(),
            annotations=["annotated", "annotated"],
            stored_files={"stored-1": _stored_image()},
        )

        response = _run(
            tool,
            urls=["https://example.com/chart.png"],
            file_ids=["stored-1"],
        )

        rich = response.rich_response
        assert [f.filename for f in rich.files] == ["chart.png", "stored.png"]
        assert file_store.save_file.call_count == 1
        assert [img.content for img in rich.tool_images] == [JPEG_BYTES, PNG_BYTES]

    def test_missing_file_id_reported_as_failure(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        tool, _ = _make_tool(
            {},
            monkeypatch,
            stored_files={"gone-1": FileNotFoundError("nope")},
        )

        response = _run(tool, file_ids=["gone-1"])

        assert response.rich_response is None
        assert "gone-1" in response.llm_facing_response
        assert "not found in the file store" in response.llm_facing_response

    def test_non_image_file_id_fails(self, monkeypatch: pytest.MonkeyPatch) -> None:
        tool, _ = _make_tool(
            {},
            monkeypatch,
            stored_files={
                "doc.pdf": _stored_image(
                    file_id="doc.pdf",
                    content=b"%PDF-1.4 fake pdf",
                    file_type=ChatFileType.DOC,
                )
            },
        )

        response = _run(tool, file_ids=["doc.pdf"])

        assert response.rich_response is None
        assert "the file is not an image" in response.llm_facing_response

    def test_unsupported_image_format_file_id_fails(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """IMAGE-typed files whose bytes cannot be verified are rejected."""
        tool, _ = _make_tool(
            {},
            monkeypatch,
            stored_files={"odd.svg": _stored_image(content=b"<svg/>")},
        )

        response = _run(tool, file_ids=["odd.svg"])

        assert response.rich_response is None
        assert "not a supported image format" in response.llm_facing_response

    def test_combined_cap_across_urls_and_file_ids(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        results = {
            f"https://example.com/{index}.png": _fetched(PNG_BYTES, "image/png")
            for index in range(3)
        }
        stored: dict[str, InMemoryChatFile | Exception] = {
            f"stored-{index}": _stored_image(file_id=f"stored-{index}")
            for index in range(4)
        }
        tool, file_store = _make_tool(results, monkeypatch, stored_files=stored)

        loaded: list[str] = []

        def _recording_load(file_id: str) -> InMemoryChatFile:
            loaded.append(file_id)
            entry = stored[file_id]
            if isinstance(entry, Exception):
                raise entry
            return entry

        monkeypatch.setattr(analyze_image_tool, "load_chat_file_by_id", _recording_load)

        response = _run(
            tool,
            urls=[f"https://example.com/{index}.png" for index in range(3)],
            file_ids=[f"stored-{index}" for index in range(4)],
        )

        # URLs fill their slots first; only 2 file_ids fit the cap of 5
        assert file_store.save_file.call_count == 3
        assert loaded == ["stored-0", "stored-1"]

        llm_payload = json.loads(response.llm_facing_response)
        assert len(llm_payload["images"]) == 5
        assert (
            f"{ANALYZE_MAX_IMAGES} images were attempted" in llm_payload["note"]
            and "2 were skipped" in llm_payload["note"]
        )

    def test_file_id_cap_alone(self, monkeypatch: pytest.MonkeyPatch) -> None:
        stored: dict[str, InMemoryChatFile | Exception] = {
            f"stored-{index}": _stored_image(file_id=f"stored-{index}")
            for index in range(ANALYZE_MAX_IMAGES + 2)
        }
        tool, _ = _make_tool(
            {}, monkeypatch, stored_files=stored, annotations=["a"] * len(stored)
        )

        response = _run(
            tool, file_ids=[f"stored-{index}" for index in range(len(stored))]
        )

        rich = response.rich_response
        assert len(rich.files) == ANALYZE_MAX_IMAGES
