import json
import threading
import time
from typing import Any
from unittest.mock import MagicMock

import pytest

from onyx.server.query_and_chat.placement import Placement
from onyx.tools.models import ToolCallException
from onyx.tools.tool_implementations.download import download_tool
from onyx.tools.tool_implementations.download.download_tool import (
    DOWNLOAD_MAX_URLS,
    DownloadFileTool,
    filename_from_url,
    sniff_mime_type,
)
from onyx.tools.tool_implementations.open_url.models import FailedFetch
from onyx.tools.tool_implementations.open_url.onyx_web_crawler import (
    FetchedFile,
)

PNG_MAGIC = b"\x89PNG\r\n\x1a\n"

PNG_BYTES = PNG_MAGIC + b"pngdata"
JPEG_BYTES = b"\xff\xd8\xff\xe0jpegdata"


def _fetched(content: bytes, content_type: str | None) -> FetchedFile:
    return FetchedFile(content=content, content_type=content_type)


def _make_tool(
    results: dict[str, Any],
    monkeypatch: pytest.MonkeyPatch,
) -> tuple[DownloadFileTool, MagicMock]:
    """Build a DownloadFileTool whose crawler returns the given per-URL results
    and whose file store assigns sequential file ids.

    By default no vision model is configured (annotations skipped)."""
    tool = DownloadFileTool(tool_id=1, emitter=MagicMock())

    def _download(
        url: str,
        **kwargs: Any,  # noqa: ARG001
    ) -> FetchedFile | FailedFetch:
        return results[url]

    monkeypatch.setattr(tool._crawler, "download_file_bytes", _download)  # noqa: SLF001

    saved_ids = [f"file-{index}" for index in range(1, 100)]
    file_store = MagicMock()
    file_store.save_file.side_effect = saved_ids
    monkeypatch.setattr(download_tool, "get_default_file_store", lambda: file_store)
    monkeypatch.setattr(
        download_tool,
        "build_full_frontend_file_url",
        lambda file_id: f"https://frontend.test/files/{file_id}",
    )
    monkeypatch.setattr(download_tool, "get_tool_vision_llm", lambda: None)
    return tool, file_store


def _run(tool: DownloadFileTool, urls: list[str]) -> Any:
    placement = Placement(turn_index=0, tab_index=0)
    return tool.run(placement=placement, override_kwargs=None, urls=urls)


class TestSniffMimeType:
    def test_png(self) -> None:
        assert sniff_mime_type(PNG_BYTES) == "image/png"

    def test_jpeg(self) -> None:
        assert sniff_mime_type(JPEG_BYTES) == "image/jpeg"

    def test_pdf(self) -> None:
        assert sniff_mime_type(b"%PDF-1.7 data") == "application/pdf"

    def test_unknown(self) -> None:
        assert sniff_mime_type(b"random bytes") is None


class TestFilenameFromUrl:
    def test_uses_url_basename(self) -> None:
        assert (
            filename_from_url("https://example.com/images/cat photo.jpg", None, 0)
            == "cat_photo.jpg"
        )

    def test_falls_back_to_numbered_name_with_mime_extension(self) -> None:
        assert (
            filename_from_url("https://example.com/", "image/png", 0)
            == "download-1.png"
        )

    def test_falls_back_without_extension_when_mime_unknown(self) -> None:
        assert filename_from_url("https://example.com", None, 2) == "download-3"


class TestDownloadFileToolRun:
    def test_saves_files_and_builds_responses(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        tool, file_store = _make_tool(
            {
                "https://example.com/cat.jpg": _fetched(JPEG_BYTES, "image/jpeg"),
                "https://example.com/dog.png": _fetched(PNG_BYTES, None),
            },
            monkeypatch,
        )

        response = _run(
            tool,
            ["https://example.com/cat.jpg", "https://example.com/dog.png"],
        )

        assert file_store.save_file.call_count == 2
        rich = response.rich_response
        assert [f.filename for f in rich.files] == ["cat.jpg", "dog.png"]
        # Second file had no Content-Type; the MIME type was sniffed from bytes
        assert [f.mime_type for f in rich.files] == ["image/jpeg", "image/png"]
        assert rich.failures == []

        llm_payload = json.loads(response.llm_facing_response)
        assert llm_payload["files"][0]["file_id"] == "file-1"
        assert (
            llm_payload["files"][0]["file_url"] == "https://frontend.test/files/file-1"
        )
        assert llm_payload["failures"] == []

    def test_failures_are_reported_not_raised(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        tool, _ = _make_tool(
            {
                "https://example.com/blocked.jpg": FailedFetch(
                    url="https://example.com/blocked.jpg",
                    failure_reason="blocked by a Cloudflare bot challenge",
                ),
                "https://example.com/ok.jpg": _fetched(JPEG_BYTES, "image/jpeg"),
            },
            monkeypatch,
        )

        response = _run(
            tool,
            ["https://example.com/blocked.jpg", "https://example.com/ok.jpg"],
        )

        assert len(response.rich_response.files) == 1
        assert response.rich_response.failures[0].url == (
            "https://example.com/blocked.jpg"
        )
        llm_payload = json.loads(response.llm_facing_response)
        assert llm_payload["failures"][0]["failure_reason"] == (
            "blocked by a Cloudflare bot challenge"
        )

    def test_all_failures_returns_failure_message(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        tool, _ = _make_tool(
            {
                "https://example.com/a.jpg": FailedFetch(
                    url="https://example.com/a.jpg", failure_reason="HTTP 404"
                ),
            },
            monkeypatch,
        )

        response = _run(tool, ["https://example.com/a.jpg"])

        assert response.rich_response is None
        assert "HTTP 404" in response.llm_facing_response

    def test_missing_urls_raises_tool_call_exception(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        tool, _ = _make_tool({}, monkeypatch)
        placement = Placement(turn_index=0, tab_index=0)

        with pytest.raises(ToolCallException):
            tool.run(placement=placement, override_kwargs=None)

    def test_downloads_run_concurrently_and_keep_order(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Different-provider URLs download in parallel, and results stay in
        input order even when the first download finishes last."""
        tool, file_store = _make_tool({}, monkeypatch)
        slow_download_done = threading.Event()

        def _download(url: str, **kwargs: Any) -> Any:  # noqa: ARG001
            if "slow" in url:
                assert slow_download_done.wait(timeout=5), (
                    "downloads did not run concurrently"
                )
                return _fetched(JPEG_BYTES + b"slow", "image/jpeg")
            time.sleep(0.2)  # let the slow download start and block first
            slow_download_done.set()
            return _fetched(JPEG_BYTES + b"fast", "image/jpeg")

        monkeypatch.setattr(tool._crawler, "download_file_bytes", _download)  # noqa: SLF001

        response = _run(
            tool,
            ["https://imgur.com/slow.jpg", "https://reddit.com/fast.jpg"],
        )

        rich = response.rich_response
        # Order matches the input, not completion order
        assert [f.filename for f in rich.files] == ["slow.jpg", "fast.jpg"]
        assert rich.files[0].annotation is None
        assert file_store.save_file.call_count == 2

    def test_url_cap_applied(self, monkeypatch: pytest.MonkeyPatch) -> None:
        results = {
            f"https://example.com/{index}.jpg": _fetched(JPEG_BYTES, "image/jpeg")
            for index in range(DOWNLOAD_MAX_URLS + 3)
        }
        tool, file_store = _make_tool(results, monkeypatch)

        urls = [f"https://example.com/{index}.jpg" for index in range(len(results))]
        _run(tool, urls)

        assert file_store.save_file.call_count == DOWNLOAD_MAX_URLS

    def test_url_cap_reported_in_success_response(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """URLs dropped by the per-call cap are called out to the LLM, so it
        knows the unattempted URLs are not dead."""
        results = {
            f"https://example.com/{index}.jpg": _fetched(JPEG_BYTES, "image/jpeg")
            for index in range(DOWNLOAD_MAX_URLS + 3)
        }
        tool, _ = _make_tool(results, monkeypatch)

        urls = [f"https://example.com/{index}.jpg" for index in range(len(results))]
        response = _run(tool, urls)

        llm_payload = json.loads(response.llm_facing_response)
        notes = llm_payload["notes"]
        assert any(
            f"{DOWNLOAD_MAX_URLS} URLs were attempted" in note
            and "3 were skipped" in note
            for note in notes
        )

    def test_url_cap_reported_when_all_fail(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        results = {
            f"https://example.com/{index}.jpg": FailedFetch(
                url=f"https://example.com/{index}.jpg", failure_reason="HTTP 404"
            )
            for index in range(DOWNLOAD_MAX_URLS + 2)
        }
        tool, _ = _make_tool(results, monkeypatch)

        urls = [f"https://example.com/{index}.jpg" for index in range(len(results))]
        response = _run(tool, urls)

        assert response.rich_response is None
        assert "2 were skipped" in response.llm_facing_response

    def test_usage_notes_instruct_embedding_once(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The response tells the LLM to embed images exactly once (no
        link-wrapped duplicates) and to link non-image files."""
        tool, _ = _make_tool(
            {
                "https://example.com/cat.jpg": _fetched(JPEG_BYTES, "image/jpeg"),
                "https://example.com/report.pdf": _fetched(
                    b"%PDF-1.7 data", "application/pdf"
                ),
            },
            monkeypatch,
        )

        response = _run(
            tool,
            ["https://example.com/cat.jpg", "https://example.com/report.pdf"],
        )

        llm_payload = json.loads(response.llm_facing_response)
        notes = llm_payload["notes"]
        assert any(
            "exactly once" in note and "Never wrap the embed in a link" in note
            for note in notes
        )
        assert any("[filename](file_url)" in note for note in notes)


class TestDownloadFileAnnotation:
    def test_images_are_annotated(self, monkeypatch: pytest.MonkeyPatch) -> None:
        tool, _ = _make_tool(
            {
                "https://example.com/cat.jpg": _fetched(JPEG_BYTES, "image/jpeg"),
                "https://example.com/report.pdf": _fetched(
                    b"%PDF-1.7 data", "application/pdf"
                ),
            },
            monkeypatch,
        )
        monkeypatch.setattr(download_tool, "get_tool_vision_llm", lambda: MagicMock())
        monkeypatch.setattr(
            download_tool,
            "annotate_images_in_parallel",
            lambda _llm, images, _question=None: (
                ["a cat sitting on a mat"]
                if len(images) == 1 and images[0][0] == "cat.jpg"
                else []
            ),
        )

        response = _run(
            tool,
            ["https://example.com/cat.jpg", "https://example.com/report.pdf"],
        )

        rich = response.rich_response
        assert rich.files[0].annotation == "a cat sitting on a mat"
        # Non-image files get no annotation
        assert rich.files[1].annotation is None
        assert rich.tool_images[0].content == JPEG_BYTES
        assert rich.tool_images[0].file_id == "file-1"

        llm_payload = json.loads(response.llm_facing_response)
        assert llm_payload["files"][0]["annotation"] == "a cat sitting on a mat"
        assert llm_payload["files"][1]["annotation"] is None

    def test_no_vision_model_degrades_to_no_annotation(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        tool, _ = _make_tool(
            {"https://example.com/cat.jpg": _fetched(JPEG_BYTES, "image/jpeg")},
            monkeypatch,
        )

        response = _run(tool, ["https://example.com/cat.jpg"])

        rich = response.rich_response
        assert rich.files[0].annotation is None
        # The image is still attached for direct replay to vision-capable models
        assert rich.tool_images[0].content == JPEG_BYTES
