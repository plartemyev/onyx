import json
import mimetypes
import re
from concurrent.futures import ThreadPoolExecutor
from io import BytesIO
from typing import Any
from urllib.parse import unquote, urlparse

from sqlalchemy.orm import Session
from typing_extensions import override

from onyx.chat.emitter import Emitter
from onyx.configs.constants import FileOrigin
from onyx.file_store.utils import build_full_frontend_file_url, get_default_file_store
from onyx.server.query_and_chat.placement import Placement
from onyx.server.query_and_chat.streaming_models import (
    DownloadToolFile,
    DownloadToolFinal,
    DownloadToolStart,
    Packet,
)
from onyx.tools.interface import Tool
from onyx.tools.models import (
    DownloadedFile,
    DownloadFailure,
    DownloadToolRichResponse,
    ToolCallException,
    ToolResponse,
    ToolResponseImage,
)
from onyx.tools.tool_implementations.image_analysis.shared import (
    annotate_images_in_parallel,
    get_tool_vision_llm,
)
from onyx.tools.tool_implementations.open_url.models import FailedFetch
from onyx.tools.tool_implementations.open_url.onyx_web_crawler import (
    OnyxWebCrawler,
    sniff_mime_type,
)
from onyx.utils.logger import setup_logger

logger = setup_logger()

URLS_FIELD = "urls"

# Per-call cap so one tool call cannot flood the context or the file store.
# Models routinely list ~8-10 candidate images per call; 8 keeps that to a
# single call (the skipped-URLs note drives a second call otherwise) while
# still bounding response size.
DOWNLOAD_MAX_URLS = 8

_SANITIZE_FILENAME_PATTERN = re.compile(r"[^A-Za-z0-9._-]+")

# Re-exported for convenience: callers (and tests) import the MIME sniffer
# from this module, though it lives with the crawler that owns fetching.
__all__ = [
    "DOWNLOAD_MAX_URLS",
    "DownloadFileTool",
    "filename_from_url",
    "sniff_mime_type",
]

IMAGE_EMBED_NOTE = (
    "Embed each downloaded image in your reply exactly once as "
    "![filename](file_url). Never wrap the embed in a link."
)
FILE_LINK_NOTE = "Link other downloaded files as [filename](file_url)."


def skipped_urls_note(tool_name: str, max_urls: int, skipped_url_count: int) -> str:
    """LLM-facing notice when URLs were dropped by a per-call cap.

    Shared by download_file and analyze_image, which apply the same cap
    pattern over different constants.
    """
    return (
        f"Only the first {max_urls} URLs were attempted; "
        f"{skipped_url_count} were skipped. Call the {tool_name} tool again "
        "with the remaining URLs if you still need them."
    )


def _skipped_urls_note(skipped_url_count: int) -> str | None:
    """download_file-specific wrapper over skipped_urls_note."""
    if not skipped_url_count:
        return None
    return skipped_urls_note(
        tool_name="download_file",
        max_urls=DOWNLOAD_MAX_URLS,
        skipped_url_count=skipped_url_count,
    )


def _usage_notes(files: list[DownloadedFile]) -> list[str]:
    """LLM-facing instructions for sharing downloaded files in the reply."""
    notes: list[str] = []
    has_images = any(f.mime_type.startswith("image/") for f in files)
    has_other = any(not f.mime_type.startswith("image/") for f in files)
    if has_images:
        notes.append(IMAGE_EMBED_NOTE)
    if has_other:
        notes.append(FILE_LINK_NOTE)
    return notes


def filename_from_url(url: str, mime_type: str | None, index: int) -> str:
    """Derive a safe filename from the URL, falling back to a numbered name
    with an extension guessed from the MIME type."""
    path = urlparse(url).path
    basename = unquote(path.rsplit("/", 1)[-1]) if path else ""
    basename = _SANITIZE_FILENAME_PATTERN.sub("_", basename).strip("._")
    if basename:
        return basename

    extension = mimetypes.guess_extension(mime_type or "") or ""
    return f"download-{index + 1}{extension}"


class DownloadFileTool(Tool[None]):
    """Server-side file downloader with bot-protection fallbacks.

    The Python sandbox downloads with urllib/requests, which is frequently
    fingerprint-blocked (non-browser TLS, default User-Agent). This tool
    fetches from the backend via the Onyx crawler's fast path and falls back
    to a stealth headless browser on bot challenges, then saves the bytes to
    the file store so they display in chat. URLs are downloaded in parallel,
    except that requests sharing a provider are serialized and spaced by the
    shared request pacer.
    """

    NAME = "download_file"
    DESCRIPTION = "Download one or more files (images, PDFs, ...) from direct URLs."
    DISPLAY_NAME = "Download File"

    def __init__(self, tool_id: int, emitter: Emitter) -> None:
        super().__init__(emitter=emitter)
        self._id = tool_id
        self._crawler = OnyxWebCrawler()

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
    def is_available(cls, db_session: Session) -> bool:  # noqa: ARG003
        return True

    def tool_definition(self) -> dict:
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": (
                    "Download files from direct URLs (like https://example.com/image.jpg) "
                    "and attach them to the chat so the user can see them. Use this instead "
                    "of fetching with the Python tool when a download is blocked or fails — "
                    "it runs with full browser-like headers and can fall back to a real "
                    "browser for bot-protected hosts. This tool does not tell you what the "
                    "files contain; if you need to understand an image yourself, use the "
                    "analyze_image tool instead. At most "
                    f"{DOWNLOAD_MAX_URLS} URLs are downloaded per call."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        URLS_FIELD: {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": (
                                "Direct file URLs to download. These must point at the file "
                                "itself, not a web page containing the file."
                            ),
                        },
                    },
                    "required": [URLS_FIELD],
                },
            },
        }

    def emit_start(self, placement: Placement) -> None:
        self.emitter.emit(
            Packet(
                placement=placement,
                obj=DownloadToolStart(),
            )
        )

    def run(
        self,
        placement: Placement,
        override_kwargs: None,  # noqa: ARG002
        **llm_kwargs: Any,
    ) -> ToolResponse:
        urls: list[str] = []
        raw_urls = llm_kwargs.get(URLS_FIELD)
        if isinstance(raw_urls, str):
            raw_urls = [raw_urls]
        if isinstance(raw_urls, list):
            for raw_url in raw_urls:
                url = str(raw_url).strip()
                if url and url not in urls:
                    urls.append(url)
        if not urls:
            raise ToolCallException(
                message=f"Missing required '{URLS_FIELD}' parameter in download_file tool call",
                llm_facing_message=(
                    f"The download_file tool requires a '{URLS_FIELD}' parameter "
                    f"containing an array of direct file URLs. Please provide "
                    f'like: {{"urls": ["https://example.com/image.jpg"]}}'
                ),
            )
        skipped_url_count = max(len(urls) - DOWNLOAD_MAX_URLS, 0)
        if skipped_url_count:
            urls = urls[:DOWNLOAD_MAX_URLS]

        # Fetch in parallel: different providers download concurrently while
        # the crawler's shared request pacer serializes same-provider URLs
        # (no burst against reddit/imgur/...). executor.map keeps the input
        # order, so results line up with `urls` below.
        max_workers = min(len(urls), DOWNLOAD_MAX_URLS)
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            fetched_list = list(executor.map(self._crawler.download_file_bytes, urls))

        file_store = get_default_file_store()
        files: list[DownloadedFile] = []
        failures: list[DownloadFailure] = []
        # Image files saved below, with the DownloadedFile each caption belongs to
        images_to_annotate: list[tuple[str, bytes, DownloadedFile]] = []
        vision_llm = get_tool_vision_llm()

        for index, url in enumerate(urls):
            fetched = fetched_list[index]
            if isinstance(fetched, FailedFetch):
                failures.append(
                    DownloadFailure(url=url, failure_reason=fetched.failure_reason)
                )
                continue

            mime_type = (
                (
                    fetched.content_type
                    if fetched.content_type
                    and fetched.content_type != "application/octet-stream"
                    else None
                )
                or sniff_mime_type(fetched.content)
                or "application/octet-stream"
            )
            filename = filename_from_url(url, mime_type, index)
            try:
                file_id = file_store.save_file(
                    content=BytesIO(fetched.content),
                    display_name=filename,
                    file_origin=FileOrigin.CHAT_IMAGE_GEN,
                    file_type=mime_type,
                )
            except Exception:
                logger.exception("Failed to save downloaded file from %s", url)
                failures.append(
                    DownloadFailure(
                        url=url, failure_reason="failed to save the downloaded file"
                    )
                )
                continue
            downloaded_file = DownloadedFile(
                url=url,
                file_id=file_id,
                file_url=build_full_frontend_file_url(file_id),
                filename=filename,
                mime_type=mime_type,
            )
            files.append(downloaded_file)
            if mime_type.startswith("image/"):
                images_to_annotate.append((filename, fetched.content, downloaded_file))

        # Describe fetched images with the configured captioning model so the
        # agent learns what they contain. Degrades to no annotation when no
        # vision model is available.
        annotations: list[str | None] = [None] * len(images_to_annotate)
        if vision_llm is not None and images_to_annotate:
            annotations = annotate_images_in_parallel(
                vision_llm,
                [(filename, content) for filename, content, _ in images_to_annotate],
            )
        for (_, _, downloaded_file), annotation in zip(
            images_to_annotate, annotations, strict=True
        ):
            downloaded_file.annotation = annotation

        # Emit the final packet even when everything failed so the UI block closes
        self.emitter.emit(
            Packet(
                placement=placement,
                obj=DownloadToolFinal(
                    files=[
                        DownloadToolFile(filename=f.filename, file_id=f.file_id)
                        for f in files
                    ],
                    failures=[
                        f"{failure.url} ({failure.failure_reason})"
                        if failure.failure_reason
                        else failure.url
                        for failure in failures
                    ],
                ),
            )
        )

        if not files:
            failure_parts = [
                f"{failure.url} ({failure.failure_reason})"
                if failure.failure_reason
                else failure.url
                for failure in failures
            ]
            llm_facing_response = "All downloads failed: " + ", ".join(failure_parts)
            skipped_note = _skipped_urls_note(skipped_url_count)
            if skipped_note:
                llm_facing_response += f"\n\nNote: {skipped_note}"
            return ToolResponse(
                rich_response=None,
                llm_facing_response=llm_facing_response,
            )

        notes = _usage_notes(files)
        skipped_note = _skipped_urls_note(skipped_url_count)
        if skipped_note:
            notes.append(skipped_note)

        llm_facing_response = json.dumps(
            {
                "files": [
                    {
                        "file_id": f.file_id,
                        "filename": f.filename,
                        "url": f.url,
                        "file_url": f.file_url,
                        "annotation": f.annotation,
                    }
                    for f in files
                ],
                "failures": [
                    {"url": f.url, "failure_reason": f.failure_reason} for f in failures
                ],
                "notes": notes,
            },
            indent=2,
        )

        return ToolResponse(
            rich_response=DownloadToolRichResponse(
                files=files,
                failures=failures,
                tool_images=[
                    ToolResponseImage(
                        filename=downloaded_file.filename,
                        file_id=downloaded_file.file_id,
                        content=content,
                    )
                    for _, content, downloaded_file in images_to_annotate
                ],
            ),
            llm_facing_response=llm_facing_response,
        )
