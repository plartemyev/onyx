import json
from io import BytesIO
from typing import Any

from sqlalchemy.orm import Session
from typing_extensions import override

from onyx.chat.emitter import Emitter
from onyx.configs.constants import FileOrigin
from onyx.file_store.models import ChatFileType
from onyx.file_store.utils import (
    build_full_frontend_file_url,
    get_default_file_store,
    load_chat_file_by_id,
)
from onyx.server.query_and_chat.placement import Placement
from onyx.server.query_and_chat.streaming_models import (
    AnalyzeImageFile,
    AnalyzeImageFinal,
    AnalyzeImageStart,
    Packet,
)
from onyx.tools.interface import Tool
from onyx.tools.models import (
    AnalyzeImageToolRichResponse,
    DownloadedFile,
    DownloadFailure,
    ToolCallException,
    ToolResponse,
    ToolResponseImage,
)
from onyx.tools.tool_implementations.download.download_tool import (
    filename_from_url,
    sniff_mime_type,
)
from onyx.tools.tool_implementations.image_analysis.shared import (
    annotate_images_in_parallel,
    get_tool_vision_llm,
)
from onyx.tools.tool_implementations.open_url.models import FailedFetch
from onyx.tools.tool_implementations.open_url.onyx_web_crawler import (
    FetchedFile,
    OnyxWebCrawler,
)
from onyx.utils.logger import setup_logger

logger = setup_logger()

URLS_FIELD = "urls"
FILE_IDS_FIELD = "file_ids"
QUESTION_FIELD = "question"

# Per-call cap so one tool call cannot flood the context or the file store
ANALYZE_MAX_IMAGES = 5

_NO_VISION_MODEL_NOTICE = (
    "No vision model is configured, so the images were attached but not analyzed."
)


def _string_list(raw: Any) -> list[str]:
    """Normalize a tool-call argument into a deduped list of strings."""
    if isinstance(raw, str):
        raw = [raw]
    if not isinstance(raw, list):
        return []
    values: list[str] = []
    for item in raw:
        value = str(item).strip()
        if value and value not in values:
            values.append(value)
    return values


def _skipped_images_note(skipped_image_count: int) -> str:
    """LLM-facing notice when images were dropped by the per-call cap."""
    return (
        f"Only the first {ANALYZE_MAX_IMAGES} images were attempted; "
        f"{skipped_image_count} were skipped. Call the analyze_image tool "
        "again with the remaining images if you still need them."
    )


class AnalyzeImageTool(Tool[None]):
    """Describes images with the vision model.

    Accepts direct web URLs (fetched with the Onyx crawler's fast path plus a
    stealth headless browser fallback, same fetching as download_file) and
    file_ids of images already in the file store. Analyzed images display in
    chat and are annotated with the admin-configured captioning LLM (default
    vision model)."""

    NAME = "analyze_image"
    DESCRIPTION = (
        "Analyze images with a vision model: from direct web URLs and/or by file_id."
    )
    DISPLAY_NAME = "Analyze Image"

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
                    "Analyze images with a vision model. Pass direct image URLs (like "
                    "https://example.com/photo.jpg), file_ids of images already saved in "
                    "this chat, or both. Use this when you need to know what an image "
                    "shows — from search results, web pages, user attachments, or earlier "
                    "tool results. Pass a `question` to focus the analysis on one detail "
                    "you need to know. The image is also shown to the user in chat. URLs "
                    "must point at the image file itself, not a web page containing the "
                    "image. Never invent a file_id — use one from an `[attached image — "
                    "file_id: <id>]` tag or from a previous tool result. "
                    f"At most {ANALYZE_MAX_IMAGES} images are analyzed per call. If you "
                    "only want to give the file to the user without looking at it, use "
                    "the download_file tool instead."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        URLS_FIELD: {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": (
                                "Direct image URLs to analyze. These must point at the image "
                                "file itself, not a web page containing the image."
                            ),
                        },
                        FILE_IDS_FIELD: {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": (
                                "file_ids of images already saved in this chat to analyze "
                                "(user-attached images, or images from download_file, "
                                "generate_image, analyze_image, or run_python results). "
                                "Use these instead of re-downloading from the web."
                            ),
                        },
                        QUESTION_FIELD: {
                            "type": "string",
                            "description": (
                                "Optional question to focus the analysis on, e.g. 'What "
                                "values does the chart show for 2024?'. When omitted, a "
                                "general description is returned."
                            ),
                        },
                    },
                },
            },
        }

    def emit_start(self, placement: Placement) -> None:
        self.emitter.emit(
            Packet(
                placement=placement,
                obj=AnalyzeImageStart(),
            )
        )

    def run(
        self,
        placement: Placement,
        override_kwargs: None,  # noqa: ARG002
        **llm_kwargs: Any,
    ) -> ToolResponse:
        urls = _string_list(llm_kwargs.get(URLS_FIELD))
        file_ids = _string_list(llm_kwargs.get(FILE_IDS_FIELD))
        if not urls and not file_ids:
            raise ToolCallException(
                message=(
                    f"Missing required '{URLS_FIELD}' or '{FILE_IDS_FIELD}' "
                    "parameter in analyze_image tool call"
                ),
                llm_facing_message=(
                    f"The analyze_image tool requires at least one of '{URLS_FIELD}' "
                    f"or '{FILE_IDS_FIELD}'. Please provide like: "
                    f'{{"{URLS_FIELD}": ["https://example.com/photo.jpg"]}}'
                ),
            )

        # URLs take the available slots first, then file_ids fill the rest
        skipped_image_count = max(len(urls) + len(file_ids) - ANALYZE_MAX_IMAGES, 0)
        if skipped_image_count:
            urls = urls[:ANALYZE_MAX_IMAGES]
            file_ids = file_ids[: max(ANALYZE_MAX_IMAGES - len(urls), 0)]

        question = llm_kwargs.get(QUESTION_FIELD)
        if not isinstance(question, str) or not question.strip():
            question = None

        file_store = get_default_file_store()
        files: list[DownloadedFile] = []
        failures: list[DownloadFailure] = []
        images_to_annotate: list[tuple[str, bytes, DownloadedFile]] = []
        vision_llm = get_tool_vision_llm()

        for index, url in enumerate(urls):
            fetched: FetchedFile | FailedFetch = self._crawler.download_file_bytes(url)
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
                or fetched.sniffed_mime_type()
                or ""
            )
            if not mime_type.startswith("image/"):
                failures.append(
                    DownloadFailure(
                        url=url,
                        failure_reason=(
                            f"the URL does not point at an image "
                            f"(content type: {mime_type or 'unknown'})"
                        ),
                    )
                )
                continue

            filename = filename_from_url(url, mime_type, index)
            try:
                file_id = file_store.save_file(
                    content=(
                        fetched.content_file
                        if fetched.content_file is not None
                        else BytesIO(fetched.content)
                    ),
                    display_name=filename,
                    file_origin=FileOrigin.CHAT_IMAGE_GEN,
                    file_type=mime_type,
                )
            except Exception:
                logger.exception("Failed to save image from %s", url)
                failures.append(
                    DownloadFailure(url=url, failure_reason="failed to save the image")
                )
                continue

            analyzed_file = DownloadedFile(
                url=url,
                file_id=file_id,
                file_url=build_full_frontend_file_url(file_id),
                filename=filename,
                mime_type=mime_type,
            )
            files.append(analyzed_file)
            # Only in-memory images can be captioned (a disk-spilled payload
            # is far past what a vision model should receive).
            if fetched.content_file is None:
                images_to_annotate.append((filename, fetched.content, analyzed_file))

        for index, file_id in enumerate(file_ids):
            try:
                loaded_file = load_chat_file_by_id(file_id)
            except Exception:
                logger.exception("Failed to load image file %s", file_id)
                failures.append(
                    DownloadFailure(
                        url=file_id, failure_reason="not found in the file store"
                    )
                )
                continue

            if loaded_file.file_type != ChatFileType.IMAGE:
                failures.append(
                    DownloadFailure(
                        url=file_id, failure_reason="the file is not an image"
                    )
                )
                continue

            mime_type = sniff_mime_type(loaded_file.content) or ""
            if not mime_type.startswith("image/"):
                failures.append(
                    DownloadFailure(
                        url=file_id,
                        failure_reason="the file is not a supported image format",
                    )
                )
                continue

            filename = loaded_file.filename or f"image-{index + 1}"
            # Already in the file store — no re-save. Point `url` at the
            # stored copy so the LLM can embed it in its reply.
            file_url = build_full_frontend_file_url(file_id)
            analyzed_file = DownloadedFile(
                url=file_url,
                file_id=file_id,
                file_url=file_url,
                filename=filename,
                mime_type=mime_type,
            )
            files.append(analyzed_file)
            images_to_annotate.append((filename, loaded_file.content, analyzed_file))

        # Annotate with the configured captioning model (targeted at the
        # agent's question when one was passed). With no vision model, the
        # images still attach to the chat so a vision-capable chat model can
        # replay them, and the LLM-facing result says so.
        annotations: list[str | None] = []
        if images_to_annotate and vision_llm is not None:
            annotations = annotate_images_in_parallel(
                vision_llm,
                [(filename, content) for filename, content, _ in images_to_annotate],
                question=question,
            )
            for (_, _, analyzed_file), annotation in zip(
                images_to_annotate, annotations, strict=True
            ):
                analyzed_file.annotation = annotation

        # Emit the final packet even when everything failed so the UI block closes
        self.emitter.emit(
            Packet(
                placement=placement,
                obj=AnalyzeImageFinal(
                    files=[
                        AnalyzeImageFile(
                            filename=f.filename,
                            file_id=f.file_id,
                            annotation=f.annotation,
                        )
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
            llm_facing_response = "All images failed: " + ", ".join(failure_parts)
            if skipped_image_count:
                llm_facing_response += "\n\nNote: " + _skipped_images_note(
                    skipped_image_count
                )
            return ToolResponse(
                rich_response=None,
                llm_facing_response=llm_facing_response,
            )

        llm_result: dict[str, Any] = {
            "images": [
                {
                    "url": f.url,
                    "file_id": f.file_id,
                    "filename": f.filename,
                    "annotation": f.annotation,
                }
                for f in files
            ],
            "failures": [
                {"url": f.url, "failure_reason": f.failure_reason} for f in failures
            ],
        }
        if skipped_image_count:
            llm_result["note"] = _skipped_images_note(skipped_image_count)
        if vision_llm is None:
            llm_result["notice"] = _NO_VISION_MODEL_NOTICE

        return ToolResponse(
            rich_response=AnalyzeImageToolRichResponse(
                files=files,
                failures=failures,
                tool_images=[
                    ToolResponseImage(
                        filename=analyzed_file.filename,
                        file_id=analyzed_file.file_id,
                        content=content,
                    )
                    for _, content, analyzed_file in images_to_annotate
                ],
            ),
            llm_facing_response=json.dumps(llm_result, indent=2),
        )
