"""Shared image-annotation helpers for agent tools.

Tools that fetch images from the web (analyze_image, download_file,
run_python) use these helpers to describe images with the admin-configured
captioning LLM (the default vision model), so the agent can reason about
images it cannot view directly.
"""

from onyx.configs.chat_configs import IMAGE_SUMMARIZATION_TIMEOUT
from onyx.configs.llm_configs import get_image_extraction_and_analysis_enabled
from onyx.db.image_caption import (
    get_cached_image_caption,
    image_caption_content_hash,
    image_caption_prompt_hash,
    store_image_caption,
)
from onyx.file_processing.image_summarization import (
    UnsupportedImageFormatError,
    summarize_image_pipeline,
)
from onyx.llm.factory import get_default_llm_with_vision
from onyx.llm.interfaces import LLM
from onyx.prompts.image_analysis import AGENT_IMAGE_ANNOTATION_SYSTEM_PROMPT
from onyx.utils.logger import setup_logger
from onyx.utils.threadpool_concurrency import run_functions_tuples_in_parallel

logger = setup_logger()

# Upper bound on images annotated per tool call, so one call cannot trigger an
# unbounded number of vision-LLM requests.
MAX_ANNOTATED_IMAGES = 5

_ANNOTATION_MAX_WORKERS = 4


def get_tool_vision_llm() -> LLM | None:
    """The LLM used to annotate images in tool results.

    Returns None when image analysis is disabled in the workspace settings or
    no vision-capable model is configured — callers must degrade gracefully
    (download/attach the image, but without a caption)."""
    if not get_image_extraction_and_analysis_enabled():
        return None
    return get_default_llm_with_vision(timeout=IMAGE_SUMMARIZATION_TIMEOUT)


def annotate_image(
    llm: LLM,
    image_data: bytes,
    context_name: str,
    question: str | None = None,
) -> str | None:
    """Describe one image with the vision LLM.

    When `question` is given, the description focuses on answering it.
    Identical (bytes, model, prompt) triples reuse the stored caption instead
    of paying the vision call again — the same image reappears across agent
    cycles, retries, and turns. Cache problems never fail the annotation.
    Returns None on failure — a missing caption must never fail the tool."""
    content_hash = image_caption_content_hash(image_data)
    prompt_hash = image_caption_prompt_hash(
        AGENT_IMAGE_ANNOTATION_SYSTEM_PROMPT, question
    )
    model_name = llm.config.model_name

    try:
        cached = get_cached_image_caption(content_hash, model_name, prompt_hash)
    except Exception:
        logger.exception("Caption cache lookup failed for %s", context_name)
        cached = None
    if cached is not None:
        logger.debug("Reusing stored caption for %s", context_name)
        return cached

    try:
        caption = summarize_image_pipeline(
            llm,
            image_data,
            query=question,
            system_prompt=AGENT_IMAGE_ANNOTATION_SYSTEM_PROMPT,
        )
    except UnsupportedImageFormatError:
        logger.info(
            "Skipping image annotation for %s: unsupported format", context_name
        )
        return None
    except Exception:
        logger.exception("Image annotation failed for %s", context_name)
        return None

    try:
        store_image_caption(content_hash, model_name, prompt_hash, caption)
    except Exception:
        logger.exception("Failed to store caption for %s", context_name)
    return caption


def annotate_images_in_parallel(
    llm: LLM,
    images: list[tuple[str, bytes]],
    question: str | None = None,
) -> list[str | None]:
    """Annotate (filename, bytes) pairs in parallel, preserving order.

    The result always has the same length as `images`; images beyond
    MAX_ANNOTATED_IMAGES get None."""
    if not images:
        return []
    if len(images) > MAX_ANNOTATED_IMAGES:
        logger.warning(
            "Capping image annotation at %d of %d images",
            MAX_ANNOTATED_IMAGES,
            len(images),
        )
        results = run_functions_tuples_in_parallel(
            [
                (annotate_image, (llm, image_data, filename, question))
                for filename, image_data in images[:MAX_ANNOTATED_IMAGES]
            ],
            allow_failures=True,
            max_workers=_ANNOTATION_MAX_WORKERS,
        )
        return results + [None] * (len(images) - MAX_ANNOTATED_IMAGES)

    return run_functions_tuples_in_parallel(
        [
            (annotate_image, (llm, image_data, filename, question))
            for filename, image_data in images
        ],
        allow_failures=True,
        max_workers=_ANNOTATION_MAX_WORKERS,
    )
