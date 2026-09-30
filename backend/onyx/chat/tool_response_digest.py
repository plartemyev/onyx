"""Digests for in-turn compaction of tool responses.

When the in-turn tail no longer fits the context window, tool responses are
stubbed (:py:func:`onyx.chat.llm_loop._compact_in_turn_tail`). A bare
"result was removed" notice makes small models re-run the identical call, so
stubs carry a digest of the dropped output instead:

- Tier 1 (deterministic, default on): for the tools' ``{"results": [...]}``
  shape, the result titles and URLs; for anything else, a head excerpt cut
  at a line boundary. No LLM, no I/O, stable across cycles.
- Tier 2 (config-gated, off by default): the chat model summarizes the
  output into a compact digest. Any failure or timeout falls back to tier 1.

Digests are computed at most once per tool call and cached by
``tool_call_id`` for the rest of the turn (see
:py:func:`maybe_digest_tool_response`).
"""

from __future__ import annotations

import json
import logging
from concurrent.futures import ThreadPoolExecutor
from concurrent.futures import TimeoutError as FutureTimeoutError
from typing import Callable

from onyx.configs.model_configs import (
    GEN_AI_TOOL_RESPONSE_DIGEST_EAGER_MIN_CHARS,
    GEN_AI_TOOL_RESPONSE_DIGEST_MAX_CHARS,
    GEN_AI_TOOL_RESPONSE_DIGESTS,
    GEN_AI_TOOL_RESPONSE_SUMMARIZATION,
    GEN_AI_TOOL_RESPONSE_SUMMARIZATION_TIMEOUT_SECONDS,
)
from onyx.llm.context_budgets import TOOL_RESPONSE_DIGEST_SUMMARY_OUTPUT, scale
from onyx.llm.interfaces import LLM
from onyx.llm.models import ChatCompletionMessage, ReasoningEffort, UserMessage
from onyx.prompts.chat_prompts import (
    TOOL_CALL_RESPONSE_COMPACTED_DIGEST,
    TOOL_RESPONSE_SUMMARIZATION_PROMPT,
)
from onyx.tracing.flows import LLMFlow
from onyx.tracing.llm_utils import llm_generation_span, record_llm_response

logger = logging.getLogger(__name__)

# Single-line caps for the extractive search-result digest: titles and URLs
# can each run long (tracking params, percent-encoded Thai), which would
# spend the whole digest budget on one entry.
_RESULT_LINE_TITLE_MAX_CHARS = 150
_RESULT_LINE_URL_MAX_CHARS = 200
# The extractive result digest keeps at most this many entries even when the
# size cap would allow more: past ~a dozen lines, a stub stops helping the
# model and starts duplicating the search it came from.
_RESULT_DIGEST_MAX_ENTRIES = 12

# LLM summary output cap, as a fraction of the model's context window. A
# digest is only useful while it is much smaller than the response it
# replaced.
_SUMMARY_MAX_TOKENS_FRACTION = TOOL_RESPONSE_DIGEST_SUMMARY_OUTPUT

# The stub wraps the digest in a ~250-character notice; a digest must beat
# the bare notice by a real margin, or the "compacted" message would not be
# smaller than the response it replaced and compaction would save nothing.
_DIGEST_MIN_SAVING_CHARS = 350


class _SummaryError(Exception):
    """Any failure of the LLM summarization path; callers fall back to the
    deterministic digest."""


def _truncate(text: str, max_chars: int) -> str:
    return text if len(text) <= max_chars else text[:max_chars]


def _results_digest(text: str, max_chars: int) -> str | None:
    """Title/URL lines for the ``{"results": [{"title", "url", ...}]}`` shape
    our web tools emit, or None when the text does not match the shape.

    Any surprise in the shape (non-dict entries, an entry with neither title
    nor URL) returns None: a wrong digest is worse than a generic one.
    """
    try:
        parsed = json.loads(text)
    except (ValueError, TypeError):
        return None
    if not isinstance(parsed, dict):
        return None
    results = parsed.get("results")
    if not isinstance(results, list) or not results:
        return None

    lines: list[str] = []
    size = 0
    for entry in results[:_RESULT_DIGEST_MAX_ENTRIES]:
        if not isinstance(entry, dict):
            return None
        title = str(entry.get("title") or "").strip()
        url = str(entry.get("url") or "").strip()
        if not title and not url:
            return None
        line = f"- {_truncate(title, _RESULT_LINE_TITLE_MAX_CHARS)}"
        if url:
            line += f" — {_truncate(url, _RESULT_LINE_URL_MAX_CHARS)}"
        if lines and size + len(line) > max_chars:
            break
        lines.append(line)
        size += len(line) + 1
    return "\n".join(lines) if lines else None


def _head_digest(text: str, max_chars: int) -> str:
    """The first ``max_chars`` of the output, cut at a line boundary so the
    excerpt does not end mid-sentence or mid-JSON."""
    if len(text) <= max_chars:
        return text
    cut = text[:max_chars]
    newline = cut.rfind("\n")
    if newline > max_chars // 2:
        cut = cut[:newline]
    return cut + "\n…"


def extractive_digest(text: str) -> str | None:
    """Tier 1: the deterministic digest of a tool response.

    Returns None when the text is empty, digests are disabled, or the text
    is too small to digest away a meaningful amount of space — in those
    cases the caller keeps the bare compaction notice.
    """
    if not GEN_AI_TOOL_RESPONSE_DIGESTS or not text or not text.strip():
        return None
    max_chars = GEN_AI_TOOL_RESPONSE_DIGEST_MAX_CHARS
    digest = _results_digest(text, max_chars)
    if digest is None and len(text) > max_chars:
        digest = _head_digest(text, max_chars)
    if digest is None or len(text) - len(digest) < _DIGEST_MIN_SAVING_CHARS:
        return None
    return digest


def _summary_messages(text: str) -> list[ChatCompletionMessage]:
    return [
        UserMessage(content=TOOL_RESPONSE_SUMMARIZATION_PROMPT.format(content=text))
    ]


def _summarize(text: str, llm: LLM) -> str:
    """Tier 2: one LLM call summarizing the dropped output.

    Runs in a worker thread under a hard wall-clock cap — a digest must
    never hold the turn hostage; on timeout the request is abandoned (it
    dies at the LLM client's own read timeout). Raises
    :class:`_SummaryError` on any failure so the caller can fall back to the
    deterministic digest.
    """
    messages = _summary_messages(text)
    executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="tool-digest")
    try:
        with llm_generation_span(
            llm=llm,
            flow=LLMFlow.TOOL_RESPONSE_DIGEST,
            input_messages=messages,
        ) as span:
            future = executor.submit(
                llm.invoke,
                messages,
                max_tokens=scale(
                    llm.config.max_input_tokens, _SUMMARY_MAX_TOKENS_FRACTION
                ),
                temperature=0.0,
                reasoning_effort=ReasoningEffort.OFF,
            )
            try:
                response = future.result(
                    timeout=GEN_AI_TOOL_RESPONSE_SUMMARIZATION_TIMEOUT_SECONDS
                )
            except FutureTimeoutError as e:
                raise _SummaryError(
                    f"digest summarization timed out after "
                    f"{GEN_AI_TOOL_RESPONSE_SUMMARIZATION_TIMEOUT_SECONDS}s"
                ) from e
            record_llm_response(span, response)
    finally:
        executor.shutdown(wait=False)

    content = response.choice.message.content
    if not content or not content.strip():
        raise _SummaryError("digest summarization returned an empty summary")
    return content.strip()


def compute_tool_response_digest(text: str, llm: LLM | None = None) -> str | None:
    """The digest for one tool response, computed on demand.

    Uses the LLM summary when one is available (tier 2, config-gated) and
    falls back to the deterministic digest (tier 1) on any failure. Never
    raises. Returns None when digests are disabled or the text is empty.
    """
    if not GEN_AI_TOOL_RESPONSE_DIGESTS or not text or not text.strip():
        return None
    if llm is not None and GEN_AI_TOOL_RESPONSE_SUMMARIZATION:
        try:
            return _summarize(text, llm)
        except Exception:
            logger.warning(
                "Tool response digest: LLM summarization failed; using the "
                "deterministic digest",
                exc_info=True,
            )
    return extractive_digest(text)


def maybe_digest_tool_response(
    text: str,
    tool_call_id: str,
    cache: dict[str, str],
    llm: LLM | None = None,
) -> None:
    """Eagerly digest a large tool response at creation time (the optional
    refinement): compaction later reuses the cached digest instead of doing
    the work — for tier 2, an LLM call — under context pressure.

    Cheap by construction: below the eager threshold nothing happens, and a
    tool call is never digested twice.
    """
    if not GEN_AI_TOOL_RESPONSE_DIGESTS:
        return
    if len(text) < GEN_AI_TOOL_RESPONSE_DIGEST_EAGER_MIN_CHARS:
        return
    if tool_call_id in cache:
        return
    digest = compute_tool_response_digest(text, llm=llm)
    if digest is not None:
        cache[tool_call_id] = digest


def stub_message_for(
    tool_call_id: str | None,
    original_message: str,
    cache: dict[str, str] | None,
    token_counter: Callable[[str], int] | None,
) -> tuple[str, int] | None:
    """The stub text and its token count for one compacted tool response.

    Returns None when no digest applies and the caller should keep the bare
    compaction notice: digests are disabled, there is no ``tool_call_id``,
    nothing is extractable, or no token counter exists (the fixed fallback
    token count only covers the bare notice, so a sized digest without a
    counter would undercount and overflow the real window).
    """
    if (
        not GEN_AI_TOOL_RESPONSE_DIGESTS
        or not token_counter
        or not tool_call_id
        or not original_message.strip()
    ):
        return None
    digest = (cache or {}).get(tool_call_id)
    if digest is None:
        digest = extractive_digest(original_message)
    if not digest:
        return None
    message = TOOL_CALL_RESPONSE_COMPACTED_DIGEST.format(digest=digest)
    return message, token_counter(message)


__all__ = [
    "compute_tool_response_digest",
    "extractive_digest",
    "maybe_digest_tool_response",
    "stub_message_for",
]
