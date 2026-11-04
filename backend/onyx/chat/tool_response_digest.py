"""Digests for in-turn compaction of tool responses.

When the in-turn tail no longer fits the context window, tool responses are
stubbed (:py:func:`onyx.chat.llm_loop._compact_in_turn_tail`). A bare
"result was removed" notice makes small models re-run the identical call, so
stubs carry a digest of the dropped output instead:

- Tier 1 (deterministic, default on): for the tools' ``{"results": [...]}``
  shape, the result titles and URLs; for anything else, a head excerpt cut
  at a line boundary. No LLM, no I/O, stable across cycles.
- Tier 2 (config-gated): the chat model summarizes the
  output into a compact digest. Computed lazily, only for responses that
  were actually stubbed: the stub carries the tier-1 digest immediately
  (the fit never waits on an LLM) while a background call replaces the
  cached digest with the LLM summary — later cycles' re-compaction picks
  it up. Any failure or timeout leaves the tier-1 digest in place.

Digests are derived at most once per tool call and cached by
``tool_call_id`` (see :py:func:`stub_message_for`).
"""

from __future__ import annotations

import json
import logging
import threading
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor
from concurrent.futures import TimeoutError as FutureTimeoutError
from typing import Callable

from onyx.configs.chat_configs import dr_is_thinking_model, dr_step_generation_budget
from onyx.configs.model_configs import (
    GEN_AI_TOOL_RESPONSE_DIGEST_MAX_CHARS,
    GEN_AI_TOOL_RESPONSE_DIGESTS,
    GEN_AI_TOOL_RESPONSE_SUMMARIZATION,
    GEN_AI_TOOL_RESPONSE_SUMMARIZATION_MAX_INPUT_CHARS,
    GEN_AI_TOOL_RESPONSE_SUMMARIZATION_TIMEOUT_SECONDS,
)
from onyx.db.engine.sql_engine import get_session_with_current_tenant
from onyx.llm.context_budgets import TOOL_RESPONSE_DIGEST_SUMMARY_OUTPUT, scale
from onyx.llm.factory import get_secondary_llm
from onyx.llm.interfaces import LLM
from onyx.llm.model_capabilities import model_is_reasoning_model
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


class _EmptySummaryError(_SummaryError):
    """The LLM answered with usable finish but no summary text.

    Small local models do this occasionally on large inputs; a fresh call
    usually succeeds, so this is the one failure worth retrying."""


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


# Marker placed where the input cap cut the middle of the payload.
_SUMMARY_INPUT_CUT_MARKER = "\n\n[…middle of the response truncated…]\n\n"


def _cap_summary_input(text: str) -> str:
    """Bound the summarization input.

    With large fetch caps the payload can approach the context window, and
    the summary call must stay well inside it. Keeps the head and the tail
    (the informative ends of page/list output) and cuts the middle.
    """
    max_chars = GEN_AI_TOOL_RESPONSE_SUMMARIZATION_MAX_INPUT_CHARS
    if len(text) <= max_chars:
        return text
    keep = max_chars - len(_SUMMARY_INPUT_CUT_MARKER)
    head = keep * 7 // 10
    tail = keep - head
    return text[:head] + _SUMMARY_INPUT_CUT_MARKER + text[len(text) - tail :]


def _summarize_once(text: str, llm: LLM) -> str:
    """One LLM call summarizing the dropped output.

    Runs in a worker thread under a hard wall-clock cap — a digest must
    never hold the turn hostage; on timeout the request is abandoned (it
    dies at the LLM client's own read timeout). Raises
    :class:`_SummaryError` on any failure (specifically
    :class:`_EmptySummaryError` when the model returns no text) so the
    caller can retry or fall back to the deterministic digest.
    """
    text = _cap_summary_input(text)
    messages = _summary_messages(text)
    # max_tokens counts native reasoning tokens: on a thinking model the
    # cap must include the thinking reserve on top of the answer budget,
    # or the whole cap burns inside the think trace and the summary comes
    # back empty (observed live: finish_reason=length with empty content,
    # llama.cpp ignoring reasoning_effort=OFF). Same scheme as the DR
    # steps; thinking-off engines never touch the reserve. The name
    # override matters here: on the chat-model fallback path the thinking
    # primary is a custom gguf the capability registry cannot classify.
    answer_budget = scale(llm.config.max_input_tokens, _SUMMARY_MAX_TOKENS_FRACTION)
    is_thinking_model = dr_is_thinking_model(
        llm.config.model_name,
        model_is_reasoning_model(
            llm.config.model_name,
            llm.config.model_provider,
            api_base=llm.config.api_base,
        ),
    )
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
                max_tokens=dr_step_generation_budget(
                    answer_budget, is_thinking_model, llm.config.max_input_tokens
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
        raise _EmptySummaryError("digest summarization returned an empty summary")
    return content.strip()


# An empty summary is retried with an identical request: small local models
# occasionally emit nothing for a large input and succeed on the next call.
# Only the empty case retries — a timed-out call already spent its wall-clock
# budget, and retrying it could hold the turn hostage for twice the timeout.
_EMPTY_SUMMARY_ATTEMPTS = 2


def _summarize(text: str, llm: LLM) -> str:
    """Tier 2: the LLM summary of the dropped output, retrying an empty
    summary once. Raises :class:`_SummaryError` on final failure so the
    caller can fall back to the deterministic digest.
    """
    last_error: _SummaryError | None = None
    for attempt in range(_EMPTY_SUMMARY_ATTEMPTS):
        try:
            return _summarize_once(text, llm)
        except _EmptySummaryError as e:
            last_error = e
            if attempt + 1 < _EMPTY_SUMMARY_ATTEMPTS:
                logger.warning(
                    "Tool response digest: empty LLM summary (attempt "
                    "%d/%d); retrying once",
                    attempt + 1,
                    _EMPTY_SUMMARY_ATTEMPTS,
                )
    assert last_error is not None
    raise last_error


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


# At most this many background LLM digest upgrades run at once; past the
# cap, stubs keep the deterministic digest (it already fits the budget).
_MAX_INFLIGHT_DIGEST_UPGRADES = 4

# Guard for the in-flight set, the shared digest caches that the
# background upgrades write into, and the cross-turn store below.
_DIGEST_STATE_LOCK = threading.Lock()
_INFLIGHT_DIGEST_UPGRADES: set[str] = set()

# Cross-turn digest store: a response stubbed in an earlier turn replays its
# digest in later turns instead of re-deriving (or, with tier 2, re-summarizing)
# it. Bounded and in-process — a restart just re-derives on demand.
_DIGEST_STORE_MAX_ENTRIES = 1024
_DIGEST_STORE: OrderedDict[str, str] = OrderedDict()


def _stored_digest(tool_call_id: str) -> str | None:
    with _DIGEST_STATE_LOCK:
        digest = _DIGEST_STORE.get(tool_call_id)
        if digest is not None:
            _DIGEST_STORE.move_to_end(tool_call_id)
        return digest


def _store_digest(tool_call_id: str, digest: str) -> None:
    with _DIGEST_STATE_LOCK:
        _DIGEST_STORE[tool_call_id] = digest
        _DIGEST_STORE.move_to_end(tool_call_id)
        while len(_DIGEST_STORE) > _DIGEST_STORE_MAX_ENTRIES:
            _DIGEST_STORE.popitem(last=False)


def _run_digest_upgrade(
    tool_call_id: str,
    text: str,
    cache: dict[str, str],
    llm: LLM,
) -> None:
    """Thread body: replace the cached deterministic digest with the LLM
    summary. Any failure keeps the deterministic digest in place."""
    try:
        summary = compute_tool_response_digest(text, llm=llm)
        if summary is not None:
            with _DIGEST_STATE_LOCK:
                cache[tool_call_id] = summary
            _store_digest(tool_call_id, summary)
            logger.info(
                "Tool response digest: background summary replaced the "
                "deterministic digest for %s",
                tool_call_id,
            )
    except Exception:
        logger.debug(
            "Tool response digest: background summary failed for %s; "
            "keeping the deterministic digest",
            tool_call_id,
            exc_info=True,
        )
    finally:
        with _DIGEST_STATE_LOCK:
            _INFLIGHT_DIGEST_UPGRADES.discard(tool_call_id)


def _resolve_upgrade_llm(fallback_llm: LLM) -> LLM:
    """The engine a tier-2 upgrade should run on: the secondary model when
    the deployment designates one, else the passed chat model.

    Upgrades are auxiliary background work: on the primary engine they
    contend with the chat/DR loop for its slot (observed: upgrades queue
    behind whole DR cycles and time out into the deterministic fallback),
    and a thinking primary burns the output cap inside its think trace —
    summaries come back empty. A designated secondary carries the
    deployment's no-thinking tuning and serves exactly this kind of
    auxiliary flow.
    """
    try:
        with get_session_with_current_tenant() as db_session:
            secondary = get_secondary_llm(db_session)
    except Exception:
        logger.warning(
            "Tool response digest: secondary model lookup failed; the "
            "upgrade will run on the chat model",
            exc_info=True,
        )
        return fallback_llm
    return secondary or fallback_llm


def _schedule_digest_upgrade(
    tool_call_id: str,
    text: str,
    cache: dict[str, str] | None,
    llm: LLM | None,
) -> None:
    """Queue the tier-2 summary of one stubbed response in the background.

    Fire-and-forget by design: the stub already fits the budget with the
    deterministic digest, so the upgrade must never hold up a request (and
    on a single-cache-slot model server it must not jump ahead of the next
    chat request). No-op when summarization is off, no LLM or cache was
    provided, or too many upgrades are already running.
    """
    if not GEN_AI_TOOL_RESPONSE_SUMMARIZATION or llm is None or cache is None:
        return
    with _DIGEST_STATE_LOCK:
        if tool_call_id in _INFLIGHT_DIGEST_UPGRADES:
            return
        if len(_INFLIGHT_DIGEST_UPGRADES) >= _MAX_INFLIGHT_DIGEST_UPGRADES:
            logger.warning(
                "Tool response digest: %d upgrades already in flight; "
                "keeping the deterministic digest for %s",
                _MAX_INFLIGHT_DIGEST_UPGRADES,
                tool_call_id,
            )
            return
        _INFLIGHT_DIGEST_UPGRADES.add(tool_call_id)
    # Resolve the upgrade engine here, in the caller's thread, so the
    # secondary-model lookup sees the request's tenant context; the
    # background thread only calls the engine.
    upgrade_llm = _resolve_upgrade_llm(llm)
    threading.Thread(
        target=_run_digest_upgrade,
        args=(tool_call_id, text, cache, upgrade_llm),
        name=f"digest-upgrade-{tool_call_id[:8]}",
        daemon=True,
    ).start()


def stub_message_for(
    tool_call_id: str | None,
    original_message: str,
    cache: dict[str, str] | None,
    token_counter: Callable[[str], int] | None,
    llm: LLM | None = None,
) -> tuple[str, int] | None:
    """The stub text and its token count for one compacted tool response.

    Returns None when no digest applies and the caller should keep the bare
    compaction notice: digests are disabled, there is no ``tool_call_id``,
    nothing is extractable, or no token counter exists (the fixed fallback
    token count only covers the bare notice, so a sized digest without a
    counter would undercount and overflow the real window).

    On a cache miss the digest is derived on the fly (deterministic tier 1)
    and stored — both in the caller's cache and in the process-wide cross-turn
    store — so later cycles and later turns reuse it; with tier-2
    summarization enabled and an ``llm`` provided, a background call then
    replaces the cached digest with the LLM summary — later re-compactions
    pick it up. The request that triggered the stub never waits on the LLM.
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
        digest = _stored_digest(tool_call_id)
        if digest is not None and cache is not None:
            with _DIGEST_STATE_LOCK:
                cache[tool_call_id] = digest
    if digest is None:
        digest = extractive_digest(original_message)
        if not digest:
            return None
        if cache is not None:
            with _DIGEST_STATE_LOCK:
                cache[tool_call_id] = digest
        _store_digest(tool_call_id, digest)
        _schedule_digest_upgrade(tool_call_id, original_message, cache, llm)
    message = TOOL_CALL_RESPONSE_COMPACTED_DIGEST.format(digest=digest)
    return message, token_counter(message)


__all__ = [
    "compute_tool_response_digest",
    "extractive_digest",
    "stub_message_for",
]
