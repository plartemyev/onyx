import fnmatch
import logging
import os

NUM_RETURNED_HITS = 50

# May be less depending on model
MAX_CHUNKS_FED_TO_CHAT = int(os.environ.get("MAX_CHUNKS_FED_TO_CHAT") or 25)

# Maximum number of LLM cycles (one tool-call round-trip per cycle) before the
# agent is forced to answer. Default 6 covers the common search → open_url
# pattern documented at the call site; raise via env when integrating with
# tool-heavy MCPs that legitimately need more turns. Very high values are
# warned about (not clamped): the per-cycle history reserve divides the
# context headroom by this number, so extreme values erode it.
MAX_LLM_CYCLES: int = int(os.environ.get("MAX_LLM_CYCLES") or 6)
_MAX_LLM_CYCLES_WARN_THRESHOLD = 50
if MAX_LLM_CYCLES > _MAX_LLM_CYCLES_WARN_THRESHOLD:
    logging.getLogger(__name__).warning(
        "MAX_LLM_CYCLES=%s is very high; the per-cycle history reserve "
        "shrinks proportionally and long turns risk context overflow",
        MAX_LLM_CYCLES,
    )

# Wall-clock budget for a whole chat turn (all LLM cycles + tool calls). When
# exceeded, the loop jumps to the forced-final-answer cycle so the user gets a
# partial answer instead of an unbounded crawl. Must stay above the per-tool
# budget below so at least one forced answer cycle is reachable.
CHAT_TURN_BUDGET_SECONDS: int = int(os.environ.get("CHAT_TURN_BUDGET_SECONDS") or 1800)

# Wall-clock budget for one tool call (applied to the batch of parallel tool
# calls). Must sit ABOVE the code-interpreter executor's per-execution timeout
# (CODE_INTERPRETER_DEFAULT_TIMEOUT_MS) so the executor's structured
# timed_out result reaches the model instead of a generic tombstone.
TOOL_EXECUTION_TIMEOUT_SECONDS: int = int(
    os.environ.get("TOOL_EXECUTION_TIMEOUT_SECONDS") or 240
)

# 1 / (1 + DOC_TIME_DECAY * doc-age-in-years), set to 0 to have no decay
# Capped in Vespa at 0.5
DOC_TIME_DECAY = float(
    os.environ.get("DOC_TIME_DECAY") or 0.5  # Hits limit at 2 years by default
)
# For the highest matching base size chunk, how many chunks above and below do we pull in by default
# Note this is not in any of the deployment configs yet
# Currently only applies to search flow not chat
CONTEXT_CHUNKS_ABOVE = int(os.environ.get("CONTEXT_CHUNKS_ABOVE") or 1)
CONTEXT_CHUNKS_BELOW = int(os.environ.get("CONTEXT_CHUNKS_BELOW") or 1)
# Fairly long but this is to account for edge cases where the LLM pauses for much longer than usual
# The alternative is to fail the request completely so this is intended to be fairly lenient.
LLM_SOCKET_READ_TIMEOUT = int(
    os.environ.get("LLM_SOCKET_READ_TIMEOUT") or "60"
)  # 60 seconds
# Total per-call timeout for image summarization. Unlike LLM_SOCKET_READ_TIMEOUT
# (per-packet gap), this bounds the whole call so a keepalive-only stream can't
# wedge a docprocessing thread. A generous backstop against hangs.
IMAGE_SUMMARIZATION_TIMEOUT = int(
    os.environ.get("IMAGE_SUMMARIZATION_TIMEOUT") or "300"
)  # 300 seconds (5 min)
# Same backstop for contextual-RAG doc/chunk summaries. These are short,
# non-reasoning calls, so this is generous headroom.
CONTEXTUAL_RAG_LLM_TIMEOUT = int(
    os.environ.get("CONTEXTUAL_RAG_LLM_TIMEOUT") or "180"
)  # 180 seconds
# Max silent gap before the chat stream emits a keepalive packet; must stay below
# the smallest proxy idle timeout in front (ALBs default to 60s).
CHAT_HEARTBEAT_INTERVAL_S = int(os.environ.get("CHAT_HEARTBEAT_INTERVAL_S") or "15")
# Extra attempts when a streaming completion errors before its first chunk.
# Never retried after partial output.
LLM_FIRST_CHUNK_MAX_RETRIES = max(
    0, int(os.environ.get("LLM_FIRST_CHUNK_MAX_RETRIES") or "2")
)
# Socket-read timeout for deep-research report calls — bounds inter-chunk gaps
# (including a zero-chunk stall), not total generation time.
DR_REPORT_LLM_TIMEOUT_S = int(os.environ.get("DR_REPORT_LLM_TIMEOUT_S") or "60")
# Deep Research wall-clock limits. Defaults assume reasonably fast inference;
# raise via env when running on slow hardware.
# Overall budget before the orchestrator forces final report generation. The
# run may still exceed it: a research cycle that starts just before the cutoff
# runs to completion.
DR_FORCE_REPORT_S = int(os.environ.get("DR_FORCE_REPORT_S") or 30 * 60)
# Orchestrator rounds (plan → research → …). The final report is forced at the
# last cycle even if time remains, so this bounds total work alongside
# DR_FORCE_REPORT_S.
DR_MAX_ORCHESTRATOR_CYCLES = int(os.environ.get("DR_MAX_ORCHESTRATOR_CYCLES") or "8")
DR_MAX_ORCHESTRATOR_CYCLES_REASONING = int(
    os.environ.get("DR_MAX_ORCHESTRATOR_CYCLES_REASONING") or "4"
)
# Per research-agent call: overall wall-clock timeout (a timed-out agent
# returns a placeholder report), and time before its intermediate report is
# forced.
DR_RESEARCH_AGENT_TIMEOUT_S = int(
    os.environ.get("DR_RESEARCH_AGENT_TIMEOUT_S") or 30 * 60
)
DR_RESEARCH_AGENT_FORCE_REPORT_S = int(
    os.environ.get("DR_RESEARCH_AGENT_FORCE_REPORT_S") or 12 * 60
)


# Per-call sampling temperatures for the Deep Research phases. `None` (the
# default) keeps the session/model-configured temperature for every step.
# When set, each phase gets a value that fits its job instead of one number
# for everything: planning benefits from diversity, tool-calling cycles need
# format reliability, and report writing benefits from near-determinism so
# figures and wording stay consistent.
def _optional_float_env(name: str) -> float | None:
    value = os.environ.get(name)
    return float(value) if value else None


def _optional_int_env(name: str) -> int | None:
    value = os.environ.get(name)
    return int(value) if value else None


DR_TEMPERATURE_PLAN = _optional_float_env("DR_TEMPERATURE_PLAN")
DR_TEMPERATURE_ORCHESTRATOR = _optional_float_env("DR_TEMPERATURE_ORCHESTRATOR")
DR_TEMPERATURE_RESEARCH_AGENT = _optional_float_env("DR_TEMPERATURE_RESEARCH_AGENT")
DR_TEMPERATURE_REPORT = _optional_float_env("DR_TEMPERATURE_REPORT")

# Cap on the tokens a research sub-agent's prompt may occupy. Sub-agent
# search results accumulate in their history cycle over cycle; without a cap
# the prompt grows toward the model's max input tokens, which on slow
# hardware turns every sub-agent cycle into a long prefill. Oldest messages
# are dropped first when the cap is hit. `None` = uncapped.
DR_SUBAGENT_CONTEXT_TOKENS = _optional_int_env("DR_SUBAGENT_CONTEXT_TOKENS")
# Cap on a research sub-agent's intermediate report output tokens. These
# reports are consumed by the orchestrator, not by users, so a large cap
# mostly costs generation time on slow inference. `None` = the built-in
# default (10000) in research_agent.py.
DR_MAX_INTERMEDIATE_REPORT_TOKENS = _optional_int_env(
    "DR_MAX_INTERMEDIATE_REPORT_TOKENS"
)
# Generation caps for DR tool-calling steps (orchestrator and research
# sub-agent cycles). `max_tokens` is an upfront provider cap over ALL
# generated tokens — thinking, tool-call JSON, and answer text together — and
# no provider API caps "answer only". Thinking-capable models burn a big
# share of a tight cap on reasoning and get cut off mid-tool-call-JSON, which
# Ollama's server rejects as an invalid tool call. The cap therefore buys the
# answer budget its own headroom: reasoning models get the answer budget plus
# a separate thinking reserve; plain models keep the tight cap because their
# entire output is answer. The stream-level reasoning/answer split in
# llm_step is what keeps these two distinguishable in logs.
DR_TOOL_CALL_ANSWER_TOKENS = int(os.environ.get("DR_TOOL_CALL_ANSWER_TOKENS") or "1024")
DR_THINKING_TOKEN_RESERVE = int(os.environ.get("DR_THINKING_TOKEN_RESERVE") or "3072")


def dr_tool_call_max_tokens(is_reasoning_model: bool) -> int:
    """Total generation cap for a DR tool-calling step on the given model."""
    return DR_TOOL_CALL_ANSWER_TOKENS + (
        DR_THINKING_TOKEN_RESERVE if is_reasoning_model else 0
    )


# Model names/patterns for thinking-capable models that Onyx's capability
# registry cannot classify — custom Ollama pulls have no litellm entry, so
# model_is_reasoning_model returns False and the thinking reserve would never
# apply. Comma-separated exact names or fnmatch patterns, e.g.
# "ornith-1.5:9b,qwen3*-thinking*". Matched case-insensitively against the
# model name.
DR_THINKING_MODEL_OVERRIDES = tuple(
    s.strip().lower()
    for s in os.environ.get("DR_THINKING_MODEL_OVERRIDES", "").split(",")
    if s.strip()
)


def dr_is_thinking_model(
    model_name: str,
    is_reasoning_model: bool,
    api_base: str | None = None,
    model_provider: str | None = None,
) -> bool:
    """Whether a DR tool-calling step on this model needs the thinking reserve.

    Resolution order:
    1. The static registry flag (model_is_reasoning_model).
    2. The engine itself — Ollama-style /api/show capabilities, authoritative
       for custom pulled local models no registry knows.
    3. Name heuristics and DR_THINKING_MODEL_OVERRIDES, for engines that
       expose no capability API or are unreachable.

    Scoped to DR on purpose: the global reasoning flag also flips prompt
    selection and provider-side temperature handling, which we do not want to
    change for a local model that merely thinks natively.
    """
    if is_reasoning_model:
        return True

    if api_base and model_provider in ("ollama", "ollama_chat"):
        # Lazy import: chat_configs sits at the base of the config graph and
        # must stay import-light.
        from onyx.llm.engine_capabilities import ollama_model_capabilities

        capabilities = ollama_model_capabilities(api_base, model_name)
        if capabilities is not None:
            return "thinking" in capabilities

    name = model_name.lower()
    if "thinking" in name or "reason" in name:
        return True
    return any(fnmatch.fnmatch(name, pat) for pat in DR_THINKING_MODEL_OVERRIDES)


# Maximum research sub-agents running at the same time. 0 = unbounded (one
# thread per dispatched call — the historical behavior). The orchestrator
# prompt is intentionally unaware of this limit: it fans out one call per
# entity in a single turn, and this harness queues the rest, running at most
# `max_parallel` at once with each call's timeout measured from its own start.
DR_MAX_PARALLEL_SUBAGENTS = int(os.environ.get("DR_MAX_PARALLEL_SUBAGENTS") or "0")
# Timeout for non-streaming secondary LLM flows (e.g. search section-relevance
# classification and section-expansion selection). These are short, low-effort
# calls; the bound exists so a stalled provider connection fails fast into the
# existing graceful fallback instead of hanging a worker until liveness kills it.
SECONDARY_LLM_FLOW_TIMEOUT_S = int(
    os.environ.get("SECONDARY_LLM_FLOW_TIMEOUT_S") or "60"
)
# Live buffer TTL. Refreshed per write.
CHAT_STREAM_BUFFER_TTL_S = int(os.environ.get("CHAT_STREAM_BUFFER_TTL_S") or 3600)
# Retention after the run is done.
CHAT_STREAM_BUFFER_DONE_TTL_S = int(
    os.environ.get("CHAT_STREAM_BUFFER_DONE_TTL_S") or 600
)
# Cap on compressed buffer bytes.
CHAT_STREAM_BUFFER_MAX_BYTES = int(
    os.environ.get("CHAT_STREAM_BUFFER_MAX_BYTES") or 16 * 1024 * 1024
)
# Resume poll cadence.
CHAT_RESUME_POLL_INTERVAL_S = float(
    os.environ.get("CHAT_RESUME_POLL_INTERVAL_S") or 0.2
)
# Weighting factor between vector and keyword Search; 1 for completely vector
# search, 0 for keyword. Enforces a valid range of [0, 1]. A supplied value from
# the env outside of this range will be clipped to the respective end of the
# range. Defaults to 0.5.
HYBRID_ALPHA = max(0, min(1, float(os.environ.get("HYBRID_ALPHA") or 0.5)))
# Weighting factor between Title and Content of documents during search, 1 for completely
# Title based. Default heavily favors Content because Title is also included at the top of
# Content. This is to avoid cases where the Content is very relevant but it may not be clear
# if the title is separated out. Title is most of a "boost" than a separate field.
TITLE_CONTENT_RATIO = max(
    0, min(1, float(os.environ.get("TITLE_CONTENT_RATIO") or 0.10))
)

# Stops streaming answers back to the UI if this pattern is seen:
STOP_STREAM_PAT = os.environ.get("STOP_STREAM_PAT") or None

# Set this to "true" to hard delete chats
# This will make chats unviewable by admins after a user deletes them
# As opposed to soft deleting them, which just hides them from non-admin users
HARD_DELETE_CHATS = os.environ.get("HARD_DELETE_CHATS", "").lower() == "true"

# Internet Search
NUM_INTERNET_SEARCH_RESULTS = int(os.environ.get("NUM_INTERNET_SEARCH_RESULTS") or 10)
NUM_INTERNET_SEARCH_CHUNKS = int(os.environ.get("NUM_INTERNET_SEARCH_CHUNKS") or 50)

# SearXNG request timeouts. Browser-backed SearXNG instances fetch engine
# pages through a real Chromium (multi-second fetches, challenge warm-up
# renders, per-origin human pacing before the fetch), and the pool may wait
# out lane contention (bounded, with browsing preemption). The read timeout
# must cover queue wait + search time: worst attempt is ~225s (burst pacing
# cap + bounded lane wait + a slow engine fetch), so this is set to 2x that.
# Retries are paced (20s/40s backoff) like a person re-querying.
SEARXNG_CONNECT_TIMEOUT_SECONDS = float(
    os.environ.get("SEARXNG_CONNECT_TIMEOUT_SECONDS") or 10
)
SEARXNG_READ_TIMEOUT_SECONDS = float(
    os.environ.get("SEARXNG_READ_TIMEOUT_SECONDS") or 450
)

VESPA_SEARCHER_THREADS = int(os.environ.get("VESPA_SEARCHER_THREADS") or 2)

# Whether or not to use the semantic & keyword search expansions for Basic Search
USE_SEMANTIC_KEYWORD_EXPANSIONS_BASIC_SEARCH = (
    os.environ.get("USE_SEMANTIC_KEYWORD_EXPANSIONS_BASIC_SEARCH", "false").lower()
    == "true"
)

# Chat History Compression
# Trigger compression when history exceeds this ratio of available context window
COMPRESSION_TRIGGER_RATIO = float(os.environ.get("COMPRESSION_TRIGGER_RATIO", "0.75"))

SKIP_DEEP_RESEARCH_CLARIFICATION = (
    os.environ.get("SKIP_DEEP_RESEARCH_CLARIFICATION", "false").lower() == "true"
)
