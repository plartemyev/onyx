"""Unit tests for tool response digests used by in-turn compaction."""

import json
import time
from collections.abc import Iterator
from unittest.mock import MagicMock

import pytest

from onyx.chat import tool_response_digest
from onyx.chat.tool_response_digest import (
    compute_tool_response_digest,
    extractive_digest,
    stub_message_for,
)
from onyx.llm.context_budgets import (
    DR_THINKING_RESERVE_BUDGET,
    TOOL_RESPONSE_DIGEST_SUMMARY_OUTPUT,
    scale,
)


def _search_docs_payload(num_results: int, snippet_chars: int = 500) -> str:
    results = [
        {
            "document": index,
            "title": f"Result title {index} about Pantip threads",
            "url": f"https://example.com/page/{index}",
            "content": "x" * snippet_chars,
        }
        for index in range(1, num_results + 1)
    ]
    return json.dumps({"results": results})


def _counter(text: str) -> int:
    return max(1, len(text) // 4)


# extractive_digest: the search-docs shape digests to title/URL lines


def test_results_shape_digest_keeps_titles_and_urls_drops_snippets() -> None:
    text = _search_docs_payload(num_results=8)
    digest = extractive_digest(text)
    assert digest is not None
    assert "- Result title 1 about Pantip threads — https://example.com/page/1" in (
        digest
    )
    assert "xxx" not in digest  # snippets are gone
    assert len(digest) < len(text) / 4


def test_results_shape_digest_caps_entries_and_line_length() -> None:
    long_title = "T" * 400
    results = [
        {"document": index, "title": long_title, "url": f"https://e.com/{index}"}
        for index in range(30)
    ]
    digest = extractive_digest(json.dumps({"results": results}))
    assert digest is not None
    lines = digest.split("\n")
    assert len(lines) <= tool_response_digest._RESULT_DIGEST_MAX_ENTRIES
    for line in lines:
        assert len(line) <= (
            tool_response_digest._RESULT_LINE_TITLE_MAX_CHARS
            + tool_response_digest._RESULT_LINE_URL_MAX_CHARS
            + 10
        )


def test_small_text_gets_no_digest() -> None:
    """Below the saving threshold the bare compaction notice wins: a tiny
    response whose digest would nearly duplicate it is not worth stubbing."""
    assert extractive_digest("Huge result") is None
    bare = json.dumps(
        {"results": [{"document": 1, "title": "T", "url": "https://e.com/1"}]}
    )
    assert extractive_digest(bare) is None


def test_long_plain_text_gets_head_digest_cut_at_line_boundary(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        tool_response_digest, "GEN_AI_TOOL_RESPONSE_DIGEST_MAX_CHARS", 1200
    )
    body = "\n".join(f"line {index} " + "y" * 80 for index in range(40))
    digest = extractive_digest(body)
    assert digest is not None
    assert digest.endswith("…")
    assert len(digest) < len(body)


def test_non_json_short_text_is_untouched() -> None:
    assert extractive_digest("plain") is None


def test_digests_disabled(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(tool_response_digest, "GEN_AI_TOOL_RESPONSE_DIGESTS", False)
    assert extractive_digest(_search_docs_payload(num_results=8)) is None


def test_malformed_results_shape_falls_back_to_generic_head(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A JSON blob claiming "results" but breaking the shape must not get
    result lines; the generic head excerpt is the honest digest."""
    monkeypatch.setattr(
        tool_response_digest, "GEN_AI_TOOL_RESPONSE_DIGEST_MAX_CHARS", 1200
    )
    text = json.dumps({"results": ["not-a-dict"]}) + "x" * 2000
    digest = extractive_digest(text)
    assert digest is not None
    assert digest.startswith('{"results"')
    assert digest.endswith("…")
    assert "- " not in digest.split("\n")[0]


# compute_tool_response_digest: tier 2 with fallback


def _mock_llm(summary: str | None = None, delay: float = 0.0) -> MagicMock:
    llm = MagicMock()
    llm.config.max_input_tokens = 50_000
    response = MagicMock()
    response.choice.message.content = summary
    if delay:

        def _slow_invoke(*args: object, **kwargs: object) -> MagicMock:  # noqa: ARG001
            time.sleep(delay)
            return response

        llm.invoke.side_effect = _slow_invoke
    else:
        llm.invoke.return_value = response
    return llm


@pytest.fixture(autouse=True)
def _no_tracing(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(tool_response_digest, "llm_generation_span", MagicMock())
    monkeypatch.setattr(tool_response_digest, "record_llm_response", MagicMock())


@pytest.fixture(autouse=True)
def _clear_digest_state() -> Iterator[None]:
    """Isolate the process-wide digest state between tests."""
    with tool_response_digest._DIGEST_STATE_LOCK:
        tool_response_digest._DIGEST_STORE.clear()
        tool_response_digest._INFLIGHT_DIGEST_UPGRADES.clear()
    yield
    with tool_response_digest._DIGEST_STATE_LOCK:
        tool_response_digest._DIGEST_STORE.clear()
        tool_response_digest._INFLIGHT_DIGEST_UPGRADES.clear()


def test_summary_used_when_enabled(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        tool_response_digest, "GEN_AI_TOOL_RESPONSE_SUMMARIZATION", True
    )
    monkeypatch.setattr(
        tool_response_digest, "model_is_reasoning_model", lambda *_args, **_kwargs: False
    )
    llm = _mock_llm(summary="Pantip threads about mangrove fires; 87k-member group.")
    text = _search_docs_payload(num_results=8)
    digest = compute_tool_response_digest(text, llm=llm)
    assert digest == "Pantip threads about mangrove fires; 87k-member group."
    # summary call is bounded: temperature 0, small output, no thinking; the
    # output cap scales with the model's window (220 tokens at the
    # 50,000-token calibration reference).
    _, kwargs = llm.invoke.call_args
    assert kwargs["temperature"] == 0.0
    assert kwargs["max_tokens"] == scale(50_000, TOOL_RESPONSE_DIGEST_SUMMARY_OUTPUT)


def test_summary_max_tokens_includes_thinking_reserve(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """On a thinking model max_tokens counts native reasoning too, so the
    digest call carries the thinking reserve on top of the answer budget —
    without it the whole cap burns inside the think trace and the summary
    comes back empty (observed live 2026-11-04: finish_reason=length with
    empty content, llama.cpp ignoring reasoning_effort=OFF)."""
    monkeypatch.setattr(
        tool_response_digest, "GEN_AI_TOOL_RESPONSE_SUMMARIZATION", True
    )
    monkeypatch.setattr(
        tool_response_digest, "model_is_reasoning_model", lambda *_args, **_kwargs: True
    )
    llm = _mock_llm(summary="Pantip threads about mangrove fires; 87k-member group.")
    digest = compute_tool_response_digest(_search_docs_payload(num_results=8), llm=llm)
    assert digest == "Pantip threads about mangrove fires; 87k-member group."
    _, kwargs = llm.invoke.call_args
    assert kwargs["max_tokens"] == scale(
        50_000, TOOL_RESPONSE_DIGEST_SUMMARY_OUTPUT
    ) + scale(50_000, DR_THINKING_RESERVE_BUDGET)


def test_upgrade_routes_to_secondary_model(monkeypatch: pytest.MonkeyPatch) -> None:
    """Tier-2 upgrades run on the deployment's designated secondary model —
    they are auxiliary background work and must not contend with the chat
    or DR loop for the primary engine's slot."""
    monkeypatch.setattr(
        tool_response_digest, "GEN_AI_TOOL_RESPONSE_SUMMARIZATION", True
    )
    primary = _mock_llm(summary="primary summary")
    secondary = _mock_llm(summary="secondary summary")

    class _FakeSession:
        def __enter__(self) -> object:
            return self

        def __exit__(self, *args: object) -> None:
            pass

    monkeypatch.setattr(
        tool_response_digest,
        "get_session_with_current_tenant",
        lambda *_args, **_kwargs: _FakeSession(),
    )
    monkeypatch.setattr(
        tool_response_digest,
        "get_secondary_llm",
        lambda _db_session: secondary,
    )
    captured: list[object] = []
    monkeypatch.setattr(
        tool_response_digest,
        "_run_digest_upgrade",
        lambda _tool_call_id, _text, _cache, llm: captured.append(llm),
    )
    cache: dict[str, str] = {}
    tool_response_digest._schedule_digest_upgrade(
        "tc_1", _search_docs_payload(num_results=8), cache, primary
    )
    time.sleep(0.05)  # let the upgrade thread run
    assert captured == [secondary]
    with tool_response_digest._DIGEST_STATE_LOCK:
        tool_response_digest._INFLIGHT_DIGEST_UPGRADES.discard("tc_1")


def test_upgrade_falls_back_to_chat_model_without_secondary(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        tool_response_digest, "GEN_AI_TOOL_RESPONSE_SUMMARIZATION", True
    )
    primary = _mock_llm(summary="primary summary")

    class _FakeSession:
        def __enter__(self) -> object:
            return self

        def __exit__(self, *args: object) -> None:
            pass

    monkeypatch.setattr(
        tool_response_digest,
        "get_session_with_current_tenant",
        lambda *_args, **_kwargs: _FakeSession(),
    )
    monkeypatch.setattr(
        tool_response_digest,
        "get_secondary_llm",
        lambda _db_session: None,
    )
    captured: list[object] = []
    monkeypatch.setattr(
        tool_response_digest,
        "_run_digest_upgrade",
        lambda _tool_call_id, _text, _cache, llm: captured.append(llm),
    )
    tool_response_digest._schedule_digest_upgrade(
        "tc_1", _search_docs_payload(num_results=8), {}, primary
    )
    time.sleep(0.05)
    assert captured == [primary]
    with tool_response_digest._DIGEST_STATE_LOCK:
        tool_response_digest._INFLIGHT_DIGEST_UPGRADES.discard("tc_1")


def test_summary_disabled_uses_extractive(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        tool_response_digest, "GEN_AI_TOOL_RESPONSE_SUMMARIZATION", False
    )
    llm = _mock_llm(summary="never called")
    digest = compute_tool_response_digest(_search_docs_payload(num_results=8), llm=llm)
    assert digest is not None
    assert "Result title 1" in digest
    llm.invoke.assert_not_called()


def test_summary_failure_falls_back_to_extractive(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        tool_response_digest, "GEN_AI_TOOL_RESPONSE_SUMMARIZATION", True
    )
    llm = _mock_llm()
    llm.invoke.side_effect = RuntimeError("provider down")
    digest = compute_tool_response_digest(_search_docs_payload(num_results=8), llm=llm)
    assert digest is not None
    assert "Result title 1" in digest


def test_empty_summary_falls_back_to_extractive(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        tool_response_digest, "GEN_AI_TOOL_RESPONSE_SUMMARIZATION", True
    )
    llm = _mock_llm(summary="   ")
    digest = compute_tool_response_digest(_search_docs_payload(num_results=8), llm=llm)
    assert digest is not None
    assert "Result title 1" in digest


def test_empty_summary_retried_once_then_succeeds(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        tool_response_digest, "GEN_AI_TOOL_RESPONSE_SUMMARIZATION", True
    )
    llm = _mock_llm(summary="   ")
    empty = MagicMock()
    empty.choice.message.content = "   "
    good = MagicMock()
    good.choice.message.content = "Recovered summary on the second call."
    llm.invoke.side_effect = [empty, good]
    digest = compute_tool_response_digest(_search_docs_payload(num_results=8), llm=llm)
    assert digest == "Recovered summary on the second call."
    assert llm.invoke.call_count == 2


def test_empty_summary_retried_twice_still_empty_falls_back(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        tool_response_digest, "GEN_AI_TOOL_RESPONSE_SUMMARIZATION", True
    )
    llm = _mock_llm(summary="   ")
    digest = compute_tool_response_digest(_search_docs_payload(num_results=8), llm=llm)
    assert digest is not None
    assert "Result title 1" in digest
    assert llm.invoke.call_count == tool_response_digest._EMPTY_SUMMARY_ATTEMPTS


def test_summary_timeout_falls_back_to_extractive(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        tool_response_digest, "GEN_AI_TOOL_RESPONSE_SUMMARIZATION", True
    )
    monkeypatch.setattr(
        tool_response_digest,
        "GEN_AI_TOOL_RESPONSE_SUMMARIZATION_TIMEOUT_SECONDS",
        0.2,
    )
    llm = _mock_llm(summary="too late", delay=2.0)
    digest = compute_tool_response_digest(_search_docs_payload(num_results=8), llm=llm)
    assert digest is not None
    assert "Result title 1" in digest


# stub_message_for: the lazy tier-2 upgrade


def test_stub_cache_miss_stores_extractive_and_schedules_upgrade(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        tool_response_digest, "GEN_AI_TOOL_RESPONSE_SUMMARIZATION", True
    )
    scheduled: list[tuple[str, str, dict[str, str] | None, object]] = []

    def _record_schedule(
        tool_call_id: str,
        text: str,
        cache: dict[str, str] | None,
        llm: object,
    ) -> None:
        scheduled.append((tool_call_id, text, cache, llm))

    monkeypatch.setattr(
        tool_response_digest, "_schedule_digest_upgrade", _record_schedule
    )
    llm = _mock_llm()
    cache: dict[str, str] = {}
    big = _search_docs_payload(num_results=8, snippet_chars=1200)
    stub = stub_message_for(
        tool_call_id="tc_1",
        original_message=big,
        cache=cache,
        token_counter=_counter,
        llm=llm,
    )
    assert stub is not None
    # The deterministic digest is in the cache (and the stub) immediately —
    # the fit never waits on the LLM...
    assert "Result title 1" in cache["tc_1"]
    # ...and exactly one background upgrade was scheduled for the payload.
    assert scheduled == [("tc_1", big, cache, llm)]


def test_stub_cache_hit_does_not_schedule_upgrade(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        tool_response_digest, "GEN_AI_TOOL_RESPONSE_SUMMARIZATION", True
    )
    scheduled: list[object] = []
    monkeypatch.setattr(
        tool_response_digest,
        "_schedule_digest_upgrade",
        lambda *args: scheduled.append(args),
    )
    stub_message_for(
        tool_call_id="tc_1",
        original_message=_search_docs_payload(num_results=8),
        cache={"tc_1": "the findings, condensed"},
        token_counter=_counter,
        llm=_mock_llm(),
    )
    assert scheduled == []


def test_stub_upgrade_needs_llm_and_summarization(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import threading as _threading

    monkeypatch.setattr(
        tool_response_digest, "GEN_AI_TOOL_RESPONSE_SUMMARIZATION", True
    )
    release = _threading.Event()

    def _blocker(
        tool_call_id: str,  # noqa: ARG001
        text: str,  # noqa: ARG001
        cache: dict[str, str],  # noqa: ARG001
        llm: object,  # noqa: ARG001
    ) -> None:
        release.wait(timeout=5)

    monkeypatch.setattr(tool_response_digest, "_run_digest_upgrade", _blocker)
    payload = _search_docs_payload(num_results=8)
    cache: dict[str, str] = {}

    def _inflight() -> set[str]:
        with tool_response_digest._DIGEST_STATE_LOCK:
            return set(tool_response_digest._INFLIGHT_DIGEST_UPGRADES)

    # Summarization disabled (default): no upgrade even with an LLM.
    monkeypatch.setattr(
        tool_response_digest, "GEN_AI_TOOL_RESPONSE_SUMMARIZATION", False
    )
    tool_response_digest._schedule_digest_upgrade("tc_1", payload, cache, _mock_llm())
    assert "tc_1" not in _inflight()
    # Enabled but no LLM: still no upgrade.
    monkeypatch.setattr(
        tool_response_digest, "GEN_AI_TOOL_RESPONSE_SUMMARIZATION", True
    )
    tool_response_digest._schedule_digest_upgrade("tc_2", payload, cache, None)
    assert "tc_2" not in _inflight()
    # Enabled with an LLM but no cache to land the digest in: no upgrade.
    tool_response_digest._schedule_digest_upgrade("tc_3", payload, None, _mock_llm())
    assert "tc_3" not in _inflight()
    # Enabled, LLM and cache present: the upgrade runs (tracked in flight).
    tool_response_digest._schedule_digest_upgrade("tc_4", payload, cache, _mock_llm())
    assert "tc_4" in _inflight()
    release.set()
    time.sleep(0.05)  # let the upgrade thread exit before the env reverts
    # The monkeypatched body skips the real self-discard; clean up.
    with tool_response_digest._DIGEST_STATE_LOCK:
        tool_response_digest._INFLIGHT_DIGEST_UPGRADES.discard("tc_4")


def test_stub_without_cache_still_stubs_but_cannot_upgrade(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        tool_response_digest, "GEN_AI_TOOL_RESPONSE_SUMMARIZATION", True
    )
    stub = stub_message_for(
        tool_call_id="tc_1",
        original_message=_search_docs_payload(num_results=8),
        cache=None,
        token_counter=_counter,
        llm=_mock_llm(),
    )
    # The stub replays the deterministic digest; with no cache there is
    # nowhere to land an upgrade, and none was queued.
    assert stub is not None
    with tool_response_digest._DIGEST_STATE_LOCK:
        assert "tc_1" not in tool_response_digest._INFLIGHT_DIGEST_UPGRADES


def test_scheduler_dedups_inflight_upgrades(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        tool_response_digest, "GEN_AI_TOOL_RESPONSE_SUMMARIZATION", True
    )
    calls: list[str] = []

    def _slow_body(
        tool_call_id: str,
        text: str,  # noqa: ARG001
        cache: dict[str, str],  # noqa: ARG001
        llm: object,  # noqa: ARG001
    ) -> None:
        calls.append(tool_call_id)
        time.sleep(0.3)

    monkeypatch.setattr(tool_response_digest, "_run_digest_upgrade", _slow_body)
    llm = _mock_llm()
    cache: dict[str, str] = {}
    payload = _search_docs_payload(num_results=8)
    # The first call registers the upgrade synchronously (before the thread
    # starts), so the second call for the same tool_call_id must dedup.
    tool_response_digest._schedule_digest_upgrade("tc_1", payload, cache, llm)
    tool_response_digest._schedule_digest_upgrade("tc_1", payload, cache, llm)
    deadline = time.time() + 5
    while not calls and time.time() < deadline:
        time.sleep(0.01)
    assert calls == ["tc_1"]
    time.sleep(0.35)  # let the body exit before the env reverts
    with tool_response_digest._DIGEST_STATE_LOCK:
        tool_response_digest._INFLIGHT_DIGEST_UPGRADES.discard("tc_1")


def test_scheduler_caps_inflight_upgrades(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        tool_response_digest, "GEN_AI_TOOL_RESPONSE_SUMMARIZATION", True
    )
    with tool_response_digest._DIGEST_STATE_LOCK:
        tool_response_digest._INFLIGHT_DIGEST_UPGRADES.update(
            f"tc_{index}"
            for index in range(tool_response_digest._MAX_INFLIGHT_DIGEST_UPGRADES)
        )
    try:
        tool_response_digest._schedule_digest_upgrade(
            "tc_new",
            _search_docs_payload(num_results=8),
            {},
            _mock_llm(),
        )
        assert "tc_new" not in tool_response_digest._INFLIGHT_DIGEST_UPGRADES
    finally:
        with tool_response_digest._DIGEST_STATE_LOCK:
            tool_response_digest._INFLIGHT_DIGEST_UPGRADES.clear()


def test_upgrade_replaces_cached_extractive_with_summary(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        tool_response_digest, "GEN_AI_TOOL_RESPONSE_SUMMARIZATION", True
    )
    llm = _mock_llm(summary="Pantip threads about mangrove fires; 87k members.")
    cache = {"tc_1": "Result title 1 — https://example.com/page/1"}
    tool_response_digest._run_digest_upgrade(
        "tc_1", _search_docs_payload(num_results=8), cache, llm
    )
    assert cache["tc_1"] == "Pantip threads about mangrove fires; 87k members."
    # The upgrade also lands in the cross-turn store, so later turns replay
    # the summary without re-summarizing.
    assert (
        tool_response_digest._stored_digest("tc_1")
        == "Pantip threads about mangrove fires; 87k members."
    )


def test_upgrade_failure_keeps_a_digest_in_place(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        tool_response_digest, "GEN_AI_TOOL_RESPONSE_SUMMARIZATION", True
    )
    llm = _mock_llm(summary=None)  # empty summary -> the fallback path
    cache = {"tc_1": "Result title 1 — https://example.com/page/1"}
    tool_response_digest._run_digest_upgrade(
        "tc_1", _search_docs_payload(num_results=8), cache, llm
    )
    # The upgrade never empties the cache: a digest (here the recomputed
    # deterministic one) is always present.
    assert "Result title 1" in cache["tc_1"]


def test_summary_input_cap_keeps_head_and_tail_cuts_middle(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        tool_response_digest, "GEN_AI_TOOL_RESPONSE_SUMMARIZATION_MAX_INPUT_CHARS", 1000
    )
    text = "H" * 900 + "M" * 500 + "T" * 900
    capped = tool_response_digest._cap_summary_input(text)
    assert len(capped) <= 1000
    assert capped.startswith("HHHHHHHH")
    assert capped.endswith("TTTTTTTT")
    assert "MMMM" not in capped
    # Short input passes through untouched.
    assert tool_response_digest._cap_summary_input("short") == "short"


def test_summary_input_capped_for_huge_payloads(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        tool_response_digest, "GEN_AI_TOOL_RESPONSE_SUMMARIZATION", True
    )
    monkeypatch.setattr(
        tool_response_digest,
        "GEN_AI_TOOL_RESPONSE_SUMMARIZATION_MAX_INPUT_CHARS",
        5000,
    )
    llm = _mock_llm(summary="summary")
    text = _search_docs_payload(num_results=8, snippet_chars=5000)
    compute_tool_response_digest(text, llm=llm)
    content = llm.invoke.call_args[0][0][0].content
    assert len(content) < len(text)
    assert tool_response_digest._SUMMARY_INPUT_CUT_MARKER in content


# cross-turn digest store


def test_digest_store_reused_across_caches(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        tool_response_digest, "GEN_AI_TOOL_RESPONSE_SUMMARIZATION", True
    )
    scheduled: list[tuple[object, ...]] = []
    monkeypatch.setattr(
        tool_response_digest,
        "_schedule_digest_upgrade",
        lambda *args: scheduled.append(args),
    )
    payload = _search_docs_payload(num_results=8)
    first: dict[str, str] = {}
    stub_message_for(
        tool_call_id="tc_1",
        original_message=payload,
        cache=first,
        token_counter=_counter,
        llm=None,
    )
    assert "Result title 1" in first["tc_1"]
    # A later turn with a fresh per-turn cache reuses the stored digest and
    # does not schedule another upgrade for it.
    second: dict[str, str] = {}
    stub = stub_message_for(
        tool_call_id="tc_1",
        original_message=payload,
        cache=second,
        token_counter=_counter,
        llm=_mock_llm(),
    )
    assert stub is not None
    assert second["tc_1"] == first["tc_1"]
    # Exactly one upgrade was requested — from the first call (the full
    # miss); the store hit on the second call must not re-request it.
    assert len(scheduled) == 1
    assert scheduled[0][0] == "tc_1"


def test_digest_store_evicts_oldest(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(tool_response_digest, "_DIGEST_STORE_MAX_ENTRIES", 2)
    tool_response_digest._store_digest("a", "digest-a")
    tool_response_digest._store_digest("b", "digest-b")
    tool_response_digest._store_digest("c", "digest-c")
    assert tool_response_digest._stored_digest("a") is None
    assert tool_response_digest._stored_digest("b") == "digest-b"
    assert tool_response_digest._stored_digest("c") == "digest-c"


# stub_message_for: what compaction actually replays


def test_stub_message_uses_cached_digest() -> None:
    cache = {"tc_1": "the findings, condensed"}
    stub = stub_message_for(
        tool_call_id="tc_1",
        original_message=_search_docs_payload(num_results=8),
        cache=cache,
        token_counter=_counter,
    )
    assert stub is not None
    message, tokens = stub
    assert "the findings, condensed" in message
    assert "Repeating the identical call will not recover it" in message
    assert tokens == _counter(message)


def test_stub_message_extracts_on_the_fly() -> None:
    stub = stub_message_for(
        tool_call_id="tc_2",
        original_message=_search_docs_payload(num_results=8),
        cache={},
        token_counter=_counter,
    )
    assert stub is not None
    message, _ = stub
    assert "https://example.com/page/1" in message


def test_stub_message_none_without_counter() -> None:
    assert (
        stub_message_for(
            tool_call_id="tc_1",
            original_message=_search_docs_payload(num_results=8),
            cache={},
            token_counter=None,
        )
        is None
    )


def test_stub_message_none_for_small_response() -> None:
    assert (
        stub_message_for(
            tool_call_id="tc_1",
            original_message="Huge result",
            cache={},
            token_counter=_counter,
        )
        is None
    )


def test_stub_message_none_without_tool_call_id() -> None:
    assert (
        stub_message_for(
            tool_call_id=None,
            original_message=_search_docs_payload(num_results=8),
            cache={},
            token_counter=_counter,
        )
        is None
    )
