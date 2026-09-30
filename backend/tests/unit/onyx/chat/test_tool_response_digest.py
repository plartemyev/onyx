"""Unit tests for tool response digests used by in-turn compaction."""

import json
import time
from unittest.mock import MagicMock

import pytest

from onyx.chat import tool_response_digest
from onyx.chat.tool_response_digest import (
    compute_tool_response_digest,
    extractive_digest,
    maybe_digest_tool_response,
    stub_message_for,
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


def test_long_plain_text_gets_head_digest_cut_at_line_boundary() -> None:
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


def test_malformed_results_shape_falls_back_to_generic_head() -> None:
    """A JSON blob claiming "results" but breaking the shape must not get
    result lines; the generic head excerpt is the honest digest."""
    text = json.dumps({"results": ["not-a-dict"]}) + "x" * 2000
    digest = extractive_digest(text)
    assert digest is not None
    assert digest.startswith('{"results"')
    assert digest.endswith("…")
    assert "- " not in digest.split("\n")[0]


# compute_tool_response_digest: tier 2 with fallback


def _mock_llm(summary: str | None = None, delay: float = 0.0) -> MagicMock:
    llm = MagicMock()
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


def test_summary_used_when_enabled(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        tool_response_digest, "GEN_AI_TOOL_RESPONSE_SUMMARIZATION", True
    )
    llm = _mock_llm(summary="Pantip threads about mangrove fires; 87k-member group.")
    text = _search_docs_payload(num_results=8)
    digest = compute_tool_response_digest(text, llm=llm)
    assert digest == "Pantip threads about mangrove fires; 87k-member group."
    # summary call is bounded: temperature 0, small output, no thinking
    _, kwargs = llm.invoke.call_args
    assert kwargs["temperature"] == 0.0
    assert kwargs["max_tokens"] == tool_response_digest._SUMMARY_MAX_TOKENS


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


# maybe_digest_tool_response: the eager refinement


def test_eager_skips_small_responses(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        tool_response_digest, "GEN_AI_TOOL_RESPONSE_SUMMARIZATION", True
    )
    llm = _mock_llm(summary="summarized")
    cache: dict[str, str] = {}
    maybe_digest_tool_response("x" * 100, "tc_1", cache, llm=llm)
    assert cache == {}
    llm.invoke.assert_not_called()


def test_eager_digests_once_per_tool_call(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        tool_response_digest, "GEN_AI_TOOL_RESPONSE_SUMMARIZATION", True
    )
    llm = _mock_llm(summary="one summary")
    cache: dict[str, str] = {}
    big = _search_docs_payload(num_results=8, snippet_chars=1200)
    maybe_digest_tool_response(big, "tc_1", cache, llm=llm)
    maybe_digest_tool_response(big, "tc_1", cache, llm=llm)
    assert cache == {"tc_1": "one summary"}
    assert llm.invoke.call_count == 1


def test_eager_without_summarization_stores_extractive() -> None:
    cache: dict[str, str] = {}
    maybe_digest_tool_response(
        _search_docs_payload(num_results=8, snippet_chars=1200),
        "tc_1",
        cache,
        llm=None,
    )
    assert "tc_1" in cache
    assert "Result title 1" in cache["tc_1"]


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
