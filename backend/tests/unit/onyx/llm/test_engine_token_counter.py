from collections.abc import Callable
from types import SimpleNamespace
from typing import cast
from unittest.mock import Mock

import pytest

from onyx.llm import engine_token_counter
from onyx.llm.engine_token_counter import get_engine_token_counter
from onyx.llm.interfaces import LLM


@pytest.fixture(autouse=True)
def reset_state() -> None:
    engine_token_counter._tokenize_state.clear()


@pytest.fixture
def clock(monkeypatch: pytest.MonkeyPatch) -> list[float]:
    """Controllable time.monotonic for the module under test."""
    now = [1000.0]
    monkeypatch.setattr(
        "onyx.llm.engine_token_counter.time",
        SimpleNamespace(monotonic=lambda: now[0]),
    )
    return now


@pytest.fixture(autouse=True)
def local_counter(monkeypatch: pytest.MonkeyPatch) -> Mock:
    mock = Mock(return_value=999, name="local_counter")
    monkeypatch.setattr(
        "onyx.llm.engine_token_counter.get_llm_token_counter",
        Mock(return_value=mock),
    )
    return mock


@pytest.fixture
def engine_counts(monkeypatch: pytest.MonkeyPatch) -> Mock:
    mock = Mock(return_value=7, name="count_via_engine")
    monkeypatch.setattr("onyx.llm.engine_token_counter._count_via_engine", mock)
    return mock


def _make_counter() -> Callable[[str], int]:
    llm = cast(
        LLM,
        SimpleNamespace(
            config=SimpleNamespace(
                api_base="http://gpu-host:8011/v1", api_key="test-key"
            )
        ),
    )
    return get_engine_token_counter(llm)


def test_engine_count_preferred(
    engine_counts: Mock,  # noqa: ARG001 (patch fixture)
    local_counter: Mock,
) -> None:
    counter = _make_counter()

    assert counter("some text") == 7
    local_counter.assert_not_called()


def test_failure_falls_back_and_stays_cached_within_ttl(
    clock: list[float],
    engine_counts: Mock,
) -> None:
    engine_counts.return_value = None
    counter = _make_counter()

    assert counter("some text") == 999
    assert counter("more text") == 999
    # Negative cache: the engine is not probed again inside the TTL.
    assert engine_counts.call_count == 1

    clock[0] += engine_token_counter._NEGATIVE_CACHE_TTL_SECONDS - 1
    counter("even more text")
    assert engine_counts.call_count == 1


def test_reprobe_after_ttl_recovers_engine_count(
    clock: list[float],
    engine_counts: Mock,
    local_counter: Mock,
) -> None:
    engine_counts.return_value = None
    counter = _make_counter()
    assert counter("some text") == 999

    clock[0] += engine_token_counter._NEGATIVE_CACHE_TTL_SECONDS + 1
    engine_counts.return_value = 42
    assert counter("some text") == 42
    local_counter.assert_called_once()

    # Recovered: every call is counted by the engine again (a healthy
    # endpoint is consulted per call; only failures are TTL-cached).
    assert counter("more text") == 42
    assert engine_counts.call_count == 3


def test_reprobe_failure_rearms_negative_cache(
    clock: list[float],
    engine_counts: Mock,
) -> None:
    engine_counts.return_value = None
    counter = _make_counter()
    counter("some text")

    clock[0] += engine_token_counter._NEGATIVE_CACHE_TTL_SECONDS + 1
    counter("more text")
    assert engine_counts.call_count == 2

    # Still failing: cached again until the TTL elapses once more.
    clock[0] += 1
    counter("even more text")
    assert engine_counts.call_count == 2
