"""Wall-clock bound for a single chat LLM cycle (LLM_CYCLE_TIMEOUT_SECONDS).

Incident 2026-11-06: a thinking model spent ~3.5h decoding pure reasoning in
one cycle (no answer, no tool call) because no bound exists between the
turn budget (checked only at cycle starts) and the socket read timeout
(inter-chunk gaps only). The watchdog raises LLMCycleTimeoutError between
chunks; the chat loop then forces the final-answer cycle instead of failing
the turn.
"""

import time
from collections.abc import Generator, Iterator
from itertools import islice
from typing import Any

import pytest

from onyx.chat.llm_step import run_llm_step_pkt_generator
from onyx.llm.exceptions import LLMCycleTimeoutError
from onyx.llm.interfaces import LLM, LLMConfig, ToolChoiceOptions
from onyx.llm.model_response import Delta, ModelResponseStream, StreamingChoice
from onyx.server.query_and_chat.streaming_models import Placement


def _fake_llm_config() -> LLMConfig:
    return LLMConfig(
        model_provider="openai_compatible",
        model_name="test-model",
        temperature=0.0,
        max_input_tokens=8192,
    )


def _chunk(content: str = "x") -> ModelResponseStream:
    return ModelResponseStream(
        id="test",
        created="0",
        choice=StreamingChoice(finish_reason=None, delta=Delta(content=content)),
    )


class _SlowStreamLLM(LLM):
    """Yields a chunk every `chunk_interval_s`, effectively forever."""

    def __init__(self, chunk_interval_s: float) -> None:
        self._chunk_interval_s = chunk_interval_s

    @property
    def config(self) -> LLMConfig:  # abstract member, minimal valid value
        return _fake_llm_config()

    def stream(  # noqa: ARG002
        self,
        prompt: Any,  # noqa: ARG002
        tools: Any = None,  # noqa: ARG002
        tool_choice: Any = None,  # noqa: ARG002
        structured_response_format: dict | None = None,  # noqa: ARG002
        timeout_override: int | None = None,  # noqa: ARG002
        max_tokens: int | None = None,  # noqa: ARG002
        reasoning_effort: Any = None,  # noqa: ARG002
        user_identity: Any = None,  # noqa: ARG002
        temperature: float | None = None,  # noqa: ARG002
    ) -> Iterator[ModelResponseStream]:
        while True:
            time.sleep(self._chunk_interval_s)
            yield _chunk()


class _InstantStreamLLM(LLM):
    """Completes immediately with a short answer — must never trip the bound."""

    @property
    def config(self) -> LLMConfig:  # abstract member, minimal valid value
        return _fake_llm_config()

    def stream(  # noqa: ARG002
        self,
        prompt: Any,  # noqa: ARG002
        tools: Any = None,  # noqa: ARG002
        tool_choice: Any = None,  # noqa: ARG002
        structured_response_format: dict | None = None,  # noqa: ARG002
        timeout_override: int | None = None,  # noqa: ARG002
        max_tokens: int | None = None,  # noqa: ARG002
        reasoning_effort: Any = None,  # noqa: ARG002
        user_identity: Any = None,  # noqa: ARG002
        temperature: float | None = None,  # noqa: ARG002
    ) -> Iterator[ModelResponseStream]:
        yield _chunk("the answer")
        yield ModelResponseStream(
            id="test",
            created="0",
            choice=StreamingChoice(finish_reason="stop", delta=Delta()),
        )


def _run_stream(llm: LLM, cycle_timeout_s: float | None) -> None:
    generator: Generator[object, None, object] = run_llm_step_pkt_generator(
        history=[],
        tool_definitions=[],
        tool_choice=ToolChoiceOptions.AUTO,
        llm=llm,
        placement=Placement(turn_index=0),
        state_container=None,
        citation_processor=None,
        cycle_timeout_s=cycle_timeout_s,
    )
    # Drain like run_llm_step does; the watchdog raises mid-iteration.
    for _ in generator:
        pass


def test_slow_stream_raises_cycle_timeout() -> None:
    # The abort check is throttled to _ABORT_CHECK_INTERVAL_S (0.5s), so the
    # chunk interval must stay below the deadline to guarantee the check runs
    # after it passed.
    llm = _SlowStreamLLM(chunk_interval_s=0.1)
    with pytest.raises(LLMCycleTimeoutError) as exc_info:
        _run_stream(llm, cycle_timeout_s=0.8)
    assert exc_info.value.error_code == "LLM_CYCLE_TIMEOUT"
    assert exc_info.value.is_retryable is True


def test_fast_stream_unaffected() -> None:
    llm = _InstantStreamLLM()
    # Bound generous relative to the instant stream: completes normally.
    _run_stream(llm, cycle_timeout_s=30.0)


def test_disabled_when_unset() -> None:
    llm = _SlowStreamLLM(chunk_interval_s=0.05)
    generator = run_llm_step_pkt_generator(
        history=[],
        tool_definitions=[],
        tool_choice=ToolChoiceOptions.AUTO,
        llm=llm,
        placement=Placement(turn_index=0),
        state_container=None,
        citation_processor=None,
        cycle_timeout_s=None,
    )
    # None disables the bound; take a few packets and close — the slow stream
    # must still be running, not cancelled.
    for _ in islice(generator, 3):
        pass
    generator.close()
