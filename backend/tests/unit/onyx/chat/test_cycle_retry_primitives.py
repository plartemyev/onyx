"""Automatic turn recovery primitives: retry classifier, state rollback, and
the UI-visibility packet (LLM_CYCLE_RETRY_MAX paths in llm_loop).
"""

from litellm.exceptions import APIConnectionError as LiteLLMAPIConnectionError
from litellm.exceptions import InternalServerError as LiteLLMInternalServerError
from litellm.exceptions import ServiceUnavailableError as LiteLLMServiceUnavailableError
from litellm.exceptions import Timeout as LiteLLMTimeout

from onyx.chat.chat_state import ChatStateContainer
from onyx.chat.llm_loop import (
    FINAL_ANSWER_CONTINUATION_PROMPT,
    _restore_stream_state,
    _StreamStateSnapshot,
)
from onyx.llm.exceptions import LLMCycleTimeoutError
from onyx.llm.multi_llm import LLMTimeoutError
from onyx.llm.utils import is_retryable_transport_exception
from onyx.server.query_and_chat.streaming_models import CycleRetryNotice


def _litellm_connection_error() -> Exception:
    try:
        raise LiteLLMAPIConnectionError(
            message="connection reset mid-stream", llm_provider="p", model="m"
        )
    except LiteLLMAPIConnectionError as e:
        return e


def test_transport_faults_are_retryable() -> None:
    assert is_retryable_transport_exception(_litellm_connection_error())
    assert is_retryable_transport_exception(
        LiteLLMTimeout(model="m", llm_provider="p", message="read timed out")
    )
    assert is_retryable_transport_exception(
        LiteLLMServiceUnavailableError(
            model="m", llm_provider="p", message="overloaded"
        )
    )
    assert is_retryable_transport_exception(
        LiteLLMInternalServerError(model="m", llm_provider="p", message="500")
    )
    assert is_retryable_transport_exception(
        LLMTimeoutError("read timed out mid-stream")
    )


def test_non_transport_faults_are_not_retryable() -> None:
    assert not is_retryable_transport_exception(ValueError("bad args"))
    assert not is_retryable_transport_exception(LLMCycleTimeoutError(timeout_s=1800))
    assert not is_retryable_transport_exception(RuntimeError("boom"))


def test_failed_step_state_rolls_back_for_retry() -> None:
    container = ChatStateContainer()
    container.set_answer_tokens("answer before the failed step")
    container.set_reasoning_tokens("reasoning before")
    container.set_request_params({"model": "a"})

    snapshot = _StreamStateSnapshot(container)
    # The failed attempt streamed partials into the container.
    container.set_answer_tokens("answer before the failed stepPARTIAL")
    container.set_reasoning_tokens("reasoning beforePARTIAL")
    container.set_request_params({"model": "b"})

    _restore_stream_state(container, snapshot)

    assert container.get_answer_tokens() == "answer before the failed step"
    assert container.get_reasoning_tokens() == "reasoning before"
    assert container.get_request_params() == {"model": "a"}


def test_snapshot_of_empty_container_restores_empty() -> None:
    container = ChatStateContainer()
    snapshot = _StreamStateSnapshot(container)
    container.set_answer_tokens("partial from a failed first step")
    _restore_stream_state(container, snapshot)
    assert container.get_answer_tokens() is None


def test_retry_notice_packet_carries_machine_reason() -> None:
    notice = CycleRetryNotice(reason="cycle_timeout", attempt=1)
    data = notice.model_dump()
    assert data["type"] == "cycle_retry_notice"
    assert data["reason"] == "cycle_timeout"
    assert data["attempt"] == 1


def test_final_answer_nudge_is_imperative() -> None:
    # The nudge must forbid tools and demand plain text: the failure it
    # recovers is a model that answered with reasoning/tool calls instead.
    low = FINAL_ANSWER_CONTINUATION_PROMPT.lower()
    assert "must" in low
    assert "do not call any tools" in low
    assert "final answer" in low
