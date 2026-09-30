import pytest

from onyx.chat.llm_step import run_llm_step
from onyx.configs.chat_configs import (
    DR_THINKING_TOKEN_RESERVE,
    DR_TOOL_CALL_ANSWER_TOKENS,
    dr_step_generation_budget,
    dr_tool_call_max_tokens,
)
from onyx.llm.utils import (
    is_truncated_tool_call_exception,
    litellm_exception_to_error_msg,
)

OLLAMA_TRUNCATED_TOOL_CALL_MSG = (
    "litellm.APIConnectionError: Ollama_chatException - KeyError: 'message', "
    "Got unexpected response from Ollama: {'error': 'llama-server returned "
    'invalid tool call arguments for "research_agent": unexpected end of '
    "JSON input'}"
)


def _make_ollama_connection_error() -> Exception:
    from litellm.exceptions import APIConnectionError

    return APIConnectionError(
        message=OLLAMA_TRUNCATED_TOOL_CALL_MSG,
        llm_provider="ollama_chat",
        model="ornith-1.5:9b",
    )


def test_dr_tool_call_max_tokens_plain_model_gets_answer_budget_only(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("DR_TOOL_CALL_ANSWER_TOKENS", raising=False)
    monkeypatch.delenv("DR_THINKING_TOKEN_RESERVE", raising=False)
    # 50,000 tokens is the calibration reference window: the relative
    # defaults reproduce the historical absolute budgets there.
    assert (
        dr_tool_call_max_tokens(is_reasoning_model=False, max_input_tokens=50_000)
        == 1024
    )


def test_dr_tool_call_max_tokens_reasoning_model_gets_reserve(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("DR_TOOL_CALL_ANSWER_TOKENS", raising=False)
    monkeypatch.delenv("DR_THINKING_TOKEN_RESERVE", raising=False)
    assert dr_tool_call_max_tokens(
        is_reasoning_model=True, max_input_tokens=50_000
    ) == (1024 + 3072)


def test_dr_step_generation_budget_adds_reserve_for_thinking_models(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Any answer-bearing step (e.g. the report steps) needs the thinking
    reserve on top of its answer budget: max_tokens caps thinking + answer
    combined, and a thinking model that exhausts the cap ends with no answer
    at all (finish_reason=length, empty answer)."""
    monkeypatch.delenv("DR_THINKING_TOKEN_RESERVE", raising=False)
    assert (
        dr_step_generation_budget(
            800, is_reasoning_model=False, max_input_tokens=50_000
        )
        == 800
    )
    assert (
        dr_step_generation_budget(800, is_reasoning_model=True, max_input_tokens=50_000)
        == 800 + 3072
    )


def test_dr_tool_call_max_tokens_honors_config(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The helper reads the module-level constants, so ops can retune them
    via env (bound at import time)."""
    from onyx.configs import chat_configs

    monkeypatch.setattr(chat_configs, "DR_TOOL_CALL_ANSWER_TOKENS", 2048)
    monkeypatch.setattr(chat_configs, "DR_THINKING_TOKEN_RESERVE", 512)
    assert (
        dr_tool_call_max_tokens(is_reasoning_model=False, max_input_tokens=50_000)
        == 2048
    )
    assert (
        dr_tool_call_max_tokens(is_reasoning_model=True, max_input_tokens=50_000)
        == 2560
    )


def test_defaults_unchanged() -> None:
    """Guard against accidental default drift: with no env override, the
    budgets scale with the model's context window and reproduce the
    historical 1024 / 3072 at the calibration reference window."""
    assert DR_TOOL_CALL_ANSWER_TOKENS is None
    assert DR_THINKING_TOKEN_RESERVE is None
    from onyx.llm.context_budgets import (
        DR_THINKING_RESERVE_BUDGET,
        DR_TOOL_CALL_ANSWER_BUDGET,
        scale,
    )

    assert scale(50_000, DR_TOOL_CALL_ANSWER_BUDGET) == 1024
    assert scale(50_000, DR_THINKING_RESERVE_BUDGET) == 3072


def test_truncated_tool_call_exception_detected() -> None:
    assert is_truncated_tool_call_exception(_make_ollama_connection_error())


def test_plain_connection_error_not_flagged() -> None:
    from litellm.exceptions import APIConnectionError

    plain = APIConnectionError(
        message="Connection error.", llm_provider="ollama_chat", model="m"
    )
    assert not is_truncated_tool_call_exception(plain)


def test_safe_error_mapping_is_truthful_for_truncated_tool_calls() -> None:
    """The user-facing message must not claim a connection problem."""
    message, error_code, is_retryable = litellm_exception_to_error_msg(
        _make_ollama_connection_error(),
        None,
        custom_error_msg_mappings=None,
    )
    assert error_code == "MODEL_TOOL_CALL_MALFORMED"
    assert is_retryable is True
    assert "truncated tool call" in message
    assert "not a connection problem" in message
    assert "internet connection" not in message


def test_llm_step_exposes_retry_flag() -> None:
    """The retry lives inside run_llm_step; DR call sites just opt in."""
    import inspect

    params = inspect.signature(run_llm_step).parameters
    flag = params["retry_on_truncated_tool_call"]
    assert flag.default is False
