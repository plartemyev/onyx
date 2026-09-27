import pytest

from onyx.chat.llm_step import run_llm_step
from onyx.configs.chat_configs import (
    DR_THINKING_TOKEN_RESERVE,
    DR_TOOL_CALL_ANSWER_TOKENS,
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
    assert dr_tool_call_max_tokens(is_reasoning_model=False) == 1024


def test_dr_tool_call_max_tokens_reasoning_model_gets_reserve(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("DR_TOOL_CALL_ANSWER_TOKENS", raising=False)
    monkeypatch.delenv("DR_THINKING_TOKEN_RESERVE", raising=False)
    assert dr_tool_call_max_tokens(is_reasoning_model=True) == 1024 + 3072


def test_dr_tool_call_max_tokens_honors_config(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The helper reads the module-level constants, so ops can retune them
    via env (bound at import time)."""
    from onyx.configs import chat_configs

    monkeypatch.setattr(chat_configs, "DR_TOOL_CALL_ANSWER_TOKENS", 2048)
    monkeypatch.setattr(chat_configs, "DR_THINKING_TOKEN_RESERVE", 512)
    assert dr_tool_call_max_tokens(is_reasoning_model=False) == 2048
    assert dr_tool_call_max_tokens(is_reasoning_model=True) == 2560


def test_defaults_unchanged() -> None:
    """Guard against accidental default drift: the answer budget stays the
    historical 1024; the thinking reserve is the new 3072."""
    assert DR_TOOL_CALL_ANSWER_TOKENS == 1024
    assert DR_THINKING_TOKEN_RESERVE == 3072


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
