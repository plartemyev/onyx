from unittest.mock import Mock

import pytest

from onyx.chat.token_budget import ChatTokenBudget, resolve_chat_token_budget
from onyx.llm.interfaces import LLMConfig


def _llm(
    max_input_tokens: int,
    context_window: int | None = None,
    max_output_tokens: int | None = None,
) -> Mock:
    return Mock(
        config=LLMConfig(
            model_provider="openai",
            model_name="model",
            temperature=0,
            max_input_tokens=max_input_tokens,
            context_window=context_window,
            max_output_tokens=max_output_tokens,
        )
    )


@pytest.mark.parametrize(
    "max_input_tokens,margin,expected_input,expected_safety",
    [
        (100_000, 0.05, 95_000, 5_000),
        (100_000, 0.10, 90_000, 10_000),
        (31_345, 0.10, 28_210, 3_135),
        (1_000_000, 0.0, 1_000_000, 0),
    ],
    ids=["default-margin", "deploy-margin", "probed-window", "no-margin"],
)
def test_input_budget_applies_safety_margin(
    monkeypatch: pytest.MonkeyPatch,
    max_input_tokens: int,
    margin: float,
    expected_input: int,
    expected_safety: int,
) -> None:
    monkeypatch.setattr(
        "onyx.chat.token_budget.GEN_AI_INPUT_TOKEN_SAFETY_MARGIN", margin
    )
    budget = resolve_chat_token_budget(_llm(max_input_tokens))
    assert (budget.input_tokens, budget.safety_tokens) == (
        expected_input,
        expected_safety,
    )


def test_resolved_context_drives_output_allowance(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr("onyx.chat.token_budget.GEN_AI_INPUT_TOKEN_SAFETY_MARGIN", 0.05)
    monkeypatch.setattr(
        "onyx.llm.context_budgets.GEN_AI_NUM_RESERVED_OUTPUT_TOKENS", 1024
    )
    budget = resolve_chat_token_budget(_llm(100_000, 128_000, 16_000))
    assert budget == ChatTokenBudget(95_000, 16_000, 128_000, 5_000)
    # min(max_output, context - safety - estimated_input)
    assert budget.output_allowance(95_000) == 16_000
    assert budget.output_allowance(120_000) == 3_000
    # Below the output reserve the answer would be useless — no cap instead.
    assert budget.output_allowance(121_977) is None


def test_probed_window_caps_output_without_explicit_max_output(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # resolve_model_context returns max_output_tokens=window for probed
    # self-hosted models: fill the remainder, like n_predict=-1 but with a
    # clean stop.
    monkeypatch.setattr("onyx.chat.token_budget.GEN_AI_INPUT_TOKEN_SAFETY_MARGIN", 0.10)
    monkeypatch.setattr(
        "onyx.llm.context_budgets.GEN_AI_NUM_RESERVED_OUTPUT_TOKENS", None
    )
    budget = resolve_chat_token_budget(_llm(16_049, 16_384, 16_384))
    assert budget == ChatTokenBudget(14_444, 16_384, 16_384, 1_605)
    assert budget.output_allowance(0) == 14_779


def test_unknown_window_keeps_legacy_no_cap(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr("onyx.chat.token_budget.GEN_AI_INPUT_TOKEN_SAFETY_MARGIN", 0.05)
    budget = resolve_chat_token_budget(_llm(100_000))
    assert budget == ChatTokenBudget(95_000, None, None, 5_000)
    assert budget.output_allowance(0) is None


@pytest.mark.parametrize("estimated_input_tokens", [95, 100, 200])
def test_exhausted_context_yields_no_allowance(
    estimated_input_tokens: int,
) -> None:
    assert (
        ChatTokenBudget(95, 10, 100, 5).output_allowance(estimated_input_tokens) is None
    )


def test_negative_estimated_input_is_rejected() -> None:
    with pytest.raises(ValueError):
        ChatTokenBudget(95, 10, 100, 5).output_allowance(-1)
