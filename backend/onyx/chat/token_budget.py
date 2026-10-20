from dataclasses import dataclass

from onyx.configs.model_configs import GEN_AI_INPUT_TOKEN_SAFETY_MARGIN
from onyx.llm.context_budgets import output_token_reserve
from onyx.llm.interfaces import LLM


@dataclass(frozen=True)
class ChatTokenBudget:
    input_tokens: int
    max_output_tokens: int | None
    context_tokens: int | None
    safety_tokens: int

    def output_allowance(self, estimated_input_tokens: int) -> int | None:
        if self.max_output_tokens is None or self.context_tokens is None:
            return None
        if estimated_input_tokens < 0:
            raise ValueError("estimated_input_tokens must be non-negative")

        available_output_tokens = (
            self.context_tokens - self.safety_tokens - estimated_input_tokens
        )
        if available_output_tokens <= 0:
            return None
        if available_output_tokens < min(
            self.max_output_tokens, output_token_reserve(self.context_tokens)
        ):
            return None

        return min(self.max_output_tokens, available_output_tokens)


def resolve_chat_token_budget(llm: LLM) -> ChatTokenBudget:
    """Budget for one chat cycle, from limits resolved once at construction
    (onyx.llm.context_resolution) and carried on the LLM config.

    Unknown window (config.context_window is None) means no max_tokens is sent
    and the model server fills its own remainder, as before.
    """
    config = llm.config
    raw_input_tokens = max(0, config.max_input_tokens)
    input_tokens = max(
        0, int(raw_input_tokens * (1 - GEN_AI_INPUT_TOKEN_SAFETY_MARGIN))
    )
    safety_tokens = raw_input_tokens - input_tokens
    return ChatTokenBudget(
        input_tokens=input_tokens,
        max_output_tokens=config.max_output_tokens,
        context_tokens=config.context_window,
        safety_tokens=safety_tokens,
    )
