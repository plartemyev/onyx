"""Relative token budgets derived from the selected model's context window.

The only absolute context size in Onyx is the one configured for the selected
model: the admin-set "Max Input Tokens" on its model configuration, resolved
through the GEN_AI_MAX_TOKENS override, LiteLLM model metadata, and the
GEN_AI_MODEL_FALLBACK_MAX_TOKENS fallback. Every other token budget in the
codebase is a fraction of that window, rounded down, so switching to a model
with a different context window rescales every budget automatically.

A `TokenFraction` is an exact rational (numerator / denominator). `scale`
evaluates it with integer math only — truncating a float product can round
the wrong way by one token.

Fractions are calibrated against the hard-coded values they replace: scaling
a 50,000-token window reproduces each former absolute exactly.

This module must stay import-light (config constants only, no DB / litellm).
"""

from onyx.configs.model_configs import GEN_AI_NUM_RESERVED_OUTPUT_TOKENS

# (numerator, denominator); denominator must be positive.
TokenFraction = tuple[int, int]


def scale(window_tokens: int, fraction: TokenFraction) -> int:
    """Floor(window_tokens * numerator / denominator) with exact integer math."""
    numerator, denominator = fraction
    if denominator <= 0:
        raise ValueError(
            f"Token fraction denominator must be positive, got {denominator}"
        )
    if window_tokens <= 0:
        return 0
    return (window_tokens * numerator) // denominator


###########################################################################
# Output reserves
###########################################################################

# Tokens always kept available for the model's answer (was a fixed 1024).
OUTPUT_TOKEN_RESERVE: TokenFraction = (64, 3125)


def output_token_reserve(context_window: int) -> int:
    """Tokens to hold back for the answer; GEN_AI_NUM_RESERVED_OUTPUT_TOKENS
    pins an absolute value when set."""
    if GEN_AI_NUM_RESERVED_OUTPUT_TOKENS is not None:
        return max(1, GEN_AI_NUM_RESERVED_OUTPUT_TOKENS)
    return scale(context_window, OUTPUT_TOKEN_RESERVE)


###########################################################################
# Chat
###########################################################################

# Persona budget: reserved per tool definition (was 256).
TOKENS_RESERVED_PER_TOOL: TokenFraction = (16, 3125)
# Persona budget: reserved for the user's message (was 2000).
TOKENS_RESERVED_FOR_USER_MESSAGE: TokenFraction = (1, 25)
# Chat-session naming: history fed to the naming call (was 3000).
SESSION_NAMING_HISTORY: TokenFraction = (3, 50)
# In-turn compaction digest: LLM summary output cap (was 220).
TOOL_RESPONSE_DIGEST_SUMMARY_OUTPUT: TokenFraction = (11, 2500)


###########################################################################
# Deep Research
###########################################################################

# Final report generation cap (was 20000).
DR_FINAL_REPORT_OUTPUT: TokenFraction = (2, 5)
# Research sub-agent intermediate report cap (was 10000).
DR_INTERMEDIATE_REPORT_OUTPUT: TokenFraction = (1, 5)
# Tool-calling step answer budget (was 1024).
DR_TOOL_CALL_ANSWER_BUDGET: TokenFraction = (64, 3125)
# Extra generation reserve for thinking models (was 3072).
DR_THINKING_RESERVE_BUDGET: TokenFraction = (192, 3125)
# Minimum context window to run Deep Research at all, against the configured
# fallback window GEN_AI_MODEL_FALLBACK_MAX_TOKENS (the former hard floor of
# 50000 is 25/16 of the 32000 default fallback).
DR_MIN_WINDOW_VS_FALLBACK: TokenFraction = (25, 16)


###########################################################################
# Coding agent
###########################################################################

# Final answer generation cap (was 4000).
CODING_AGENT_FINAL_ANSWER_OUTPUT: TokenFraction = (2, 25)
# Per-step tool-calling generation cap (was 2048).
CODING_AGENT_STEP_OUTPUT: TokenFraction = (128, 3125)


###########################################################################
# Secondary LLM flows
###########################################################################

# Keyword expansion output cap (was 150).
KEYWORD_EXPANSION_OUTPUT: TokenFraction = (3, 1000)
# Search-vs-chat classification output cap (was 20).
SEARCH_FLOW_CLASSIFICATION_OUTPUT: TokenFraction = (1, 2500)
# Provider probe / warmup call output cap (was 50).
LLM_PROBE_OUTPUT: TokenFraction = (1, 1000)


###########################################################################
# Contextual RAG
###########################################################################

# Per-chunk context / per-document summary generation cap (was 100).
CONTEXTUAL_RAG_SUMMARY_OUTPUT: TokenFraction = (1, 500)
# Documents at or below this share of the window are embedded in full
# instead of summarized (was 4096).
CONTEXTUAL_RAG_FULL_DOC_INCLUSION: TokenFraction = (256, 3125)


###########################################################################
# Anthropic thinking budgets
###########################################################################

# thinking.budget_tokens per effort tier (was 1024 / 2048 / 4096).
ANTHROPIC_THINKING_BUDGET_LOW: TokenFraction = (64, 3125)
ANTHROPIC_THINKING_BUDGET_MEDIUM: TokenFraction = (128, 3125)
ANTHROPIC_THINKING_BUDGET_HIGH: TokenFraction = (256, 3125)
