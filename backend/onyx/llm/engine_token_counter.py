"""Token counting via the inference engine's own tokenizer.

LLMs served by llama.cpp (GGUF models behind an OpenAI-compatible endpoint)
expose ``POST /tokenize``; it tokenizes with the exact model vocabulary, which
no local tokenizer can match. The counter below prefers that endpoint and falls
back to Onyx's local tokenizer when the engine does not provide one.
"""

from typing import Callable

import requests

from onyx.llm.factory import get_llm_token_counter
from onyx.llm.interfaces import LLM
from onyx.utils.logger import setup_logger

logger = setup_logger()

# Connect/read timeouts for the tokenize probe. Tokenization is fast; the read
# timeout only guards a wedged server.
_CONNECT_TIMEOUT_SECONDS = 2
_READ_TIMEOUT_SECONDS = 15

# api_base (normalized, without /v1) -> endpoint usability. Negative caching
# keeps a non-llama.cpp provider (e.g. hosted OpenAI) from being probed on
# every open_url call.
_tokenize_availability: dict[str, bool] = {}


def _engine_tokenize_endpoint(api_base: str) -> str | None:
    """llama.cpp serves /tokenize at the server root, while OpenAI-compatible
    api_base values usually end in /v1. Returns None when there is no
    usable http(s) base."""
    base = api_base.rstrip("/")
    base = base.removesuffix("/v1")
    if not base.startswith(("http://", "https://")):
        return None
    return f"{base}/tokenize"


def _count_via_engine(endpoint: str, text: str, headers: dict[str, str]) -> int | None:
    """Returns the token count, or None when the endpoint is unusable."""
    try:
        response = requests.post(
            endpoint,
            json={"content": text},
            headers=headers,
            timeout=(_CONNECT_TIMEOUT_SECONDS, _READ_TIMEOUT_SECONDS),
        )
    except requests.RequestException:
        return None
    if response.status_code != 200:
        return None
    tokens = response.json().get("tokens")
    if not isinstance(tokens, list):
        return None
    return len(tokens)


def get_engine_token_counter(llm: LLM) -> Callable[[str], int]:
    """Builds a token counter that prefers the serving engine's tokenizer.

    Falls back to the standard Onyx tokenizer for the LLM when there is no
    api_base or the engine does not answer /tokenize.
    """
    fallback_counter = get_llm_token_counter(llm)
    api_base = llm.config.api_base
    if not api_base:
        return fallback_counter

    endpoint = _engine_tokenize_endpoint(api_base)
    if endpoint is None:
        return fallback_counter

    # llama.cpp servers commonly sit behind an API key check even though the
    # key itself is unused for local models.
    headers: dict[str, str] = {}
    if llm.config.api_key:
        headers["Authorization"] = f"Bearer {llm.config.api_key}"

    def counter(text: str) -> int:
        if _tokenize_availability.get(endpoint) is False:
            return fallback_counter(text)
        count = _count_via_engine(endpoint, text, headers)
        if count is not None:
            if not _tokenize_availability.get(endpoint):
                logger.info("Using engine tokenizer at %s", endpoint)
            _tokenize_availability[endpoint] = True
            return count
        if _tokenize_availability.get(endpoint):
            # Engine went away mid-session; stop probing it.
            _tokenize_availability[endpoint] = False
        else:
            logger.info(
                "No engine tokenizer at %s; falling back to local tokenizer",
                endpoint,
            )
            _tokenize_availability[endpoint] = False
        return fallback_counter(text)

    return counter
