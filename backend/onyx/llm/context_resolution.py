"""Single source of truth for a model's context window.

Resolution ladder (first hit wins):

1. DB-stored ``max_input_tokens`` (admin pin) — input budget honored as-is,
   probe skipped. Window/output still come from the LiteLLM map when known.
2. Live probe of the model server's own API (llama.cpp ``/props``,
   vLLM ``/v1/models``), TTL-cached — positive and negative. On probe
   failure the last known value is served until it can be refreshed.
3. LiteLLM model map, including deployment aliases.
4. ``GEN_AI_MODEL_FALLBACK_MAX_TOKENS`` with an unknown window.

The window is the model's TOTAL context (prompt + generation); the input
budget subtracts ``output_token_reserve`` exactly like the static fallback
path does. Downstream budgets (safety margin, per-feature fractions) keep
deriving from ``config.max_input_tokens`` as before.
"""

import threading
import time
from dataclasses import dataclass
from typing import TYPE_CHECKING

import requests

from onyx.configs.model_configs import GEN_AI_MODEL_CONTEXT_PROBE_TTL_SECONDS
from onyx.llm.constants import CONTEXT_PROBE_PROVIDERS
from onyx.llm.context_budgets import output_token_reserve
from onyx.llm.model_capabilities import (
    find_model_obj,
    get_max_input_tokens,
    get_model_map,
    model_identity_names,
)

if TYPE_CHECKING:
    from onyx.server.manage.llm.models import LLMProviderView

_PROBE_TIMEOUT_SECONDS = 3
_CACHE_LOCK = threading.Lock()


@dataclass(frozen=True)
class ModelContext:
    """Resolved context limits for one model.

    ``window`` / ``max_output_tokens`` are None when the total context is
    unknown (admin pin without a known window, or the plain fallback); the
    chat budget then sends no ``max_tokens``, exactly as before.
    """

    input_budget: int
    window: int | None
    max_output_tokens: int | None
    source: str


def _positive_int(value: object) -> int | None:
    if isinstance(value, int) and not isinstance(value, bool) and value > 0:
        return value
    return None


def _parse_props_response(payload: dict) -> int | None:
    """llama.cpp ``GET /props`` → ``default_generation_settings.n_ctx``."""
    settings = payload.get("default_generation_settings")
    if not isinstance(settings, dict):
        return None
    return _positive_int(settings.get("n_ctx"))


def _parse_models_response(payload: dict, model_name: str) -> int | None:
    """vLLM-style ``GET /v1/models`` → ``max_model_len`` of the matching id."""
    entries = payload.get("data")
    if not isinstance(entries, list):
        return None
    for entry in entries:
        if not isinstance(entry, dict):
            continue
        entry_id = entry.get("id")
        if entry_id == model_name or (
            isinstance(entry_id, str)
            and isinstance(model_name, str)
            and model_name.endswith(entry_id)
        ):
            return _positive_int(entry.get("max_model_len"))
    return None


def probe_context_window(
    api_base: str, api_key: str | None, model_name: str
) -> int | None:
    """Ask the model server for its live total context size."""
    headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}
    try:
        response = requests.get(
            f"{api_base.rstrip('/')}/props",
            headers=headers,
            timeout=_PROBE_TIMEOUT_SECONDS,
        )
        if response.ok:
            window = _parse_props_response(response.json())
            if window is not None:
                return window

        response = requests.get(
            f"{api_base.rstrip('/')}/v1/models",
            headers=headers,
            timeout=_PROBE_TIMEOUT_SECONDS,
        )
        if response.ok:
            return _parse_models_response(response.json(), model_name)
    except requests.RequestException:
        pass
    return None


@dataclass
class _CacheEntry:
    # None = negative entry: the probe found nothing / the server was down.
    window: int | None
    fetched_at: float


_probe_cache: dict[tuple[int, str], _CacheEntry] = {}


def _cached_probe_context_window(
    provider_id: int,
    model_name: str,
    api_base: str | None,
    api_key: str | None,
) -> int | None:
    """Probe with a TTL cache; stale entries are refreshed, and a failed
    refresh keeps serving the last known value (stale-while-error)."""
    if not api_base:
        return None

    key = (provider_id, model_name)
    now = time.monotonic()
    with _CACHE_LOCK:
        entry = _probe_cache.get(key)

    if (
        entry is not None
        and now - entry.fetched_at < GEN_AI_MODEL_CONTEXT_PROBE_TTL_SECONDS
    ):
        return entry.window

    window = probe_context_window(
        api_base=api_base, api_key=api_key, model_name=model_name
    )
    if window is None and entry is not None:
        # Refresh failed: keep the stale value and retry after another TTL.
        with _CACHE_LOCK:
            entry.fetched_at = now
        return entry.window

    with _CACHE_LOCK:
        _probe_cache[key] = _CacheEntry(window=window, fetched_at=now)
    return window


def _map_limits(
    model_name: str, model_provider: str, deployment_name: str | None
) -> tuple[int | None, int | None]:
    """(window, max_output_tokens) from the LiteLLM map, honoring deployment
    aliases. Mirrors the lookup that used to live in resolve_chat_token_budget."""
    model_map = get_model_map()
    for identity_name in model_identity_names(model_name, deployment_name):
        model_obj = find_model_obj(model_map, model_provider, identity_name) or {}
        window = _positive_int(model_obj.get("max_context_tokens")) or _positive_int(
            model_obj.get("max_input_tokens")
        )
        max_output = _positive_int(model_obj.get("max_output_tokens"))
        if window is not None and max_output is not None:
            return window, max_output
    return None, None


def _stored_max_input_tokens(
    llm_provider: "LLMProviderView", model_name: str
) -> int | None:
    for model_configuration in llm_provider.model_configurations:
        if model_configuration.name == model_name:
            return model_configuration.max_input_tokens
    return None


def resolve_model_context(
    llm_provider: "LLMProviderView",
    model_name: str,
) -> ModelContext:
    """Resolve one model's context limits through the precedence ladder."""
    model_provider = llm_provider.provider
    deployment_name = llm_provider.deployment_name
    stored_max_input_tokens = _stored_max_input_tokens(llm_provider, model_name)

    if stored_max_input_tokens is not None:
        window, max_output = _map_limits(model_name, model_provider, deployment_name)
        return ModelContext(
            input_budget=stored_max_input_tokens,
            window=window,
            max_output_tokens=max_output,
            source="admin",
        )

    if (
        model_provider in CONTEXT_PROBE_PROVIDERS
        and GEN_AI_MODEL_CONTEXT_PROBE_TTL_SECONDS > 0
    ):
        probed_window = _cached_probe_context_window(
            provider_id=llm_provider.id,
            model_name=model_name,
            api_base=llm_provider.api_base,
            api_key=llm_provider.api_key,
        )
        if probed_window is not None:
            return ModelContext(
                input_budget=probed_window - output_token_reserve(probed_window),
                window=probed_window,
                max_output_tokens=probed_window,
                source="model_api",
            )

    window, max_output = _map_limits(model_name, model_provider, deployment_name)
    return ModelContext(
        input_budget=get_max_input_tokens(
            model_name=model_name, model_provider=model_provider
        ),
        window=window,
        max_output_tokens=max_output,
        source="litellm" if window is not None else "fallback",
    )
