"""Query inference engines for per-model capabilities at runtime.

Static capability registries cannot know about custom pulled local models — a
custom Ollama model has no litellm entry, so registry-based checks answer
False for models that think natively. Ollama and similar engines expose
per-model capability data over their API; this module queries and caches it so
capability checks can trust the engine itself.

Fail-safe by design: when the engine cannot answer, callers fall back to
their static/heuristic logic.
"""

import os
import threading
import time

import requests

from onyx.utils.logger import setup_logger

logger = setup_logger()

# A pulled model's capabilities only change when the model is re-pulled, so
# successes are held for a long TTL; failures (engine briefly down, model not
# present) are retried after a short TTL so a transient outage does not wedge
# every later step into the fallback path.
_SUCCESS_TTL_S = 3600.0
_FAILURE_TTL_S = 60.0

_LOCK = threading.Lock()
# (api_base, model_name) -> (read_time, capabilities or None)
_CACHE: dict[tuple[str, str], tuple[float, list[str] | None]] = {}


def ollama_model_capabilities(
    api_base: str, model_name: str, timeout_s: float = 5.0
) -> list[str] | None:
    """Capabilities for a model, per the Ollama server's /api/show endpoint.

    Returns e.g. ["completion", "tools", "thinking"], or None when the engine
    cannot answer (down, model missing, unexpected response shape). Results
    are cached per (api_base, model).
    """
    cache_key = (api_base.rstrip("/"), model_name)
    now = time.monotonic()
    with _LOCK:
        cached = _CACHE.get(cache_key)
        if cached is not None:
            read_time, capabilities = cached
            ttl = _SUCCESS_TTL_S if capabilities is not None else _FAILURE_TTL_S
            if now - read_time < ttl:
                return capabilities

    capabilities = _fetch_ollama_capabilities(api_base, model_name, timeout_s)
    with _LOCK:
        _CACHE[cache_key] = (now, capabilities)
    if capabilities is not None:
        logger.info("Ollama reports capabilities for %s: %s", model_name, capabilities)
    return capabilities


def _fetch_ollama_capabilities(
    api_base: str, model_name: str, timeout_s: float
) -> list[str] | None:
    try:
        response = requests.post(
            f"{api_base.rstrip('/')}/api/show",
            json={"model": model_name},
            timeout=timeout_s,
        )
        payload = response.json()
    except (requests.RequestException, ValueError):
        # Connection error, timeout, or a non-JSON body.
        return None
    capabilities = payload.get("capabilities")
    if isinstance(capabilities, list) and all(
        isinstance(item, str) for item in capabilities
    ):
        return capabilities
    # Missing/invalid field: model not present, or an engine whose /api/show
    # does not report capabilities.
    return None


_DB_BASE_TTL_S = 300.0
_DB_BASE_LOCK = threading.Lock()
_DB_BASE_CACHE: tuple[float, str | None] | None = None


def resolve_ollama_api_base(explicit: str | None = None) -> str | None:
    """Best-effort Ollama base URL for capability probes.

    Order: an explicitly provided base (from the LLM config at hand) wins,
    then the OLLAMA_API_BASE environment variable, then the first configured
    Ollama LLM provider in the DB (cached briefly — this path serves call
    sites that only know provider/model names).
    """
    if explicit:
        return explicit

    env_base = os.environ.get("OLLAMA_API_BASE")
    if env_base:
        return env_base

    global _DB_BASE_CACHE
    now = time.monotonic()
    with _DB_BASE_LOCK:
        if _DB_BASE_CACHE is not None and now - _DB_BASE_CACHE[0] < _DB_BASE_TTL_S:
            return _DB_BASE_CACHE[1]

    api_base: str | None = None
    try:
        # Lazy imports: this module is imported from config/capability paths
        # that must not pull the DB stack (and db/llm imports this package).
        from onyx.db.engine.sql_engine import get_session_with_current_tenant
        from onyx.db.llm import fetch_ollama_llm_provider_api_base

        with get_session_with_current_tenant() as db_session:
            api_base = fetch_ollama_llm_provider_api_base(db_session)
    except Exception:
        logger.exception("Failed to resolve the Ollama api_base for capability probes")
        api_base = None

    with _DB_BASE_LOCK:
        _DB_BASE_CACHE = (now, api_base)
    return api_base
