import contextlib

import pytest
import requests

from onyx.configs import chat_configs
from onyx.configs.chat_configs import dr_is_thinking_model
from onyx.llm import engine_capabilities, model_capabilities
from onyx.llm.engine_capabilities import (
    ollama_model_capabilities,
    resolve_ollama_api_base,
)

API_BASE = "http://ollama-host:11434/"  # trailing slash must be normalized
MODEL = "ornith-1.5:9b"


class _FakeResponse:
    def __init__(self, payload: dict) -> None:
        self._payload = payload

    def json(self) -> dict:
        return self._payload


@pytest.fixture(autouse=True)
def _clear_caches():
    engine_capabilities._CACHE.clear()
    engine_capabilities._DB_BASE_CACHE = None
    yield
    engine_capabilities._CACHE.clear()
    engine_capabilities._DB_BASE_CACHE = None


# ---------------------------------------------------------------------------
# ollama_model_capabilities (the engine probe)
# ---------------------------------------------------------------------------


def test_parses_capabilities_from_engine(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[dict] = []

    def fake_post(url: str, json: dict, timeout: float) -> _FakeResponse:  # noqa: ARG001
        calls.append({"url": url, "json": json})
        return _FakeResponse({"capabilities": ["tools", "thinking", "completion"]})

    monkeypatch.setattr(engine_capabilities.requests, "post", fake_post)
    capabilities = ollama_model_capabilities(API_BASE, MODEL)
    assert capabilities == ["tools", "thinking", "completion"]
    assert calls[0]["url"] == "http://ollama-host:11434/api/show"
    assert calls[0]["json"] == {"model": MODEL}


def test_missing_model_returns_none(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        engine_capabilities.requests,
        "post",
        lambda url, json, timeout: _FakeResponse(  # noqa: ARG005
            {"error": "model not found"}
        ),
    )
    assert ollama_model_capabilities(API_BASE, "not-installed:1b") is None


def test_connection_error_returns_none(monkeypatch: pytest.MonkeyPatch) -> None:
    def fake_post(url: str, json: dict, timeout: float) -> _FakeResponse:  # noqa: ARG001
        raise requests.ConnectionError("engine down")

    monkeypatch.setattr(engine_capabilities.requests, "post", fake_post)
    assert ollama_model_capabilities(API_BASE, MODEL) is None


def test_non_json_body_returns_none(monkeypatch: pytest.MonkeyPatch) -> None:
    class _BadResponse:
        def json(self) -> dict:
            raise ValueError("not json")

    monkeypatch.setattr(
        engine_capabilities.requests,
        "post",
        lambda url, json, timeout: _BadResponse(),  # noqa: ARG005
    )
    assert ollama_model_capabilities(API_BASE, MODEL) is None


def test_results_are_cached(monkeypatch: pytest.MonkeyPatch) -> None:
    post_calls: list[int] = []

    def fake_post(url: str, json: dict, timeout: float) -> _FakeResponse:  # noqa: ARG001
        post_calls.append(1)
        return _FakeResponse({"capabilities": ["completion"]})

    monkeypatch.setattr(engine_capabilities.requests, "post", fake_post)
    assert ollama_model_capabilities(API_BASE, MODEL) == ["completion"]
    assert ollama_model_capabilities(API_BASE, MODEL) == ["completion"]
    assert len(post_calls) == 1  # second call served from cache


def test_failures_are_cached_briefly(monkeypatch: pytest.MonkeyPatch) -> None:
    post_calls: list[int] = []

    def fake_post(url: str, json: dict, timeout: float) -> _FakeResponse:  # noqa: ARG001
        post_calls.append(1)
        raise requests.ConnectionError("down")

    monkeypatch.setattr(engine_capabilities.requests, "post", fake_post)
    assert ollama_model_capabilities(API_BASE, MODEL) is None
    assert ollama_model_capabilities(API_BASE, MODEL) is None
    assert len(post_calls) == 1


# ---------------------------------------------------------------------------
# resolve_ollama_api_base
# ---------------------------------------------------------------------------


def test_resolve_explicit_base_wins(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OLLAMA_API_BASE", "http://from-env:11434")
    assert (
        resolve_ollama_api_base(explicit="http://explicit:11434")
        == "http://explicit:11434"
    )


def test_resolve_env_fallback(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OLLAMA_API_BASE", "http://from-env:11434")
    assert resolve_ollama_api_base() == "http://from-env:11434"


def test_resolve_db_fallback(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("OLLAMA_API_BASE", raising=False)
    db_calls: list[int] = []

    @contextlib.contextmanager
    def fake_session():
        yield object()

    def fake_fetch(_db_session: object) -> str:
        db_calls.append(1)
        return "http://from-db:11434"

    monkeypatch.setattr(
        "onyx.db.engine.sql_engine.get_session_with_current_tenant", fake_session
    )
    monkeypatch.setattr("onyx.db.llm.fetch_ollama_llm_provider_api_base", fake_fetch)
    assert resolve_ollama_api_base() == "http://from-db:11434"
    assert resolve_ollama_api_base() == "http://from-db:11434"
    assert len(db_calls) == 1  # DB path cached


def test_resolve_db_failure_returns_none(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("OLLAMA_API_BASE", raising=False)

    def failing_fetch(_db_session: object) -> str:
        raise RuntimeError("db down")

    monkeypatch.setattr("onyx.db.llm.fetch_ollama_llm_provider_api_base", failing_fetch)
    assert resolve_ollama_api_base() is None


# ---------------------------------------------------------------------------
# model_is_reasoning_model (system-wide engine integration)
# ---------------------------------------------------------------------------


def _patch_engine(
    monkeypatch: pytest.MonkeyPatch, capabilities: list[str] | None
) -> list[str]:
    """Point the engine probe at a canned response; returns probe targets."""
    targets: list[str] = []

    def fake_probe(_api_base: str, model_name: str, _timeout_s: float = 5.0):
        targets.append(model_name)
        return capabilities

    monkeypatch.setattr(engine_capabilities, "ollama_model_capabilities", fake_probe)
    return targets


def test_reasoning_model_engine_thinking_wins(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _patch_engine(monkeypatch, ["tools", "thinking", "completion"])
    assert model_capabilities.model_is_reasoning_model(
        MODEL, "ollama_chat", api_base=API_BASE
    )


def test_reasoning_model_engine_negative_wins(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _patch_engine(monkeypatch, ["completion", "tools"])
    assert not model_capabilities.model_is_reasoning_model(
        "mistral:7b", "ollama_chat", api_base=API_BASE
    )


def test_reasoning_model_engine_unreachable_falls_back_to_static(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _patch_engine(monkeypatch, None)  # engine cannot answer
    # Static registries empty; the litellm probe is the last authority.
    monkeypatch.setattr(model_capabilities, "get_model_map", dict)
    monkeypatch.setattr(
        model_capabilities, "_litellm_supports_reasoning", lambda _name: True
    )
    assert model_capabilities.model_is_reasoning_model(
        MODEL, "ollama_chat", api_base=API_BASE
    )


def test_reasoning_model_resolves_base_from_db_fallback(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Call sites without an LLM config still reach the engine via the
    DB/env resolution."""
    targets = _patch_engine(monkeypatch, ["thinking"])
    monkeypatch.setattr(
        engine_capabilities, "resolve_ollama_api_base", lambda _explicit=None: API_BASE
    )
    assert model_capabilities.model_is_reasoning_model(MODEL, "ollama_chat")
    assert targets == [MODEL]


def test_reasoning_model_non_ollama_not_probed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fail_probe(_api_base: str, _model_name: str, _timeout_s: float = 5.0):
        raise AssertionError("must not probe non-ollama providers")

    monkeypatch.setattr(engine_capabilities, "ollama_model_capabilities", fail_probe)
    monkeypatch.setattr(model_capabilities, "get_model_map", dict)
    monkeypatch.setattr(
        model_capabilities, "_litellm_supports_reasoning", lambda _name: False
    )
    assert not model_capabilities.model_is_reasoning_model(
        "gpt-4o", "openai", api_base="https://api.openai.com/v1"
    )


# ---------------------------------------------------------------------------
# dr_is_thinking_model (DR-layer fallbacks on top of the system-wide flag)
# ---------------------------------------------------------------------------


def test_dr_thinking_flag_short_circuits() -> None:
    assert dr_is_thinking_model(MODEL, True)


def test_dr_thinking_name_markers() -> None:
    assert dr_is_thinking_model("qwen3:30b-thinking", False)
    assert dr_is_thinking_model("qwen3-reasoning:8b", False)
    assert not dr_is_thinking_model("mistral:7b", False)


def test_dr_thinking_env_overrides(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(chat_configs, "DR_THINKING_MODEL_OVERRIDES", ("ornith*",))
    assert dr_is_thinking_model(MODEL, False)
    assert not dr_is_thinking_model("mistral:7b", False)
