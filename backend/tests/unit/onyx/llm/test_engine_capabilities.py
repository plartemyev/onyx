import pytest
import requests

from onyx.configs import chat_configs
from onyx.configs.chat_configs import dr_is_thinking_model
from onyx.llm import engine_capabilities
from onyx.llm.engine_capabilities import ollama_model_capabilities

API_BASE = "http://ollama-host:11434/"  # trailing slash must be normalized
MODEL = "ornith-1.5:9b"


class _FakeResponse:
    def __init__(self, payload: dict) -> None:
        self._payload = payload

    def json(self) -> dict:
        return self._payload


@pytest.fixture(autouse=True)
def _clear_cache():
    engine_capabilities._CACHE.clear()
    yield
    engine_capabilities._CACHE.clear()


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


def test_reasoning_flag_short_circuits_without_probe(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fail_post(url: str, json: dict, timeout: float) -> _FakeResponse:  # noqa: ARG001
        raise AssertionError("must not probe")

    monkeypatch.setattr(engine_capabilities.requests, "post", fail_post)
    assert dr_is_thinking_model(
        MODEL, True, api_base=API_BASE, model_provider="ollama_chat"
    )


def test_engine_thinking_capability_wins(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        engine_capabilities.requests,
        "post",
        lambda url, json, timeout: _FakeResponse(  # noqa: ARG005
            {"capabilities": ["tools", "thinking", "completion"]}
        ),
    )
    assert dr_is_thinking_model(
        MODEL, False, api_base=API_BASE, model_provider="ollama_chat"
    )


def test_engine_negative_is_authoritative(monkeypatch: pytest.MonkeyPatch) -> None:
    """A model the engine says cannot think stays False even if its name
    contains a thinking marker."""
    monkeypatch.setattr(
        engine_capabilities.requests,
        "post",
        lambda url, json, timeout: _FakeResponse(  # noqa: ARG005
            {"capabilities": ["completion"]}
        ),
    )
    assert not dr_is_thinking_model(
        "qwen3:30b-thinking", False, api_base=API_BASE, model_provider="ollama_chat"
    )


def test_falls_back_to_name_markers_without_engine(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fail_post(url: str, json: dict, timeout: float) -> _FakeResponse:  # noqa: ARG001
        raise requests.ConnectionError()

    monkeypatch.setattr(engine_capabilities.requests, "post", fail_post)
    assert dr_is_thinking_model("qwen3:30b-thinking", False, api_base=API_BASE)
    assert not dr_is_thinking_model("mistral:7b", False, api_base=API_BASE)


def test_falls_back_to_env_overrides(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(chat_configs, "DR_THINKING_MODEL_OVERRIDES", ("ornith*",))
    assert dr_is_thinking_model(MODEL, False)  # no api_base: no probe
    assert not dr_is_thinking_model("mistral:7b", False)


def test_non_ollama_providers_are_not_probed(monkeypatch: pytest.MonkeyPatch) -> None:
    def fail_post(url: str, json: dict, timeout: float) -> _FakeResponse:  # noqa: ARG001
        raise AssertionError("must not probe non-ollama providers")

    monkeypatch.setattr(engine_capabilities.requests, "post", fail_post)
    assert not dr_is_thinking_model(
        "gpt-4o", False, api_base="https://api.openai.com/v1", model_provider="openai"
    )
