from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from onyx.llm.constants import LlmProviderNames
from onyx.llm.context_resolution import (
    ModelContext,
    _parse_models_response,
    _parse_props_response,
    _probe_cache,
    resolve_model_context,
)
from onyx.server.manage.llm.models import LLMProviderView, ModelConfigurationView

MODEL_NAME = "/models/Qwen3.6-35B-A3B-UD-Q4_K_XL.gguf"


def _provider(
    provider: str = LlmProviderNames.OPENAI_COMPATIBLE,
    stored: int | None = None,
    api_key: str | None = "test-key",
    provider_id: int = 1,
) -> LLMProviderView:
    return LLMProviderView(
        id=provider_id,
        name="test-provider",
        provider=provider,
        model_configurations=[
            ModelConfigurationView(
                name=MODEL_NAME,
                is_visible=True,
                max_input_tokens=stored,
                supports_image_input=False,
            )
        ],
        api_key=api_key,
        api_base="http://gpu-host:8011",
        api_version=None,
        custom_config=None,
        is_public=True,
        is_auto_mode=False,
        groups=[],
        personas=[],
        deployment_name=None,
    )


@pytest.fixture(autouse=True)
def clear_probe_cache() -> None:
    _probe_cache.clear()


@pytest.fixture(autouse=True)
def frozen_reserve(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        "onyx.llm.context_budgets.GEN_AI_NUM_RESERVED_OUTPUT_TOKENS", 1024
    )


@pytest.fixture
def model_map(monkeypatch: pytest.MonkeyPatch) -> dict:
    models: dict = {}
    monkeypatch.setattr("onyx.llm.context_resolution.get_model_map", lambda: models)
    monkeypatch.setattr("onyx.llm.model_capabilities.get_model_map", lambda: models)
    return models


@pytest.fixture
def probe(monkeypatch: pytest.MonkeyPatch) -> Mock:
    mock = Mock(return_value=None, name="probe_context_window")
    monkeypatch.setattr("onyx.llm.context_resolution.probe_context_window", mock)
    return mock


def test_admin_pin_wins_and_skips_probe(model_map: dict, probe: Mock) -> None:
    model_map["openai_compatible" + "/" + MODEL_NAME] = {
        "max_input_tokens": 200_000,
        "max_output_tokens": 8_000,
    }
    context = resolve_model_context(_provider(stored=16_384), MODEL_NAME)

    probe.assert_not_called()
    assert context == ModelContext(
        input_budget=16_384, window=200_000, max_output_tokens=8_000, source="admin"
    )


@pytest.mark.usefixtures("model_map")
def test_admin_pin_with_unknown_map_window_keeps_no_cap(
    probe: Mock,
) -> None:
    context = resolve_model_context(_provider(stored=16_384), MODEL_NAME)

    probe.assert_not_called()
    assert context == ModelContext(
        input_budget=16_384, window=None, max_output_tokens=None, source="admin"
    )


def test_probe_hit_beats_map(model_map: dict, probe: Mock) -> None:
    model_map["openai_compatible" + "/" + MODEL_NAME] = {
        "max_input_tokens": 200_000,
        "max_output_tokens": 8_000,
    }
    probe.return_value = 16_384

    context = resolve_model_context(_provider(), MODEL_NAME)

    # n_ctx is the total window: reserve the output share from the input side.
    assert context == ModelContext(
        input_budget=16_384 - 1024,
        window=16_384,
        max_output_tokens=16_384,
        source="model_api",
    )
    probe.assert_called_once_with(
        api_base="http://gpu-host:8011", api_key="test-key", model_name=MODEL_NAME
    )


def test_probe_skipped_for_non_probe_provider(model_map: dict, probe: Mock) -> None:
    model_map["openai" + "/model"] = {
        "max_input_tokens": 128_000,
        "max_output_tokens": 16_000,
    }

    context = resolve_model_context(
        _provider(provider=LlmProviderNames.OPENAI, api_key="k"), "model"
    )

    probe.assert_not_called()
    assert context == ModelContext(
        input_budget=128_000 - 1024,
        window=128_000,
        max_output_tokens=16_000,
        source="litellm",
    )


@pytest.mark.usefixtures("model_map")
def test_fallback_when_probe_fails_and_map_misses(probe: Mock) -> None:
    probe.return_value = None

    context = resolve_model_context(_provider(), MODEL_NAME)

    # Unknown window → no cap; input = 32000 fallback minus the output reserve.
    assert context == ModelContext(
        input_budget=32_000 - 1024,
        window=None,
        max_output_tokens=None,
        source="fallback",
    )


def test_deployment_alias_resolves_through_map(model_map: dict, probe: Mock) -> None:
    provider = _provider(provider=LlmProviderNames.AZURE)
    provider.deployment_name = "alias"
    model_map["azure/alias"] = {
        "max_input_tokens": 128_000,
        "max_output_tokens": 16_000,
    }

    context = resolve_model_context(provider, "model")

    probe.assert_not_called()
    # Legacy quirk preserved: the input budget comes from the non-alias
    # lookup (32000 fallback), while window/output honor the alias.
    assert context == ModelContext(
        input_budget=32_000 - 1024,
        window=128_000,
        max_output_tokens=16_000,
        source="litellm",
    )


def _freeze_time(monkeypatch: pytest.MonkeyPatch, start: float) -> list[float]:
    clock = [start]
    monkeypatch.setattr(
        "onyx.llm.context_resolution.time",
        SimpleNamespace(monotonic=lambda: clock[0]),
    )
    return clock


@pytest.mark.usefixtures("model_map")
def test_cache_serves_within_ttl_and_refreshes_after(
    probe: Mock, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        "onyx.llm.context_resolution.GEN_AI_MODEL_CONTEXT_PROBE_TTL_SECONDS", 300
    )
    clock = _freeze_time(monkeypatch, 1_000.0)
    probe.side_effect = [16_384, 131_072]

    provider = _provider()
    first = resolve_model_context(provider, MODEL_NAME)
    clock[0] += 299
    cached = resolve_model_context(provider, MODEL_NAME)
    clock[0] += 2  # past the TTL now
    refreshed = resolve_model_context(provider, MODEL_NAME)

    assert (first.window, cached.window, refreshed.window) == (
        16_384,
        16_384,
        131_072,
    )
    assert probe.call_count == 2


@pytest.mark.usefixtures("model_map")
def test_cache_follows_context_lowering(
    probe: Mock, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        "onyx.llm.context_resolution.GEN_AI_MODEL_CONTEXT_PROBE_TTL_SECONDS", 300
    )
    clock = _freeze_time(monkeypatch, 1_000.0)
    probe.side_effect = [131_072, 16_384]

    provider = _provider()
    raised = resolve_model_context(provider, MODEL_NAME)
    clock[0] += 301
    lowered = resolve_model_context(provider, MODEL_NAME)

    assert (raised.window, lowered.window) == (131_072, 16_384)
    assert lowered.input_budget == 16_384 - 1024


@pytest.mark.usefixtures("model_map")
def test_stale_value_served_when_refresh_fails(
    probe: Mock, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        "onyx.llm.context_resolution.GEN_AI_MODEL_CONTEXT_PROBE_TTL_SECONDS", 300
    )
    clock = _freeze_time(monkeypatch, 1_000.0)
    probe.side_effect = [16_384, None, None]

    provider = _provider()
    assert resolve_model_context(provider, MODEL_NAME).window == 16_384
    clock[0] += 301
    # Server stopped answering: last known good survives the refresh attempt.
    assert resolve_model_context(provider, MODEL_NAME).window == 16_384
    clock[0] += 299
    # Refreshed timestamp means no new probe attempt within the new TTL.
    assert resolve_model_context(provider, MODEL_NAME).window == 16_384
    assert probe.call_count == 2


@pytest.mark.usefixtures("model_map")
def test_negative_result_cached_within_ttl(
    probe: Mock, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        "onyx.llm.context_resolution.GEN_AI_MODEL_CONTEXT_PROBE_TTL_SECONDS", 300
    )
    clock = _freeze_time(monkeypatch, 1_000.0)
    provider = _provider()

    assert resolve_model_context(provider, MODEL_NAME).window is None
    clock[0] += 100
    assert resolve_model_context(provider, MODEL_NAME).window is None
    assert probe.call_count == 1


@pytest.mark.usefixtures("model_map")
def test_ttl_zero_disables_probing(
    probe: Mock, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        "onyx.llm.context_resolution.GEN_AI_MODEL_CONTEXT_PROBE_TTL_SECONDS", 0
    )

    assert resolve_model_context(_provider(), MODEL_NAME).source == "fallback"
    probe.assert_not_called()


@pytest.mark.usefixtures("model_map")
def test_missing_api_base_skips_probing(probe: Mock) -> None:
    provider = _provider()
    provider.api_base = None

    assert resolve_model_context(provider, MODEL_NAME).source == "fallback"
    probe.assert_not_called()


def test_parse_props_response() -> None:
    assert (
        _parse_props_response({"default_generation_settings": {"n_ctx": 16_384}})
        == 16_384
    )
    assert _parse_props_response({"default_generation_settings": None}) is None
    assert _parse_props_response({"default_generation_settings": {"n_ctx": 0}}) is None
    assert _parse_props_response({}) is None


def test_parse_models_response() -> None:
    payload = {
        "data": [
            {"id": "other-model", "max_model_len": 4_096},
            {"id": "Qwen3.6-35B", "max_model_len": 131_072},
        ]
    }
    assert _parse_models_response(payload, "Qwen3.6-35B") == 131_072
    assert _parse_models_response(payload, "missing") is None
    assert _parse_models_response({"data": None}, "Qwen3.6-35B") is None
