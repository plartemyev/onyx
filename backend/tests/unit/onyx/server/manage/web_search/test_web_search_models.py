import pytest
from pydantic import ValidationError

from onyx.server.manage.web_search.models import (
    WebSearchProviderUpsertRequest,
    WebSearchProviderView,
)
from shared_configs.enums import WebSearchProviderType


def _view_config(config: dict[str, object] | None) -> dict[str, str] | None:
    view = WebSearchProviderView(
        id=1,
        name="SearXNG",
        provider_type=WebSearchProviderType.SEARXNG,
        is_active=True,
        config=config,
    )
    return view.config


def test_view_config_accepts_json_boolean() -> None:
    # Rows edited outside the API (e.g. SQL) may store JSON booleans.
    config = _view_config({"searxng_base_url": "http://searxng:8080", "flag": False})
    assert config == {"searxng_base_url": "http://searxng:8080", "flag": "false"}


def test_view_config_accepts_json_boolean_true_and_numbers() -> None:
    config = _view_config({"a": True, "b": 3, "c": 1.5})
    assert config == {"a": "true", "b": "3", "c": "1.5"}


def test_view_config_keeps_strings_and_none() -> None:
    assert _view_config(None) is None
    assert _view_config({"k": "v"}) == {"k": "v"}


def test_upsert_request_accepts_json_boolean() -> None:
    request = WebSearchProviderUpsertRequest(
        name="SearXNG",
        provider_type=WebSearchProviderType.SEARXNG,
        config={
            "searxng_base_url": "http://searxng:8080",
            "enable_image_search": False,
        },
    )
    assert request.config is not None
    assert request.config["enable_image_search"] == "false"


def test_view_config_still_rejects_nested_structures() -> None:
    with pytest.raises(ValidationError):
        _view_config({"k": ["list"]})
