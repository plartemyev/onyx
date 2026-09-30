import pytest
from typing_extensions import cast

from onyx.tools.tool_implementations.web_search.clients.brave_client import BraveClient
from onyx.tools.tool_implementations.web_search.clients.searxng_client import (
    SearXNGClient,
)
from onyx.tools.tool_implementations.web_search.providers import (
    build_search_provider_from_config,
    provider_requires_api_key,
)
from shared_configs.enums import WebSearchProviderType


def test_provider_requires_api_key() -> None:
    """Test that provider_requires_api_key correctly identifies which providers need API keys."""
    assert provider_requires_api_key(WebSearchProviderType.EXA) is True
    assert provider_requires_api_key(WebSearchProviderType.BRAVE) is True
    assert provider_requires_api_key(WebSearchProviderType.SERPER) is True
    assert provider_requires_api_key(WebSearchProviderType.GOOGLE_PSE) is True
    assert provider_requires_api_key(WebSearchProviderType.SEARXNG) is False


def test_build_searxng_provider_without_api_key() -> None:
    """Test that SearXNG provider can be built without an API key."""
    provider = build_search_provider_from_config(
        provider_type=WebSearchProviderType.SEARXNG,
        api_key=None,
        config={"searxng_base_url": "http://localhost:8080"},
    )
    assert provider is not None


def test_build_searxng_image_search_enabled_by_default() -> None:
    provider = build_search_provider_from_config(
        provider_type=WebSearchProviderType.SEARXNG,
        api_key=None,
        config={"searxng_base_url": "http://localhost:8080"},
    )
    assert isinstance(provider, SearXNGClient)
    assert provider._enable_image_search is True  # noqa: SLF001


@pytest.mark.parametrize("raw_value", ["false", "False", "0", "no", "off", False])
def test_build_searxng_image_search_disabled(raw_value: str | bool) -> None:
    """String spellings and real booleans both disable the image fan-out.

    The provider config column is JSON, so real booleans reach this code
    too; the dict[str, str] annotation on the builder is a standing TODO.
    """
    provider = build_search_provider_from_config(
        provider_type=WebSearchProviderType.SEARXNG,
        api_key=None,
        config=cast(
            dict[str, str],
            {
                "searxng_base_url": "http://localhost:8080",
                "enable_image_search": raw_value,
            },
        ),
    )
    assert isinstance(provider, SearXNGClient)
    assert provider._enable_image_search is False  # noqa: SLF001


@pytest.mark.parametrize("raw_value", ["true", "1", "yes", True])
def test_build_searxng_image_search_enabled(raw_value: str | bool) -> None:
    provider = build_search_provider_from_config(
        provider_type=WebSearchProviderType.SEARXNG,
        api_key=None,
        config=cast(
            dict[str, str],
            {
                "searxng_base_url": "http://localhost:8080",
                "enable_image_search": raw_value,
            },
        ),
    )
    assert isinstance(provider, SearXNGClient)
    assert provider._enable_image_search is True  # noqa: SLF001


def test_build_searxng_rejects_invalid_bool() -> None:
    with pytest.raises(ValueError, match="Invalid boolean"):
        build_search_provider_from_config(
            provider_type=WebSearchProviderType.SEARXNG,
            api_key=None,
            config={
                "searxng_base_url": "http://localhost:8080",
                "enable_image_search": "maybe",
            },
        )


def test_build_searxng_provider_requires_base_url() -> None:
    """Test that SearXNG provider requires a base URL."""
    with pytest.raises(ValueError, match="Please provide a URL"):
        build_search_provider_from_config(
            provider_type=WebSearchProviderType.SEARXNG,
            api_key=None,
            config={},
        )


def test_build_exa_provider_requires_api_key() -> None:
    """Test that Exa provider requires an API key."""
    with pytest.raises(ValueError, match="API key is required"):
        build_search_provider_from_config(
            provider_type=WebSearchProviderType.EXA,
            api_key=None,
            config={},
        )


def test_build_brave_provider_requires_api_key() -> None:
    """Test that Brave provider requires an API key."""
    with pytest.raises(ValueError, match="API key is required"):
        build_search_provider_from_config(
            provider_type=WebSearchProviderType.BRAVE,
            api_key=None,
            config={},
        )


def test_build_brave_provider_with_optional_config() -> None:
    provider = build_search_provider_from_config(
        provider_type=WebSearchProviderType.BRAVE,
        api_key="test-api-key",
        config={
            "country": "us",
            "search_lang": "en",
            "ui_lang": "en-US",
            "safesearch": "strict",
            "freshness": "pm",
            "timeout_seconds": "12",
        },
    )
    assert isinstance(provider, BraveClient)
    assert provider._country == "US"  # noqa: SLF001
    assert provider._search_lang == "en"  # noqa: SLF001
    assert provider._ui_lang == "en-US"  # noqa: SLF001
    assert provider._safesearch == "strict"  # noqa: SLF001
    assert provider._freshness == "pm"  # noqa: SLF001
    assert provider._timeout_seconds == 12  # noqa: SLF001


def test_build_brave_provider_rejects_invalid_timeout() -> None:
    with pytest.raises(ValueError, match="timeout_seconds"):
        build_search_provider_from_config(
            provider_type=WebSearchProviderType.BRAVE,
            api_key="test-api-key",
            config={"timeout_seconds": "not-an-int"},
        )


def test_build_serper_provider_requires_api_key() -> None:
    """Test that Serper provider requires an API key."""
    with pytest.raises(ValueError, match="API key is required"):
        build_search_provider_from_config(
            provider_type=WebSearchProviderType.SERPER,
            api_key=None,
            config={},
        )


def test_build_google_pse_provider_requires_api_key() -> None:
    """Test that Google PSE provider requires an API key."""
    with pytest.raises(ValueError, match="API key is required"):
        build_search_provider_from_config(
            provider_type=WebSearchProviderType.GOOGLE_PSE,
            api_key=None,
            config={"search_engine_id": "test-cx"},
        )


def test_build_google_pse_provider_requires_search_engine_id() -> None:
    """Test that Google PSE provider requires a search engine ID."""
    with pytest.raises(ValueError, match="search engine id"):
        build_search_provider_from_config(
            provider_type=WebSearchProviderType.GOOGLE_PSE,
            api_key="test-api-key",
            config={},
        )
