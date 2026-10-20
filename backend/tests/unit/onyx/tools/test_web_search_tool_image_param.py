"""The web_search tool's image-search parameter: exposed only when the
provider supports it, routed to the provider's image-only search, and
rejected when a model emits it against a provider without the capability."""

from unittest.mock import MagicMock, patch

import pytest

from onyx.server.query_and_chat.placement import Placement
from onyx.tools.models import WebSearchToolOverrideKwargs
from onyx.tools.tool_implementations.web_search.models import (
    WebSearchProvider,
    WebSearchResult,
)
from onyx.tools.tool_implementations.web_search.web_search_tool import (
    SEARCH_IMAGES_FIELD,
    WebSearchTool,
)
from shared_configs.enums import WebSearchProviderType

WEB_RESULT = WebSearchResult(
    title="A web page", link="https://example.com/page", snippet="text"
)
IMAGE_RESULT = WebSearchResult(
    title="An image",
    link="https://example.com/page",
    snippet="1200 x 675",
    image_urls=["https://cdn.example.com/img.jpg"],
)


class StubProvider(WebSearchProvider):
    def __init__(self, supports_images: bool) -> None:
        self._supports_images = supports_images
        self.search_calls: list[tuple[str, str | None]] = []
        self.image_calls: list[tuple[str, str | None]] = []

    @property
    def supports_image_search(self) -> bool:
        return self._supports_images

    def search(self, query: str, language: str | None = None) -> list[WebSearchResult]:
        self.search_calls.append((query, language))
        return [WEB_RESULT]

    def search_images(
        self, query: str, language: str | None = None
    ) -> list[WebSearchResult]:
        self.image_calls.append((query, language))
        return [IMAGE_RESULT]

    def test_connection(self) -> dict[str, str]:
        return {"status": "ok"}


def _build_tool(supports_images: bool) -> WebSearchTool:
    stub = StubProvider(supports_images=supports_images)
    with (
        patch(
            "onyx.tools.tool_implementations.web_search.web_search_tool"
            ".get_session_with_current_tenant",
            MagicMock(),
        ),
        patch(
            "onyx.tools.tool_implementations.web_search.web_search_tool"
            ".fetch_active_web_search_provider"
        ) as fetch,
        patch(
            "onyx.tools.tool_implementations.web_search.web_search_tool"
            ".build_search_provider_from_config",
            return_value=stub,
        ),
    ):
        fetch.return_value = MagicMock(
            provider_type=WebSearchProviderType.SEARXNG.value,
            api_key=None,
            config={},
        )
        tool = WebSearchTool(tool_id=1, emitter=MagicMock())
    tool._provider = stub  # noqa: SLF001
    return tool


def _override_kwargs() -> WebSearchToolOverrideKwargs:
    return WebSearchToolOverrideKwargs(starting_citation_num=1)


def _placement() -> Placement:
    return Placement(turn_index=0)


def test_image_param_exposed_only_when_supported() -> None:
    supported = _build_tool(supports_images=True).tool_definition()
    unsupported = _build_tool(supports_images=False).tool_definition()

    assert SEARCH_IMAGES_FIELD in supported["function"]["parameters"]["properties"]
    assert (
        SEARCH_IMAGES_FIELD not in unsupported["function"]["parameters"]["properties"]
    )
    # the standard parameters are always present
    for definition in (supported, unsupported):
        assert "queries" in definition["function"]["parameters"]["properties"]
        assert "language" in definition["function"]["parameters"]["properties"]


def test_standard_call_hits_search_never_images() -> None:
    tool = _build_tool(supports_images=True)

    tool.run(
        placement=_placement(),
        override_kwargs=_override_kwargs(),
        queries=["drake meme"],
    )

    stub = tool._provider
    assert isinstance(stub, StubProvider)
    assert stub.search_calls == [("drake meme", None)]
    assert stub.image_calls == []


def test_image_call_hits_images_never_search() -> None:
    tool = _build_tool(supports_images=True)

    tool.run(
        placement=_placement(),
        override_kwargs=_override_kwargs(),
        queries=["drake meme"],
        search_images=True,
    )

    stub = tool._provider
    assert isinstance(stub, StubProvider)
    assert stub.image_calls == [("drake meme", None)]
    assert stub.search_calls == []


def test_image_call_on_unsupported_provider_is_rejected() -> None:
    """The parameter is only advertised for capable providers; a model that
    emits it anyway gets a targeted correction instead of a search."""
    tool = _build_tool(supports_images=False)

    with pytest.raises(Exception, match="does not support it") as exc_info:
        tool.run(
            placement=_placement(),
            override_kwargs=_override_kwargs(),
            queries=["drake meme"],
            search_images=True,
        )

    assert "search_images" in str(exc_info.value.llm_facing_message)
    stub = tool._provider
    assert isinstance(stub, StubProvider)
    assert stub.search_calls == []
    assert stub.image_calls == []


def test_string_boolean_spellings_accepted() -> None:
    tool = _build_tool(supports_images=True)

    tool.run(
        placement=_placement(),
        override_kwargs=_override_kwargs(),
        queries=["drake meme"],
        search_images="true",
    )

    stub = tool._provider
    assert isinstance(stub, StubProvider)
    assert stub.image_calls == [("drake meme", None)]
