from unittest.mock import call, patch
from urllib.parse import parse_qs, unquote, urlsplit

import pytest
import requests

from onyx.configs.chat_configs import (
    SEARXNG_CONNECT_TIMEOUT_SECONDS,
    SEARXNG_READ_TIMEOUT_SECONDS,
)
from onyx.tools.tool_implementations.custom.openapi_parsing import MethodSpec
from onyx.tools.tool_implementations.web_search.clients.searxng_client import (
    SearXNGClient,
)

# (connect, read) pair the client must pass to every HTTP call.
_SEARXNG_TIMEOUT = (SEARXNG_CONNECT_TIMEOUT_SECONDS, SEARXNG_READ_TIMEOUT_SECONDS)


@pytest.mark.parametrize("value", ["a/b?admin=true#fragment", "café +&role=admin%20"])
def test_custom_tool_parameters_remain_values(value: str) -> None:
    method = MethodSpec(
        name="lookup",
        raw_name="lookup",
        summary="",
        path="/items/{item}",
        method="get",
        spec={},
    )
    query = {"q": value, "key&injected": "value=with&separators"}
    url = method.build_url("https://example.com", {"item": value}, query)
    prepared = requests.Request("GET", url).prepare()
    assert prepared.url is not None
    parsed = urlsplit(prepared.url)
    assert parsed.netloc == "example.com"
    assert len(parsed.path.split("/")) == 3
    assert unquote(parsed.path.removeprefix("/items/")) == value
    assert parse_qs(parsed.query) == {name: [value] for name, value in query.items()}
    assert parsed.fragment == ""


def test_searxng_search_has_connect_and_read_timeouts() -> None:
    client = SearXNGClient("https://example.com")
    with patch(
        "onyx.tools.tool_implementations.web_search.clients.searxng_client.requests.post"
    ) as post:
        post.return_value.json.return_value = {"results": []}
        assert client.search("query") == []
        # Both the general search and the image-category search must carry
        # explicit connect + read timeouts
        assert post.call_args_list == [
            call(
                "https://example.com/search",
                data={"q": "query", "format": "json"},
                timeout=_SEARXNG_TIMEOUT,
            ),
            call(
                "https://example.com/search",
                data={"q": "query", "format": "json", "categories": "images"},
                timeout=_SEARXNG_TIMEOUT,
            ),
        ]


def test_searxng_connection_has_connect_and_read_timeouts() -> None:
    client = SearXNGClient("https://example.com")
    with (
        patch(
            "onyx.tools.tool_implementations.web_search.clients.searxng_client.requests.get"
        ) as get,
        patch(
            "onyx.tools.tool_implementations.web_search.clients.searxng_client.requests.post"
        ) as post,
    ):
        get.return_value.json.return_value = {
            "brand": {"GIT_URL": "https://github.com/searxng/searxng"}
        }
        assert client.test_connection() == {"status": "ok"}
        get.assert_called_once_with(
            "https://example.com/config", timeout=_SEARXNG_TIMEOUT
        )
        post.assert_called_once_with(
            "https://example.com/search",
            data={"q": "test", "format": "json"},
            timeout=_SEARXNG_TIMEOUT,
        )
