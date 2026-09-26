"""Tests for the SearXNG crawl-endpoint client (`onyx.utils.playwright_fetch`).

The client is exercised with a mocked HTTP layer — no SearXNG instance or
browser involved.
"""

from __future__ import annotations

from unittest.mock import patch

import pytest
import requests as requests_lib

from onyx.utils import playwright_fetch
from onyx.utils.playwright_fetch import (
    DownloadedContent,
    RenderedPage,
    _crawl_via_searxng_bytes,
    _crawl_via_searxng_render,
)


class _FakeResponse:
    def __init__(self, status_code=200, payload=None, content=b"", headers=None):
        self.status_code = status_code
        self._payload = payload
        self.content = content
        self.text = "" if payload is None else str(payload)
        self.headers = headers or {}

    def json(self):
        return self._payload


def test_render_endpoint_maps_payload(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        playwright_fetch,
        "SEARXNG_CRAWL_ENDPOINT",
        "http://searxng:8080/crawl",
    )
    fake = _FakeResponse(
        200,
        {
            "final_url": "https://example.org/page",
            "status": 200,
            "challenge": False,
            "html": "<html><body>hello</body></html>",
        },
    )
    with patch.object(requests_lib, "get", return_value=fake) as mock_get:
        rendered = _crawl_via_searxng_render(
            "https://example.org/page", navigation_timeout_ms=30000
        )

    assert isinstance(rendered, RenderedPage)
    assert rendered.final_url == "https://example.org/page"
    assert rendered.status == 200
    assert "hello" in rendered.html
    _, kwargs = mock_get.call_args
    assert kwargs["params"]["mode"] == "render"
    assert kwargs["params"]["url"] == "https://example.org/page"


def test_render_endpoint_failure_returns_none(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        playwright_fetch,
        "SEARXNG_CRAWL_ENDPOINT",
        "http://searxng:8080/crawl",
    )
    with patch.object(
        requests_lib, "get", return_value=_FakeResponse(502, {"error": "crawl failed"})
    ):
        assert (
            _crawl_via_searxng_render(
                "https://example.org", navigation_timeout_ms=30000
            )
            is None
        )


def test_render_endpoint_unreachable_returns_none(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        playwright_fetch,
        "SEARXNG_CRAWL_ENDPOINT",
        "http://searxng:8080/crawl",
    )
    with patch.object(
        requests_lib,
        "get",
        side_effect=requests_lib.ConnectionError("no route"),
    ):
        assert (
            _crawl_via_searxng_render(
                "https://example.org", navigation_timeout_ms=30000
            )
            is None
        )


def test_bytes_endpoint_maps_headers(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        playwright_fetch,
        "SEARXNG_CRAWL_ENDPOINT",
        "http://searxng:8080/crawl",
    )
    fake = _FakeResponse(
        200,
        content=b"%PDF-1.4 fake",
        headers={
            "X-Final-URL": "https://example.org/doc.pdf",
            "X-Crawl-Status": "200",
            "content-type": "application/pdf",
        },
    )
    with patch.object(requests_lib, "get", return_value=fake) as mock_get:
        downloaded = _crawl_via_searxng_bytes(
            "https://example.org/doc.pdf", navigation_timeout_ms=30000
        )

    assert isinstance(downloaded, DownloadedContent)
    assert downloaded.content == b"%PDF-1.4 fake"
    assert downloaded.final_url == "https://example.org/doc.pdf"
    assert downloaded.content_type == "application/pdf"
    assert downloaded.status == 200
    _, kwargs = mock_get.call_args
    assert kwargs["params"]["mode"] == "bytes"


def test_endpoint_disabled_leaves_pool_path_in_place() -> None:
    # default config: no endpoint -> the module keeps its local-pool behavior
    assert playwright_fetch.SEARXNG_CRAWL_ENDPOINT == ""
