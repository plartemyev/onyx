"""Unit tests for the WebConnector's SearXNG /crawl transport.

The connector is exercised with the crawl endpoint enabled and a mocked HTTP
layer — no SearXNG instance, no local browser. `start_playwright` is patched
to fail loudly so the tests prove the local Chromium is never started.
"""

from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock

import pytest
import requests as requests_lib

from onyx.connectors.models import Document, SlimDocument, TextSection
from onyx.connectors.web.connector import (
    WEB_CONNECTOR_VALID_SETTINGS,
    WebConnector,
)
from onyx.utils import playwright_fetch

ENDPOINT = "http://searxng:8080/crawl"
BASE_URL = "http://example.com"

PAGE_HTML = (
    "<html><head><title>Test Page</title></head><body>"
    "<p>Hello world</p>"
    '<p><a href="/page2">Page 2</a></p>'
    "</body></html>"
)


@pytest.fixture(autouse=True)
def _searxng_transport(monkeypatch: pytest.MonkeyPatch) -> None:
    """Enable the SearXNG crawl endpoint (single source of truth: the
    playwright_fetch module global) and keep the tests hermetic."""
    monkeypatch.setattr(playwright_fetch, "SEARXNG_CRAWL_ENDPOINT", ENDPOINT)
    monkeypatch.setattr(
        "onyx.connectors.web.connector.protected_url_check", lambda _url: None
    )
    monkeypatch.setattr(
        "onyx.connectors.web.connector.check_internet_connection", lambda _url: None
    )
    # The local browser must never start on this transport.
    monkeypatch.setattr(
        "onyx.connectors.web.connector.start_playwright",
        MagicMock(side_effect=AssertionError("local browser started")),
    )
    # PDF sniff: every URL is plain HTML.
    head = MagicMock()
    head.headers = {"content-type": "text/html"}
    monkeypatch.setattr(
        "onyx.connectors.web.connector.requests.head", MagicMock(return_value=head)
    )


def _crawl_response(
    url_to_html: dict[str, str],
    status: int = 200,
    final_url: str | None = None,
) -> MagicMock:
    """Fake the /crawl endpoint's JSON response, resolved per target URL."""
    response = MagicMock()
    response.status_code = 200

    def _json() -> dict[str, Any]:
        target = _last_url[-1]
        html = url_to_html.get(target, "<html><body></body></html>")
        return {
            "final_url": final_url or target,
            "status": status,
            "challenge": False,
            "html": html,
        }

    response.json.side_effect = _json
    return response


_last_url: list[str] = []


def _install_crawl_client(
    monkeypatch: pytest.MonkeyPatch,
    url_to_html: dict[str, str],
    status: int = 200,
    final_url: str | None = None,
) -> MagicMock:
    """Route /crawl JSON bodies per URL; returns the mock for call checks."""

    def _get(_endpoint: str, params: dict[str, Any], **_kwargs: Any) -> MagicMock:
        _last_url.append(params["url"])
        return _crawl_response(url_to_html, status=status, final_url=final_url)

    get_mock = MagicMock(side_effect=_get)
    monkeypatch.setattr(requests_lib, "get", get_mock)
    # failure tests would otherwise sleep out the human-paced retry gap
    monkeypatch.setattr(playwright_fetch, "_crawl_retry_wait", lambda: 0.0)
    return get_mock


def _single_connector(**kwargs: Any) -> WebConnector:
    return WebConnector(
        base_url=BASE_URL + "/",
        web_connector_type=WEB_CONNECTOR_VALID_SETTINGS.SINGLE.value,
        **kwargs,
    )


def _collect_docs(connector: WebConnector) -> list[Document]:
    return [
        doc
        for batch in connector.load_from_state()
        for doc in batch
        if isinstance(doc, Document)
    ]


def test_markdown_links_preserved_via_searxng(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Non-slim docs carry the markdown render: links stay as [text](href)."""
    monkeypatch.setattr("onyx.connectors.web.connector.WEB_CONTENT_FORMAT", "markdown")
    _install_crawl_client(monkeypatch, {BASE_URL + "/": PAGE_HTML})

    docs = _collect_docs(_single_connector())

    assert len(docs) == 1
    doc = docs[0]
    assert isinstance(doc, Document)
    assert doc.id == BASE_URL + "/"
    assert doc.semantic_identifier == "Test Page"
    section = doc.sections[0]
    assert isinstance(section, TextSection)
    text = section.text
    assert "Hello world" in text
    assert "[Page 2](http://example.com/page2)" in text


def test_text_format_fallback(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """WEB_CONTENT_FORMAT=text keeps the legacy bs4 flattening (no URLs)."""
    monkeypatch.setattr("onyx.connectors.web.connector.WEB_CONTENT_FORMAT", "text")
    _install_crawl_client(monkeypatch, {BASE_URL + "/": PAGE_HTML})

    docs = _collect_docs(_single_connector())

    assert len(docs) == 1
    section = docs[0].sections[0]
    assert isinstance(section, TextSection)
    text = section.text
    assert "Hello world" in text
    assert "Page 2" in text
    assert "](http" not in text


def test_redirect_persists_under_final_url(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A remote-followed redirect stores the doc under the final URL."""
    final = BASE_URL + "/canonical"
    _install_crawl_client(monkeypatch, {BASE_URL + "/": PAGE_HTML}, final_url=final)

    docs = _collect_docs(_single_connector())

    assert len(docs) == 1
    assert docs[0].id == final
    assert docs[0].sections[0].link == final


def test_http_500_sets_retry(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A persistent 5xx render is retried (with backoff), then the crawl
    fails the same way the local transport does."""
    monkeypatch.setattr("onyx.connectors.web.connector.time.sleep", lambda _s: None)
    get_mock = _install_crawl_client(
        monkeypatch, {BASE_URL + "/": PAGE_HTML}, status=500
    )

    with pytest.raises(RuntimeError, match="SearXNG crawl endpoint"):
        [doc for batch in _single_connector().load_from_state() for doc in batch]

    # every connector-level attempt sees the client's own 2 paced tries
    # (a 5xx is a "wall" there) before coming back as a failure
    assert get_mock.call_count == 2 * WebConnector.MAX_RETRIES


def test_endpoint_failure_sets_retry(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An unreachable endpoint exhausts retries, then fails the crawl."""

    def _get(*_args: Any, **_kwargs: Any) -> MagicMock:
        raise requests_lib.ConnectionError("searxng down")

    monkeypatch.setattr(requests_lib, "get", MagicMock(side_effect=_get))
    monkeypatch.setattr(playwright_fetch, "_crawl_retry_wait", lambda: 0.0)
    monkeypatch.setattr("onyx.connectors.web.connector.time.sleep", lambda _s: None)

    with pytest.raises(RuntimeError, match="SearXNG crawl endpoint"):
        [doc for batch in _single_connector().load_from_state() for doc in batch]


def test_recursive_discovers_links_from_rendered_html(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Link discovery runs on the rendered HTML, under the base path only."""
    root = (
        "<html><head><title>Root</title></head><body>"
        '<a href="/docs/page2">Page 2</a>'
        '<a href="/docs/page3">Page 3</a>'
        '<a href="/elsewhere">Elsewhere</a>'
        "</body></html>"
    )
    child2 = (
        "<html><head><title>Child 2</title></head><body><p>leaf two</p></body></html>"
    )
    child3 = (
        "<html><head><title>Child 3</title></head><body><p>leaf three</p></body></html>"
    )
    _install_crawl_client(
        monkeypatch,
        {
            BASE_URL + "/docs": root,
            BASE_URL + "/docs/page2": child2,
            BASE_URL + "/docs/page3": child3,
        },
    )

    connector = WebConnector(
        base_url=BASE_URL + "/docs",
        web_connector_type=WEB_CONNECTOR_VALID_SETTINGS.RECURSIVE.value,
    )
    docs = _collect_docs(connector)

    ids = {doc.id for doc in docs}
    assert ids == {
        BASE_URL + "/docs",
        BASE_URL + "/docs/page2",
        BASE_URL + "/docs/page3",
    }


def test_slim_via_searxng(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Slim docs carry URL ids only, discovered from the rendered HTML."""
    _install_crawl_client(monkeypatch, {BASE_URL + "/": PAGE_HTML})

    docs = [
        doc for batch in _single_connector().retrieve_all_slim_docs() for doc in batch
    ]

    assert len(docs) == 1
    assert isinstance(docs[0], SlimDocument)
    assert docs[0].id == BASE_URL + "/"
