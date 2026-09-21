from __future__ import annotations

import time
from collections.abc import Callable
from typing import Any
from unittest.mock import MagicMock, patch

import pytest
from pydantic import BaseModel

import onyx.tools.tool_implementations.open_url.onyx_web_crawler as crawler_module
from onyx.server.security.models import SSRFProtectionLevel
from onyx.server.security.store import _build_env_defaults
from onyx.tools.tool_implementations.open_url.onyx_web_crawler import (
    DEFAULT_CONNECT_TIMEOUT_SECONDS,
    DEFAULT_READ_TIMEOUT_SECONDS,
    OnyxWebCrawler,
)
from onyx.utils.request_pacer import NullPacer


@pytest.fixture(autouse=True)
def _disable_request_pacing(monkeypatch: pytest.MonkeyPatch) -> None:
    """Keep real per-provider sleeps (0.7-5s) out of unit tests. Tests that
    exercise pacing inject a recording pacer via the crawler constructor."""
    monkeypatch.setattr(crawler_module, "get_default_pacer", lambda: NullPacer())


class FakeResponse(BaseModel):
    status_code: int
    headers: dict[str, str]
    content: bytes
    text: str = ""
    apparent_encoding: str | None = None
    encoding: str | None = None


def test_fetch_url_extracts_image_urls(monkeypatch: pytest.MonkeyPatch) -> None:
    crawler = OnyxWebCrawler()
    html = (
        "<html><body><p>Some readable content.</p>"
        '<img src="/images/hero.jpg?w=800">'
        '<img src="https://cdn.example.com/abs.png">'
        '<img src="data:image/png;base64,AAAA">'
        "</body></html>"
    ).encode()
    response = FakeResponse(
        status_code=200,
        headers={"Content-Type": "text/html; charset=utf-8"},
        content=html,
    )

    monkeypatch.setattr(
        crawler_module,
        "ssrf_safe_get",
        lambda *args, **kwargs: response,  # noqa: ARG005
    )

    result = crawler._fetch_url("https://example.com/page")

    assert result.scrape_successful is True
    assert result.image_urls == [
        "https://example.com/images/hero.jpg?w=800",
        "https://cdn.example.com/abs.png",
    ]


def test_fetch_url_pdf_with_content_type(monkeypatch: pytest.MonkeyPatch) -> None:
    crawler = OnyxWebCrawler()
    response = FakeResponse(
        status_code=200,
        headers={"Content-Type": "application/pdf"},
        content=b"%PDF-1.4 mock",
    )

    monkeypatch.setattr(
        crawler_module,
        "ssrf_safe_get",
        lambda *args, **kwargs: response,  # noqa: ARG005
    )
    monkeypatch.setattr(
        crawler_module,
        "extract_pdf_text",
        lambda *args, **kwargs: ("pdf text", {"Title": "Doc Title"}),  # noqa: ARG005
    )

    result = crawler._fetch_url("https://example.com/report.pdf")

    assert result.full_content == "pdf text"
    assert result.title == "Doc Title"
    assert result.scrape_successful is True


def test_fetch_url_pdf_with_signature(monkeypatch: pytest.MonkeyPatch) -> None:
    crawler = OnyxWebCrawler()
    response = FakeResponse(
        status_code=200,
        headers={"Content-Type": "application/octet-stream"},
        content=b"%PDF-1.7 mock",
    )

    monkeypatch.setattr(
        crawler_module,
        "ssrf_safe_get",
        lambda *args, **kwargs: response,  # noqa: ARG005
    )
    monkeypatch.setattr(
        crawler_module,
        "extract_pdf_text",
        lambda *args, **kwargs: ("pdf text", {}),  # noqa: ARG005
    )

    result = crawler._fetch_url("https://example.com/files/file.pdf")

    assert result.full_content == "pdf text"
    assert result.title == "file.pdf"
    assert result.scrape_successful is True


def test_fetch_url_decodes_html_bytes(monkeypatch: pytest.MonkeyPatch) -> None:
    crawler = OnyxWebCrawler()
    html_bytes = b"<html><body>caf\xe9</body></html>"
    response = FakeResponse(
        status_code=200,
        headers={"Content-Type": "text/html; charset=iso-8859-1"},
        content=html_bytes,
        text="caf\u00ef\u00bf\u00bd",
    )

    monkeypatch.setattr(
        crawler_module,
        "ssrf_safe_get",
        lambda *args, **kwargs: response,  # noqa: ARG005
    )

    result = crawler._fetch_url("https://example.com/page.html")

    assert "caf\u00e9" in result.full_content
    assert result.scrape_successful is True


def test_fetch_url_pdf_exceeds_size_limit(monkeypatch: pytest.MonkeyPatch) -> None:
    """PDF content exceeding max_pdf_size_bytes should be rejected."""
    crawler = OnyxWebCrawler(max_pdf_size_bytes=100)
    response = FakeResponse(
        status_code=200,
        headers={"Content-Type": "application/pdf"},
        content=b"%PDF-1.4 " + b"x" * 200,  # 209 bytes, exceeds 100 limit
    )

    monkeypatch.setattr(
        crawler_module,
        "ssrf_safe_get",
        lambda *args, **kwargs: response,  # noqa: ARG005
    )

    result = crawler._fetch_url("https://example.com/large.pdf")

    assert result.full_content == ""
    assert result.scrape_successful is False
    assert result.link == "https://example.com/large.pdf"


def test_fetch_url_pdf_within_size_limit(monkeypatch: pytest.MonkeyPatch) -> None:
    """PDF content within max_pdf_size_bytes should be processed normally."""
    crawler = OnyxWebCrawler(max_pdf_size_bytes=500)
    response = FakeResponse(
        status_code=200,
        headers={"Content-Type": "application/pdf"},
        content=b"%PDF-1.4 mock",  # Small content
    )

    monkeypatch.setattr(
        crawler_module,
        "ssrf_safe_get",
        lambda *args, **kwargs: response,  # noqa: ARG005
    )
    monkeypatch.setattr(
        crawler_module,
        "extract_pdf_text",
        lambda *args, **kwargs: ("pdf text", {"Title": "Doc Title"}),  # noqa: ARG005
    )

    result = crawler._fetch_url("https://example.com/small.pdf")

    assert result.full_content == "pdf text"
    assert result.scrape_successful is True


def test_fetch_url_html_exceeds_size_limit(monkeypatch: pytest.MonkeyPatch) -> None:
    """HTML content exceeding max_html_size_bytes should be rejected."""
    crawler = OnyxWebCrawler(max_html_size_bytes=50)
    html_bytes = b"<html><body>" + b"x" * 100 + b"</body></html>"  # Exceeds 50 limit
    response = FakeResponse(
        status_code=200,
        headers={"Content-Type": "text/html"},
        content=html_bytes,
    )

    monkeypatch.setattr(
        crawler_module,
        "ssrf_safe_get",
        lambda *args, **kwargs: response,  # noqa: ARG005
    )

    result = crawler._fetch_url("https://example.com/large.html")

    assert result.full_content == ""
    assert result.scrape_successful is False
    assert result.link == "https://example.com/large.html"


def test_fetch_url_html_within_size_limit(monkeypatch: pytest.MonkeyPatch) -> None:
    """HTML content within max_html_size_bytes should be processed normally."""
    crawler = OnyxWebCrawler(max_html_size_bytes=500)
    html_bytes = b"<html><body>hello world</body></html>"
    response = FakeResponse(
        status_code=200,
        headers={"Content-Type": "text/html"},
        content=html_bytes,
    )

    monkeypatch.setattr(
        crawler_module,
        "ssrf_safe_get",
        lambda *args, **kwargs: response,  # noqa: ARG005
    )

    result = crawler._fetch_url("https://example.com/small.html")

    assert "hello world" in result.full_content
    assert result.scrape_successful is True


# ---------------------------------------------------------------------------
# Helpers for parallel / failure-isolation / timeout tests
# ---------------------------------------------------------------------------


def _make_mock_response(
    *,
    status_code: int = 200,
    content: bytes = b"<html><body>Hello</body></html>",
    content_type: str = "text/html",
    delay: float = 0.0,
) -> MagicMock:
    """Create a mock response that behaves like a requests.Response."""
    resp = MagicMock()
    resp.status_code = status_code
    resp.headers = {"Content-Type": content_type}

    if delay:
        original_content = content

        @property
        def _delayed_content(_self: object) -> bytes:
            time.sleep(delay)
            return original_content

        type(resp).content = _delayed_content
    else:
        resp.content = content

    resp.apparent_encoding = None
    resp.encoding = None

    return resp


def _respond_by_url(
    responses: dict[str, MagicMock | Exception],
) -> Callable[..., MagicMock]:
    """Build a ``side_effect`` that maps each URL to its response, raising any
    mapped exception — independent of call order.

    ``OnyxWebCrawler.contents`` fetches concurrently via a ThreadPoolExecutor, so
    a positional ``side_effect`` list binds responses to URLs nondeterministically
    (whichever thread calls ``ssrf_safe_get`` first consumes the first element).
    Keying on the URL keeps the failure-isolation assertions deterministic.
    """

    def _side_effect(url: str, *_args: object, **_kwargs: object) -> MagicMock:
        result = responses[url]
        if isinstance(result, Exception):
            raise result
        return result

    return _side_effect


class TestParallelExecution:
    """Verify that contents() fetches URLs in parallel."""

    @patch("onyx.tools.tool_implementations.open_url.onyx_web_crawler.ssrf_safe_get")
    def test_multiple_urls_fetched_concurrently(self, mock_get: MagicMock) -> None:
        """With a per-URL delay, parallel execution should be much faster than sequential."""
        per_url_delay = 0.3
        num_urls = 5
        urls = [f"http://example.com/page{i}" for i in range(num_urls)]

        mock_get.return_value = _make_mock_response(delay=per_url_delay)

        crawler = OnyxWebCrawler()
        start = time.monotonic()
        results = crawler.contents(urls)
        elapsed = time.monotonic() - start

        # Sequential would take ~1.5s; parallel should be well under that
        assert elapsed < per_url_delay * num_urls * 0.7
        assert len(results) == num_urls
        assert all(r.scrape_successful for r in results)

    @patch("onyx.tools.tool_implementations.open_url.onyx_web_crawler.ssrf_safe_get")
    def test_empty_urls_returns_empty(self, mock_get: MagicMock) -> None:
        crawler = OnyxWebCrawler()
        results = crawler.contents([])
        assert results == []
        mock_get.assert_not_called()

    @patch("onyx.tools.tool_implementations.open_url.onyx_web_crawler.ssrf_safe_get")
    def test_single_url(self, mock_get: MagicMock) -> None:
        mock_get.return_value = _make_mock_response()
        crawler = OnyxWebCrawler()
        results = crawler.contents(["http://example.com"])
        assert len(results) == 1
        assert results[0].scrape_successful


class TestFailureIsolation:
    """Verify that one URL failure doesn't affect others in the batch."""

    @patch("onyx.tools.tool_implementations.open_url.onyx_web_crawler.ssrf_safe_get")
    def test_one_failure_doesnt_kill_batch(self, mock_get: MagicMock) -> None:
        good_resp = _make_mock_response()
        bad_resp = _make_mock_response(status_code=500)

        # First and third URLs succeed, second fails. Keyed by URL because the
        # crawler fetches concurrently, so call order is nondeterministic.
        mock_get.side_effect = _respond_by_url(
            {
                "http://a.com": good_resp,
                "http://b.com": bad_resp,
                "http://c.com": good_resp,
            }
        )

        crawler = OnyxWebCrawler()
        results = crawler.contents(["http://a.com", "http://b.com", "http://c.com"])

        assert len(results) == 3
        assert results[0].scrape_successful
        assert not results[1].scrape_successful
        assert results[2].scrape_successful

    @patch("onyx.tools.tool_implementations.open_url.onyx_web_crawler.ssrf_safe_get")
    def test_exception_doesnt_kill_batch(self, mock_get: MagicMock) -> None:
        good_resp = _make_mock_response()

        # Second URL raises an exception. Keyed by URL (concurrent fetch).
        mock_get.side_effect = _respond_by_url(
            {
                "http://a.com": good_resp,
                "http://b.com": RuntimeError("connection reset"),
                "http://c.com": _make_mock_response(),
            }
        )

        crawler = OnyxWebCrawler()
        results = crawler.contents(["http://a.com", "http://b.com", "http://c.com"])

        assert len(results) == 3
        assert results[0].scrape_successful
        assert not results[1].scrape_successful
        assert results[2].scrape_successful

    @patch("onyx.tools.tool_implementations.open_url.onyx_web_crawler.ssrf_safe_get")
    def test_ssrf_exception_doesnt_kill_batch(self, mock_get: MagicMock) -> None:
        from onyx.utils.url import SSRFException

        good_resp = _make_mock_response()
        mock_get.side_effect = _respond_by_url(
            {
                "http://a.com": good_resp,
                "http://internal.local": SSRFException("blocked"),
                "http://c.com": _make_mock_response(),
            }
        )

        crawler = OnyxWebCrawler()
        results = crawler.contents(
            ["http://a.com", "http://internal.local", "http://c.com"]
        )

        assert len(results) == 3
        assert results[0].scrape_successful
        assert not results[1].scrape_successful
        assert results[2].scrape_successful


class TestTupleTimeout:
    """Verify that separate connect and read timeouts are passed correctly."""

    @patch("onyx.tools.tool_implementations.open_url.onyx_web_crawler.ssrf_safe_get")
    def test_default_tuple_timeout(self, mock_get: MagicMock) -> None:
        mock_get.return_value = _make_mock_response()

        crawler = OnyxWebCrawler()
        crawler.contents(["http://example.com"])

        call_kwargs = mock_get.call_args
        assert call_kwargs.kwargs["timeout"] == (
            DEFAULT_CONNECT_TIMEOUT_SECONDS,
            DEFAULT_READ_TIMEOUT_SECONDS,
        )

    @patch("onyx.tools.tool_implementations.open_url.onyx_web_crawler.ssrf_safe_get")
    def test_custom_tuple_timeout(self, mock_get: MagicMock) -> None:
        mock_get.return_value = _make_mock_response()

        crawler = OnyxWebCrawler(timeout_seconds=30, connect_timeout_seconds=3)
        crawler.contents(["http://example.com"])

        call_kwargs = mock_get.call_args
        assert call_kwargs.kwargs["timeout"] == (3, 30)


def _pin_level(monkeypatch: pytest.MonkeyPatch, level: SSRFProtectionLevel) -> None:
    settings = _build_env_defaults().model_copy(update={"ssrf_protection_level": level})
    monkeypatch.setattr(crawler_module, "get_security_settings", lambda: settings)


def test_should_validate_ssrf_resolves_per_request(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """With no explicit override the crawler reads the admin level on each call,
    so an admin change takes effect on an already-constructed crawler."""
    crawler = OnyxWebCrawler()  # validate_ssrf=None

    _pin_level(monkeypatch, SSRFProtectionLevel.VALIDATE_ALL)
    assert crawler._should_validate_ssrf() is True

    _pin_level(monkeypatch, SSRFProtectionLevel.DISABLED)
    assert crawler._should_validate_ssrf() is False


def test_should_validate_ssrf_override_pins(monkeypatch: pytest.MonkeyPatch) -> None:
    """An explicit validate_ssrf wins regardless of the admin level."""
    crawler = OnyxWebCrawler(validate_ssrf=True)
    _pin_level(monkeypatch, SSRFProtectionLevel.DISABLED)
    assert crawler._should_validate_ssrf() is True

    crawler = OnyxWebCrawler(validate_ssrf=False)
    _pin_level(monkeypatch, SSRFProtectionLevel.VALIDATE_ALL)
    assert crawler._should_validate_ssrf() is False


def _jpeg_response() -> FakeResponse:
    return FakeResponse(
        status_code=200,
        headers={"Content-Type": "image/jpeg"},
        content=b"\xff\xd8\xff\xe0jpegdata",
    )


def test_download_file_bytes_fast_path(monkeypatch: pytest.MonkeyPatch) -> None:
    crawler = OnyxWebCrawler()
    response = _jpeg_response()

    monkeypatch.setattr(
        crawler_module,
        "ssrf_safe_get",
        lambda *args, **kwargs: response,  # noqa: ARG005
    )

    result = crawler.download_file_bytes("https://example.com/cat.jpg")

    assert isinstance(result, crawler_module.FetchedFile)
    assert result.content == b"\xff\xd8\xff\xe0jpegdata"
    assert result.content_type == "image/jpeg"


class _RecordingPacer:
    def __init__(self) -> None:
        self.paced_urls: list[str] = []

    def pace(self, url: str) -> None:
        self.paced_urls.append(url)


def test_download_file_bytes_paces_request(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    pacer = _RecordingPacer()
    crawler = OnyxWebCrawler(playwright_fallback_enabled=False, pacer=pacer)
    monkeypatch.setattr(
        crawler_module,
        "ssrf_safe_get",
        lambda *args, **kwargs: _jpeg_response(),  # noqa: ARG005
    )

    crawler.download_file_bytes("https://i.imgur.com/AKVZMd3b.jpg")

    assert pacer.paced_urls == ["https://i.imgur.com/AKVZMd3b.jpg"]


def test_download_file_bytes_paces_playwright_fallback(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The browser retry is a fresh request to the same provider, so it is
    paced too."""
    pacer = _RecordingPacer()
    crawler = OnyxWebCrawler(playwright_fallback_enabled=True, pacer=pacer)
    response = FakeResponse(
        status_code=403,
        headers={},
        content=b"blocked",
    )
    monkeypatch.setattr(
        crawler_module,
        "ssrf_safe_get",
        lambda *args, **kwargs: response,  # noqa: ARG005
    )
    monkeypatch.setattr(
        crawler_module,
        "fetch_content_bytes",
        lambda *args, **kwargs: None,  # noqa: ARG005
    )

    result = crawler.download_file_bytes("https://i.imgur.com/AKVZMd3b.jpg")

    assert isinstance(result, crawler_module.FailedFetch)
    assert pacer.paced_urls == ["https://i.imgur.com/AKVZMd3b.jpg"] * 2


def test_fetch_url_paces_request(monkeypatch: pytest.MonkeyPatch) -> None:
    pacer = _RecordingPacer()
    crawler = OnyxWebCrawler(pacer=pacer)
    response = FakeResponse(
        status_code=200,
        headers={"Content-Type": "text/html; charset=utf-8"},
        content=b"<html><body><p>Readable text.</p></body></html>",
    )
    monkeypatch.setattr(
        crawler_module,
        "ssrf_safe_get",
        lambda *args, **kwargs: response,  # noqa: ARG005
    )

    crawler._fetch_url("https://example.com/page")

    assert pacer.paced_urls == ["https://example.com/page"]


def test_download_file_bytes_rejects_html(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    crawler = OnyxWebCrawler(playwright_fallback_enabled=False)
    response = FakeResponse(
        status_code=200,
        headers={"Content-Type": "text/html; charset=utf-8"},
        content=b"<html><body>page</body></html>",
    )

    monkeypatch.setattr(
        crawler_module,
        "ssrf_safe_get",
        lambda *args, **kwargs: response,  # noqa: ARG005
    )

    result = crawler.download_file_bytes("https://example.com/page")

    assert isinstance(result, crawler_module.FailedFetch)
    assert result.failure_reason == crawler_module.FailureReason.HTML_NOT_FILE


def test_download_file_bytes_rejects_html_after_playwright_fails(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A 200 HTML response retries via Playwright, and when the real browser
    also yields no usable bytes, the failure reason stays HTML_NOT_FILE."""
    crawler = OnyxWebCrawler(playwright_fallback_enabled=True)
    response = FakeResponse(
        status_code=200,
        headers={"Content-Type": "text/html"},
        content=b"<html><body>blocked</body></html>",
    )

    monkeypatch.setattr(
        crawler_module,
        "ssrf_safe_get",
        lambda *args, **kwargs: response,  # noqa: ARG005
    )
    fallback_calls: list[str] = []

    def _no_bytes(url: str, **kwargs: Any) -> None:  # noqa: ARG001
        fallback_calls.append(url)
        return None

    monkeypatch.setattr(crawler_module, "fetch_content_bytes", _no_bytes)

    result = crawler.download_file_bytes("https://example.com/page")

    assert fallback_calls == ["https://example.com/page"]
    assert isinstance(result, crawler_module.FailedFetch)
    assert result.failure_reason == crawler_module.FailureReason.HTML_NOT_FILE


def test_download_file_bytes_html_page_falls_back_to_playwright(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """HTTP 200 + HTML (the imgur/reddit antibot block pattern) retries through
    the headless browser, which can still fetch the real file bytes."""
    crawler = OnyxWebCrawler(playwright_fallback_enabled=True)
    response = FakeResponse(
        status_code=200,
        headers={"Content-Type": "text/html"},
        content=b"<html><body>blocked</body></html>",
    )

    monkeypatch.setattr(
        crawler_module,
        "ssrf_safe_get",
        lambda *args, **kwargs: response,  # noqa: ARG005
    )
    downloaded = crawler_module.DownloadedContent(
        content=b"\xff\xd8\xff\xe0jpegdata",
        final_url="https://example.com/cat.jpg",
        content_type="image/jpeg",
        status=200,
    )
    monkeypatch.setattr(
        crawler_module,
        "fetch_content_bytes",
        lambda *args, **kwargs: downloaded,  # noqa: ARG005
    )

    result = crawler.download_file_bytes("https://example.com/cat.jpg")

    assert isinstance(result, crawler_module.FetchedFile)
    assert result.content == b"\xff\xd8\xff\xe0jpegdata"
    assert result.content_type == "image/jpeg"


def test_download_file_bytes_trusts_binary_magic_bytes_over_html_header(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Servers behind antibot walls sometimes mislabel binaries as text/html;
    the magic bytes win so the file is still downloadable."""
    crawler = OnyxWebCrawler(playwright_fallback_enabled=False)
    response = FakeResponse(
        status_code=200,
        headers={"Content-Type": "text/html"},
        content=b"\x89PNG\r\n\x1a\npngdata",
    )

    monkeypatch.setattr(
        crawler_module,
        "ssrf_safe_get",
        lambda *args, **kwargs: response,  # noqa: ARG005
    )
    monkeypatch.setattr(
        crawler_module,
        "fetch_content_bytes",
        lambda *args, **kwargs: pytest.fail(  # noqa: ARG005
            "Playwright fallback must not be needed"
        ),
    )

    result = crawler.download_file_bytes("https://example.com/cat.png")

    assert isinstance(result, crawler_module.FetchedFile)
    assert result.content_type == "image/png"
    assert result.content == b"\x89PNG\r\n\x1a\npngdata"


def test_download_file_bytes_429_falls_back_to_playwright(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """429 (antibot rate limiting without CF headers) is worth one browser
    retry for downloads."""
    crawler = OnyxWebCrawler(playwright_fallback_enabled=True)
    response = FakeResponse(
        status_code=429,
        headers={"Content-Type": "text/html"},
        content=b"slow down",
    )

    monkeypatch.setattr(
        crawler_module,
        "ssrf_safe_get",
        lambda *args, **kwargs: response,  # noqa: ARG005
    )
    downloaded = crawler_module.DownloadedContent(
        content=b"\x89PNG\r\n\x1a\npngdata",
        final_url="https://example.com/cat.png",
        content_type="image/png",
        status=200,
    )
    monkeypatch.setattr(
        crawler_module,
        "fetch_content_bytes",
        lambda *args, **kwargs: downloaded,  # noqa: ARG005
    )

    result = crawler.download_file_bytes("https://example.com/cat.png")

    assert isinstance(result, crawler_module.FetchedFile)
    assert result.content_type == "image/png"


def test_download_file_bytes_enforces_size_cap(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    crawler = OnyxWebCrawler()
    response = FakeResponse(
        status_code=200,
        headers={"Content-Type": "image/png"},
        content=b"\x89PNG\r\n\x1a\n" + b"x" * 200,
    )

    monkeypatch.setattr(
        crawler_module,
        "ssrf_safe_get",
        lambda *args, **kwargs: response,  # noqa: ARG005
    )

    result = crawler.download_file_bytes(
        "https://example.com/cat.png", max_file_size_bytes=100
    )

    assert isinstance(result, crawler_module.FailedFetch)
    assert result.failure_reason == crawler_module.FailureReason.OVERSIZED_FILE


def test_download_file_bytes_403_falls_back_to_playwright(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    crawler = OnyxWebCrawler(playwright_fallback_enabled=True)
    response = FakeResponse(
        status_code=403,
        headers={"cf-ray": "some-ray"},
        content=b"blocked",
    )

    monkeypatch.setattr(
        crawler_module,
        "ssrf_safe_get",
        lambda *args, **kwargs: response,  # noqa: ARG005
    )

    downloaded = crawler_module.DownloadedContent(
        content=b"\x89PNG\r\n\x1a\npngdata",
        final_url="https://example.com/cat.png",
        content_type="image/png",
        status=200,
    )
    monkeypatch.setattr(
        crawler_module,
        "fetch_content_bytes",
        lambda *args, **kwargs: downloaded,  # noqa: ARG005
    )

    result = crawler.download_file_bytes("https://example.com/cat.png")

    assert isinstance(result, crawler_module.FetchedFile)
    assert result.content == b"\x89PNG\r\n\x1a\npngdata"
    assert result.content_type == "image/png"


def test_download_file_bytes_403_challenge_not_resolved(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    crawler = OnyxWebCrawler(playwright_fallback_enabled=True)
    response = FakeResponse(
        status_code=403,
        headers={"cf-ray": "some-ray"},
        content=b"blocked",
    )

    monkeypatch.setattr(
        crawler_module,
        "ssrf_safe_get",
        lambda *args, **kwargs: response,  # noqa: ARG005
    )

    downloaded = crawler_module.DownloadedContent(
        content=b"<html>Just a moment...</html>",
        final_url="https://example.com/cat.png",
        content_type="text/html",
        status=403,
    )
    monkeypatch.setattr(
        crawler_module,
        "fetch_content_bytes",
        lambda *args, **kwargs: downloaded,  # noqa: ARG005
    )

    result = crawler.download_file_bytes("https://example.com/cat.png")

    assert isinstance(result, crawler_module.FailedFetch)
    assert result.failure_reason == crawler_module.FailureReason.CLOUDFLARE_CHALLENGE


def test_download_file_bytes_403_fallback_fails(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    crawler = OnyxWebCrawler(playwright_fallback_enabled=True)
    response = FakeResponse(
        status_code=403,
        headers={},
        content=b"blocked",
    )

    monkeypatch.setattr(
        crawler_module,
        "ssrf_safe_get",
        lambda *args, **kwargs: response,  # noqa: ARG005
    )
    monkeypatch.setattr(
        crawler_module,
        "fetch_content_bytes",
        lambda *args, **kwargs: None,  # noqa: ARG005
    )

    result = crawler.download_file_bytes("https://example.com/cat.jpg")

    assert isinstance(result, crawler_module.FailedFetch)
    assert result.failure_reason == crawler_module.FailureReason.HTTP_403_BLOCKED


def test_download_file_bytes_ssrf_blocked(monkeypatch: pytest.MonkeyPatch) -> None:
    crawler = OnyxWebCrawler()

    def _raise_ssrf(*args: object, **kwargs: object) -> None:  # noqa: ARG001
        raise crawler_module.SSRFException("internal address")

    monkeypatch.setattr(crawler_module, "ssrf_safe_get", _raise_ssrf)

    result = crawler.download_file_bytes("http://127.0.0.1:8080/secret")

    assert isinstance(result, crawler_module.FailedFetch)
    assert result.failure_reason == crawler_module.FailureReason.SSRF_BLOCKED


class TestLooksLikeImageUrl:
    @pytest.mark.parametrize(
        "url",
        [
            "https://i.imgur.com/AKVZMd3b.jpg",
            # Image CDN hosts with no extension in the URL
            "https://i.imgur.com/AKVZMd3",
            "https://i.redd.it/7mw4q5vkt5vb1",
            "https://preview.redd.it/abc.jpg?auto=webp&s=deadbeef",
            "https://external-preview.redd.it/abc",
            "https://media.i.imgur.com/abc.gif",
            # Generic hosts with an image extension
            "https://example.com/cat.jpg",
            "https://example.com/images/CAT.PNG",
            "https://example.com/pic.svg?w=800",
        ],
    )
    def test_image_urls(self, url: str) -> None:
        assert crawler_module.looks_like_image_url(url) is True

    @pytest.mark.parametrize(
        "url",
        [
            "https://example.com/page",
            "https://example.com/report.pdf",
            "https://example.com/photo.jpgx",
            # Viewer pages, not the image CDN hosts
            "https://imgur.com/trending",
            "https://www.reddit.com/r/pics/top/.json",
        ],
    )
    def test_non_image_urls(self, url: str) -> None:
        assert crawler_module.looks_like_image_url(url) is False


def _capture_download_headers(
    monkeypatch: pytest.MonkeyPatch, response: FakeResponse
) -> dict[str, str]:
    """Stub ssrf_safe_get and return the headers it was called with."""
    captured: dict[str, str] = {}

    def _fake_get(*_args: Any, **kwargs: Any) -> FakeResponse:
        captured.update(kwargs.get("headers") or {})
        return response

    monkeypatch.setattr(crawler_module, "ssrf_safe_get", _fake_get)
    return captured


def test_download_file_bytes_image_url_uses_image_headers(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Image URLs are fetched the way a browser <img> load is, because image
    CDNs (imgur, reddit) redirect navigation-style requests to HTML pages."""
    crawler = OnyxWebCrawler(playwright_fallback_enabled=False)
    headers = _capture_download_headers(monkeypatch, _jpeg_response())

    result = crawler.download_file_bytes("https://i.imgur.com/AKVZMd3b.jpg")

    assert isinstance(result, crawler_module.FetchedFile)
    assert headers["Accept"].startswith("image/")
    assert headers["Sec-Fetch-Dest"] == "image"


def test_download_file_bytes_page_url_uses_navigation_headers(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    crawler = OnyxWebCrawler(playwright_fallback_enabled=False)
    response = FakeResponse(
        status_code=200,
        headers={"Content-Type": "image/jpeg"},
        content=b"\xff\xd8\xff\xe0jpegdata",
    )
    headers = _capture_download_headers(monkeypatch, response)

    # Extension-less, non-image-host URL keeps the navigation-style request
    result = crawler.download_file_bytes("https://example.com/file")

    assert isinstance(result, crawler_module.FetchedFile)
    assert headers["Accept"].startswith("text/html")
    assert headers["Sec-Fetch-Dest"] == "document"


def test_download_file_bytes_image_playwright_render_also_html(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """When even the browser render of an image URL lands on an HTML page
    (the imgur/reddit viewer-page pattern for deleted images), the failure
    names the image problem instead of the generic not-a-file reason."""
    crawler = OnyxWebCrawler(playwright_fallback_enabled=True)
    response = FakeResponse(
        status_code=200,
        headers={"Content-Type": "text/html"},
        content=b"<html><body>viewer page</body></html>",
    )
    _capture_download_headers(monkeypatch, response)
    downloaded = crawler_module.DownloadedContent(
        content=b"<html><body>viewer page</body></html>",
        final_url="https://imgur.com/deletedimage",
        content_type="text/html",
        status=200,
    )
    monkeypatch.setattr(
        crawler_module,
        "fetch_content_bytes",
        lambda *args, **kwargs: downloaded,  # noqa: ARG005
    )

    result = crawler.download_file_bytes("https://i.imgur.com/deletedimage.jpg")

    assert isinstance(result, crawler_module.FailedFetch)
    assert result.failure_reason == crawler_module.FailureReason.IMAGE_NOT_AVAILABLE


def test_download_file_bytes_page_playwright_render_also_html(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Non-image URLs that end up HTML keep the generic not-a-file reason."""
    crawler = OnyxWebCrawler(playwright_fallback_enabled=True)
    response = FakeResponse(
        status_code=200,
        headers={"Content-Type": "text/html"},
        content=b"<html><body>page</body></html>",
    )
    _capture_download_headers(monkeypatch, response)
    downloaded = crawler_module.DownloadedContent(
        content=b"<html><body>page</body></html>",
        final_url="https://example.com/page",
        content_type="text/html",
        status=200,
    )
    monkeypatch.setattr(
        crawler_module,
        "fetch_content_bytes",
        lambda *args, **kwargs: downloaded,  # noqa: ARG005
    )

    result = crawler.download_file_bytes("https://example.com/page")

    assert isinstance(result, crawler_module.FailedFetch)
    assert result.failure_reason == crawler_module.FailureReason.HTML_NOT_FILE


def test_download_file_bytes_image_html_without_playwright(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """With the fallback disabled, an image URL that returns HTML still gets
    the image-specific reason."""
    crawler = OnyxWebCrawler(playwright_fallback_enabled=False)
    response = FakeResponse(
        status_code=200,
        headers={"Content-Type": "text/html"},
        content=b"<html><body>viewer page</body></html>",
    )
    _capture_download_headers(monkeypatch, response)

    result = crawler.download_file_bytes("https://i.imgur.com/deletedimage.jpg")

    assert isinstance(result, crawler_module.FailedFetch)
    assert result.failure_reason == crawler_module.FailureReason.IMAGE_NOT_AVAILABLE


def test_download_file_bytes_image_html_keeps_playwright_rescue(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A 200 HTML response for an image URL still retries through the
    headless browser, which rescues the file when the block is a bot
    challenge rather than a viewer page."""
    crawler = OnyxWebCrawler(playwright_fallback_enabled=True)
    response = FakeResponse(
        status_code=200,
        headers={"Content-Type": "text/html"},
        content=b"<html><body>challenge</body></html>",
    )
    _capture_download_headers(monkeypatch, response)
    downloaded = crawler_module.DownloadedContent(
        content=b"\xff\xd8\xff\xe0jpegdata",
        final_url="https://i.imgur.com/AKVZMd3b.jpg",
        content_type="image/jpeg",
        status=200,
    )
    monkeypatch.setattr(
        crawler_module,
        "fetch_content_bytes",
        lambda *args, **kwargs: downloaded,  # noqa: ARG005
    )

    result = crawler.download_file_bytes("https://i.imgur.com/AKVZMd3b.jpg")

    assert isinstance(result, crawler_module.FetchedFile)
    assert result.content == b"\xff\xd8\xff\xe0jpegdata"
