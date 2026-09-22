from __future__ import annotations

import threading
import time
from collections.abc import Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from urllib.parse import urlparse

import requests

from onyx.configs.app_configs import (
    OPEN_URL_FETCH_MODE,
    OPEN_URL_PLAYWRIGHT_FALLBACK_ENABLED,
    WEB_CRAWLER_USER_AGENT,
)
from onyx.file_processing.html_utils import (
    ParsedHTML,
    extract_image_urls,
    web_html_cleanup,
)
from onyx.server.security.models import outbound_allow_private_network
from onyx.server.security.store import get_security_settings
from onyx.tools.tool_implementations.open_url.models import (
    FailedFetch,
    WebContent,
    WebContentProvider,
)
from onyx.utils.logger import setup_logger
from onyx.utils.playwright_fetch import (
    DEFAULT_HEADERS,
    IMAGE_FETCH_HEADERS,
    DownloadedContent,
    RenderedPage,
    fetch_content_bytes,
    fetch_rendered_html,
    looks_like_cloudflare_challenge,
)
from onyx.utils.request_pacer import Pacer, get_default_pacer, provider_key
from onyx.utils.url import SSRFException, ssrf_safe_get
from onyx.utils.web_content import (
    decode_html_bytes,
    extract_pdf_text,
    is_pdf_resource,
    title_from_pdf_metadata,
    title_from_url,
)

logger = setup_logger()

# Fetch strategies for OnyxWebCrawler. "auto" keeps the Python-requests fast
# path (with the browser as fallback); "playwright" routes every fetch through
# the real browser. See OPEN_URL_FETCH_MODE.
FETCH_MODE_AUTO = "auto"
FETCH_MODE_PLAYWRIGHT = "playwright"
_FETCH_MODES = (FETCH_MODE_AUTO, FETCH_MODE_PLAYWRIGHT)

DEFAULT_READ_TIMEOUT_SECONDS = 15
DEFAULT_CONNECT_TIMEOUT_SECONDS = 5
# Browser-consistent UA for the Python-requests fast path (see
# WEB_CRAWLER_USER_AGENT). The old crawler-branded UA was an instant bot
# signal that poisoned the IP's reputation with antibot services.
DEFAULT_USER_AGENT = WEB_CRAWLER_USER_AGENT
DEFAULT_MAX_PDF_SIZE_BYTES = 50 * 1024 * 1024  # 50 MB
DEFAULT_MAX_HTML_SIZE_BYTES = 20 * 1024 * 1024  # 20 MB
DEFAULT_MAX_DOWNLOAD_SIZE_BYTES = 50 * 1024 * 1024  # 50 MB
DEFAULT_MAX_WORKERS = 5

# Headers that, when present on a 4xx response, signal that the upstream
# is a Cloudflare-style bot challenge (vs. a real auth/not-found error)
# and that retrying via a headless browser is likely to succeed.
_CLOUDFLARE_HEADER_NAMES = ("cf-ray", "cf-mitigated")


# Failure-reason strings surfaced to the LLM. Centralized so we don't drift
# wording across call sites and so the LLM sees consistent text to reason
# over (e.g. "don't bother retrying this URL").
class FailureReason:
    CLOUDFLARE_CHALLENGE = (
        "blocked by a Cloudflare bot challenge that the built-in crawler "
        "cannot solve — try a different URL or configure Firecrawl as the "
        "web content provider"
    )
    # Generic 403 with no Cloudflare evidence. Kept distinct from
    # CLOUDFLARE_CHALLENGE so we don't tell the LLM to "configure Firecrawl"
    # for what's actually an auth wall, expired presigned URL, private repo, etc.
    HTTP_403_BLOCKED = (
        "upstream returned HTTP 403 — the URL likely requires authentication "
        "or is otherwise restricted from the built-in crawler"
    )
    SSRF_BLOCKED = "blocked by SSRF protection (URL resolves to an internal address)"
    NETWORK_ERROR = "network error while fetching the URL"
    OVERSIZED_HTML = "HTML response exceeded the configured maximum size"
    OVERSIZED_PDF = "PDF response exceeded the configured maximum size"
    OVERSIZED_FILE = "file exceeded the configured maximum download size"
    DECODE_ERROR = "could not decode the response body"
    EMPTY_OR_UNPARSEABLE = "response could not be parsed into readable text"
    HTML_NOT_FILE = "the URL returned a web page, not a downloadable file"
    IMAGE_NOT_AVAILABLE = (
        "the image URL returned a web page instead of the image file — the "
        "image is likely deleted, private, or not directly downloadable"
    )

    @staticmethod
    def http_status(status_code: int) -> str:
        return f"upstream returned HTTP {status_code}"


def _failed_result(url: str, failure_reason: str | None = None) -> WebContent:
    return WebContent(
        title="",
        link=url,
        full_content="",
        published_date=None,
        scrape_successful=False,
        failure_reason=failure_reason,
    )


# Known-challenger memory: providers that recently served a bot challenge
# (Cloudflare interstitial, Reddit's network-security block, ...) to the
# Python-requests fast path. Those blocks are TLS-fingerprint-driven, so
# retrying requests against them is guaranteed to fail again — and every
# doomed attempt costs IP reputation with the antibot service. Within the
# TTL window we skip the fast path for such providers and go straight to
# the real-browser fetch.
_CHALLENGER_TTL_SECONDS = 1800.0
_challenger_last_seen: dict[str, float] = {}
_challenger_lock = threading.Lock()


def _remember_challenger(url: str) -> None:
    with _challenger_lock:
        _challenger_last_seen[provider_key(url)] = time.monotonic()


def _is_known_challenger(url: str) -> bool:
    key = provider_key(url)
    with _challenger_lock:
        seen = _challenger_last_seen.get(key)
    if seen is None:
        return False
    if time.monotonic() - seen > _CHALLENGER_TTL_SECONDS:
        with _challenger_lock:
            _challenger_last_seen.pop(key, None)
        return False
    return True


@dataclass
class FetchedFile:
    """Binary content fetched from a URL, with its declared content type."""

    content: bytes
    content_type: str | None


def primary_content_type(header_value: str | None) -> str | None:
    """Normalize a Content-Type header to its bare MIME type."""
    if not header_value:
        return None
    return header_value.split(";", 1)[0].strip().lower() or None


def sniff_mime_type(content: bytes) -> str | None:
    """Best-effort MIME detection from magic bytes, for servers that send a
    generic Content-Type."""
    if content.startswith(b"\x89PNG\r\n\x1a\n"):
        return "image/png"
    if content.startswith(b"\xff\xd8\xff"):
        return "image/jpeg"
    if content.startswith((b"GIF87a", b"GIF89a")):
        return "image/gif"
    if content.startswith(b"RIFF") and content[8:12] == b"WEBP":
        return "image/webp"
    if content.startswith(b"%PDF-"):
        return "application/pdf"
    return None


# Image CDN hosts that serve file bytes directly. Their URLs often carry no
# extension (e.g. https://i.imgur.com/AbCdEf), so the extension check alone
# is not enough.
_IMAGE_HOSTS = (
    "i.imgur.com",
    "i.redd.it",
    "preview.redd.it",
    "external-preview.redd.it",
    "i.redditmedia.com",
    "styles.redditmedia.com",
)

_IMAGE_EXTENSIONS = (
    ".avif",
    ".bmp",
    ".gif",
    ".ico",
    ".jpeg",
    ".jpg",
    ".png",
    ".svg",
    ".tiff",
    ".webp",
)


def looks_like_image_url(url: str) -> bool:
    """True when the URL points at a directly served image, by image CDN host
    or file extension. Used to fetch with an <img>-style request instead of a
    page-navigation one (see IMAGE_FETCH_HEADERS)."""
    parsed = urlparse(url)
    host = parsed.hostname or ""
    if any(
        host == image_host or host.endswith(f".{image_host}")
        for image_host in _IMAGE_HOSTS
    ):
        return True
    path = parsed.path.lower()
    return any(path.endswith(extension) for extension in _IMAGE_EXTENSIONS)


def has_cloudflare_signals(response: requests.Response) -> bool:
    """True iff the response carries actual Cloudflare-specific markers.

    Strict on purpose — only used to choose between the CF-specific failure
    reason (which tells admins to configure Firecrawl) vs. the generic 403
    failure reason (which points at auth / access). A bare 403 with no
    `cf-ray` / `cf-mitigated` / `Server: cloudflare` headers is treated
    as "not Cloudflare" here.
    """
    headers = response.headers
    if any(name in headers for name in _CLOUDFLARE_HEADER_NAMES):
        return True
    server = headers.get("Server", "").lower()
    return server.startswith("cloudflare")


def should_try_playwright_fallback(response: requests.Response) -> bool:
    """True if a Playwright render is plausibly worth attempting.

    Broader than `has_cloudflare_signals` — any 403 is cheap insurance to
    retry through a real browser (some sites serve JS-protected interstitials
    without CF headers). 429 is included because antibot WAFs rate-limit
    datacenter clients before serving a challenge. Real 401/404/410/5xx
    errors fall through unchanged.
    """
    return response.status_code >= 300 and (
        response.status_code in (403, 429) or has_cloudflare_signals(response)
    )


def failure_reason_for_status(status_code: int) -> str:
    """Pick the LLM-facing failure reason for a 4xx/5xx upstream response.

    A bare 403 without Cloudflare evidence is far more often an auth wall
    or an access-restricted resource than a bot challenge, so it gets the
    generic 403 reason instead of the Cloudflare one.
    """
    if status_code == 403:
        return FailureReason.HTTP_403_BLOCKED
    return FailureReason.http_status(status_code)


def _parse_html_to_web_content(url: str, html: str) -> WebContent:
    """Run cleanup on raw HTML and shape the result into a WebContent.

    Used by both the fast `requests` path and the Playwright fallback, so
    they emit identical-shape results.
    """
    try:
        parsed: ParsedHTML = web_html_cleanup(html)
        text_content = parsed.cleaned_text or ""
        title = parsed.title or ""
    except Exception as exc:
        logger.warning(
            "Onyx crawler failed to parse %s (%s)", url, exc.__class__.__name__
        )
        return _failed_result(url, FailureReason.EMPTY_OR_UNPARSEABLE)

    if not text_content.strip():
        return _failed_result(url, FailureReason.EMPTY_OR_UNPARSEABLE)

    return WebContent(
        title=title,
        link=url,
        full_content=text_content,
        published_date=None,
        scrape_successful=True,
        image_urls=extract_image_urls(html, url),
    )


class OnyxWebCrawler(WebContentProvider):
    """
    Built-in crawler that fetches web content and extracts readable text.
    Acts as the default content provider when no external crawler (e.g. Firecrawl)
    is configured.

    Two fetch strategies (OPEN_URL_FETCH_MODE / `fetch_mode`):

    - "auto" (default): a Python-requests fast path with browser-consistent
      headers tries first. On trouble it falls back to a real-browser fetch
      via `playwright_fetch`: Cloudflare/bot-challenge responses (HTTP 403,
      `cf-ray` / `cf-mitigated` headers), providers remembered from earlier
      challenges, and unparseable 200 bodies (Reddit-style JS shells). The
      fallback is controlled by `OPEN_URL_PLAYWRIGHT_FALLBACK_ENABLED`.

    - "playwright": every fetch goes through the real browser — no requests
      attempt at all. Maximum stealth for TLS-fingerprint-driven bot walls.
    """

    def __init__(
        self,
        *,
        timeout_seconds: int = DEFAULT_READ_TIMEOUT_SECONDS,
        connect_timeout_seconds: int = DEFAULT_CONNECT_TIMEOUT_SECONDS,
        user_agent: str = DEFAULT_USER_AGENT,
        max_pdf_size_bytes: int | None = None,
        max_html_size_bytes: int | None = None,
        playwright_fallback_enabled: bool = OPEN_URL_PLAYWRIGHT_FALLBACK_ENABLED,
        fetch_mode: str = OPEN_URL_FETCH_MODE,
        validate_ssrf: bool | None = None,
        pacer: Pacer | None = None,
    ) -> None:
        if fetch_mode not in _FETCH_MODES:
            raise ValueError(
                f"fetch_mode must be one of {_FETCH_MODES}, got '{fetch_mode}'"
            )
        self._read_timeout_seconds = timeout_seconds
        self._connect_timeout_seconds = connect_timeout_seconds
        self._max_pdf_size_bytes = max_pdf_size_bytes
        self._max_html_size_bytes = max_html_size_bytes
        self._playwright_fallback_enabled = playwright_fallback_enabled
        self._fetch_mode = fetch_mode
        # None => resolve from the admin SSRF Protection setting per fetch (see
        # _should_validate_ssrf); a non-None caller value pins it.
        self._validate_ssrf_override = validate_ssrf
        # Shared by default so every crawler in the process paces jointly.
        self._pacer = pacer if pacer is not None else get_default_pacer()
        self._headers = {**DEFAULT_HEADERS, "User-Agent": user_agent}

    def _should_validate_ssrf(self) -> bool:
        """Whether to enforce SSRF validation for this fetch. Resolved per
        request (like the MCP transport guard) so an admin SSRF Protection
        change takes effect on an already-constructed crawler. Validated on
        every level except DISABLED; the open_url path keeps its loopback floor
        even when disabled (it is LLM-controlled)."""
        if self._validate_ssrf_override is not None:
            return self._validate_ssrf_override
        return not outbound_allow_private_network(
            get_security_settings().ssrf_protection_level
        )

    def _ssrf_safe_get_with_retry(
        self, url: str, headers: dict[str, str] | None = None
    ) -> requests.Response:
        """Fast-path GET with one retry on transient network failures
        (connection resets, DNS hiccups — residential links see these).
        The retry is paced like any other same-provider request.
        """
        try:
            return self._ssrf_safe_get(url, headers)
        except SSRFException:
            raise
        except Exception as exc:
            logger.warning(
                "Fast-path fetch of %s failed once (%s); retrying",
                url,
                exc.__class__.__name__,
            )
        self._pacer.pace(url)
        return self._ssrf_safe_get(url, headers)

    def _ssrf_safe_get(
        self, url: str, headers: dict[str, str] | None = None
    ) -> requests.Response:
        return ssrf_safe_get(
            url,
            headers=headers if headers is not None else self._headers,
            timeout=(self._connect_timeout_seconds, self._read_timeout_seconds),
            allow_private_network=not self._should_validate_ssrf(),
        )

    def contents(self, urls: Sequence[str]) -> list[WebContent]:
        if not urls:
            return []

        max_workers = min(DEFAULT_MAX_WORKERS, len(urls))
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            return list(executor.map(self._fetch_url_safe, urls))

    def _fetch_url_safe(self, url: str) -> WebContent:
        """Wrapper that catches all exceptions so one bad URL doesn't kill the batch."""
        try:
            return self._fetch_url(url)
        except Exception as exc:
            logger.warning(
                "Onyx crawler unexpected error for %s (%s)",
                url,
                exc.__class__.__name__,
            )
            return _failed_result(url, FailureReason.NETWORK_ERROR)

    def _fetch_url(self, url: str) -> WebContent:
        self._pacer.pace(url)
        if self._fetch_mode == FETCH_MODE_PLAYWRIGHT:
            return self._fetch_url_via_browser(url)
        return self._fetch_url_requests_then_browser(url)

    def _fetch_url_via_browser(self, url: str) -> WebContent:
        """Browser-only page fetch (OPEN_URL_FETCH_MODE=playwright).

        Renders the page first (post-JS DOM — bot-walls and SPAs resolve like
        in a real browser), then falls back to a fetch-style bytes read for
        resources a renderer cannot represent (PDF, plain text).
        """
        rendered: RenderedPage | None = fetch_rendered_html(
            url, allow_private_network=not self._should_validate_ssrf()
        )
        if rendered is None:
            return _failed_result(url, FailureReason.NETWORK_ERROR)

        if (
            self._max_html_size_bytes is not None
            and len(rendered.html) > self._max_html_size_bytes
        ):
            logger.warning(
                "Rendered HTML too large (%d chars) for %s, max is %d",
                len(rendered.html),
                url,
                self._max_html_size_bytes,
            )
            return _failed_result(url, FailureReason.OVERSIZED_HTML)

        if looks_like_cloudflare_challenge(rendered.html):
            logger.info(
                "Browser fetch of %s landed on the Cloudflare challenge page; "
                "treating as Cloudflare failure",
                url,
            )
            return _failed_result(url, FailureReason.CLOUDFLARE_CHALLENGE)

        result = _parse_html_to_web_content(url, rendered.html)
        if result.scrape_successful:
            return result

        if rendered.status is not None and rendered.status >= 400:
            return _failed_result(url, failure_reason_for_status(rendered.status))

        # The renderer drew nothing readable: the URL may be a non-HTML
        # resource (PDF, plain text) that Chromium shows in a viewer shell.
        # Read the raw bytes fetch-style and handle the non-HTML cases.
        downloaded = fetch_content_bytes(
            url, allow_private_network=not self._should_validate_ssrf()
        )
        if downloaded is not None:
            content_type = primary_content_type(downloaded.content_type) or ""
            if is_pdf_resource(url, content_type, downloaded.content[:1024]):
                return self._handle_pdf_response(url, downloaded.content)
            if content_type.startswith("text/"):
                try:
                    decoded = decode_html_bytes(
                        downloaded.content, content_type=content_type
                    )
                except Exception:
                    decoded = None
                if decoded:
                    text_result = _parse_html_to_web_content(url, decoded)
                    if text_result.scrape_successful:
                        return text_result
        return result

    def _fetch_url_requests_then_browser(self, url: str) -> WebContent:
        """Default 'auto' strategy: requests fast path first, browser on trouble."""
        # Known challenge-heavy provider: skip the doomed requests attempt and
        # go straight to the real-browser fetch.
        if self._playwright_fallback_enabled and _is_known_challenger(url):
            logger.info("Skipping requests fast path for known challenger %s", url)
            fallback = self._fetch_via_playwright(url)
            if fallback is not None:
                return fallback
            return _failed_result(url, FailureReason.CLOUDFLARE_CHALLENGE)
        try:
            response = self._ssrf_safe_get_with_retry(url)
        except SSRFException as exc:
            logger.error(
                "SSRF protection blocked request to %s (%s)",
                url,
                exc.__class__.__name__,
            )
            return _failed_result(url, FailureReason.SSRF_BLOCKED)
        except Exception as exc:
            logger.warning(
                "Onyx crawler failed to fetch %s (%s)",
                url,
                exc.__class__.__name__,
            )
            return _failed_result(url, FailureReason.NETWORK_ERROR)

        if response.status_code >= 400:
            # Decide separately:
            #   - whether to attempt the Playwright fallback (broad, any 403
            #     is cheap insurance — some sites serve JS interstitials
            #     without CF-specific headers)
            #   - what failure reason to surface when nothing works (strict;
            #     only claim "Cloudflare" when we have actual CF evidence,
            #     either headers or a CF body returned by the render. A bare
            #     403 from e.g. a private GitHub repo or expired presigned
            #     S3 URL gets the generic-403 reason instead).
            has_cf_signals = has_cloudflare_signals(response)
            try_fallback = (
                self._playwright_fallback_enabled
                and should_try_playwright_fallback(response)
            )

            if try_fallback:
                # The fast path was challenge-blocked; remember it so later
                # fetches skip straight to the browser.
                _remember_challenger(url)
                logger.info(
                    "Onyx crawler got %s for %s; retrying via Playwright "
                    "(cf_signals=%s)",
                    response.status_code,
                    url,
                    has_cf_signals,
                )
                fallback = self._fetch_via_playwright(url)
                if fallback is not None:
                    # Either a successful render OR a definitive CF-challenge
                    # signal from the rendered body itself. Either way the
                    # fallback's own result is the truth.
                    return fallback

            logger.warning("Onyx crawler received %s for %s", response.status_code, url)
            return _failed_result(
                url,
                FailureReason.CLOUDFLARE_CHALLENGE
                if has_cf_signals
                else failure_reason_for_status(response.status_code),
            )

        content_type = response.headers.get("Content-Type", "")
        content = response.content

        content_sniff = content[:1024] if content else None
        if is_pdf_resource(url, content_type, content_sniff):
            return self._handle_pdf_response(url, content)

        if (
            self._max_html_size_bytes is not None
            and len(content) > self._max_html_size_bytes
        ):
            logger.warning(
                "HTML content too large (%d bytes) for %s, max is %d",
                len(content),
                url,
                self._max_html_size_bytes,
            )
            return _failed_result(url, FailureReason.OVERSIZED_HTML)

        try:
            decoded_html = decode_html_bytes(
                content,
                content_type=content_type,
                fallback_encoding=response.apparent_encoding or response.encoding,
            )
        except Exception as exc:
            logger.warning(
                "Onyx crawler failed to decode %s (%s)", url, exc.__class__.__name__
            )
            return _failed_result(url, FailureReason.DECODE_ERROR)

        result = _parse_html_to_web_content(url, decoded_html)
        if result.scrape_successful or not self._playwright_fallback_enabled:
            return result
        # Some bot-walls (e.g. Reddit) answer non-browser TLS with HTTP 200
        # and a JS-only shell: no error status triggers the fallback, but
        # there is nothing readable either. Try a real-browser render.
        logger.info(
            "Onyx crawler got an unparseable body for %s; retrying via Playwright",
            url,
        )
        fallback = self._fetch_via_playwright(url)
        if fallback is not None and fallback.scrape_successful:
            return fallback
        return result

    def _handle_pdf_response(self, url: str, content: bytes) -> WebContent:
        if (
            self._max_pdf_size_bytes is not None
            and len(content) > self._max_pdf_size_bytes
        ):
            logger.warning(
                "PDF content too large (%d bytes) for %s, max is %d",
                len(content),
                url,
                self._max_pdf_size_bytes,
            )
            return _failed_result(url, FailureReason.OVERSIZED_PDF)
        text_content, metadata = extract_pdf_text(content)
        title = title_from_pdf_metadata(metadata) or title_from_url(url)
        if not text_content.strip():
            return _failed_result(url, FailureReason.EMPTY_OR_UNPARSEABLE)
        return WebContent(
            title=title,
            link=url,
            full_content=text_content,
            published_date=None,
            scrape_successful=True,
        )

    def download_file_bytes(
        self,
        url: str,
        *,
        max_file_size_bytes: int = DEFAULT_MAX_DOWNLOAD_SIZE_BYTES,
    ) -> FetchedFile | FailedFetch:
        """Download binary content for a URL with bot-protection fallbacks.

        "auto" mode: SSRF-safe GET with browser-like headers first —
        navigation-style for page-like URLs, `<img>`-style for image URLs
        (image CDNs like imgur and reddit redirect navigation requests for
        direct image URLs to their HTML viewer pages). On 403 / Cloudflare
        signals (or any fallback-worthy 4xx), retries through the pooled
        real-browser fetch, which carries the browser's TLS fingerprint,
        User-Agent, and cookies.

        "playwright" mode: everything goes through the pooled real-browser
        fetch directly (fetch-style `context.request.get` with a page
        warm-up retry on challenges).

        Returns:
            FetchedFile on success. FailedFetch with an LLM-facing reason
            otherwise (auth walls, challenges, oversized files, network
            errors, or URLs that return a web page instead of a file).
        """
        # Image CDNs (imgur, reddit, ...) redirect navigation-style requests
        # for direct image URLs to their HTML viewer pages, which this method
        # rejects as HTML_NOT_FILE. Fetch image-like URLs the way a browser
        # <img> load does; those CDNs serve the bytes for that.
        self._pacer.pace(url)
        if self._fetch_mode == FETCH_MODE_PLAYWRIGHT:
            result = self._download_via_playwright(
                url, max_file_size_bytes=max_file_size_bytes
            )
            if result is None:
                # The browser fetch failed entirely (navigation hard-errored).
                return FailedFetch(url=url, failure_reason=FailureReason.NETWORK_ERROR)
            return result
        # Known challenge-heavy provider: the requests attempt would be
        # TLS-fingerprint-blocked again; fetch through the real browser.
        if self._playwright_fallback_enabled and _is_known_challenger(url):
            logger.info("Skipping requests fast path for known challenger %s", url)
            fallback = self._download_via_playwright(
                url, max_file_size_bytes=max_file_size_bytes
            )
            if fallback is not None:
                return fallback
            return FailedFetch(
                url=url, failure_reason=FailureReason.CLOUDFLARE_CHALLENGE
            )

        headers = IMAGE_FETCH_HEADERS if looks_like_image_url(url) else DEFAULT_HEADERS
        try:
            response = self._ssrf_safe_get_with_retry(url, headers=headers)
        except SSRFException:
            logger.error("SSRF protection blocked download of %s", url)
            return FailedFetch(url=url, failure_reason=FailureReason.SSRF_BLOCKED)
        except Exception as exc:
            logger.warning(
                "Download fetch failed for %s (%s)", url, exc.__class__.__name__
            )
            return FailedFetch(url=url, failure_reason=FailureReason.NETWORK_ERROR)

        content_type = primary_content_type(response.headers.get("Content-Type"))
        content = b""
        if response.status_code < 400:
            content = response.content
        else:
            has_cf_signals = has_cloudflare_signals(response)
            try_fallback = self._playwright_fallback_enabled and (
                should_try_playwright_fallback(response)
            )
            if try_fallback:
                # Fast path was challenge-blocked; remember the provider.
                _remember_challenger(url)
                logger.info(
                    "Download got HTTP %s for %s; retrying via Playwright",
                    response.status_code,
                    url,
                )
                fallback = self._download_via_playwright(
                    url, max_file_size_bytes=max_file_size_bytes
                )
                if fallback is not None:
                    return fallback
            return FailedFetch(
                url=url,
                failure_reason=FailureReason.CLOUDFLARE_CHALLENGE
                if has_cf_signals
                else failure_reason_for_status(response.status_code),
            )

        if len(content) > max_file_size_bytes:
            return FailedFetch(url=url, failure_reason=FailureReason.OVERSIZED_FILE)
        if not content:
            return FailedFetch(url=url, failure_reason=FailureReason.NETWORK_ERROR)
        if content_type and (
            content_type.startswith("text/html")
            or content_type == "application/xhtml+xml"
        ):
            # Antibot-protected hosts (imgur, reddit, ...) commonly answer
            # direct file fetches with an HTTP 200 HTML block page instead of
            # an error status, so this is the *most likely* failure mode for
            # major sites. First trust magic bytes over a mislabelled header;
            # then retry through a real browser before giving up.
            sniffed_type = sniff_mime_type(content)
            if sniffed_type:
                logger.info(
                    "Download of %s had %s Content-Type but binary magic bytes; "
                    "trusting the bytes",
                    url,
                    content_type,
                )
                return FetchedFile(content=content, content_type=sniffed_type)
            if self._playwright_fallback_enabled:
                logger.info(
                    "Download of %s got an HTML page with HTTP %s; retrying via "
                    "Playwright",
                    url,
                    response.status_code,
                )
                fallback = self._download_via_playwright(
                    url, max_file_size_bytes=max_file_size_bytes
                )
                if fallback is not None:
                    return fallback
            # For image URLs this is the imgur/reddit viewer-page pattern: the
            # CDN served its HTML page instead of the file, so the link points
            # at content the user cannot download directly.
            return FailedFetch(
                url=url,
                failure_reason=(
                    FailureReason.IMAGE_NOT_AVAILABLE
                    if looks_like_image_url(url)
                    else FailureReason.HTML_NOT_FILE
                ),
            )
        return FetchedFile(content=content, content_type=content_type)

    def _download_via_playwright(
        self, url: str, *, max_file_size_bytes: int
    ) -> FetchedFile | FailedFetch | None:
        """One-shot headless-Chromium binary fetch. Returns None when the
        fallback gave no new information (caller keeps its status-based reason).
        """
        # The browser fetch is a fresh request to the same provider, made
        # seconds after the fast-path attempt — pace it separately.
        self._pacer.pace(url)
        rendered_content: DownloadedContent | None = fetch_content_bytes(
            url, allow_private_network=not self._should_validate_ssrf()
        )
        if rendered_content is None:
            return None

        if len(rendered_content.content) > max_file_size_bytes:
            return FailedFetch(url=url, failure_reason=FailureReason.OVERSIZED_FILE)

        content_type = primary_content_type(rendered_content.content_type)
        if content_type and (
            content_type.startswith("text/html")
            or content_type == "application/xhtml+xml"
        ):
            # The challenge did not resolve (or the URL genuinely serves HTML).
            snippet = rendered_content.content[:4096].decode("utf-8", errors="ignore")
            if looks_like_cloudflare_challenge(snippet):
                _remember_challenger(url)
                return FailedFetch(
                    url=url, failure_reason=FailureReason.CLOUDFLARE_CHALLENGE
                )
            # For image URLs the render typically lands on the host's viewer
            # page (imgur/reddit pattern) — say that instead of the generic
            # not-a-file reason.
            return FailedFetch(
                url=url,
                failure_reason=(
                    FailureReason.IMAGE_NOT_AVAILABLE
                    if looks_like_image_url(url)
                    else FailureReason.HTML_NOT_FILE
                ),
            )

        return FetchedFile(content=rendered_content.content, content_type=content_type)

    def _fetch_via_playwright(self, url: str) -> WebContent | None:
        """Try a one-shot headless render.

        Returns:
            - Successful `WebContent` on success.
            - Failed `WebContent` with `failure_reason=CLOUDFLARE_CHALLENGE`
              when the render came back as the CF challenge interstitial
              itself (a definitive signal we can pass up regardless of what
              headers the original response carried).
            - `None` when the fallback gave us no new information (Chromium
              failed to launch, navigation hard-errored, content oversized,
              or rendered HTML didn't parse to anything). Caller should fall
              back to its own status-based failure reason.
        """
        # The browser fetch is a fresh request to the same provider, made
        # seconds after the fast-path attempt — pace it separately.
        self._pacer.pace(url)
        rendered: RenderedPage | None = fetch_rendered_html(
            url, allow_private_network=not self._should_validate_ssrf()
        )
        if rendered is None:
            return None

        if (
            self._max_html_size_bytes is not None
            and len(rendered.html) > self._max_html_size_bytes
        ):
            logger.warning(
                "Rendered HTML too large (%d chars) for %s, max is %d",
                len(rendered.html),
                url,
                self._max_html_size_bytes,
            )
            return None

        # If the render came back as a CF challenge interstitial, surface
        # that as a definitive CF failure (parsing it would just leak
        # "Just a moment..." text to the LLM). This is the one case where
        # Playwright actually adds information vs. the original 4xx.
        if looks_like_cloudflare_challenge(rendered.html):
            logger.info(
                "Playwright fallback rendered the Cloudflare challenge page "
                "itself for %s; treating as Cloudflare failure",
                url,
            )
            _remember_challenger(url)
            return _failed_result(url, FailureReason.CLOUDFLARE_CHALLENGE)

        result = _parse_html_to_web_content(url, rendered.html)
        if not result.scrape_successful:
            return None
        logger.info("Playwright fallback succeeded for %s", url)
        return result
