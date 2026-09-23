"""Playwright-based fetching helpers.

Centralizes browser-launch tuning and bot-detection-aware navigation logic.

Three consumers:
- `WebConnector` (long-lived `BrowserContext` reused across many pages
  in a single crawl) uses `start_playwright()` directly.
- `OnyxWebCrawler` (open_url / download_file / analyze_image tools) uses
  `fetch_rendered_html()` and `fetch_content_bytes()`, which run on the
  process-wide per-provider browser pool (`browser_pool`) so cookies —
  including solved-challenge clearance cookies — persist across fetches.

Fingerprint posture (what actually moves the needle with antibot services):
- A distro-packaged Chromium (apt/pacman build) is driven via
  `executable_path` instead of Playwright's bundled fork, which ships
  automation-friendly defaults detectors fingerprint. Point
  `CHROMIUM_EXECUTABLE_PATH` at the binary or rely on standard paths.
- Playwright's automation-flavored default launch args (`--enable-automation`,
  `--disable-component-update`, ...) are stripped via `ignore_default_args`.
- The browser runs headed under an auto-started Xvfb display when possible;
  headless Chromium is a strong bot signal even in "new" headless mode.
- UA and Client Hints derive from the real binary version, and the page-side
  init script aligns `navigator.platform`, WebGL vendor/renderer, plugins,
  and `userAgentData` with the claims (a single consistent identity rather
  than macOS-claims-on-Linux mismatches).
"""

import atexit
import os
import re
import shutil
import subprocess
import threading
import time
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from typing import Any, TypeVar

from playwright.sync_api import BrowserContext, Playwright, sync_playwright
from playwright.sync_api import TimeoutError as PlaywrightTimeoutError
from pydantic import BaseModel

from onyx.configs.app_configs import (
    CHROMIUM_EXECUTABLE_PATH,
    WEB_BROWSER_CHROME_MAJOR_VERSION,
    WEB_CONNECTOR_OAUTH_CLIENT_ID,
    WEB_CONNECTOR_OAUTH_CLIENT_SECRET,
    WEB_CONNECTOR_OAUTH_TOKEN_URL,
    WEB_CRAWLER_USER_AGENT,
    WEB_FETCH_HEADED,
)
from onyx.utils.logger import setup_logger

logger = setup_logger()

# Static (Python-requests fast path) headers. Claims Windows + the image's
# Chromium major so the UA and Client Hints tell one consistent story.
DEFAULT_USER_AGENT = WEB_CRAWLER_USER_AGENT

DEFAULT_HEADERS: dict[str, str] = {
    "User-Agent": DEFAULT_USER_AGENT,
    "Accept": (
        "text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,"
        "image/webp,image/apng,*/*;q=0.8,application/signed-exchange;v=b3;q=0.7"
    ),
    "Accept-Language": "en-US,en;q=0.9",
    # Brotli decoding has been flaky in brotlicffi/httpx for certain chunked responses;
    # stick to gzip/deflate to keep connectivity checks stable.
    "Accept-Encoding": "gzip, deflate",
    "Connection": "keep-alive",
    "Upgrade-Insecure-Requests": "1",
    "Sec-Fetch-Dest": "document",
    "Sec-Fetch-Mode": "navigate",
    "Sec-Fetch-Site": "none",
    "Sec-Fetch-User": "?1",
    "Sec-CH-UA": (
        f'"Chromium";v="{WEB_BROWSER_CHROME_MAJOR_VERSION}", '
        f'"Google Chrome";v="{WEB_BROWSER_CHROME_MAJOR_VERSION}", "Not:A-Brand";v="24"'
    ),
    "Sec-CH-UA-Mobile": "?0",
    "Sec-CH-UA-Platform": '"Windows"',
}

# Header set mirroring a browser <img> load. Image CDNs (imgur, reddit, ...)
# treat navigation-style requests to direct image URLs as page visits and
# redirect them to HTML viewer pages; an image-dest request gets the bytes.
IMAGE_FETCH_HEADERS: dict[str, str] = {
    "User-Agent": DEFAULT_USER_AGENT,
    "Accept": "image/avif,image/webp,image/apng,image/svg+xml,image/*,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.9",
    "Accept-Encoding": "gzip, deflate",
    "Connection": "keep-alive",
    "Sec-Fetch-Dest": "image",
    "Sec-Fetch-Mode": "no-cors",
    "Sec-Fetch-Site": "cross-site",
}

# Grace period after page navigation to allow bot-detection challenges
# (Cloudflare / Imperva / etc.) and SPA content rendering to complete.
DEFAULT_BOT_CHALLENGE_GRACE_MS = 5000

# Total per-navigation budget for Playwright `goto` / wait_for_load_state.
# Generous because we *want* to absorb a Cloudflare interstitial.
DEFAULT_NAVIGATION_TIMEOUT_MS = 30000

# Total budget for binary downloads (download_file). Covers the 50 MB
# download cap at ~430 KB/s; slower links are trickles worth aborting rather
# than waiting out — the in-sandbox equivalent of this failure was a run of
# 10-minute tool timeouts per download attempt.
BINARY_DOWNLOAD_TIMEOUT_MS = 120000

# Common distro Chromium locations (Debian/Arch/Fedora packages).
_CHROMIUM_CANDIDATE_PATHS = (
    "/usr/bin/chromium",
    "/usr/bin/chromium-browser",
    "/usr/bin/google-chrome-stable",
    "/usr/bin/google-chrome",
)

_XVFB_DISPLAY = ":99"
_XVFB_GEOMETRY = "1440x900x24"

_xvfb_process: subprocess.Popen | None = None
_xvfb_lock = threading.Lock()


def _discover_chromium() -> str | None:
    """Locate a Chromium binary to drive: explicit config, then distro paths.

    Returns None to let Playwright use its own bundled build.
    """
    if CHROMIUM_EXECUTABLE_PATH:
        if os.path.isfile(CHROMIUM_EXECUTABLE_PATH):
            return CHROMIUM_EXECUTABLE_PATH
        logger.warning(
            "CHROMIUM_EXECUTABLE_PATH=%s does not exist; falling back to "
            "auto-discovery",
            CHROMIUM_EXECUTABLE_PATH,
        )
    for candidate in _CHROMIUM_CANDIDATE_PATHS:
        if os.path.isfile(candidate):
            return candidate
    return None


def _ensure_display() -> str | None:
    """Return an X display for a headed browser, starting Xvfb if needed.

    Returns None when headed mode is impossible (no DISPLAY and no Xvfb), in
    which case the caller falls back to headless.
    """
    global _xvfb_process
    existing_display = os.environ.get("DISPLAY")
    if existing_display:
        return existing_display
    if not WEB_FETCH_HEADED:
        return None
    with _xvfb_lock:
        if _xvfb_process is not None and _xvfb_process.poll() is None:
            return _XVFB_DISPLAY
        xvfb = shutil.which("Xvfb")
        if xvfb is None:
            return None
        try:
            # The X11 socket directory is created by the x11-common package
            # at image build; recreate it when absent (ephemeral /tmp, odd
            # base images).
            os.makedirs("/tmp/.X11-unix", exist_ok=True)  # noqa: S108
            _xvfb_process = subprocess.Popen(
                [
                    xvfb,
                    _XVFB_DISPLAY,
                    "-screen",
                    "0",
                    _XVFB_GEOMETRY,
                    "-nolisten",
                    "tcp",
                ],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
        except OSError:
            logger.warning("Failed to start Xvfb; falling back to headless browser")
            return None
        # Standard X11 socket path convention (not application temp storage).
        socket_path = f"/tmp/.X11-unix/X{_XVFB_DISPLAY.lstrip(':')}"  # noqa: S108
        for _ in range(50):  # up to 5s for the display socket to appear
            if os.path.exists(socket_path):
                break
            time.sleep(0.1)
        atexit.register(_shutdown_xvfb)
        logger.info("Started Xvfb on %s for headed browser fetches", _XVFB_DISPLAY)
        return _XVFB_DISPLAY


def _shutdown_xvfb() -> None:
    global _xvfb_process
    with _xvfb_lock:
        if _xvfb_process is not None and _xvfb_process.poll() is None:
            _xvfb_process.terminate()
        _xvfb_process = None


# Playwright's default launch args tilt toward automation and test farms.
# Dropping these makes the launched browser arg-for-arg closer to a
# user-started one. (Entries Playwright doesn't pass are simply never
# matched, so the list is safe across Playwright versions.)
_OMIT_DEFAULT_ARGS = [
    "--enable-automation",
    "--disable-background-networking",
    "--disable-extensions",
    "--disable-dev-shm-usage",
    "--disable-default-apps",
    "--disable-component-update",
    "--disable-client-side-phishing-detection",
    "--disable-breakpad",
    "--disable-back-forward-cache",
    "--disable-backgrounding-occluded-windows",
    "--disable-background-timer-throttling",
    "--disable-component-extensions-with-background-pages",
    "--disable-ipc-flooding-protection",
    "--disable-popup-blocking",
    "--disable-prompt-on-repost",
    "--disable-renderer-backgrounding",
    "--use-mock-keychain",
    "--unsafely-disable-devtools-self-xss-warnings",
    "--password-store=basic",
    "--disable-search-engine-choice-screen",
    "--export-tagged-pdf",
    "--no-service-autorun",
    "--no-first-run",
    "--metrics-recording-only",
    "--force-color-profile=srgb",
    "--disable-hang-monitor",
    "--allow-pre-commit-input",
    "--disable-field-trial-config",
    (
        "--disable-features=AcceptCHFrame,AutoExpandDetailsElement,"
        "AvoidUnnecessaryBeforeUnloadCheckSync,"
        "CertificateTransparencyComponentUpdater,DestroyProfileOnBrowserClose,"
        "DialMediaRouteProvider,ExtensionManifestV2Disabled,"
        "GlobalMediaControls,HttpsUpgrades,ImprovedCookieControls,"
        "LazyFrameLoading,LensOverlay,MediaRouter,PaintHolding,"
        "ThirdPartyStoragePartitioning,Translate"
    ),
]

_LAUNCH_ARGS = [
    "--no-sandbox",
    "--disable-dev-shm-usage",
    "--disable-blink-features=AutomationControlled",
    "--disable-infobars",
]


def _stealth_init_script(chrome_major: str, chrome_full: str) -> str:
    """Init script aligning every JS-visible surface with a Windows Chrome
    `{chrome_major}` identity. Claims must match the UA/Client Hints set on
    the context."""
    return """
    Object.defineProperty(navigator, 'webdriver', {get: () => undefined});
    Object.defineProperty(navigator, 'languages', {get: () => ['en-US', 'en']});
    Object.defineProperty(navigator, 'platform', {get: () => 'Win32'});
    window.chrome = { runtime: {}, loadTimes: function () {}, csi: function () {} };
    const fakePlugin = (name) => {
        const p = { name, description: name,
                    filename: name.toLowerCase().replaceAll(' ', '_'), length: 1 };
        p[0] = { type: 'application/pdf', suffixes: 'pdf', description: name };
        return p;
    };
    Object.defineProperty(navigator, 'plugins', {
        get: () => {
            const arr = [fakePlugin('PDF Viewer'), fakePlugin('Chrome PDF Viewer'),
                         fakePlugin('Chromium PDF Viewer'),
                         fakePlugin('Microsoft Edge PDF Viewer'),
                         fakePlugin('WebKit built-in PDF')];
            arr.namedItem = (n) => arr.find(p => p.name === n) || null;
            arr.item = (i) => arr[i] || null;
            arr.refresh = () => {};
            return arr;
        }
    });
    Object.defineProperty(navigator, 'mimeTypes', {
        get: () => {
            const arr = [{ type: 'application/pdf', suffixes: 'pdf', description: '' }];
            arr.namedItem = (n) => arr.find(m => m.type === n) || null;
            arr.item = (i) => arr[i] || null;
            return arr;
        }
    });
    if (window.Notification) {
        Object.defineProperty(Notification, 'permission', { get: () => 'default' });
    }
    const patchGL = (proto) => {
        const orig = proto.getParameter;
        proto.getParameter = function (param) {
            // UNMASKED_VENDOR_WEBGL / UNMASKED_RENDERER_WEBGL: headless and VM
            // builds report SwiftShader/llvmpipe, a top bot signal.
            if (param === 37445) return 'Google Inc. (NVIDIA)';
            if (param === 37446) {
                return 'ANGLE (NVIDIA, NVIDIA GeForce GTX 1650 Direct3D11 vs_5_0 ps_5_0, D3D11)';
            }
            return orig.call(this, param);
        };
    };
    if (window.WebGLRenderingContext) patchGL(WebGLRenderingContext.prototype);
    if (window.WebGL2RenderingContext) patchGL(WebGL2RenderingContext.prototype);
    const brands = [
        { brand: 'Chromium', version: '__MAJOR__' },
        { brand: 'Google Chrome', version: '__MAJOR__' },
        { brand: 'Not:A-Brand', version: '24' },
    ];
    Object.defineProperty(navigator, 'userAgentData', {
        get: () => ({
            brands,
            mobile: false,
            platform: 'Windows',
            getHighEntropyValues: (hints) => Promise.resolve({
                architecture: 'x86',
                bitness: '64',
                model: '',
                mobile: false,
                platform: 'Windows',
                platformVersion: '15.0.0',
                uaFullVersion: '__FULL__',
                fullVersionList: brands,
                wow64: false,
            }),
            toJSON: () => ({ brands, mobile: false, platform: 'Windows' }),
        }),
    });
    """.replace("__MAJOR__", chrome_major).replace("__FULL__", chrome_full)


class RenderedPage(BaseModel):
    """Result of a successful Playwright navigation."""

    html: str
    final_url: str
    last_modified: str | None = None
    status: int | None = None


class DownloadedContent(BaseModel):
    """Binary content fetched via Playwright (bot-protected hosts)."""

    content: bytes
    final_url: str
    content_type: str | None = None
    status: int | None = None


def start_playwright() -> tuple[Playwright, BrowserContext]:
    """Launch a Playwright-driven Chromium that looks like a real browser.

    Prefers a distro-packaged Chromium over Playwright's bundled fork, strips
    Playwright's automation default args, and runs headed under Xvfb when a
    display is available (headless otherwise). The context's User-Agent and
    Client Hints derive from the launched binary's real version, so claims
    always match the actual engine.

    Used by both the long-lived web-connector crawl and (via the browser
    pool) the tool fetchers. Caller owns lifecycle and must call
    `context.close()` + `playwright.stop()` when done.
    """
    playwright = sync_playwright().start()

    executable_path = _discover_chromium()
    if executable_path is None:
        logger.warning(
            "No distro Chromium found (looked in %s) and CHROMIUM_EXECUTABLE_PATH "
            "is unset; using Playwright's bundled browser, which is easier for "
            "bot detectors to fingerprint",
            ", ".join(_CHROMIUM_CANDIDATE_PATHS),
        )

    display = _ensure_display()
    headed = display is not None
    env: dict[str, str | int | float] = dict(os.environ)
    if display:
        env["DISPLAY"] = display

    browser = playwright.chromium.launch(
        headless=not headed,
        executable_path=executable_path,
        ignore_default_args=_OMIT_DEFAULT_ARGS,
        args=_LAUNCH_ARGS,
        env=env,
    )

    # Build the claimed identity from the real binary version.
    chrome_full = browser.version  # e.g. "153.0.8010.52"
    chrome_major = chrome_full.split(".", 1)[0]
    user_agent = (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
        f"(KHTML, like Gecko) Chrome/{chrome_major}.0.0.0 Safari/537.36"
    )
    sec_ch_ua = (
        f'"Chromium";v="{chrome_major}", "Google Chrome";v="{chrome_major}", '
        '"Not:A-Brand";v="24"'
    )

    context = browser.new_context(
        user_agent=user_agent,
        viewport={"width": 1440, "height": 900},
        locale="en-US",
        timezone_id="America/Los_Angeles",
        has_touch=False,
        java_script_enabled=True,
        color_scheme="light",
        ignore_https_errors=True,
    )

    context.set_extra_http_headers(
        {
            "Accept": DEFAULT_HEADERS["Accept"],
            "Accept-Language": DEFAULT_HEADERS["Accept-Language"],
            "Sec-Fetch-Dest": DEFAULT_HEADERS["Sec-Fetch-Dest"],
            "Sec-Fetch-Mode": DEFAULT_HEADERS["Sec-Fetch-Mode"],
            "Sec-Fetch-Site": DEFAULT_HEADERS["Sec-Fetch-Site"],
            "Sec-Fetch-User": DEFAULT_HEADERS["Sec-Fetch-User"],
            "Sec-CH-UA": sec_ch_ua,
            "Sec-CH-UA-Mobile": DEFAULT_HEADERS["Sec-CH-UA-Mobile"],
            "Sec-CH-UA-Platform": DEFAULT_HEADERS["Sec-CH-UA-Platform"],
        }
    )

    context.add_init_script(_stealth_init_script(chrome_major, chrome_full))

    if (
        WEB_CONNECTOR_OAUTH_CLIENT_ID
        and WEB_CONNECTOR_OAUTH_CLIENT_SECRET
        and WEB_CONNECTOR_OAUTH_TOKEN_URL
    ):
        # Imported lazily so the OAuth deps don't get pulled in unless configured.
        from oauthlib.oauth2 import BackendApplicationClient
        from requests_oauthlib import OAuth2Session

        client = BackendApplicationClient(client_id=WEB_CONNECTOR_OAUTH_CLIENT_ID)
        oauth = OAuth2Session(client=client)
        token = oauth.fetch_token(
            token_url=WEB_CONNECTOR_OAUTH_TOKEN_URL,
            client_id=WEB_CONNECTOR_OAUTH_CLIENT_ID,
            client_secret=WEB_CONNECTOR_OAUTH_CLIENT_SECRET,
        )
        context.set_extra_http_headers(
            {"Authorization": "Bearer {}".format(token["access_token"])}
        )

    return playwright, context


@contextmanager
def playwright_session() -> Iterator[BrowserContext]:
    """Context-manager wrapper around `start_playwright()` for one-shot use.

    Yields a `BrowserContext` and guarantees both the context and the
    underlying `Playwright` instance are torn down when the `with` block
    exits, including when setup itself raises (e.g. missing Chromium binary).
    Use this for short-lived fetches that own their Playwright lifecycle
    end-to-end. Most fetchers should use the process-wide browser pool
    instead (see `browser_pool.get_browser_pool`).
    """
    playwright: Playwright | None = None
    context: BrowserContext | None = None
    try:
        playwright, context = start_playwright()
        yield context
    finally:
        if context is not None:
            try:
                context.close()
            except Exception:
                logger.debug("Failed to close Playwright context", exc_info=True)
        if playwright is not None:
            try:
                playwright.stop()
            except Exception:
                logger.debug("Failed to stop Playwright", exc_info=True)


def _looks_like_bot_challenge(
    status: int | None, cf_ray_header: str | None, cf_mitigated: str | None = None
) -> bool:
    """Heuristic: did this response look like a Cloudflare/Imperva challenge?

    We trigger the post-navigation grace period when *any* signal is
    present so JS challenges have time to resolve before we read the DOM.
    """
    if cf_ray_header is not None or status in (403, 429):
        return True
    return (cf_mitigated or "").lower() == "challenge"


# A Cloudflare challenge interstitial is identified by its page title and by
# Cloudflare-specific strings in the page body. Bare references to
# `challenges.cloudflare.com` / `/cdn-cgi/challenge-platform/` inside
# <script> tags are NOT signals: they appear on every Cloudflare-proxied
# page, including ones fetched successfully after the challenge resolved.
# Those tags are stripped before marker matching.
_CF_CHALLENGE_TITLE_RE = re.compile(
    r"<title[^>]*>(.*?)</title>", re.IGNORECASE | re.DOTALL
)
_CF_CHALLENGE_TITLE_MARKERS = (
    "just a moment...",
    "attention required",
    "please wait",
    "checking your browser",
    "verifying you are human",
)
_CF_CHALLENGE_BODY_MARKERS = (
    "challenges.cloudflare.com",
    "/cdn-cgi/challenge-platform/",
    "cf-chl-bypass",
    "Just a moment...",
    "Verifying you are human",
    'id="challenge-form"',
    'id="challenge-error-text"',
    'id="cf-challenge-running"',
    'id="challenge-running"',
    'class="cf-turnstile"',
)
_SCRIPT_BLOCK_RE = re.compile(r"<script\b.*?</script>", re.IGNORECASE | re.DOTALL)


def looks_like_cloudflare_challenge(html: str) -> bool:
    """Did this HTML come back as an unresolved Cloudflare challenge page?

    Used by callers to distinguish "we got real content" from "the render
    landed on the challenge interstitial because CF didn't let us through".
    The latter must NOT be returned to the LLM as if it were the page.
    """
    if not html:
        return False
    title_match = _CF_CHALLENGE_TITLE_RE.search(html)
    if title_match:
        title = title_match.group(1).strip().lower()
        if any(marker in title for marker in _CF_CHALLENGE_TITLE_MARKERS):
            return True
    body = _SCRIPT_BLOCK_RE.sub("", html)
    return any(marker in body for marker in _CF_CHALLENGE_BODY_MARKERS)


T = TypeVar("T")


def _run_on_pool(provider: str, job: Callable[[BrowserContext], T]) -> T:
    from onyx.utils.browser_pool import get_browser_pool

    return get_browser_pool().run(provider, job)


def _warm_up_and_retry_fetch(
    context: BrowserContext,
    url: str,
    *,
    navigation_timeout_ms: int,
    bot_challenge_grace_ms: int,
    headers: dict[str, str],
) -> tuple[Any, bytes] | None:
    """Navigate the challenge in a real page, then re-fetch fetch-style.

    The page visit lets the challenge JS run (and set clearance cookies on
    the context); the follow-up `context.request.get` reuses them. Returns
    (response, body) or None when the retry was not attempted.
    """
    try:
        page = context.new_page()
        try:
            response = page.goto(
                url, timeout=navigation_timeout_ms, wait_until="commit"
            )
            status = response.status if response else None
            cf_mitigated = response.header_value("cf-mitigated") if response else None
            cf_ray = response.header_value("cf-ray") if response else None
            if _looks_like_bot_challenge(status, cf_ray, cf_mitigated):
                page.wait_for_timeout(bot_challenge_grace_ms)
            try:
                page.wait_for_load_state("networkidle", timeout=bot_challenge_grace_ms)
            except PlaywrightTimeoutError:
                pass
        finally:
            page.close()
    except Exception:
        logger.warning("Challenge warm-up navigation failed for %s", url, exc_info=True)
        return None

    try:
        retry_response = context.request.get(
            url, headers=headers, timeout=navigation_timeout_ms
        )
        return retry_response, retry_response.body()
    except Exception:
        logger.warning("Post-warm-up refetch failed for %s", url, exc_info=True)
        return None


def _is_html_type(content_type: str | None) -> bool:
    if not content_type:
        return False
    bare = content_type.split(";", 1)[0].strip().lower()
    return bare in ("text/html", "application/xhtml+xml")


def _fetch_bytes_job(
    context: BrowserContext,
    url: str,
    *,
    navigation_timeout_ms: int,
    bot_challenge_grace_ms: int,
    image_accept: bool,
) -> DownloadedContent:
    """Fetch-style (browser TLS + cookies) binary fetch with a challenge
    warm-up retry. Runs on the provider's browser-pool lane thread."""
    headers = (
        {"Accept": IMAGE_FETCH_HEADERS["Accept"], "Accept-Language": "en-US,en;q=0.9"}
        if image_accept
        else {
            "Accept": DEFAULT_HEADERS["Accept"],
            "Accept-Language": DEFAULT_HEADERS["Accept-Language"],
        }
    )

    response = context.request.get(url, headers=headers, timeout=navigation_timeout_ms)
    status = response.status
    content_type = response.headers.get("content-type")
    body = response.body()

    needs_retry = _looks_like_bot_challenge(
        status, response.headers.get("cf-ray"), response.headers.get("cf-mitigated")
    )
    if (
        not needs_retry
        and image_accept
        and _is_html_type(content_type)
        and looks_like_cloudflare_challenge(
            body[:16384].decode("utf-8", errors="ignore")
        )
    ):
        needs_retry = True

    if needs_retry:
        retried = _warm_up_and_retry_fetch(
            context,
            url,
            navigation_timeout_ms=navigation_timeout_ms,
            bot_challenge_grace_ms=bot_challenge_grace_ms,
            headers=headers,
        )
        if retried is not None:
            response, body = retried
            status = response.status
            content_type = response.headers.get("content-type")

    return DownloadedContent(
        content=body,
        final_url=url,
        content_type=content_type,
        status=status,
    )


def fetch_content_bytes(
    url: str,
    *,
    navigation_timeout_ms: int = DEFAULT_NAVIGATION_TIMEOUT_MS,
    bot_challenge_grace_ms: int = DEFAULT_BOT_CHALLENGE_GRACE_MS,
    allow_private_network: bool = False,
) -> DownloadedContent | None:
    """Fetch raw bytes for a URL through a pooled real-browser context.

    Primary mechanism is a fetch-style `context.request.get` (browser TLS
    fingerprint + cookies, no page render). If that hits a bot challenge,
    the URL is first navigated in a real page so the challenge JS resolves,
    then the fetch is retried with the clearance cookies.

    Image-like URLs are requested with an <img>-style Accept header. Callers
    must check `content_type`: a challenge that did not resolve still
    returns a response, but with an HTML body.

    Runs on the per-provider browser-pool lane, so cookies persist across
    calls for the same provider.
    """
    from onyx.utils.request_pacer import provider_key
    from onyx.utils.url import SSRFException, validate_outbound_http_url

    # Playwright bypasses our `requests`-level SSRF protection, so revalidate
    # the URL here before letting the browser touch it.
    try:
        validate_outbound_http_url(
            url,
            allow_private_network=allow_private_network,
            block_loopback_and_link_local=True,
        )
    except (SSRFException, ValueError) as exc:
        logger.warning(
            "Refusing Playwright binary fetch for %s (%s)", url, exc.__class__.__name__
        )
        return None

    from onyx.tools.tool_implementations.open_url.onyx_web_crawler import (
        looks_like_image_url,
    )

    try:
        return _run_on_pool(
            provider_key(url),
            lambda context: _fetch_bytes_job(
                context,
                url,
                navigation_timeout_ms=navigation_timeout_ms,
                bot_challenge_grace_ms=bot_challenge_grace_ms,
                image_accept=looks_like_image_url(url),
            ),
        )
    except Exception as exc:
        msg = str(exc)
        if "Executable doesn't exist" in msg:
            # Friendlier message for the "no browser binary" footgun.
            logger.warning(
                "Playwright binary fetch unavailable for %s: no Chromium binary "
                "installed. Set CHROMIUM_EXECUTABLE_PATH or install one.",
                url,
            )
        else:
            logger.warning(
                "Playwright binary fetch failed for %s (%s: %s)",
                url,
                exc.__class__.__name__,
                msg.splitlines()[0] if msg else "",
            )
        return None


def _render_job(
    context: BrowserContext,
    url: str,
    *,
    navigation_timeout_ms: int,
    bot_challenge_grace_ms: int,
) -> RenderedPage:
    page = context.new_page()
    try:
        # Use "commit" instead of "domcontentloaded" to avoid hanging
        # on bot-detection pages that may never fire domcontentloaded.
        response = page.goto(url, timeout=navigation_timeout_ms, wait_until="commit")

        status = response.status if response else None
        cf_ray = response.header_value("cf-ray") if response else None
        cf_mitigated = response.header_value("cf-mitigated") if response else None

        if _looks_like_bot_challenge(status, cf_ray, cf_mitigated):
            page.wait_for_timeout(bot_challenge_grace_ms)

        # Best-effort wait for network to settle (SPA / CF challenge JS).
        try:
            page.wait_for_load_state("networkidle", timeout=bot_challenge_grace_ms)
        except PlaywrightTimeoutError:
            pass

        html = page.content()
        final_url = page.url
        last_modified = response.header_value("Last-Modified") if response else None
        return RenderedPage(
            html=html,
            final_url=final_url,
            last_modified=last_modified,
            status=status,
        )
    finally:
        page.close()


def fetch_rendered_html(
    url: str,
    *,
    navigation_timeout_ms: int = DEFAULT_NAVIGATION_TIMEOUT_MS,
    bot_challenge_grace_ms: int = DEFAULT_BOT_CHALLENGE_GRACE_MS,
    allow_private_network: bool = False,
) -> RenderedPage | None:
    """Render a single URL in the pooled real-browser context and return the
    final HTML.

    Uses the process-wide per-provider browser pool (persistent cookies —
    solved challenges benefit later fetches of the same provider).

    When ``allow_private_network`` is True, the private-IP guard is skipped
    so operators on trusted networks can render URLs that resolve to RFC1918
    addresses. Scheme/credential/blocked-hostname checks still apply.

    Returns:
        RenderedPage on success, or None if navigation failed entirely
        (including SSRF rejection of the URL). A non-None return with a
        4xx/5xx `status` is still possible — the caller can decide whether
        to use the rendered HTML (challenge pages often render real content
        after JS executes despite the original 4xx status code).
    """
    from onyx.utils.request_pacer import provider_key
    from onyx.utils.url import SSRFException, validate_outbound_http_url

    try:
        validate_outbound_http_url(
            url,
            allow_private_network=allow_private_network,
            block_loopback_and_link_local=True,
        )
    except (SSRFException, ValueError) as exc:
        logger.warning(
            "Refusing Playwright fallback for %s (%s)", url, exc.__class__.__name__
        )
        return None

    try:
        return _run_on_pool(
            provider_key(url),
            lambda context: _render_job(
                context,
                url,
                navigation_timeout_ms=navigation_timeout_ms,
                bot_challenge_grace_ms=bot_challenge_grace_ms,
            ),
        )
    except Exception as exc:
        msg = str(exc)
        if "Executable doesn't exist" in msg:
            # Friendlier message for the "no browser binary" footgun.
            logger.warning(
                "Playwright fallback unavailable for %s: no Chromium binary "
                "installed. Set CHROMIUM_EXECUTABLE_PATH or install one.",
                url,
            )
        else:
            logger.warning(
                "Playwright fallback failed to render %s (%s: %s)",
                url,
                exc.__class__.__name__,
                msg.splitlines()[0] if msg else "",
            )
        return None
