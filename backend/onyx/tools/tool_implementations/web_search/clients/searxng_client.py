from urllib.parse import urljoin

import requests
from fastapi import HTTPException

from onyx.configs.chat_configs import (
    SEARXNG_CONNECT_TIMEOUT_SECONDS,
    SEARXNG_READ_TIMEOUT_SECONDS,
)
from onyx.tools.tool_implementations.web_search.models import (
    WebSearchProvider,
    WebSearchResult,
)
from onyx.utils.logger import setup_logger
from onyx.utils.retry_wrapper import retry_builder

logger = setup_logger()

# (connect, read). The read side must cover pacing-proxy queue wait plus a
# browser-backed SearXNG search (see chat_configs for tuning).
_SEARXNG_TIMEOUT_SECONDS = (
    SEARXNG_CONNECT_TIMEOUT_SECONDS,
    SEARXNG_READ_TIMEOUT_SECONDS,
)

# Max results returned by an image-only search
_MAX_IMAGE_RESULTS = 5


def _absolute_http_url(url: str, base_url: str) -> str | None:
    """Resolve `url` against `base_url` and keep only http(s) URLs."""
    if not url:
        return None
    absolute_url = urljoin(base_url, url.strip())
    if not absolute_url.startswith(("http://", "https://")):
        return None
    return absolute_url


class SearXNGClient(WebSearchProvider):
    def __init__(
        self,
        searxng_base_url: str,
        num_results: int = 10,
        language: str | None = None,
        enable_image_search: bool = True,
    ) -> None:
        logger.debug("Initializing SearXNGClient with base URL: %s", searxng_base_url)
        self._searxng_base_url = searxng_base_url
        self._num_results = num_results
        # Optional SearXNG UI language / locale (e.g. "en", "en-US", "de").
        # Without it, results follow the instance's own configured language,
        # which can localize snippets (and skew engines) unexpectedly.
        # Serves as the default for `search(language=None)` calls.
        self._language = language or None
        # Image searches cost a second full search per query. On instances
        # that fetch engines through a browser pool (challenge solving,
        # human-paced requests), that doubles the load the instance must
        # serve for every query, so image search is a separate, opt-in
        # request the agent makes explicitly; this flag gates the
        # capability entirely.
        self._enable_image_search = enable_image_search

    @property
    def supports_image_search(self) -> bool:
        return self._enable_image_search

    # Paced retries: a challenge-hit search fails on the SearXNG side while
    # the lane keeps solving in the background. Re-querying 20s/40s later
    # (a person refining a search) gives that clearance time to land; a
    # 1-2s machine burst would just re-lose. Worst total ~3x read timeout.
    @retry_builder(tries=3, delay=20, backoff=2)
    def search(self, query: str, language: str | None = None) -> list[WebSearchResult]:
        """Standard search: web results only, never image results."""
        payload = {
            "q": query,
            "format": "json",
        }
        # A per-call language (from the agent's tool call) wins over the
        # instance-level default.
        language = language or self._language
        if language:
            payload["language"] = language
        logger.debug(
            "Searching with payload: %s to %s/search", payload, self._searxng_base_url
        )
        response = requests.post(
            f"{self._searxng_base_url}/search",
            data=payload,
            timeout=_SEARXNG_TIMEOUT_SECONDS,
        )
        response.raise_for_status()

        results = response.json()
        result_list = results.get("results", [])
        if not result_list and results.get("unresponsive_engines"):
            # a challenge-hit search comes back HTTP 200 with zero results:
            # surface it so the paced retry can pick up the lane's banked
            # clearance a couple of human re-query gaps later
            raise requests.RequestException(
                "SearXNG returned no results; unresponsive engines: "
                f"{results.get('unresponsive_engines')}"
            )
        # SearXNG doesn't support limiting results via API parameters,
        # so we limit client-side after receiving the response
        limited_results = result_list[: self._num_results]
        return [self._parse_web_result(result) for result in limited_results]

    def _parse_web_result(self, result: dict) -> WebSearchResult:
        """Build a WebSearchResult from a general (web) search hit.

        Attaches the hit's `img_src` thumbnail as a direct image URL when present.
        """
        image_url = _absolute_http_url(result.get("img_src") or "", result["url"])
        return WebSearchResult(
            title=result["title"],
            link=result["url"],
            snippet=result["content"],
            image_urls=[image_url] if image_url else [],
        )

    def search_images(
        self, query: str, language: str | None = None
    ) -> list[WebSearchResult]:
        """Image-only search: every result carries a direct image URL in
        `image_urls` and `link` points at the source page. Raises on
        failure so the paced retry and the tool's error capture apply —
        in this mode the images request is the whole result."""
        if not self._enable_image_search:
            raise ValueError("Image search is disabled for this SearXNG provider.")
        return self._search_images_with_retries(query, language)

    @retry_builder(tries=3, delay=20, backoff=2)
    def _search_images_with_retries(
        self, query: str, language: str | None = None
    ) -> list[WebSearchResult]:
        payload = {
            "q": query,
            "format": "json",
            "categories": "images",
        }
        if language or self._language:
            payload["language"] = language or self._language
        response = requests.post(
            f"{self._searxng_base_url}/search",
            data=payload,
            timeout=_SEARXNG_TIMEOUT_SECONDS,
        )
        response.raise_for_status()
        results = response.json()
        result_list = results.get("results", [])
        if not result_list and results.get("unresponsive_engines"):
            # parity with general search: surface challenge hits so the
            # paced retry can pick up the lane's banked clearance
            raise requests.RequestException(
                "SearXNG image search returned no results; unresponsive engines: "
                f"{results.get('unresponsive_engines')}"
            )

        image_results: list[WebSearchResult] = []
        seen_image_urls: set[str] = set()
        for result in result_list:
            if len(image_results) >= _MAX_IMAGE_RESULTS:
                break
            source_url = result.get("url") or ""
            image_url = _absolute_http_url(
                result.get("img_src") or result.get("thumbnail_src") or "",
                source_url,
            )
            if not image_url or image_url in seen_image_urls:
                continue
            seen_image_urls.add(image_url)
            snippet_parts = [
                str(part)
                for part in (result.get("resolution"), result.get("content"))
                if part
            ]
            image_results.append(
                WebSearchResult(
                    title=result.get("title") or "",
                    link=source_url or image_url,
                    snippet=" | ".join(snippet_parts),
                    image_urls=[image_url],
                )
            )
        return image_results

    def test_connection(self) -> dict[str, str]:
        try:
            logger.debug("Testing connection to %s/config", self._searxng_base_url)
            response = requests.get(
                f"{self._searxng_base_url}/config",
                timeout=_SEARXNG_TIMEOUT_SECONDS,
            )
            logger.debug("Response: %s, text: %s", response.status_code, response.text)
            response.raise_for_status()
        except requests.HTTPError as e:
            status_code = e.response.status_code
            logger.debug(
                "HTTPError: status_code=%s, e.response=%s, error=%s",
                status_code,
                e.response.status_code if e.response else None,
                e,
            )
            if status_code == 429:
                raise HTTPException(
                    status_code=400,
                    detail=(
                        "This SearXNG instance does not allow API requests. "
                        "Use a private instance and configure it to allow bots."
                    ),
                ) from e
            elif status_code == 404:
                raise HTTPException(
                    status_code=400,
                    detail="This SearXNG instance was not found. Please check the URL and try again.",
                ) from e
            else:
                raise HTTPException(
                    status_code=400,
                    detail=f"SearXNG connection failed (status {status_code}): {str(e)}",
                ) from e

        # Not a sure way to check if this is a SearXNG instance as opposed to some other website that
        # happens to have a /config endpoint containing a "brand" key with a "GIT_URL" key with value
        # "https://github.com/searxng/searxng". I don't think that would happen by coincidence, so I
        # think this is a good enough check for now. I'm open for suggestions on improvements.
        config = response.json()
        if (
            config.get("brand", {}).get("GIT_URL")
            != "https://github.com/searxng/searxng"
        ):
            raise HTTPException(
                status_code=400,
                detail="This does not appear to be a SearXNG instance. Please check the URL and try again.",
            )

        # Test that JSON mode is enabled by performing a simple search
        self._test_json_mode()

        logger.info("Web search provider test succeeded for SearXNG.")
        return {"status": "ok"}

    def _test_json_mode(self) -> None:
        """Test that JSON format is enabled in SearXNG settings.

        SearXNG requires JSON format to be explicitly enabled in settings.yml.
        If it's not enabled, the search endpoint returns a 403.
        """
        try:
            payload = {
                "q": "test",
                "format": "json",
            }
            response = requests.post(
                f"{self._searxng_base_url}/search",
                data=payload,
                timeout=_SEARXNG_TIMEOUT_SECONDS,
            )
            response.raise_for_status()
        except requests.HTTPError as e:
            status_code = e.response.status_code if e.response is not None else None
            if status_code == 403:
                raise HTTPException(
                    status_code=400,
                    detail=(
                        "Got a 403 response when trying to reach SearXNG. This likely means that "
                        "JSON format is not enabled on this SearXNG instance. "
                        "Please enable JSON format in your SearXNG settings.yml file by adding "
                        "'json' to the 'search.formats' list."
                    ),
                ) from e
            raise HTTPException(
                status_code=400,
                detail=f"Failed to test search on SearXNG instance (status {status_code}): {str(e)}",
            ) from e
