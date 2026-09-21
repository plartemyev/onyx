"""Per-provider pacing for outbound fetches.

Mirrors the SearXNG pacing proxy (deployment/docker_compose/pacer/pacer.py):
serialize requests and space them a random gap apart so providers do not see
bursts. Two differences: requests are grouped by provider (reddit, imgur, ...)
instead of a single global queue, and pacing runs in-process — a proxy in the
request path would re-resolve DNS outside `ssrf_safe_get`'s validation.
"""

import random
import threading
import time
from typing import Protocol
from urllib.parse import urlparse

from onyx.configs.app_configs import (
    REQUEST_PACING_ENABLED,
    REQUEST_PACING_MAX_GAP_SECONDS,
    REQUEST_PACING_MAX_WAIT_SECONDS,
    REQUEST_PACING_MIN_GAP_SECONDS,
)
from onyx.utils.logger import setup_logger

logger = setup_logger()

# CDN hosts pace together with their origin site even though the registrable
# domains differ. Keyed by registrable domain, value is the shared bucket.
_PROVIDER_ALIASES: dict[str, str] = {
    "redd.it": "reddit.com",
    "redditmedia.com": "reddit.com",
    "redditstatic.com": "reddit.com",
    "twitter.com": "x.com",
}

# Two-level public suffixes that would otherwise collapse unrelated sites
# (everything under *.co.uk would share one bucket without this).
_TWO_LEVEL_SUFFIXES: frozenset[str] = frozenset(
    {
        "co.uk",
        "org.uk",
        "ac.uk",
        "gov.uk",
        "co.jp",
        "or.jp",
        "ne.jp",
        "com.au",
        "net.au",
        "org.au",
        "com.br",
        "com.mx",
        "co.in",
        "com.cn",
        "com.tr",
    }
)


def provider_key(url: str) -> str:
    """Bucket a URL by service provider: registrable domain (eTLD+1, with a
    small two-level-suffix list), plus aliases so CDN hosts pace together
    with their origin (i.redd.it -> reddit.com)."""
    host = (urlparse(url).hostname or "").lower().rstrip(".")
    labels = host.split(".") if host else []
    if len(labels) >= 3 and ".".join(labels[-2:]) in _TWO_LEVEL_SUFFIXES:
        registrable = ".".join(labels[-3:])
    elif len(labels) >= 2:
        registrable = ".".join(labels[-2:])
    else:
        registrable = host
    return _PROVIDER_ALIASES.get(registrable, registrable) or "unknown"


class Pacer(Protocol):
    def pace(self, url: str) -> None: ...


class DomainPacer:
    """Space out requests sharing a provider; different providers never block.

    Each caller of `pace` queues on a per-provider lock and starts its request
    a uniform random gap after the previous same-provider request start. The
    first request to a provider starts immediately. The wait is capped so a
    same-provider pile-up cannot stall a caller forever.
    """

    def __init__(
        self,
        *,
        min_gap_seconds: float = REQUEST_PACING_MIN_GAP_SECONDS,
        max_gap_seconds: float = REQUEST_PACING_MAX_GAP_SECONDS,
        max_wait_seconds: float = REQUEST_PACING_MAX_WAIT_SECONDS,
    ) -> None:
        if min_gap_seconds < 0 or max_gap_seconds < min_gap_seconds:
            raise ValueError(
                "require 0 <= min_gap_seconds <= max_gap_seconds, got "
                f"{min_gap_seconds}..{max_gap_seconds}"
            )
        self._min_gap_seconds = min_gap_seconds
        self._max_gap_seconds = max_gap_seconds
        self._max_wait_seconds = max_wait_seconds
        self._gates: dict[str, threading.Lock] = {}
        self._gates_lock = threading.Lock()
        self._last_start: dict[str, float] = {}

    def _gate(self, key: str) -> threading.Lock:
        with self._gates_lock:
            gate = self._gates.get(key)
            if gate is None:
                gate = threading.Lock()
                self._gates[key] = gate
            return gate

    def pace(self, url: str) -> None:
        """Block until `url` may be requested, respecting the per-provider gap."""
        key = provider_key(url)
        with self._gate(key):
            now = time.monotonic()
            gap = random.uniform(self._min_gap_seconds, self._max_gap_seconds)
            target = max(now, self._last_start.get(key, 0.0) + gap)
            wait = min(target - now, self._max_wait_seconds)
            if wait > 0:
                time.sleep(wait)
            # Record the actual start, so a capped wait shortens the next gap.
            self._last_start[key] = now + wait
            if wait >= 1.0:
                logger.debug("Request pacer: %.1fs wait for %s", wait, key)


class NullPacer:
    """No-op pacer (pacing disabled, or unit tests)."""

    def pace(self, url: str) -> None:  # noqa: ARG002
        return None


_pacer: Pacer | None = None
_pacer_lock = threading.Lock()


def get_default_pacer() -> Pacer:
    """Process-wide pacer shared by every crawler, so all tools pace jointly."""
    global _pacer
    with _pacer_lock:
        if _pacer is None:
            _pacer = DomainPacer() if REQUEST_PACING_ENABLED else NullPacer()
        return _pacer
