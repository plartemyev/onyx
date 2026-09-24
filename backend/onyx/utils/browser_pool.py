"""Per-provider pooled browser contexts for Playwright fetches.

One browser per provider (reddit.com, imgur.com, ...) is kept alive across
fetches so cookies persist: a Cloudflare/Reddit JS challenge solved on the
first visit sets clearance cookies that every later same-provider fetch
reuses. This is both faster (no per-URL browser launch) and less suspicious
(a "browser" that keeps its session looks human; ten fresh browsers hitting
ten pages in a burst does not).

Playwright's sync API is thread-affine — its objects must only be touched by
the thread that started them. Each lane therefore owns a dedicated worker
thread: callers submit a job (a callable receiving the BrowserContext) and
block on its Future. All Playwright operations for a lane run on that lane's
thread, sequentially, so jobs to one provider queue up behind each other
while different providers run in parallel.

Lanes cap out at MAX_LANES (least-recently-used is retired) and self-exit
after LANE_IDLE_TTL_SECONDS without work. A lane whose browser process
died (OOM kill, crash) is rebuilt before the next job is served, and any
job that still hits the dead browser tears the lane down for the same
reason.
"""

from __future__ import annotations

import atexit
import queue
import threading
import time
from collections.abc import Callable
from concurrent.futures import Future
from typing import Any, Protocol, TypeVar

from onyx.utils.logger import setup_logger

logger = setup_logger()

T = TypeVar("T")


class BrowserSession(Protocol):
    """What a lane needs from a browser session holder."""

    context: Any

    def close(self) -> None: ...

    def is_alive(self) -> bool: ...


class _Shutdown:
    """Sentinel telling a lane thread to tear down and exit."""


_SHUTDOWN = _Shutdown()


class PooledBrowserSession:
    """One Playwright instance + BrowserContext owned by a lane."""

    def __init__(self, context_factory: Callable[[], tuple[Any, Any]]) -> None:
        self.playwright, self.context = context_factory()

    @classmethod
    def from_start_playwright(cls) -> PooledBrowserSession:
        from onyx.utils.playwright_fetch import start_playwright

        return cls(start_playwright)

    def is_alive(self) -> bool:
        """False once the underlying browser process has gone away.

        `BrowserContext.browser` is None only for contexts created outside a
        normal browser (Android/Electron), never for ours.
        """
        browser = self.context.browser
        return browser is not None and browser.is_connected()

    def close(self) -> None:
        try:
            self.context.close()
        except Exception:
            logger.debug("Failed to close pooled browser context", exc_info=True)
        try:
            self.playwright.stop()
        except Exception:
            logger.debug("Failed to stop pooled Playwright", exc_info=True)


# How long an idle lane stays alive before its browser is torn down.
LANE_IDLE_TTL_SECONDS = 600.0
# Most-recently-used lanes kept alive at once. Each lane is one Chromium
# process; three covers a typical multi-provider download batch.
MAX_LANES = 3
# Upper bound for a single job (navigation + grace + retries stay well below).
JOB_TIMEOUT_SECONDS = 120.0

_POLL_INTERVAL_SECONDS = 2.0


class _BrowserLane:
    """A worker thread owning one Playwright instance + BrowserContext."""

    def __init__(self, provider: str, factory: Callable[[], BrowserSession]) -> None:
        self.provider = provider
        self._factory = factory
        self._jobs: queue.Queue[
            tuple[Callable[[Any], Any], Future[Any]] | _Shutdown
        ] = queue.Queue()
        # Serializes callers within one lane: each job runs to completion
        # before the next same-provider job starts.
        self._lock = threading.Lock()
        self._thread: threading.Thread | None = None
        self._last_used = time.monotonic()
        self._session: BrowserSession | None = None

    def run(self, job: Callable[[Any], T]) -> T:
        """Run `job(context)` on the lane thread and return its result.

        Same-provider jobs are serialized (by `_lock`); the browser context
        and its cookies persist across jobs.
        """
        with self._lock:
            self._ensure_thread()
            future: Future[T] = Future()
            self._jobs.put((job, future))
            self._last_used = time.monotonic()
            return future.result(timeout=JOB_TIMEOUT_SECONDS)

    def is_alive(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    def last_used(self) -> float:
        return self._last_used

    def shutdown(self) -> None:
        """Ask the lane to exit; non-blocking (teardown happens on its thread)."""
        if self.is_alive():
            self._jobs.put(_SHUTDOWN)

    def _ensure_thread(self) -> None:
        if self.is_alive():
            return
        # Thread died (idle TTL or fatal error) — start a fresh one with a
        # fresh browser. Any stale session handle is discarded.
        self._session = None
        self._thread = threading.Thread(
            target=self._worker, name=f"browser-lane-{self.provider}", daemon=True
        )
        self._thread.start()

    def _worker(self) -> None:
        deadline = time.monotonic() + LANE_IDLE_TTL_SECONDS
        while True:
            try:
                item = self._jobs.get(timeout=_POLL_INTERVAL_SECONDS)
            except queue.Empty:
                if time.monotonic() >= deadline:
                    self._teardown()
                    return
                continue

            if isinstance(item, _Shutdown):
                self._teardown()
                return

            job, future = item
            deadline = time.monotonic() + LANE_IDLE_TTL_SECONDS
            try:
                if self._session is None:
                    self._session = self._factory()
                elif not self._session.is_alive():
                    # The browser process died between jobs (OOM kill,
                    # driver crash). Rebuild before serving so the caller
                    # does not get a request failed against dead handles.
                    logger.warning(
                        "Browser lane %s browser process is gone; rebuilding",
                        self.provider,
                    )
                    self._teardown()
                    self._session = self._factory()
                future.set_result(job(self._session.context))
            except BaseException as exc:  # noqa: BLE001 — surfaced to the caller
                future.set_exception(exc)
                # A failed session build or a dead browser poisons every later
                # job; rebuild from scratch on the next checkout.
                if self._session is None or _is_fatal_browser_error(exc):
                    logger.warning(
                        "Browser lane %s is unhealthy (%s); rebuilding",
                        self.provider,
                        exc.__class__.__name__,
                    )
                    self._teardown()

    def _teardown(self) -> None:
        session = self._session
        self._session = None
        if session is None:
            return
        session.close()


def _is_fatal_browser_error(exc: BaseException) -> bool:
    """Did this exception likely leave the browser dead?"""
    name = exc.__class__.__name__
    if name in ("Error", "TargetClosedError"):  # playwright sync API errors
        return True
    return "Target closed" in str(exc) or "Browser closed" in str(exc)


class BrowserPool:
    """Provider-keyed lanes with an LRU cap."""

    def __init__(
        self,
        session_factory: Callable[[], BrowserSession],
        *,
        max_lanes: int = MAX_LANES,
    ) -> None:
        self._session_factory = session_factory
        self._max_lanes = max_lanes
        self._lanes: dict[str, _BrowserLane] = {}
        self._registry_lock = threading.Lock()

    def run(self, provider: str, job: Callable[[Any], T]) -> T:
        """Run `job(context)` on the provider's lane."""
        lane = self._get_lane(provider)
        try:
            return lane.run(job)
        finally:
            self._maybe_retire_lanes()

    def _get_lane(self, provider: str) -> _BrowserLane:
        with self._registry_lock:
            lane = self._lanes.get(provider)
            if lane is None or not lane.is_alive():
                lane = _BrowserLane(provider, self._session_factory)
                self._lanes[provider] = lane
            return lane

    def _maybe_retire_lanes(self) -> None:
        with self._registry_lock:
            live = [lane for lane in self._lanes.values() if lane.is_alive()]
            if len(live) <= self._max_lanes:
                return
            live.sort(key=lambda lane: lane.last_used())
            for lane in live[: len(live) - self._max_lanes]:
                lane.shutdown()
                del self._lanes[lane.provider]

    def close_all(self) -> None:
        with self._registry_lock:
            for lane in self._lanes.values():
                lane.shutdown()
            self._lanes.clear()


_pool: BrowserPool | None = None
_pool_lock = threading.Lock()


def get_browser_pool() -> BrowserPool:
    """Process-wide pool shared by every fetcher, so cookies persist across
    tool calls and connectors."""
    global _pool
    with _pool_lock:
        if _pool is None:
            pool = BrowserPool(PooledBrowserSession.from_start_playwright)
            atexit.register(_shutdown_pool)
            _pool = pool
        return _pool


def _shutdown_pool() -> None:
    global _pool
    with _pool_lock:
        if _pool is not None:
            _pool.close_all()
            _pool = None
