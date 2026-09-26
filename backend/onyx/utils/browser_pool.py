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
import fcntl
import os
import queue
import threading
import time
import zlib
from collections.abc import Callable
from concurrent.futures import Future
from typing import Any, Protocol, TypeVar

from onyx.configs.app_configs import (
    BROWSER_PROFILE_DIR,
    BROWSER_PROFILE_FIRST_INDEX,
    BROWSER_PROFILE_LANES,
)
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


class _ProfileLease:
    """Exclusive use of one shared persistent-profile lane.

    The profile dir may be shared with another browser operator (e.g. the
    SearXNG browser lanes on a common docker volume). Chromium keeps a
    SingletonLock per profile, so two browsers on one dir would corrupt it;
    an flock on a sibling `lane-<N>.lock` file makes the allocation
    exclusive across processes instead. A lease whose `user_data_dir` is
    None means "no lane was free" — the session then runs an ephemeral
    context, same as the disabled default.
    """

    __slots__ = ("user_data_dir", "_handle")

    def __init__(self, user_data_dir: str | None, handle: Any = None) -> None:
        self.user_data_dir = user_data_dir
        self._handle = handle

    def release(self) -> None:
        handle = self._handle
        self._handle = None
        if handle is None:
            return
        try:
            fcntl.flock(handle, fcntl.LOCK_UN)
        except OSError:
            logger.debug("Failed to unlock profile lane", exc_info=True)
        try:
            handle.close()
        except OSError:
            pass


def _stable_lane_index(provider: str, lanes: int) -> int:
    """Crash-stable preferred lane for a provider, so one provider's cookies
    concentrate in one shared profile across restarts."""
    return zlib.crc32(provider.encode("utf-8")) % lanes


def _acquire_profile_lease(provider: str | None) -> _ProfileLease:
    """Take the provider's preferred shared profile lane, or any free one.

    Returns an ephemeral lease (no dir) when profiles are disabled or every
    lane of the configured range is in use — fetches must never fail for
    lack of a profile.
    """
    if not BROWSER_PROFILE_DIR or BROWSER_PROFILE_LANES <= 0:
        return _ProfileLease(None)

    order = list(
        range(
            BROWSER_PROFILE_FIRST_INDEX,
            BROWSER_PROFILE_FIRST_INDEX + BROWSER_PROFILE_LANES,
        )
    )
    if provider:
        preferred = _stable_lane_index(provider, BROWSER_PROFILE_LANES)
        order.remove(preferred)
        order.insert(0, preferred)

    for index in order:
        lock_path = os.path.join(BROWSER_PROFILE_DIR, f"lane-{index}.lock")
        lane_path = os.path.join(BROWSER_PROFILE_DIR, f"lane-{index}")
        try:
            handle = open(lock_path, "a", encoding="utf-8")  # noqa: SIM115
            # the volume may be shared with another container running as a
            # different UID; a lock file it created must stay lockable here
            os.chmod(lock_path, 0o666)  # noqa: S103 — shared-volume lock, by design
        except OSError:
            continue
        try:
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError:
            handle.close()
            continue
        try:
            os.makedirs(lane_path, exist_ok=True)
            os.chmod(lane_path, 0o777)  # noqa: S103 — shared-volume profile dir, by design
        except OSError:
            fcntl.flock(handle, fcntl.LOCK_UN)
            handle.close()
            continue
        return _ProfileLease(lane_path, handle)

    logger.debug(
        "All %d shared browser profile lanes are busy; running ephemeral",
        BROWSER_PROFILE_LANES,
    )
    return _ProfileLease(None)


class PooledBrowserSession:
    """One Playwright instance + BrowserContext owned by a lane."""

    def __init__(self, context_factory: Callable[[], tuple[Any, Any]]) -> None:
        self.playwright, self.context = context_factory()
        self._lease: _ProfileLease | None = None

    @classmethod
    def from_start_playwright(
        cls, user_data_dir: str | None = None
    ) -> PooledBrowserSession:
        from onyx.utils.playwright_fetch import start_playwright

        return cls(lambda: start_playwright(user_data_dir))

    def adopt_lease(self, lease: "_ProfileLease | None") -> None:
        """Attach the profile lease held while this browser is up."""
        self._lease = lease

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
        finally:
            if self._lease is not None:
                self._lease.release()
                self._lease = None


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

    def __init__(self, provider: str, factory: Callable[[str], BrowserSession]) -> None:
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
                    self._session = self._factory(self.provider)
                elif not self._session.is_alive():
                    # The browser process died between jobs (OOM kill,
                    # driver crash). Rebuild before serving so the caller
                    # does not get a request failed against dead handles.
                    logger.warning(
                        "Browser lane %s browser process is gone; rebuilding",
                        self.provider,
                    )
                    self._teardown()
                    self._session = self._factory(self.provider)
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
        session_factory: Callable[[str], BrowserSession],
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
            pool = BrowserPool(_leased_session_factory)
            atexit.register(_shutdown_pool)
            _pool = pool
        return _pool


def _leased_session_factory(provider: str) -> BrowserSession:
    """Build a lane session on a shared persistent profile when configured.

    The profile lease (flock on lane-<N>.lock) is held for the browser's
    lifetime and released on teardown. The provider's stable preferred lane
    is tried first so its cookies concentrate in one profile; a busy lane
    falls back to any free one, and a fully busy pool runs ephemeral.
    """
    lease = _acquire_profile_lease(provider)
    try:
        session = PooledBrowserSession.from_start_playwright(lease.user_data_dir)
    except Exception:
        lease.release()
        raise
    session.adopt_lease(lease)
    if lease.user_data_dir:
        logger.info(
            "Browser lane %s launched on shared profile %s",
            provider,
            lease.user_data_dir,
        )
    return session


def _shutdown_pool() -> None:
    global _pool
    with _pool_lock:
        if _pool is not None:
            _pool.close_all()
            _pool = None
