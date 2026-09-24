"""Tests for the per-provider browser pool (`onyx.utils.browser_pool`).

Uses fake session objects — no real Playwright/browser involved. Verifies
lane persistence (same session reused), serialization, idle teardown,
and the LRU retirement cap.
"""

from __future__ import annotations

import threading
import time

import pytest

from onyx.utils.browser_pool import BrowserPool, PooledBrowserSession


class FakeSession:
    """Stand-in for PooledBrowserSession: counts builds and closes."""

    def __init__(self, alive: bool = True) -> None:
        self.closed = False
        self.alive = alive
        self.context = object()  # what jobs receive
        self.build_lock = threading.Lock()

    def close(self) -> None:
        self.closed = True

    def is_alive(self) -> bool:
        return self.alive


@pytest.fixture
def sessions() -> list[FakeSession]:
    return []


@pytest.fixture
def pool(sessions: list[FakeSession]) -> BrowserPool:
    def factory() -> FakeSession:
        session = FakeSession()
        sessions.append(session)
        return session

    return BrowserPool(factory, max_lanes=2)


def test_same_provider_reuses_one_session(
    pool: BrowserPool, sessions: list[FakeSession]
) -> None:
    pool.run("reddit.com", lambda ctx: ("a", ctx))
    pool.run("reddit.com", lambda ctx: ("b", ctx))
    assert len(sessions) == 1


def test_job_receives_session_context(
    pool: BrowserPool, sessions: list[FakeSession]
) -> None:
    seen: list[object] = []
    pool.run("reddit.com", lambda ctx: seen.append(ctx) or "done")  # type: ignore[func-returns-value]
    assert seen == [sessions[0].context]


def test_different_providers_get_own_sessions(
    pool: BrowserPool, sessions: list[FakeSession]
) -> None:
    pool.run("reddit.com", lambda ctx: str(ctx))  # noqa: ARG005
    pool.run("imgur.com", lambda ctx: str(ctx))  # noqa: ARG005
    assert len(sessions) == 2


def test_job_exception_propagates_to_caller(
    pool: BrowserPool, sessions: list[FakeSession]
) -> None:
    def boom(ctx: object) -> None:  # noqa: ARG001
        raise RuntimeError("fetch failed")

    with pytest.raises(RuntimeError):
        pool.run("example.com", boom)
    # The session stays alive for benign errors.
    assert not sessions[0].closed


def test_dead_browser_is_rebuilt_before_serving(
    pool: BrowserPool, sessions: list[FakeSession]
) -> None:
    pool.run("reddit.com", lambda ctx: "first")  # noqa: ARG005
    # The browser process dies between jobs (OOM kill, crash).
    sessions[0].alive = False

    # The next job must be served on a fresh session without failing.
    assert pool.run("reddit.com", lambda ctx: "second")  # noqa: ARG005 == "second"
    assert len(sessions) == 2
    assert sessions[0].closed
    assert not sessions[1].closed


def test_dead_browser_rebuild_failure_surfaces() -> None:
    built: list[FakeSession] = []

    def factory() -> FakeSession:
        # First build succeeds; every later build fails (e.g. no Chromium).
        if built:
            raise RuntimeError("no chromium")
        session = FakeSession()
        built.append(session)
        return session

    pool = BrowserPool(factory, max_lanes=2)
    pool.run("reddit.com", lambda ctx: "first")  # noqa: ARG005
    built[0].alive = False

    # The dead session is torn down, then the factory error surfaces to the
    # caller instead of a confusing in-job browser failure.
    with pytest.raises(RuntimeError, match="no chromium"):
        pool.run("reddit.com", lambda ctx: "second")  # noqa: ARG005
    assert built[0].closed


def test_job_against_dead_browser_tears_lane_down(
    pool: BrowserPool, sessions: list[FakeSession]
) -> None:
    # The session reports alive but the browser dies mid-job: the job raises
    # a fatal playwright-style error and the lane must rebuild afterwards.
    def target_closed(ctx: object) -> None:  # noqa: ARG001
        raise RuntimeError("Target closed")

    pool.run("reddit.com", lambda ctx: "first")  # noqa: ARG005
    with pytest.raises(RuntimeError, match="Target closed"):
        pool.run("reddit.com", target_closed)
    time.sleep(0.2)  # teardown happens on the lane thread
    assert sessions[0].closed
    # Next job gets a fresh session.
    assert pool.run("reddit.com", lambda ctx: "second")  # noqa: ARG005 == "second"
    assert len(sessions) == 2


def test_lru_cap_retires_oldest_lanes(
    pool: BrowserPool, sessions: list[FakeSession]
) -> None:
    pool.run("a.com", lambda ctx: 1)  # noqa: ARG005
    time.sleep(0.01)
    pool.run("b.com", lambda ctx: 2)  # noqa: ARG005
    time.sleep(0.01)
    # max_lanes=2 — touching a third provider retires the LRU lane ("a.com").
    pool.run("c.com", lambda ctx: 3)  # noqa: ARG005
    time.sleep(0.2)  # shutdown is async on the lane thread
    assert sessions[0].closed
    assert not sessions[1].closed
    assert not sessions[2].closed


def test_pooled_browser_session_closes_both_handles() -> None:
    class FakePlaywright:
        def __init__(self) -> None:
            self.stopped = False

        def stop(self) -> None:
            self.stopped = True

    class FakeContext:
        def __init__(self) -> None:
            self.closed = False

        def close(self) -> None:
            self.closed = True

    fake_pw, fake_ctx = FakePlaywright(), FakeContext()
    session = PooledBrowserSession(lambda: (fake_pw, fake_ctx))
    assert session.context is fake_ctx
    session.close()
    assert fake_ctx.closed
    assert fake_pw.stopped
