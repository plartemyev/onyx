"""Tests for `ScrapeSessionContext` lifecycle hardening.

The web connector rebuilds its Playwright session after every failed page
(`load_from_state` retry loop), so the session holder must stay consistent
when closes or launches fail — otherwise the node driver process leaks and
later rebuilds run against stale handles.
"""

import pytest

from onyx.connectors.web.connector import ScrapeSessionContext


class _ExplodingContext:
    """BrowserContext stand-in whose close() fails (browser already dead)."""

    def close(self) -> None:
        raise RuntimeError("Target closed")


class _FakePlaywright:
    def __init__(self) -> None:
        self.stopped = False

    def stop(self) -> None:
        self.stopped = True


def test_stop_survives_context_close_failure() -> None:
    session = ScrapeSessionContext("https://example.com", ["https://example.com"])
    playwright = _FakePlaywright()
    session.playwright_context = _ExplodingContext()  # ty: ignore[invalid-assignment]
    session.playwright = playwright  # ty: ignore[invalid-assignment]

    session.stop()

    # close() raised, but stop() must still tear down and reset both handles.
    assert playwright.stopped
    assert session.playwright_context is None
    assert session.playwright is None


def test_failed_initialize_leaves_consistent_state(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    session = ScrapeSessionContext("https://example.com", ["https://example.com"])

    def failing_start_playwright() -> tuple[object, object]:
        raise RuntimeError("browser pool init failed")

    monkeypatch.setattr(
        "onyx.connectors.web.connector.start_playwright", failing_start_playwright
    )

    with pytest.raises(RuntimeError, match="browser pool init failed"):
        session.initialize()

    # Handles stay None so a later initialize() starts from a clean state.
    assert session.playwright is None
    assert session.playwright_context is None
