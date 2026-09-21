import threading
import time

import pytest

from onyx.utils.request_pacer import DomainPacer, NullPacer, provider_key


@pytest.mark.parametrize(
    "url, expected",
    [
        # Image CDN hosts group with their origin
        ("https://i.imgur.com/AKVZMd3b.jpg", "imgur.com"),
        ("https://imgur.com/trending", "imgur.com"),
        ("https://i.redd.it/abc.jpg", "reddit.com"),
        ("https://preview.redd.it/abc.jpg?auto=webp", "reddit.com"),
        ("https://external-preview.redd.it/abc", "reddit.com"),
        ("https://www.reddit.com/r/pics/", "reddit.com"),
        # x.com and twitter.com are one provider
        ("https://x.com/user/status/1", "x.com"),
        ("https://twitter.com/user/status/1", "x.com"),
        # Plain registrable domains
        ("https://example.com/cat.jpg", "example.com"),
        ("https://api.example.com/v1/file", "example.com"),
        # Two-level public suffixes keep distinct sites distinct
        ("https://www.bbc.co.uk/news", "bbc.co.uk"),
        ("https://www.mit.edu/", "mit.edu"),
        # Scheme-relative / junk still buckets somewhere stable
        ("//cdn.example.org/a.png", "example.org"),
    ],
)
def test_provider_key(url: str, expected: str) -> None:
    assert provider_key(url) == expected


class TestDomainPacer:
    def test_first_request_does_not_sleep(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        sleeps: list[float] = []
        monkeypatch.setattr(time, "sleep", lambda s: sleeps.append(s))
        pacer = DomainPacer(min_gap_seconds=0.7, max_gap_seconds=5.0)

        pacer.pace("https://imgur.com/a.jpg")

        assert sleeps == []

    def test_second_same_provider_request_sleeps_within_gap(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        sleeps: list[float] = []
        monkeypatch.setattr(time, "sleep", lambda s: sleeps.append(s))
        pacer = DomainPacer(min_gap_seconds=0.7, max_gap_seconds=5.0)

        pacer.pace("https://imgur.com/a.jpg")
        pacer.pace("https://i.imgur.com/b.jpg")

        assert len(sleeps) == 1
        assert 0.7 <= sleeps[0] <= 5.0

    def test_different_providers_do_not_sleep(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        sleeps: list[float] = []
        monkeypatch.setattr(time, "sleep", lambda s: sleeps.append(s))
        pacer = DomainPacer(min_gap_seconds=0.7, max_gap_seconds=5.0)

        pacer.pace("https://i.imgur.com/a.jpg")
        pacer.pace("https://i.redd.it/b.jpg")
        pacer.pace("https://x.com/c.jpg")

        assert sleeps == []

    def test_cdn_host_shares_gate_with_origin(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        sleeps: list[float] = []
        monkeypatch.setattr(time, "sleep", lambda s: sleeps.append(s))
        pacer = DomainPacer(min_gap_seconds=0.7, max_gap_seconds=5.0)

        pacer.pace("https://i.redd.it/a.jpg")
        pacer.pace("https://www.reddit.com/b")

        assert len(sleeps) == 1

    def test_wait_capped_at_max_wait(self, monkeypatch: pytest.MonkeyPatch) -> None:
        sleeps: list[float] = []
        monkeypatch.setattr(time, "sleep", lambda s: sleeps.append(s))
        pacer = DomainPacer(
            min_gap_seconds=0.7, max_gap_seconds=5.0, max_wait_seconds=2.0
        )
        # Simulate a provider whose next slot is far in the future.
        key = provider_key("https://imgur.com/a.jpg")
        pacer._last_start[key] = time.monotonic() + 10_000  # noqa: SLF001

        pacer.pace("https://imgur.com/a.jpg")

        assert sleeps == [2.0]

    def test_threads_are_spaced_per_provider(self) -> None:
        """Concurrent same-provider callers start at least min_gap apart,
        while a different provider is not delayed by them."""
        pacer = DomainPacer(min_gap_seconds=0.05, max_gap_seconds=0.1)
        starts: dict[str, float] = {}

        def _record(url: str) -> None:
            pacer.pace(url)
            starts[url] = time.monotonic()

        t1 = threading.Thread(target=_record, args=("https://i.imgur.com/a.jpg",))
        t2 = threading.Thread(target=_record, args=("https://i.imgur.com/b.jpg",))
        t3 = threading.Thread(target=_record, args=("https://i.redd.it/c.jpg",))
        t1.start()
        t2.start()
        t3.start()
        t1.join(timeout=10)
        t2.join(timeout=10)
        t3.join(timeout=10)

        assert set(starts) == {
            "https://i.imgur.com/a.jpg",
            "https://i.imgur.com/b.jpg",
            "https://i.redd.it/c.jpg",
        }
        first, second = sorted(
            (starts["https://i.imgur.com/a.jpg"], starts["https://i.imgur.com/b.jpg"])
        )
        assert second - first >= 0.05

    def test_invalid_gap_range_rejected(self) -> None:
        with pytest.raises(ValueError):
            DomainPacer(min_gap_seconds=5.0, max_gap_seconds=0.7)


def test_null_pacer_is_immediate() -> None:
    pacer = NullPacer()
    start = time.monotonic()
    pacer.pace("https://imgur.com/a.jpg")
    assert time.monotonic() - start < 0.1
