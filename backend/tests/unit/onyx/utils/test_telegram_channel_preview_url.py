"""Tests for the Telegram channel URL helpers (`onyx.utils.url`).

The bare ``t.me/<channel>`` URL is an app hand-off stub; the readable
public history of the same channel lives at ``t.me/s/<channel>``. The
SearXNG crawl pipeline serves crawls of the bare URL from that preview
transparently, so Onyx needs to recognize the rewrite both to produce it
and to keep the requested URL as a fetched page's identity.
"""

from __future__ import annotations

from onyx.utils.url import telegram_channel_preview_url, unwrap_telegram_preview_url


def test_bare_channel_url_maps_to_preview() -> None:
    assert telegram_channel_preview_url("https://t.me/thinkaboutism") == (
        "https://t.me/s/thinkaboutism"
    )


def test_mirror_hosts_canonicalize_to_t_me() -> None:
    assert telegram_channel_preview_url("https://telegram.me/Durov") == (
        "https://t.me/s/Durov"
    )
    assert telegram_channel_preview_url("http://www.telegram.me/Durov?before=42") == (
        "https://t.me/s/Durov?before=42"
    )


def test_preview_and_post_urls_pass_through() -> None:
    assert telegram_channel_preview_url("https://t.me/s/thinkaboutism") == (
        "https://t.me/s/thinkaboutism"
    )
    assert telegram_channel_preview_url("https://t.me/thinkaboutism/1234") == (
        "https://t.me/thinkaboutism/1234"
    )


def test_utility_paths_and_non_telegram_pass_through() -> None:
    assert telegram_channel_preview_url("https://t.me/share/url?url=x") == (
        "https://t.me/share/url?url=x"
    )
    assert telegram_channel_preview_url("https://example.org/thinkaboutism") == (
        "https://example.org/thinkaboutism"
    )


def test_preview_final_keeps_requested_bare_url() -> None:
    assert (
        unwrap_telegram_preview_url(
            "https://t.me/s/thinkaboutism", "https://t.me/thinkaboutism"
        )
        == "https://t.me/thinkaboutism"
    )


def test_mirror_request_keeps_requested_mirror_url() -> None:
    assert (
        unwrap_telegram_preview_url("https://t.me/s/Durov", "https://telegram.me/Durov")
        == "https://telegram.me/Durov"
    )


def test_non_preview_final_passes_through() -> None:
    # e.g. the pre-rewrite behavior: the stub page itself was rendered
    assert (
        unwrap_telegram_preview_url(
            "https://t.me/thinkaboutism", "https://t.me/thinkaboutism"
        )
        == "https://t.me/thinkaboutism"
    )
    assert (
        unwrap_telegram_preview_url(
            "https://t.me/otherchannel", "https://t.me/thinkaboutism"
        )
        == "https://t.me/otherchannel"
    )


def test_unrelated_urls_pass_through() -> None:
    assert (
        unwrap_telegram_preview_url(
            "https://example.com/redirected", "https://example.com/original"
        )
        == "https://example.com/redirected"
    )
