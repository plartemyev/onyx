"""Tests for `unwrap_google_translate_url` (`onyx.utils.url`).

Google serves auto-translated copies of foreign pages on
`<dashed-host>.translate.goog` wrappers. The original host is encoded
locally in the subdomain, so the reverse mapping must need no request.
"""

from __future__ import annotations

from onyx.utils.url import unwrap_google_translate_url


def test_non_translate_url_passes_through() -> None:
    url = "https://example.org/page?a=1"
    assert unwrap_google_translate_url(url) == url


def test_translate_goog_wrapped_page_reverts_to_original() -> None:
    assert (
        unwrap_google_translate_url(
            "https://energysavingtrust-org-uk.translate.goog"
            "/advice/in-depth-guide?_x_tr_sl=en&_x_tr_tl=th&_x_tr_hl=th"
        )
        == "https://energysavingtrust.org.uk/advice/in-depth-guide"
    )


def test_translate_goog_strips_language_prefix() -> None:
    assert (
        unwrap_google_translate_url(
            "https://en-m-wikipedia-org.translate.goog/wiki/Heat_pump"
            "?_x_tr_sl=en&_x_tr_tl=th"
        )
        == "https://m.wikipedia.org/wiki/Heat_pump"
    )


def test_translate_goog_drops_x_tr_params() -> None:
    unwrapped = unwrap_google_translate_url(
        "https://www-iea-org.translate.goog/reports/x?keep=1&_x_tr_sl=auto&_x_tr_tl=th"
    )
    assert unwrapped == "https://www.iea.org/reports/x?keep=1"


def test_legacy_translate_google_com_resolves_u_param() -> None:
    assert (
        unwrap_google_translate_url(
            "https://translate.google.com/translate"
            "?u=https%3A%2F%2Fexample.com%2Fpage&sl=en&tl=th"
        )
        == "https://example.com/page"
    )


def test_unknown_subdomain_shape_returns_input() -> None:
    url = "https://x.translate.goog/path"
    assert unwrap_google_translate_url(url) == url
