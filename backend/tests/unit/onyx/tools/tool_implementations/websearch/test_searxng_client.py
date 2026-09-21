from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock

import pytest

from onyx.tools.tool_implementations.web_search.clients import searxng_client
from onyx.tools.tool_implementations.web_search.clients.searxng_client import (
    SearXNGClient,
    _interleave_image_results,
)
from onyx.tools.tool_implementations.web_search.models import WebSearchResult

GENERAL_PAYLOAD = {
    "results": [
        {
            "title": "Meme history",
            "url": "https://knowyourmeme.com/memes/drake",
            "content": "Drakeposting history",
            "img_src": "https://knowyourmeme.com/thumb.jpg?w=200",
        },
        {
            "title": "No image result",
            "url": "https://example.com/page",
            "content": "plain",
        },
    ]
}

IMAGE_PAYLOAD = {
    "results": [
        {
            "title": "drakeposting.jpg",
            "url": "https://knowyourmeme.com/memes/drake",
            "img_src": "https://i.kym-cdn.com/drake.jpg",
            "resolution": "1200 x 675",
        },
        {
            "title": "relative image",
            "url": "https://knowyourmeme.com/memes/other",
            "img_src": "/photos/relative.png",
        },
        {
            "title": "no image url",
            "url": "https://knowyourmeme.com/memes/empty",
        },
    ]
}


def _fake_post(payloads: dict[str, dict[str, Any]]) -> Any:
    def _post(
        url: str,  # noqa: ARG001
        data: dict[str, str],
        timeout: Any = None,  # noqa: ARG001
    ) -> Any:
        response = MagicMock()
        response.raise_for_status.return_value = None
        key = "images" if data.get("categories") == "images" else "general"
        response.json.return_value = payloads[key]
        return response

    return _post


def test_search_surfaces_direct_image_urls(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        searxng_client.requests,
        "post",
        _fake_post({"general": GENERAL_PAYLOAD, "images": IMAGE_PAYLOAD}),
    )
    client = SearXNGClient("http://localhost:8080", num_results=10)

    results = client.search("drake meme")

    # General result carries its page thumbnail as a direct image URL,
    # with query params intact
    assert results[0].title == "Meme history"
    assert results[0].image_urls == ["https://knowyourmeme.com/thumb.jpg?w=200"]

    # Image-category hits become results with the direct image URL and the
    # source page as link; relative img_src is resolved against the source page
    image_urls = {r.image_urls[0]: r for r in results if r.image_urls}
    assert "https://i.kym-cdn.com/drake.jpg" in image_urls
    assert (
        image_urls["https://i.kym-cdn.com/drake.jpg"].link
        == "https://knowyourmeme.com/memes/drake"
    )
    assert "https://knowyourmeme.com/photos/relative.png" in image_urls
    assert (
        "https://i.kym-cdn.com/drake.jpg"
        not in image_urls["https://knowyourmeme.com/photos/relative.png"].image_urls
    )


def test_search_survives_image_search_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def _post(
        url: str,  # noqa: ARG001
        data: dict[str, str],
        timeout: Any = None,  # noqa: ARG001
    ) -> Any:
        if data.get("categories") == "images":
            raise RuntimeError("images endpoint down")
        response = MagicMock()
        response.raise_for_status.return_value = None
        response.json.return_value = GENERAL_PAYLOAD
        return response

    monkeypatch.setattr(searxng_client.requests, "post", _post)
    client = SearXNGClient("http://localhost:8080")

    results = client.search("drake meme")

    assert [r.title for r in results] == ["Meme history", "No image result"]


def test_search_still_fails_when_general_search_fails(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def _post(
        url: str,  # noqa: ARG001
        data: dict[str, str],
        timeout: Any = None,  # noqa: ARG001
    ) -> Any:
        if data.get("categories") == "images":
            response = MagicMock()
            response.raise_for_status.return_value = None
            response.json.return_value = IMAGE_PAYLOAD
            return response
        raise RuntimeError("general search down")

    monkeypatch.setattr(searxng_client.requests, "post", _post)
    client = SearXNGClient("http://localhost:8080")

    with pytest.raises(RuntimeError, match="general search down"):
        client.search("drake meme")


def test_interleave_positions() -> None:
    general = [
        WebSearchResult(
            title=f"t{index}", link=f"https://example.com/{index}", snippet="s"
        )
        for index in range(10)
    ]
    images = [
        WebSearchResult(
            title=f"img{index}",
            link=f"https://example.com/page{index}",
            snippet="",
            image_urls=[f"https://cdn.example.com/{index}.jpg"],
        )
        for index in range(3)
    ]

    combined = _interleave_image_results(general, images)

    # One image after every 4 general results, leftovers appended at the end
    assert combined[0].title == "t0"
    assert combined[4].image_urls == ["https://cdn.example.com/0.jpg"]
    assert combined[9].image_urls == ["https://cdn.example.com/1.jpg"]
    assert combined[12].image_urls == ["https://cdn.example.com/2.jpg"]
    assert len(combined) == 13


def test_interleave_no_images_returns_general_unchanged() -> None:
    general = [WebSearchResult(title="t0", link="https://example.com/0", snippet="s")]
    assert _interleave_image_results(general, []) is general


def test_language_config_passed_to_searxng(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A configured language rides along on both the general and image
    search payloads, so a mis-localized instance can be overridden."""
    captured_payloads: list[dict[str, str]] = []

    def _post(
        url: str,  # noqa: ARG001
        data: dict[str, str],
        timeout: Any = None,  # noqa: ARG001
    ) -> Any:
        captured_payloads.append(dict(data))
        response = MagicMock()
        response.raise_for_status.return_value = None
        key = "images" if data.get("categories") == "images" else "general"
        response.json.return_value = payloads[key]
        return response

    payloads = {"general": GENERAL_PAYLOAD, "images": IMAGE_PAYLOAD}
    monkeypatch.setattr(searxng_client.requests, "post", _post)
    client = SearXNGClient("http://localhost:8080", num_results=10, language="en")

    client.search("drake meme")

    assert len(captured_payloads) == 2
    assert all(payload.get("language") == "en" for payload in captured_payloads)


def test_language_omitted_when_not_configured(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured_payloads: list[dict[str, str]] = []

    def _post(
        url: str,  # noqa: ARG001
        data: dict[str, str],
        timeout: Any = None,  # noqa: ARG001
    ) -> Any:
        captured_payloads.append(dict(data))
        response = MagicMock()
        response.raise_for_status.return_value = None
        key = "images" if data.get("categories") == "images" else "general"
        response.json.return_value = payloads[key]
        return response

    payloads = {"general": GENERAL_PAYLOAD, "images": IMAGE_PAYLOAD}
    monkeypatch.setattr(searxng_client.requests, "post", _post)
    client = SearXNGClient("http://localhost:8080", num_results=10)

    client.search("drake meme")

    assert all("language" not in payload for payload in captured_payloads)
