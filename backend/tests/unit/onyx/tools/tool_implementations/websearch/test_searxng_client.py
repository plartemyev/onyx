from __future__ import annotations

import time
from typing import Any
from unittest.mock import MagicMock

import pytest
import requests

from onyx.tools.tool_implementations.web_search.clients import searxng_client
from onyx.tools.tool_implementations.web_search.clients.searxng_client import (
    SearXNGClient,
)

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


def _capture_post(
    captured_payloads: list[dict[str, str]], payloads: dict[str, dict[str, Any]]
) -> Any:
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

    return _post


def test_search_surfaces_direct_image_urls(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A general hit carries its page thumbnail as a direct image URL."""
    monkeypatch.setattr(
        searxng_client.requests,
        "post",
        _fake_post({"general": GENERAL_PAYLOAD, "images": IMAGE_PAYLOAD}),
    )
    client = SearXNGClient("http://localhost:8080", num_results=10)

    results = client.search("drake meme")

    assert results[0].title == "Meme history"
    assert results[0].image_urls == ["https://knowyourmeme.com/thumb.jpg?w=200"]
    assert results[1].image_urls == []


def test_default_search_makes_no_images_request(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A standard search costs exactly one request and returns web results
    only — the image fan-out must be an explicit agent request."""
    captured_payloads: list[dict[str, str]] = []
    monkeypatch.setattr(
        searxng_client.requests,
        "post",
        _capture_post(
            captured_payloads, {"general": GENERAL_PAYLOAD, "images": IMAGE_PAYLOAD}
        ),
    )
    client = SearXNGClient("http://localhost:8080", num_results=10)

    results = client.search("drake meme")

    assert [payload.get("categories") for payload in captured_payloads] == [None]
    assert [r.title for r in results] == ["Meme history", "No image result"]


def test_search_images_returns_only_image_results(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured_payloads: list[dict[str, str]] = []
    monkeypatch.setattr(
        searxng_client.requests,
        "post",
        _capture_post(
            captured_payloads, {"general": GENERAL_PAYLOAD, "images": IMAGE_PAYLOAD}
        ),
    )
    client = SearXNGClient("http://localhost:8080", num_results=10)

    results = client.search_images("drake meme")

    # One request, aimed at the images category only
    assert len(captured_payloads) == 1
    assert captured_payloads[0].get("categories") == "images"

    # No web pages mixed in: hits without a usable image URL are dropped,
    # relative img_src is resolved against the source page
    assert [r.title for r in results] == ["drakeposting.jpg", "relative image"]
    assert all(len(r.image_urls) == 1 for r in results)
    by_title = {r.title: r for r in results}
    assert by_title["drakeposting.jpg"].image_urls == [
        "https://i.kym-cdn.com/drake.jpg"
    ]
    assert by_title["drakeposting.jpg"].link == "https://knowyourmeme.com/memes/drake"
    assert by_title["relative image"].image_urls == [
        "https://knowyourmeme.com/photos/relative.png"
    ]


def test_search_images_passes_language(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured_payloads: list[dict[str, str]] = []
    monkeypatch.setattr(
        searxng_client.requests,
        "post",
        _capture_post(
            captured_payloads, {"general": GENERAL_PAYLOAD, "images": IMAGE_PAYLOAD}
        ),
    )
    client = SearXNGClient("http://localhost:8080", language="en")

    client.search_images("drake meme", language="de")

    assert captured_payloads[0].get("language") == "de"


def test_search_images_retries_then_raises_on_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """In image-only mode the images request is the whole result, so a
    failure surfaces (through the paced retry) instead of degrading to []."""
    monkeypatch.setattr(time, "sleep", lambda _s: None)
    calls = {"n": 0}

    def _post(
        url: str,  # noqa: ARG001
        data: dict[str, str],  # noqa: ARG001
        timeout: Any = None,  # noqa: ARG001
    ) -> Any:
        calls["n"] += 1
        raise RuntimeError("images endpoint down")

    monkeypatch.setattr(searxng_client.requests, "post", _post)
    client = SearXNGClient("http://localhost:8080")

    with pytest.raises(RuntimeError, match="images endpoint down"):
        client.search_images("drake meme")

    assert calls["n"] == 3


def test_search_images_surfaces_a_challenge_hit(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Empty results with unresponsive engines is a challenge hit, not an
    empty answer: raise so the paced retry can pick up the clearance."""
    monkeypatch.setattr(time, "sleep", lambda _s: None)

    def _post(
        url: str,  # noqa: ARG001
        data: dict[str, str],  # noqa: ARG001
        timeout: Any = None,  # noqa: ARG001
    ) -> Any:
        response = MagicMock()
        response.raise_for_status.return_value = None
        response.json.return_value = {
            "results": [],
            "unresponsive_engines": [{"engine": "google", "error": "challenge"}],
        }
        return response

    monkeypatch.setattr(searxng_client.requests, "post", _post)
    client = SearXNGClient("http://localhost:8080")

    with pytest.raises(requests.RequestException):
        client.search_images("drake meme")


def test_search_images_raises_when_capability_disabled() -> None:
    client = SearXNGClient("http://localhost:8080", enable_image_search=False)

    with pytest.raises(ValueError, match="Image search is disabled"):
        client.search_images("drake meme")


def test_search_still_fails_when_general_search_fails(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # keep the paced retry from actually sleeping out its 20s/40s gaps
    monkeypatch.setattr(time, "sleep", lambda _s: None)

    def _post(
        url: str,  # noqa: ARG001
        data: dict[str, str],  # noqa: ARG001
        timeout: Any = None,  # noqa: ARG001
    ) -> Any:
        raise RuntimeError("general search down")

    monkeypatch.setattr(searxng_client.requests, "post", _post)
    client = SearXNGClient("http://localhost:8080")

    with pytest.raises(RuntimeError, match="general search down"):
        client.search("drake meme")


def test_search_retries_a_challenge_hit_and_then_delivers(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A challenge-hit search comes back HTTP 200 with zero results and
    all engines unresponsive: the client surfaces that (instead of
    silently delivering nothing) and the paced retry — spaced like a
    person re-querying — picks up the lane's banked clearance."""
    monkeypatch.setattr(time, "sleep", lambda _s: None)
    challenge_payload = {
        "results": [],
        "unresponsive_engines": [{"engine": "google", "error": "challenge"}],
    }
    calls = {"n": 0}

    def _post(
        url: str,  # noqa: ARG001
        data: dict[str, str],  # noqa: ARG001
        timeout: Any = None,  # noqa: ARG001
    ) -> Any:
        calls["n"] += 1
        response = MagicMock()
        response.raise_for_status.return_value = None
        response.json.return_value = (
            challenge_payload if calls["n"] < 3 else GENERAL_PAYLOAD
        )
        return response

    monkeypatch.setattr(searxng_client.requests, "post", _post)
    client = SearXNGClient("http://localhost:8080")

    results = client.search("drake meme")

    assert calls["n"] == 3
    assert [r.title for r in results] == ["Meme history", "No image result"]


def test_language_config_passed_to_searxng(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A configured language rides along on the search payload, so a
    mis-localized instance can be overridden."""
    captured_payloads: list[dict[str, str]] = []
    monkeypatch.setattr(
        searxng_client.requests,
        "post",
        _capture_post(
            captured_payloads, {"general": GENERAL_PAYLOAD, "images": IMAGE_PAYLOAD}
        ),
    )
    client = SearXNGClient("http://localhost:8080", num_results=10, language="en")

    client.search("drake meme")

    assert len(captured_payloads) == 1
    assert captured_payloads[0].get("language") == "en"


def test_language_omitted_when_not_configured(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured_payloads: list[dict[str, str]] = []
    monkeypatch.setattr(
        searxng_client.requests,
        "post",
        _capture_post(
            captured_payloads, {"general": GENERAL_PAYLOAD, "images": IMAGE_PAYLOAD}
        ),
    )
    client = SearXNGClient("http://localhost:8080", num_results=10)

    client.search("drake meme")

    assert all("language" not in payload for payload in captured_payloads)


def test_per_call_language_overrides_configured_language(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A per-call language (from the agent's tool call) wins over the
    instance-level default."""
    captured_payloads: list[dict[str, str]] = []
    monkeypatch.setattr(
        searxng_client.requests,
        "post",
        _capture_post(
            captured_payloads, {"general": GENERAL_PAYLOAD, "images": IMAGE_PAYLOAD}
        ),
    )
    client = SearXNGClient("http://localhost:8080", num_results=10, language="en")

    client.search("drake meme", language="de")

    assert len(captured_payloads) == 1
    assert captured_payloads[0].get("language") == "de"


def test_per_call_language_used_without_configured_language(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured_payloads: list[dict[str, str]] = []
    monkeypatch.setattr(
        searxng_client.requests,
        "post",
        _capture_post(
            captured_payloads, {"general": GENERAL_PAYLOAD, "images": IMAGE_PAYLOAD}
        ),
    )
    client = SearXNGClient("http://localhost:8080", num_results=10)

    client.search("drake meme", language="pt-BR")

    assert len(captured_payloads) == 1
    assert captured_payloads[0].get("language") == "pt-BR"
