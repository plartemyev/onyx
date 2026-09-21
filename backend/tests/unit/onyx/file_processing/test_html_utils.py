from onyx.file_processing.html_utils import MAX_IMAGE_URLS_PER_PAGE, extract_image_urls

BASE_URL = "https://example.com/articles/memes"


def test_absolute_urls_passed_through() -> None:
    html = (
        '<img src="https://cdn.example.com/a.jpg">'
        '<img src="https://cdn.example.com/b.png">'
    )
    assert extract_image_urls(html, BASE_URL) == [
        "https://cdn.example.com/a.jpg",
        "https://cdn.example.com/b.png",
    ]


def test_relative_urls_resolved_against_base() -> None:
    html = '<img src="/img/relative.jpg"><img src="../up.png">'
    assert extract_image_urls(html, BASE_URL) == [
        "https://example.com/img/relative.jpg",
        "https://example.com/up.png",
    ]


def test_query_params_preserved() -> None:
    html = '<img src="https://cdn.example.com/a.jpg?w=640&amp;sig=abc">'
    assert extract_image_urls(html, BASE_URL) == [
        "https://cdn.example.com/a.jpg?w=640&sig=abc"
    ]


def test_data_uris_and_non_http_schemes_skipped() -> None:
    html = (
        '<img src="data:image/png;base64,AAAA">'
        '<img src="javascript:void(0)">'
        '<img src="https://cdn.example.com/ok.jpg">'
    )
    assert extract_image_urls(html, BASE_URL) == ["https://cdn.example.com/ok.jpg"]


def test_lazy_load_attributes_used_as_fallback() -> None:
    html = (
        '<img src="" data-src="https://cdn.example.com/lazy.jpg">'
        '<img data-lazy-src="https://cdn.example.com/lazy2.jpg">'
        '<img data-original="https://cdn.example.com/lazy3.jpg">'
    )
    assert extract_image_urls(html, BASE_URL) == [
        "https://cdn.example.com/lazy.jpg",
        "https://cdn.example.com/lazy2.jpg",
        "https://cdn.example.com/lazy3.jpg",
    ]


def test_duplicates_removed() -> None:
    html = (
        '<img src="/a.jpg"><img src="/a.jpg"><img src="https://cdn.example.com/a.jpg">'
    )
    assert extract_image_urls(html, BASE_URL) == [
        "https://example.com/a.jpg",
        "https://cdn.example.com/a.jpg",
    ]


def test_tracking_pixels_skipped() -> None:
    html = (
        '<img src="https://t.example.com/pixel.gif" width="1" height="1">'
        '<img src="https://cdn.example.com/real.jpg">'
    )
    assert extract_image_urls(html, BASE_URL) == ["https://cdn.example.com/real.jpg"]


def test_cap_applied() -> None:
    html = "".join(
        f'<img src="https://cdn.example.com/{index}.jpg">' for index in range(20)
    )
    urls = extract_image_urls(html, BASE_URL)
    assert len(urls) == MAX_IMAGE_URLS_PER_PAGE
    assert urls[0] == "https://cdn.example.com/0.jpg"


def test_empty_html_returns_empty_list() -> None:
    assert extract_image_urls("", BASE_URL) == []
