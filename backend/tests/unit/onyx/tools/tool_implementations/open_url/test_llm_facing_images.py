from onyx.tools.tool_implementations.open_url.open_url_tool import (
    MAX_IMAGES_PER_RESULT,
    _llm_facing_image_urls,
)


def test_drops_data_uris_svg_and_php() -> None:
    urls = [
        "https://example.com/photo.jpg",
        "data:image/png;base64,AAAA",
        "https://cdn.example.com/icon.svg",
        "https://lvs.truehits.in.th/goggen.php?hc=d0006264&rand=86250&x=1",
        "https://example.com/chart.png",
    ]
    assert _llm_facing_image_urls(urls) == [
        "https://example.com/photo.jpg",
        "https://example.com/chart.png",
    ]


def test_query_params_do_not_hide_a_content_image() -> None:
    urls = ["https://cdn.example.com/img.jpg?w=640&h=480"]
    assert _llm_facing_image_urls(urls) == urls


def test_caps_per_result() -> None:
    urls = [f"https://cdn.example.com/p{i}.jpg" for i in range(30)]
    kept = _llm_facing_image_urls(urls)
    assert len(kept) == MAX_IMAGES_PER_RESULT
    assert kept == urls[:MAX_IMAGES_PER_RESULT]


def test_empty_input() -> None:
    assert _llm_facing_image_urls([]) == []
