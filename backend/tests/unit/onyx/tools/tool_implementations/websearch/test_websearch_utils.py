from pathlib import Path

from onyx.tools.tool_implementations.open_url.models import WebContent
from onyx.tools.tool_implementations.web_search.models import WebSearchResult
from onyx.tools.tool_implementations.web_search.utils import (
    inference_section_from_internet_page_scrape,
    inference_section_from_internet_search_result,
)

CONTENT_FILE = Path(__file__).parent / "data" / "tartan.txt"

# inference_section_from_internet_page_scrape will cull the content to 15000 characters
MAX_NUM_CHARS_WEB_CONTENT = 15000
TRUNCATED_CONTENT_SUFFIX = " [...truncated]"
TRUNCATED_CONTENT_PREFIX = "[...truncated] "


def get_text_from_file(file_path: Path) -> str:
    with open(file_path, "r") as file:
        return file.read()


def get_tartan_text() -> str:
    return get_text_from_file(CONTENT_FILE)


def create_web_content_object(text: str) -> WebContent:
    return WebContent(
        full_content=text,
        title="Tartan",
        link="https://en.wikipedia.org/wiki/Tartan",
        published_date=None,
        scrape_successful=True,
    )


def test_no_snippet_provided() -> None:
    tartan_text = get_tartan_text()
    web_content = create_web_content_object(tartan_text)

    section = inference_section_from_internet_page_scrape(web_content, "")

    # Section will be of length min(MAX_NUM_CHARS_WEB_CONTENT, len(tartan_text))
    assert len(section.combined_content) == MAX_NUM_CHARS_WEB_CONTENT + len(
        TRUNCATED_CONTENT_SUFFIX
    )

    # Get the combined_content without the truncated suffix
    combined_content_without_suffix = section.combined_content[
        :MAX_NUM_CHARS_WEB_CONTENT
    ]

    # Check that we have the first 15000 characters of the tartan text
    assert combined_content_without_suffix == tartan_text[:MAX_NUM_CHARS_WEB_CONTENT]
    assert (
        section.combined_content
        == tartan_text[:MAX_NUM_CHARS_WEB_CONTENT] + TRUNCATED_CONTENT_SUFFIX
    )


def test_snippet_lower_bound_() -> None:
    tartan_text = get_tartan_text()
    web_content = create_web_content_object(tartan_text)

    snippet = (
        'Close-up view of traditional tartan cloth, showing pattern of diagonal "ribs" of colour; '
        "this is a five-colour tartan, in scarlet red, black, yellow..."
    )

    section = inference_section_from_internet_page_scrape(web_content, snippet)

    assert len(section.combined_content) == MAX_NUM_CHARS_WEB_CONTENT + len(
        TRUNCATED_CONTENT_SUFFIX
    )

    no_suffix = section.combined_content[:MAX_NUM_CHARS_WEB_CONTENT]

    assert no_suffix == tartan_text[:MAX_NUM_CHARS_WEB_CONTENT]
    assert section.combined_content == no_suffix + TRUNCATED_CONTENT_SUFFIX


def test_snippet_provided_after_limit() -> None:
    tartan_text = get_tartan_text()
    web_content = create_web_content_object(tartan_text)

    snippet = (
        'Transmutations of the Tartan: Attributed Meanings to Tartan Design"]. '
        "_Textiles as Primary Sources: Proceedings_. First Textile Society of America Symposium."
    )

    section = inference_section_from_internet_page_scrape(web_content, snippet)

    assert (
        len(section.combined_content)
        == len(TRUNCATED_CONTENT_PREFIX) + MAX_NUM_CHARS_WEB_CONTENT
    )

    no_prefix = section.combined_content[len(TRUNCATED_CONTENT_PREFIX) :]
    # We should get the last 15000 characters of the tartan text
    index = len(tartan_text) - MAX_NUM_CHARS_WEB_CONTENT

    assert no_prefix == tartan_text[index:]
    assert section.combined_content == TRUNCATED_CONTENT_PREFIX + no_prefix


def test_snippet_provided_in_middle() -> None:
    tartan_text = get_tartan_text()
    web_content = create_web_content_object(tartan_text)

    snippet = "marketing as a district tartan for Ulster, Scottish weavers (and in two cases English, and in another American)"

    SNIPPET_START_LOCATION_IN_TEXT = 215398

    section = inference_section_from_internet_page_scrape(web_content, snippet)

    assert len(section.combined_content) == len(
        TRUNCATED_CONTENT_PREFIX
    ) + MAX_NUM_CHARS_WEB_CONTENT + len(TRUNCATED_CONTENT_SUFFIX)

    no_prefix = section.combined_content[len(TRUNCATED_CONTENT_PREFIX) :]
    no_affix = no_prefix[:MAX_NUM_CHARS_WEB_CONTENT]

    # expected start index of the snippet
    expected_start_idx = SNIPPET_START_LOCATION_IN_TEXT
    expected_end_idx = expected_start_idx + len(snippet) - 1

    top_padding = (MAX_NUM_CHARS_WEB_CONTENT - len(snippet)) // 2
    bottom_padding = MAX_NUM_CHARS_WEB_CONTENT - len(snippet) - top_padding

    assert (
        no_affix
        == tartan_text[
            expected_start_idx - top_padding : expected_end_idx + bottom_padding + 1
        ]
    )

    assert section.combined_content == (
        TRUNCATED_CONTENT_PREFIX
        + tartan_text[
            expected_start_idx - top_padding : expected_end_idx + bottom_padding + 1
        ]
        + TRUNCATED_CONTENT_SUFFIX
    )


def test_bad_snippet() -> None:
    tartan_text = get_tartan_text()
    web_content = create_web_content_object(tartan_text)

    snippet = "This is a bad snippet"
    # We expect the fallback (from top) to occur
    section = inference_section_from_internet_page_scrape(web_content, snippet)

    # Section will be of length min(MAX_NUM_CHARS_WEB_CONTENT, len(tartan_text))
    assert len(section.combined_content) == MAX_NUM_CHARS_WEB_CONTENT + len(
        TRUNCATED_CONTENT_SUFFIX
    )

    # Get the combined_content without the truncated suffix
    combined_content_without_suffix = section.combined_content[
        :MAX_NUM_CHARS_WEB_CONTENT
    ]

    # Check that we have the first 15000 characters of the tartan text
    assert combined_content_without_suffix == tartan_text[:MAX_NUM_CHARS_WEB_CONTENT]
    assert (
        section.combined_content
        == tartan_text[:MAX_NUM_CHARS_WEB_CONTENT] + TRUNCATED_CONTENT_SUFFIX
    )


def test_similar_snippet_in_middle_fuzzy_match() -> None:
    tartan_text = get_tartan_text()
    web_content = create_web_content_object(tartan_text)

    # In the actual text, the word "English" is used instead of "British"
    # This is very similar though, so we expect a fuzzy match to occur
    snippet = "marketing as a district tartan for Ulster, Scottish weavers (and in two cases British, and in another American)"

    SNIPPET_START_LOCATION_IN_TEXT = 215398

    section = inference_section_from_internet_page_scrape(web_content, snippet)

    assert len(section.combined_content) == len(
        TRUNCATED_CONTENT_PREFIX
    ) + MAX_NUM_CHARS_WEB_CONTENT + len(TRUNCATED_CONTENT_SUFFIX)

    no_prefix = section.combined_content[len(TRUNCATED_CONTENT_PREFIX) :]
    no_affix = no_prefix[:MAX_NUM_CHARS_WEB_CONTENT]

    # expected start index of the snippet
    expected_start_idx = SNIPPET_START_LOCATION_IN_TEXT
    expected_end_idx = expected_start_idx + len(snippet) - 1

    top_padding = (MAX_NUM_CHARS_WEB_CONTENT - len(snippet)) // 2
    bottom_padding = MAX_NUM_CHARS_WEB_CONTENT - len(snippet) - top_padding

    assert (
        no_affix
        == tartan_text[
            expected_start_idx - top_padding : expected_end_idx + bottom_padding + 1
        ]
    )

    assert section.combined_content == (
        TRUNCATED_CONTENT_PREFIX
        + tartan_text[
            expected_start_idx - top_padding : expected_end_idx + bottom_padding + 1
        ]
        + TRUNCATED_CONTENT_SUFFIX
    )


def test_inference_section_from_search_result_carries_image_urls() -> None:
    result = WebSearchResult(
        title="Drakeposting",
        link="https://knowyourmeme.com/memes/drake",
        snippet="Meme history",
        image_urls=["https://i.kym-cdn.com/drake.jpg"],
    )

    section = inference_section_from_internet_search_result(result, rank=0)

    assert section.center_chunk.image_urls == ["https://i.kym-cdn.com/drake.jpg"]


def test_inference_section_from_search_result_no_image_urls() -> None:
    result = WebSearchResult(
        title="Drakeposting",
        link="https://knowyourmeme.com/memes/drake",
        snippet="Meme history",
    )

    section = inference_section_from_internet_search_result(result, rank=0)

    assert section.center_chunk.image_urls is None


def test_inference_section_from_page_scrape_carries_image_urls() -> None:
    web_content = WebContent(
        title="Tartan",
        link="https://en.wikipedia.org/wiki/Tartan",
        full_content="Some page text.",
        scrape_successful=True,
        image_urls=["https://upload.wikimedia.org/tartan.jpg"],
    )

    section = inference_section_from_internet_page_scrape(web_content, "")

    assert section.center_chunk.image_urls == [
        "https://upload.wikimedia.org/tartan.jpg"
    ]
