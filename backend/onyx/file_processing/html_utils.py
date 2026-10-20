import re
from copy import copy
from dataclasses import dataclass
from io import BytesIO
from typing import IO

import bs4

from onyx.configs.app_configs import (
    HTML_BASED_CONNECTOR_TRANSFORM_LINKS_STRATEGY,
    PARSE_WITH_TRAFILATURA,
    WEB_CONNECTOR_IGNORED_CLASSES,
    WEB_CONNECTOR_IGNORED_ELEMENTS,
)
from onyx.file_processing.enums import HtmlBasedConnectorTransformLinksStrategy
from onyx.utils.logger import setup_logger

logger = setup_logger()

MINTLIFY_UNWANTED = ["sticky", "hidden"]
_ANCHOR_ELEMENT = "a"
_HREF_ATTRIBUTE = "href"
_TABLE_ELEMENT = "table"

# Icon-font glyph carriers: the DOM text is the ligature name ("message",
# "add_box"), which the font renders as a glyph but text extraction reads
# as content. Class tokens cover the common icon libraries; the generic
# naming conventions (icon / icon-* / *-icon[s]) catch custom icon fonts.
_ICON_CLASS_TOKEN_RE = re.compile(
    r"^(?:material-icons|material-symbols(?:-outlined|-rounded|-sharp)?"
    r"|glyphicon(?:-[a-z0-9-]+)?|iconfont|icofont(?:-[a-z0-9-]+)?"
    r"|typcn(?:-[a-z0-9-]+)?|octicon(?:-[a-z0-9-]+)?"
    r"|lucide(?:-[a-z0-9-]+)?|feather(?:-[a-z0-9-]+)?|tabler-icon"
    r"|fa(?:s|r|l|b|d|ss|sr|sl|t)?|fa-[a-z0-9-]+"
    r"|bi(?:-[a-z0-9-]+)?|ion-[a-z0-9-]+|icons?"
    r"|icon-[a-z0-9-]+|[a-z0-9]+-icons?)$"
)
_ICON_GLYPH_TAGS = ("i", "em", "span", "b", "svg", "use")
# Ligature names are short; a longer body means the class is decoration on
# real content, not a glyph slot.
_ICON_GLYPH_MAX_TEXT_CHARS = 40

# Private Use Area codepoints: icon fonts map glyph slots there. They
# render as tofu or nothing and carry no text value.
_PUA_CHARS_RE = re.compile("[\ue000-\uf8ff\U000f0000-\U000ffffd\U00100000-\U0010fffd]")

# Action controls: javascript: anchors and <button> elements trigger UI
# behavior; their labels ("Reply", "Like", "Share") are chrome in every
# locale, and the crawler cannot follow them anyway.
_ACTION_HREF_RE = re.compile(r"^\s*javascript:", re.IGNORECASE)

# Identical short text repeated across a page is UI boilerplate (button
# labels, action menus), never content. Locale neutral: no word lists,
# just occurrence counting with shape guards. Numbers are exempt (vote
# counts repeat legitimately) and so is text inside real links (listing
# rows repeat tag names as linked content).
_REPEATED_TEXT_MAX_OCCURRENCES = 10
_REPEATED_TEXT_MIN_CHARS = 3
_REPEATED_TEXT_MAX_CHARS = 40
_REAL_LINK_HREF_RE = re.compile(r"^https?://", re.IGNORECASE)


@dataclass
class ParsedHTML:
    title: str | None
    cleaned_text: str


def strip_pua_chars(text: str) -> str:
    """Drop private-use-area characters (icon-font glyph slots)."""
    return _PUA_CHARS_RE.sub("", text)


def _drop_chrome_tag(tag: bs4.element.Tag) -> None:
    """Remove a chrome element, leaving a whitespace boundary.

    Adjacent inline text must not fuse once the removed element is gone:
    dropping the icon between a comment count and a vote count fused
    "119" and "17" into "11917". The space is harmless in block context
    (whitespace runs collapse).
    """
    tag.replace_with(bs4.element.NavigableString(" "))


def _is_icon_glyph(tag: bs4.element.Tag) -> bool:
    """True for elements whose only text is an icon-font ligature name."""
    tokens = tag.get("class")
    if isinstance(tokens, str):
        tokens = tokens.split()
    if not tokens or not any(_ICON_CLASS_TOKEN_RE.match(t) for t in tokens):
        return False
    if tag.find(_ANCHOR_ELEMENT) is not None:
        return False
    return len(tag.get_text(strip=True)) <= _ICON_GLYPH_MAX_TEXT_CHARS


def _drop_action_controls(soup: bs4.BeautifulSoup) -> None:
    """Remove buttons and javascript: anchors with their labels."""
    for tag in soup.find_all("button"):
        _drop_chrome_tag(tag)
    for anchor in soup.find_all(_ANCHOR_ELEMENT, href=_ACTION_HREF_RE):
        _drop_chrome_tag(anchor)


def _drop_repeated_boilerplate(soup: bs4.BeautifulSoup) -> int:
    """Drop text repeated across the page more than the threshold times.

    Counts leaf text nodes by normalized value. Pure numbers, too-short or
    too-long strings, and anything inside a real (http/s) link are exempt.
    Returns the number of occurrences removed.
    """
    nodes = [
        node
        for node in soup.find_all(string=True)
        if type(node) is bs4.element.NavigableString
    ]
    counts: dict[str, int] = {}
    for node in nodes:
        text = re.sub(r"\s+", " ", str(node)).strip()
        if _REPEATED_TEXT_MIN_CHARS <= len(text) <= _REPEATED_TEXT_MAX_CHARS and any(
            char.isalpha() for char in text
        ):
            counts[text] = counts.get(text, 0) + 1
    boilerplate = {
        text for text, count in counts.items() if count > _REPEATED_TEXT_MAX_OCCURRENCES
    }
    if not boilerplate:
        return 0
    dropped = 0
    for node in nodes:
        text = re.sub(r"\s+", " ", str(node)).strip()
        if text not in boilerplate:
            continue
        if node.find_parent(_ANCHOR_ELEMENT, href=_REAL_LINK_HREF_RE) is not None:
            continue
        node.replace_with(bs4.element.NavigableString(" "))
        dropped += 1
    return dropped


def strip_excessive_newlines_and_spaces(document: str) -> str:
    # collapse repeated spaces into one
    document = re.sub(r" +", " ", document)
    # remove trailing spaces
    document = re.sub(r" +[\n\r]", "\n", document)
    # remove repeated newlines
    document = re.sub(r"[\n\r]+", "\n", document)
    return document.strip()


def strip_newlines(document: str) -> str:
    # HTML might contain newlines which are just whitespaces to a browser
    return re.sub(r"[\n\r]+", " ", document)


def format_element_text(element_text: str, link_href: str | None) -> str:
    element_text_no_newlines = strip_newlines(element_text)

    if (
        not link_href
        or HTML_BASED_CONNECTOR_TRANSFORM_LINKS_STRATEGY
        == HtmlBasedConnectorTransformLinksStrategy.STRIP
    ):
        return element_text_no_newlines

    return f"[{element_text_no_newlines}]({link_href})"


def _get_ancestor_link_href(
    element: bs4.element.PageElement, in_table: bool
) -> str | None:
    if in_table:
        return None

    link = element.find_parent(_ANCHOR_ELEMENT)
    if not link:
        return None

    href = link.get(_HREF_ATTRIBUTE)
    return href[0] if isinstance(href, list) else href


def parse_html_with_trafilatura(html_content: str) -> str:
    """Parse HTML content using trafilatura."""
    import trafilatura
    from trafilatura.settings import use_config

    config = use_config()
    config.set("DEFAULT", "include_links", "True")
    config.set("DEFAULT", "include_tables", "True")
    config.set("DEFAULT", "include_images", "True")
    config.set("DEFAULT", "include_formatting", "True")

    extracted_text = trafilatura.extract(html_content, config=config)
    return strip_excessive_newlines_and_spaces(extracted_text) if extracted_text else ""


def format_document_soup(
    document: bs4.BeautifulSoup, table_cell_separator: str = "\t"
) -> str:
    """Format html to a flat text document.

    The following goals:
    - Newlines from within the HTML are removed (as browser would ignore them as well).
    - Repeated newlines/spaces are removed (as browsers would ignore them).
    - Newlines only before and after headlines and paragraphs or when explicit (br or pre tag)
    - Table columns/rows are separated by newline
    - List elements are separated by newline and start with a hyphen
    """
    text = ""
    list_element_start = False
    verbatim_output = 0
    last_added_newline = False

    for e in document.descendants:
        verbatim_output -= 1
        if isinstance(e, bs4.element.NavigableString):
            if isinstance(e, (bs4.element.Comment, bs4.element.Doctype)):
                continue
            in_table = e.find_parent(_TABLE_ELEMENT) is not None
            element_text = e.text
            if in_table:
                # Tables are represented in natural language with rows separated by newlines
                # Can't have newlines then in the table elements
                element_text = element_text.replace("\n", " ").strip()

            # Some tags are translated to spaces but in the logic underneath this section, we
            # translate them to newlines as a browser should render them such as with br
            # This logic here avoids a space after newline when it shouldn't be there.
            if last_added_newline and element_text.startswith(" "):
                element_text = element_text[1:]
                last_added_newline = False

            if element_text:
                content_to_add = (
                    element_text
                    if verbatim_output > 0
                    else format_element_text(
                        element_text, _get_ancestor_link_href(e, in_table)
                    )
                )

                # Don't join separate elements without any spacing
                if (text and not text[-1].isspace()) and (
                    content_to_add and not content_to_add[0].isspace()
                ):
                    text += " "

                text += content_to_add

                list_element_start = False
        elif isinstance(e, bs4.element.Tag):
            in_table = e.find_parent(_TABLE_ELEMENT) is not None
            # tr is for rows
            if e.name == "tr" and in_table:
                text += "\n"
            # td for data cell, th for header
            elif e.name in ["td", "th"] and in_table:
                text += table_cell_separator
            elif in_table:
                # don't handle other cases while in table
                pass
            elif e.name in ["p", "div"]:
                if not list_element_start:
                    text += "\n"
            elif e.name in ["h1", "h2", "h3", "h4"]:
                text += "\n"
                list_element_start = False
                last_added_newline = True
            elif e.name == "br":
                text += "\n"
                list_element_start = False
                last_added_newline = True
            elif e.name == "li":
                text += "\n- "
                list_element_start = True
            elif e.name == "pre":
                if verbatim_output <= 0:
                    verbatim_output = len(list(e.childGenerator()))
    return strip_excessive_newlines_and_spaces(text)


def parse_html_page_basic(text: str | BytesIO | IO[bytes]) -> str:
    soup = bs4.BeautifulSoup(text, "lxml")
    return format_document_soup(soup)


def clean_web_html(
    page_content: str | bs4.BeautifulSoup,
    mintlify_cleanup_enabled: bool = True,
    additional_element_types_to_discard: list[str] | None = None,
) -> tuple[bs4.BeautifulSoup, str | None]:
    """First stage shared by the web content extraction paths: parse the
    page, extract the <title>, and drop chrome (ignored element types and
    CSS classes). Returns the cleaned soup and the title."""
    if isinstance(page_content, str):
        soup = bs4.BeautifulSoup(page_content, "lxml")
    else:
        soup = page_content

    title_tag = soup.find("title")
    title = None
    if title_tag and title_tag.text:
        title = title_tag.text
        title_tag.extract()

    # Heuristics based cleaning of elements based on css classes
    unwanted_classes = copy(WEB_CONNECTOR_IGNORED_CLASSES)
    if mintlify_cleanup_enabled:
        unwanted_classes.extend(MINTLIFY_UNWANTED)
    for undesired_element in unwanted_classes:
        [
            _drop_chrome_tag(tag)
            for tag in soup.find_all(
                class_=lambda x, cls=undesired_element: x and cls in x.split()
            )
        ]

    for undesired_tag in WEB_CONNECTOR_IGNORED_ELEMENTS:
        [_drop_chrome_tag(tag) for tag in soup.find_all(undesired_tag)]

    if additional_element_types_to_discard:
        for undesired_tag in additional_element_types_to_discard:
            [_drop_chrome_tag(tag) for tag in soup.find_all(undesired_tag)]

    # Icon-font glyph names are DOM text the font renders away; they read
    # as garbage tokens in extracted text ("*message*119*add_box*17").
    for tag in soup.find_all(_ICON_GLYPH_TAGS):
        if _is_icon_glyph(tag):
            _drop_chrome_tag(tag)

    # Interactive controls ("Reply", "Like", share menus): UI labels in
    # every locale, and the crawler cannot act on them anyway.
    _drop_action_controls(soup)

    # Backstop for handler-based controls without hrefs: identical short
    # text repeated across the page is boilerplate, never content.
    _drop_repeated_boilerplate(soup)

    return soup, title


def web_html_cleanup(
    page_content: str | bs4.BeautifulSoup,
    mintlify_cleanup_enabled: bool = True,
    additional_element_types_to_discard: list[str] | None = None,
) -> ParsedHTML:
    soup, title = clean_web_html(
        page_content,
        mintlify_cleanup_enabled=mintlify_cleanup_enabled,
        additional_element_types_to_discard=additional_element_types_to_discard,
    )

    soup_string = str(soup)
    page_text = ""

    if PARSE_WITH_TRAFILATURA:
        try:
            page_text = parse_html_with_trafilatura(soup_string)
            if not page_text:
                raise ValueError("Empty content returned by trafilatura.")
        except Exception as e:
            logger.info("Trafilatura parsing failed: %s. Falling back on bs4.", e)
            page_text = format_document_soup(soup)
    else:
        page_text = format_document_soup(soup)

    # 200B is ZeroWidthSpace which we don't care for; PUA chars are
    # icon-font glyph slots with no text value.
    cleaned_text = strip_pua_chars(page_text.replace("\u200b", ""))

    return ParsedHTML(title=title, cleaned_text=cleaned_text)
