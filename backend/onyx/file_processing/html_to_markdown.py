"""HTML to GitHub-flavored Markdown for model-facing web content.

Produces the output shape of Firecrawl's converter (Turndown + GFM plugin:
ATX headings, pipe tables, fenced code blocks, inline links) without the
JS/Go machinery. Runs on the same BeautifulSoup chrome removal the text
extraction path uses, so both formats see an identical DOM. Media (images,
file links) renders inline in the body — markdownify turns <img> into
![alt](src) — so the markdown carries every resource reference the page
does. Link rendering is tunable per call, and every emitted link/media URL
is reconstructed to a fetchable address: relative references resolve
against the page URL, tracking parameters (utm_*) are stripped,
scheme-relative URLs are anchored to https, parentheses and spaces are
percent-encoded, and anything that cannot become an http(s) URL degrades
to plain text (links) or is dropped (images).
"""

from __future__ import annotations

import re
from urllib.parse import parse_qsl, urlencode, urljoin, urlsplit, urlunsplit

import bs4
from markdownify import MarkdownConverter

from onyx.file_processing.html_utils import ParsedHTML, clean_web_html

# Collapse the blank-line runs markdownify leaves between block elements.
_EXCESS_BLANK_LINES = re.compile(r"\n{3,}")

# Link hrefs that carry no navigation value for the model: in-page anchors
# and script URLs render as bare link text even in "full" mode.
_NO_VALUE_HREF_PREFIXES = ("#", "javascript:")

# Tracking parameters stripped from link and media URLs so the model sees
# the canonical address (utm_* is the dominant family; matched by prefix,
# case-insensitive, so utm_source/utm_campaign/... all go).
_TRACKING_PARAM_PREFIX = "utm_"

LINK_MODE_FULL = "full"
LINK_MODE_BARE = "bare"
_LINK_MODES = (LINK_MODE_FULL, LINK_MODE_BARE)


def _class_tokens(node: bs4.element.Tag) -> list[str]:
    """Class attribute as a token list (bs4 may hand back str or list)."""
    value = node.get("class")
    if isinstance(value, str):
        return value.split()
    if isinstance(value, list):
        return [str(token) for token in value]
    return []


def _attribute_value(node: bs4.element.Tag, name: str) -> str:
    """Attribute as a plain string (bs4 may hand back str or list)."""
    value = node.get(name)
    if isinstance(value, str):
        return value
    if isinstance(value, list):
        return " ".join(str(token) for token in value)
    return ""


def _normalize_resource_url(url: str, base_url: str | None = None) -> str:
    """Reconstruct a fetchable URL: resolve relative references against the
    page URL, anchor scheme-relative addresses to https, drop tracking
    parameters, and percent-encode parentheses and spaces. Non-http(s)
    schemes (data:, mailto:) are returned untouched."""
    url = url.strip()
    if url.startswith("//"):
        url = f"https:{url}"
    if base_url and not url.lower().startswith(("http://", "https://")):
        url = urljoin(base_url, url)
    if not url.lower().startswith(("http://", "https://")):
        return url
    parts = urlsplit(url)
    if parts.query:
        params = parse_qsl(parts.query, keep_blank_values=True)
        kept = [
            (name, value)
            for name, value in params
            if not name.lower().startswith(_TRACKING_PARAM_PREFIX)
        ]
        if len(kept) != len(params):
            url = urlunsplit(
                (
                    parts.scheme,
                    parts.netloc,
                    parts.path,
                    urlencode(kept),
                    parts.fragment,
                )
            )
    # Literal parentheses and spaces are legal in URLs but terminate
    # markdown link destinations for naive parsers and LLM URL extraction
    # (Wikipedia's "..._Cat_(film)" pattern). The server treats the
    # percent-encoded form identically.
    return url.replace("(", "%28").replace(")", "%29").replace(" ", "%20")


class WebContentMarkdownConverter(MarkdownConverter):
    """markdownify converter tuned for LLM-facing page content."""

    def __init__(
        self, *, link_mode: str = LINK_MODE_FULL, base_url: str | None = None
    ) -> None:
        super().__init__(
            heading_style="ATX",
            bullets="-",
            code_language_callback=self._code_language,
        )
        self._link_mode = link_mode
        self._base_url = base_url

    @staticmethod
    def _code_language(el: bs4.element.Tag) -> str:
        # Recover the fence language from the common highlight.js / Pygments
        # class conventions on the <pre> or its <code> child.
        nodes: list[bs4.element.Tag] = [el]
        code_el = el.find("code")
        if isinstance(code_el, bs4.element.Tag):
            nodes.append(code_el)
        for node in nodes:
            for cls in _class_tokens(node):
                if cls.startswith("language-"):
                    return cls.removeprefix("language-")
                if cls.startswith("highlight-"):
                    return cls.removeprefix("highlight-").split("-", 1)[0]
        return ""

    def convert_a(
        self,
        el: bs4.element.Tag,
        text: str,
        parent_tags: list[str] | None = None,  # noqa: ARG002  (call-site contract)
    ) -> str:
        raw_href = _attribute_value(el, "href")
        # Checked before resolution: fragment-only hrefs ("#section", table
        # of contents links) must not turn into full self-links.
        if raw_href.strip().startswith(_NO_VALUE_HREF_PREFIXES):
            return text
        if self._link_mode == LINK_MODE_BARE:
            return text
        href = _normalize_resource_url(raw_href, self._base_url)
        # Invariant: every emitted link destination is a fetchable http(s)
        # URL. Anything unresolvable (mailto:, tel:, data:, relatives with
        # no base) degrades to its text.
        if not href.lower().startswith(("http://", "https://")):
            return text
        # Inline-style link, mirroring markdownify's inlined link_style.
        title = _attribute_value(el, "title")
        title_suffix = f' "{title}"' if title else ""
        return f"[{text.strip()}]({href}{title_suffix})"

    def convert_img(
        self,
        el: bs4.element.Tag,
        text: str,  # noqa: ARG002  (call-site contract)
        parent_tags: list[str] | None = None,  # noqa: ARG002  (call-site contract)
    ) -> str:
        # Mirrors markdownify's convert_img with a normalized src.
        src = _normalize_resource_url(_attribute_value(el, "src"), self._base_url)
        # Same http(s) invariant as links; also caps data:-URI bloat.
        if not src.lower().startswith(("http://", "https://")):
            return ""
        alt = _attribute_value(el, "alt")
        if not alt:
            # Fall back to the title of a wrapping link (the Wikipedia
            # audio/video player pattern: an alt-less thumbnail inside
            # <a title="Play audio">).
            anchor = el.find_parent("a")
            if isinstance(anchor, bs4.element.Tag):
                alt = _attribute_value(anchor, "title")
        title = _attribute_value(el, "title").replace('"', '\\"')
        title_suffix = f' "{title}"' if title else ""
        return f"![{alt}]({src}{title_suffix})"


# Main-content anchoring: when the page marks its primary content with a
# single <article> or <main>, convert only that subtree — site chrome that
# survives the element/class cleanup (mega-menu dropdown lists, language
# selectors, toolbars) lives outside it. Several <article> tags usually mean
# a teaser index page, where each card is real content: keep the whole body.
_MAIN_CONTENT_SELECTORS = ("article", "main")


def _select_main_content(soup: bs4.BeautifulSoup) -> bs4.element.Tag:
    for selector in _MAIN_CONTENT_SELECTORS:
        matches = soup.find_all(selector)
        if len(matches) == 1:
            return matches[0]
    return soup.body or soup


def web_page_to_markdown(
    page_content: str | bs4.BeautifulSoup,
    *,
    base_url: str | None = None,
    link_mode: str = LINK_MODE_FULL,
) -> ParsedHTML:
    """Clean a web page (chrome dropped, <title> extracted) and convert the
    body to GitHub-flavored Markdown. `cleaned_text` carries the markdown
    body; the <title> is not part of it (title is a separate field).
    Relative links and media sources resolve against `base_url` — the
    crawler passes the post-redirect page URL."""
    if link_mode not in _LINK_MODES:
        raise ValueError(f"link_mode must be one of {_LINK_MODES}, got '{link_mode}'")

    soup, title = clean_web_html(page_content)
    root = _select_main_content(soup)
    converter = WebContentMarkdownConverter(link_mode=link_mode, base_url=base_url)
    markdown = converter.convert_soup(root)
    markdown = _EXCESS_BLANK_LINES.sub("\n\n", markdown).strip()
    return ParsedHTML(title=title, cleaned_text=markdown)
