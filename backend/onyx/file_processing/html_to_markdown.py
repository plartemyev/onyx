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
to plain text (links) or is dropped (images). Destinations are deduped
per page: the first informative occurrence stays a link and later
repeats drop altogether; text-less anchors never claim a destination —
they render only while the URL appears nowhere else, so overlay URLs
stay available for digging. Pagination controls are the one exception:
icon-only anchors with rel first/prev/next/last keep their destination
under a generated label ("page 765"), and <head> <link rel=...>
metadata supplies a one-line footer for whatever the body dropped —
both through the same seen-href dedup. Repeated-sibling card grids
(listings) are
reassembled into pipe tables from their shared item anatomy, and
icon-font glyph names plus private-use-area characters never reach the
text.
"""

from __future__ import annotations

import re
from urllib.parse import parse_qsl, urlencode, urljoin, urlsplit, urlunsplit

import bs4
from markdownify import MarkdownConverter

from onyx.file_processing.html_utils import (
    ParsedHTML,
    clean_web_html,
    strip_pua_chars,
)

# Collapse the blank-line runs markdownify leaves between block elements.
_EXCESS_BLANK_LINES = re.compile(r"\n{3,}")

# Nested block groups can both fire at one boundary; one rule between
# blocks carries the information.
_CONSECUTIVE_RULES = re.compile(r"(?m)^---[ \t]*\n(?:\n*^---[ \t]*\n)+\n?")

# Link hrefs that carry no navigation value for the model: in-page anchors
# and script URLs render as bare link text even in "full" mode.
_NO_VALUE_HREF_PREFIXES = ("#", "javascript:")

# rel tokens that mark a link as pagination metadata (visible nav anchors
# and <head> <link rel=...> alike). These are machine semantics, not
# decoration, so they survive rules that drop icon-only chrome.
_PAGINATION_RELS = frozenset({"first", "prev", "next", "last"})

# Markdown image syntax, used to tell image-only anchors from anchors
# that carry real text.
_IMAGE_MARKDOWN = re.compile(r"!\[[^\]]*\]\([^)]*\)")

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


def _pagination_anchor_label(el: bs4.element.Tag) -> str | None:
    """Label for a text-less pagination control, or None when not one.

    Forums render the first/prev/next/last buttons as icon-only anchors
    (IPS: ``<a rel="last" title="..."><i class="fa fa-..."></i></a>``) —
    exactly the shape the text-less-anchor rule drops. Their rel and
    data-page attributes are machine metadata, so such anchors keep their
    destination under a generated label instead. Anchors with text never
    reach this helper; they label themselves.
    """
    rels = {token.lower() for token in _attribute_value(el, "rel").split()}
    matched = rels & _PAGINATION_RELS
    if not matched:
        return None
    data_page = _attribute_value(el, "data-page").strip()
    if data_page:
        return f"page {data_page}"
    return min(matched)


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
        # Page-level href dedupe: card-style layouts wrap every item in
        # overlapping anchors (stretched overlays, tag chips, author
        # links), repeating each destination two or three times. The first
        # informative occurrence in document order carries the text and
        # stays a link; later duplicates of the same destination drop
        # altogether, and text-less anchors never claim a destination.
        self._seen_hrefs: set[str] = set()
        # Same rule for media: reaction icons and avatars repeat per
        # comment/row and the duplicate instances carry zero information.
        self._seen_srcs: set[str] = set()

    def pagination_link_footer(self, soup: bs4.BeautifulSoup) -> str | None:
        """One-line footer from ``<head>`` ``<link rel=first|prev|next|last>``.

        SEO pagination metadata most forum engines emit. Shares
        ``_seen_hrefs`` with the body conversion, so the pipeline dedups in
        execution order: a destination the body already linked (a numbered
        page anchor, or a rel anchor kept by the pagination exception) is
        not repeated, and the footer only supplies what the body lost —
        e.g. pages whose only nav affordance was an icon-only control.
        Comparison is fragment-insensitive: forum nav anchors point at
        ``.../page/2/#comments`` while head metadata points at
        ``.../page/2/`` — the same page, and the footer must not repeat it.
        Emitted destinations are claimed so later consumers stay deduped.
        """
        entries: list[str] = []
        seen_roots = {href.split("#", 1)[0] for href in self._seen_hrefs}
        for link in soup.find_all("link"):
            rels = {token.lower() for token in _attribute_value(link, "rel").split()}
            matched = sorted(rels & _PAGINATION_RELS)
            if not matched:
                continue
            href = _normalize_resource_url(
                _attribute_value(link, "href"), self._base_url
            )
            if not href.lower().startswith(("http://", "https://")):
                continue
            root = href.split("#", 1)[0]
            if root in seen_roots:
                continue
            seen_roots.add(root)
            self._seen_hrefs.add(href)
            entries.append(f"[{matched[0]}]({href})")
        if not entries:
            return None
        return "Pagination: " + " · ".join(entries)

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
        if href in self._seen_hrefs:
            # Later repeats of the same destination drop altogether.
            return ""
        title = _attribute_value(el, "title")
        title_suffix = f' "{title}"' if title else ""
        # An anchor whose text is only image markdown (an avatar link)
        # carries no name and must not claim the destination either.
        has_text = bool(_IMAGE_MARKDOWN.sub("", text).strip())
        if has_text:
            # Informative anchors claim the destination: this is the
            # occurrence that names it.
            self._seen_hrefs.add(href)
            return f"[{text.strip()}]({href}{title_suffix})"
        # Text-less anchors (image-only avatars, stretched-card overlays,
        # icon links) never claim: they render with their content while
        # the URL appears nowhere else, so overlay URLs stay available
        # for digging without crowding out the informative link that
        # follows them. An anchor rendering nothing at all (its image
        # deduped away, label dropped) is pure URL noise: dropped.
        # Exception: pagination controls (rel first/prev/next/last) are
        # machine metadata rendered icon-only — they claim under a
        # generated label, subject to the same seen-hrefs dedup.
        if not text.strip():
            pagination_label = _pagination_anchor_label(el)
            if pagination_label is None:
                return ""
            self._seen_hrefs.add(href)
            return f"[{pagination_label}]({href}{title_suffix})"
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
        if src in self._seen_srcs:
            return ""
        self._seen_srcs.add(src)
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

    # GFM pipe tables. markdownify's built-ins linearize cells without a
    # header separator, which renders as plain text, not a table.

    def convert_table(
        self,
        el: bs4.element.Tag,  # noqa: ARG002  (call-site contract)
        text: str,
        parent_tags: list[str] | None = None,  # noqa: ARG002  (call-site contract)
    ) -> str:
        rows = [row.strip() for row in text.strip().splitlines() if row.strip()]
        if len(rows) < 2:
            # Not enough rows for a pipe table: keep the linearized cells.
            return "\n\n" + text.strip() + "\n\n"
        rows = [row if row.startswith("|") else f"|{row}" for row in rows]
        # Escaped pipes (\|) are cell content, not column separators.
        columns = len(re.findall(r"(?<!\\)\|", rows[0])) - 1
        rows.insert(1, "|" + "---|" * columns)
        return "\n\n" + "\n".join(rows) + "\n\n"

    def convert_thead(
        self,
        el: bs4.element.Tag,  # noqa: ARG002  (call-site contract)
        text: str,
        parent_tags: list[str] | None = None,  # noqa: ARG002  (call-site contract)
    ) -> str:
        return text

    def convert_tbody(
        self,
        el: bs4.element.Tag,  # noqa: ARG002  (call-site contract)
        text: str,
        parent_tags: list[str] | None = None,  # noqa: ARG002  (call-site contract)
    ) -> str:
        return text

    def convert_tr(
        self,
        el: bs4.element.Tag,  # noqa: ARG002  (call-site contract)
        text: str,
        parent_tags: list[str] | None = None,  # noqa: ARG002  (call-site contract)
    ) -> str:
        # Cells already end with " |"; wrap the row, dropping the duplicate.
        cells = text.strip("\n").removesuffix("|")
        return f"|{cells}|\n"

    def _convert_cell(self, text: str) -> str:
        # Pipes inside cell text would break the row; newlines cannot occur.
        return " " + text.strip().replace("\n", " ").replace("|", "\\|") + " |"

    def convert_td(
        self,
        el: bs4.element.Tag,  # noqa: ARG002  (call-site contract)
        text: str,
        parent_tags: list[str] | None = None,  # noqa: ARG002  (call-site contract)
    ) -> str:
        return self._convert_cell(text)

    def convert_th(
        self,
        el: bs4.element.Tag,  # noqa: ARG002  (call-site contract)
        text: str,
        parent_tags: list[str] | None = None,  # noqa: ARG002  (call-site contract)
    ) -> str:
        return self._convert_cell(text)


# Main-content anchoring: when the page marks its primary content with a
# single <article> or <main>, convert only that subtree — site chrome that
# survives the element/class cleanup (mega-menu dropdown lists, language
# selectors, toolbars) lives outside it. Several <article> tags usually mean
# a teaser index page, where each card is real content: keep the whole body.
_MAIN_CONTENT_SELECTORS = ("article", "main")


def _marker_dom_id(element: bs4.element.Tag) -> str:
    """The marker's own id, or the nearest ancestor's (post ids usually
    live on the marker or its block wrapper)."""
    dom_id = _attribute_value(element, "id").strip()
    walk = element.parent
    for _ in range(3):
        if walk is None or dom_id:
            break
        dom_id = _attribute_value(walk, "id").strip()
        walk = walk.parent
    return dom_id


def _promote_sequential_markers(soup: bs4.BeautifulSoup) -> None:
    """Rename repeated prefix+number markers to headings, in place."""
    groups: dict[str, list[tuple[int, bs4.element.Tag]]] = {}
    for node in soup.find_all(string=True):
        if type(node) is not bs4.element.NavigableString:
            continue
        text = re.sub(r"\s+", " ", str(node)).strip()
        match = _MARKER_TEXT_RE.fullmatch(text)
        if not match:
            continue
        prefix = match.group(1).strip()
        if len(prefix) < _MARKER_PREFIX_MIN_CHARS:
            continue
        parent = node.parent
        if parent is None or parent.get_text(strip=True) != text:
            continue
        # find_parent starts above the element, so check its own name too.
        if parent.name in _NO_PROMOTE_PARENTS:
            continue
        if parent.find_parent(_NO_PROMOTE_ANCESTORS) is not None:
            continue
        groups.setdefault(prefix, []).append((int(match.group(2)), parent))
    for markers in groups.values():
        if len(markers) < _MARKER_MIN_COUNT:
            continue
        numbers = [number for number, _ in markers]
        # Arbitrary ids are expected; a repeated number is a label copy.
        if len(set(numbers)) != len(numbers):
            continue
        for _, element in markers:
            element.name = _MARKER_ELEMENT
            element.attrs[_BLOCK_MARKER_ATTR] = "1"
            dom_id = _marker_dom_id(element)
            if dom_id:
                element.append(f" ({dom_id})")


def _prose_length(item: bs4.element.Tag) -> int:
    """Item text that lives outside links: the non-navigation content."""
    total = len(item.get_text(" ", strip=True))
    linked = sum(len(a.get_text(" ", strip=True)) for a in item.find_all("a"))
    return total - linked


def _repeated_block_runs(soup: bs4.BeautifulSoup) -> int:
    """Insert <hr> between members of prose-dominant sibling groups.

    Runs top-down: once a group qualifies, every member is marked and no
    container inside a member is ever considered again — a comment's own
    paragraphs are part of the comment, not a block run of their own.
    When headings already delimit the members (promoted markers), the
    rules are redundant and only the marks are set.
    """
    inserted = 0
    for container in soup.find_all(True):
        if container.name in ("html", "body"):
            continue
        if container.find_parent(attrs={_BLOCK_MEMBER_ATTR: True}) is not None:
            continue
        children = container.find_all(True, recursive=False)
        groups: dict[tuple, list[bs4.element.Tag]] = {}
        for child in children:
            signature = _class_signature(child) or (child.name,)
            groups.setdefault(signature, []).append(child)
        for items in groups.values():
            if len(items) < _BLOCK_MIN_ITEMS:
                continue
            prose_items = [
                el for el in items if _prose_length(el) >= _BLOCK_MIN_PROSE_CHARS
            ]
            if len(prose_items) < _BLOCK_MIN_ITEMS:
                continue
            for el in items:
                el.attrs[_BLOCK_MEMBER_ATTR] = "1"
            delimited = sum(
                1
                for el in items
                if el.find(attrs={_BLOCK_MARKER_ATTR: True}) is not None
            )
            if delimited * 2 > len(items):
                continue  # headings already delimit these blocks
            for el in items[1:]:
                el.insert_before(soup.new_tag("hr"))
                inserted += 1
    return inserted


def _select_main_content(soup: bs4.BeautifulSoup) -> bs4.element.Tag:
    for selector in _MAIN_CONTENT_SELECTORS:
        matches = soup.find_all(selector)
        if len(matches) == 1:
            return matches[0]
    return soup.body or soup


# -- Repeated-sibling card grids -> pipe tables ------------------------------
#
# Listing pages (forums, news indexes, search results) render their rows as
# sibling cards: real DOM, regular anatomy, but rarely real <table> markup.
# Linearized, one row's title, author, timestamp and counts blur into the
# next. Siblings sharing one class signature and one inner anatomy can be
# reassembled into a markdown table: columns follow each text leaf's
# structural path inside the item (wrapper induction), and the title keeps
# its link.

_SIBLING_TABLE_MIN_ITEMS = 3
_SIBLING_TABLE_TITLE_MIN_CHARS = 15
_SIBLING_TABLE_MAX_ITEM_LINKS = 12
_SIBLING_TABLE_MAX_ITEM_CHARS = 600
_SIBLING_TABLE_MIN_COLUMN_COVERAGE = 0.6
_TABLE_CONTAINER_TAGS = ("ul", "ol", "div")
_TABLE_ITEM_TAGS = ("li", "div")

# Sequential markers ("ความคิดเห็นที่ 3", "Post 120599831", "Step 12"): one
# text prefix followed by a number, repeated across the page, marks content
# block boundaries (comment threads, step lists, changelog entries). The
# numbers may be arbitrary ids — random values are expected; only exact
# duplicates of a number disqualify a group (a marker identifies one
# block). Promote each to a heading so the block structure survives
# linearization, appending the marker element's DOM id (commonly the
# entry's post id) for precise citation. Locale neutral: prefix + number
# shape only, no word lists.
_MARKER_MIN_COUNT = 5
_MARKER_PREFIX_MIN_CHARS = 3
_MARKER_ELEMENT = "h3"
_MARKER_TEXT_RE = re.compile(r"^(\D{3,})[\s\u00a0]+(\d{1,10})$")
_NO_PROMOTE_ANCESTORS = ("td", "th", "h1", "h2", "h3", "h4", "h5", "h6")
# A full paragraph is prose, not a marker label.
_NO_PROMOTE_PARENTS = ("p", *_NO_PROMOTE_ANCESTORS)

# Block runs: repeated siblings carrying the page's primary prose (comment
# threads, article feeds) — the groups the table pass rejects. The
# discriminator is prose density: in a content block most of an item's
# text lives outside its links; in a listing (cards, tag rows) the text is
# the links. A horizontal rule between members keeps block boundaries
# visible after linearization; items holding a promoted marker (###) are
# already delimited and never fire a run.
_BLOCK_MIN_ITEMS = 3
_BLOCK_MIN_PROSE_CHARS = 50
_BLOCK_MARKER_ATTR = "data-md-marker"
_BLOCK_MEMBER_ATTR = "data-md-block"


def _class_signature(el: bs4.element.Tag) -> tuple[str, ...] | None:
    tokens = _class_tokens(el)
    return tuple(sorted(tokens)) if tokens else None


def _href_shape(anchor: bs4.element.Tag) -> str:
    """Link destination shape: scheme, host, first path segment."""
    parts = urlsplit(_attribute_value(anchor, "href"))
    first = parts.path.split("/", 2)[1] if parts.path.startswith("/") else ""
    return f"{parts.scheme}://{parts.netloc}/{first}"


def _dominant_href_shape(items: list[bs4.element.Tag]) -> str | None:
    """The destination shape that provides the item titles.

    Votes per item: the shape of each item's longest link (the title).
    The shape winning a majority of items carries the title column;
    total link text breaks ties.
    """
    wins: dict[str, int] = {}
    text_length: dict[str, int] = {}
    for item in items:
        links = item.find_all("a", href=True)
        if not links:
            continue
        longest = max(links, key=lambda a: len(a.get_text(" ", strip=True)))
        shape = _href_shape(longest)
        wins[shape] = wins.get(shape, 0) + 1
        for anchor in links:
            anchor_shape = _href_shape(anchor)
            text_length[anchor_shape] = text_length.get(anchor_shape, 0) + len(
                anchor.get_text(" ", strip=True)
            )
    if not wins:
        return None
    shape = max(wins, key=lambda s: (wins[s], text_length.get(s, 0), s))
    return shape if wins[shape] * 2 > len(items) else None


def _title_anchor(item: bs4.element.Tag, shape: str | None) -> bs4.element.Tag | None:
    """The item's title: its longest link on the group's dominant shape."""
    links = [
        a
        for a in item.find_all("a", href=True)
        if shape is None or _href_shape(a) == shape
    ]
    if not links:
        return None
    best = max(links, key=lambda a: len(a.get_text(" ", strip=True)))
    if len(best.get_text(" ", strip=True)) < _SIBLING_TABLE_TITLE_MIN_CHARS:
        return None
    return best


def _leaf_cells(
    item: bs4.element.Tag, title: bs4.element.Tag
) -> list[tuple[tuple[tuple[str, str], ...], str, str]]:
    """(structural path, header hint, text) for one card's text leaves.

    The path runs from the item root down to the leaf's direct parent
    (tag, first class token); the hint is the direct parent's last class
    token, the best generic column name available.
    """
    cells = []
    for node in item.descendants:
        if not isinstance(node, bs4.element.NavigableString):
            continue
        if title in node.parents:
            continue
        parent = node.parent
        if parent is None or parent.name in ("script", "style"):
            continue
        text = re.sub(r"\s+", " ", str(node)).strip()
        if not text:
            continue
        # Column name: the nearest class-carrying ancestor (the leaf's
        # own container preferred); tracking-markup classes (gtm-*, js-*)
        # say nothing about the column and are skipped.
        hint = parent.name
        walk_hint: bs4.element.Tag | None = parent
        while walk_hint is not None and walk_hint is not item:
            tokens = [
                t for t in _class_tokens(walk_hint) if not t.startswith(("gtm-", "js-"))
            ]
            if tokens:
                hint = tokens[-1]
                break
            walk_hint = walk_hint.parent
        path = []
        walk: bs4.element.Tag | None = parent
        while walk is not None and walk is not item:
            tokens = _class_tokens(walk)
            path.append((walk.name, tokens[0] if tokens else walk.name))
            walk = walk.parent
        cells.append((tuple(reversed(path)), hint, text))
    return cells


def _column_header(token: str, used: set[str]) -> str:
    """Readable column name from the leaf container's class token."""
    name = re.split(r"[-_]", token)[-1] or token
    name = name[:24] or "col"
    candidate, n = name, 2
    while candidate.lower() in used:
        candidate = f"{name}-{n}"
        n += 1
    used.add(candidate.lower())
    return candidate


def _table_schema(
    parsed: list[tuple[bs4.element.Tag, bs4.element.Tag, list]],
) -> list[tuple[tuple[tuple[str, str], ...], str]] | None:
    """Column paths present in most items, in DOM order, with headers."""
    coverage: dict[tuple, int] = {}
    hints: dict[tuple, list[str]] = {}
    for _, _, cells in parsed:
        seen: set[tuple] = set()
        for path, hint, _ in cells:
            if path in seen:
                continue
            seen.add(path)
            coverage[path] = coverage.get(path, 0) + 1
            hints.setdefault(path, []).append(hint)
    threshold = max(1, int(len(parsed) * _SIBLING_TABLE_MIN_COLUMN_COVERAGE))
    schema = [path for path, count in coverage.items() if count >= threshold]
    if not schema:
        return None
    schema.sort(key=min)
    used = {"title"}
    columns = []
    for path in schema:
        hint = max(set(hints[path]), key=hints[path].count)
        columns.append((path, _column_header(hint, used)))
    return columns


def _build_table_block(
    soup: bs4.BeautifulSoup, items: list[bs4.element.Tag]
) -> bs4.element.Tag | None:
    """One sibling group as a <table>, or None when it is not card-shaped."""
    shape = _dominant_href_shape(items)
    parsed = []
    for item in items:
        if len(item.find_all("a")) > _SIBLING_TABLE_MAX_ITEM_LINKS:
            return None
        if len(item.get_text(" ", strip=True)) > _SIBLING_TABLE_MAX_ITEM_CHARS:
            return None
        title = _title_anchor(item, shape)
        if title is None:
            return None
        parsed.append((item, title, _leaf_cells(item, title)))

    columns = _table_schema(parsed)
    if columns is None:
        return None

    table = soup.new_tag("table")
    head_row = soup.new_tag("tr")
    for name in ["Title"] + [header for _, header in columns]:
        th = soup.new_tag("th")
        th.string = name
        head_row.append(th)
    thead = soup.new_tag("thead")
    thead.append(head_row)
    table.append(thead)

    body = soup.new_tag("tbody")
    for _item, title, cells in parsed:
        row = soup.new_tag("tr")
        title_cell = soup.new_tag("td")
        title_cell.append(title.extract())
        row.append(title_cell)
        for path, _ in columns:
            cell = soup.new_tag("td")
            cell.string = " ".join(text for p, _, text in cells if p == path)
            row.append(cell)
        body.append(row)
    table.append(body)
    items[0].insert_before(table)
    for item, _, _ in parsed:
        item.decompose()
    return table


def _repeated_sibling_tables(soup: bs4.BeautifulSoup) -> None:
    """Swap repeated-sibling card groups for pipe-table markup, in place."""
    containers = sorted(
        soup.find_all(_TABLE_CONTAINER_TAGS),
        key=lambda c: -len(list(c.parents)),
    )
    consumed: set[int] = set()
    for container in containers:
        children = container.find_all(_TABLE_ITEM_TAGS, recursive=False)
        groups: dict[tuple[str, ...], list[bs4.element.Tag]] = {}
        for child in children:
            signature = _class_signature(child)
            if signature:
                groups.setdefault(signature, []).append(child)
        for items in groups.values():
            if len(items) < _SIBLING_TABLE_MIN_ITEMS:
                continue
            if any(id(item) in consumed for item in items):
                continue
            table = _build_table_block(soup, items)
            if table is not None:
                consumed.update(id(item) for item in items)


def _collapse_rules(markdown: str) -> str:
    return _CONSECUTIVE_RULES.sub("---\n\n", markdown)


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
    _repeated_sibling_tables(soup)
    _promote_sequential_markers(soup)
    _repeated_block_runs(soup)
    root = _select_main_content(soup)
    converter = WebContentMarkdownConverter(link_mode=link_mode, base_url=base_url)
    markdown = converter.convert_soup(root)
    markdown = _collapse_rules(markdown)
    markdown = _EXCESS_BLANK_LINES.sub("\n\n", markdown).strip()
    footer = converter.pagination_link_footer(soup)
    if footer:
        markdown = f"{markdown}\n\n{footer}" if markdown else footer
    return ParsedHTML(title=title, cleaned_text=strip_pua_chars(markdown))
