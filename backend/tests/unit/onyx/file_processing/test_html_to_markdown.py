from onyx.file_processing.html_to_markdown import (
    LINK_MODE_BARE,
    LINK_MODE_FULL,
    web_page_to_markdown,
)
from onyx.file_processing.html_utils import web_html_cleanup

PAGE = """
<html>
  <head><title>Test Page</title><style>.x { color: red }</style></head>
  <body>
    <script>evil()</script>
    <nav><a href="/menu">Menu item</a></nav>
    <h2>Section</h2>
    <p>See <a href="https://example.com/doc">the doc</a> and
       <a href="#anchor">an anchor</a> plus
       <a href="javascript:void(0)">a script link</a>.</p>
    <table>
      <tr><th>Key</th><th>Value</th></tr>
      <tr><td>a</td><td>1</td></tr>
    </table>
    <pre><code class="language-python">def f():
    return 1</code></pre>
    <img src="/img.png" alt="photo">
    <p>End of <b>page</b>.</p>
  </body>
</html>
"""


def test_full_link_mode() -> None:
    parsed = web_page_to_markdown(
        PAGE, base_url="https://example.test/page", link_mode=LINK_MODE_FULL
    )

    assert parsed.title == "Test Page"
    md = parsed.cleaned_text
    assert "## Section" in md
    assert "[the doc](https://example.com/doc)" in md
    # Fragment / javascript hrefs carry no navigation value: text only.
    assert "an anchor" in md and "](#anchor)" not in md
    # javascript: anchors are action controls: dropped with their text.
    assert "a script link" not in md and "javascript:" not in md
    assert "| Key | Value |" in md and "| a | 1 |" in md
    assert "```python\ndef f():\n    return 1\n```" in md
    # Media renders inline in the body and script/nav chrome is gone.
    assert "![photo](https://example.test/img.png)" in md
    assert "evil()" not in md and "Menu item" not in md


def test_bare_link_mode() -> None:
    parsed = web_page_to_markdown(PAGE, link_mode=LINK_MODE_BARE)

    assert "the doc" in parsed.cleaned_text
    assert "](https://example.com/doc)" not in parsed.cleaned_text
    # Structure survives link stripping.
    assert "## Section" in parsed.cleaned_text
    assert "```python" in parsed.cleaned_text


def test_invalid_link_mode() -> None:
    try:
        web_page_to_markdown(PAGE, link_mode="footnote")  # type: ignore[arg-type]
    except ValueError:
        pass
    else:
        raise AssertionError("expected ValueError for invalid link_mode")


def test_main_content_anchoring() -> None:
    # Chrome outside a single <main> (mega-menus, language dropdowns) is
    # dropped; the article body inside is kept.
    page = """
    <html><head><title>Anchored</title></head><body>
      <div class="menu"><ul><li>Language A</li><li>Language B</li></ul></div>
      <main><h1>Body</h1><p>Real content.</p></main>
      <footer>Footer junk</footer>
    </body></html>
    """
    parsed = web_page_to_markdown(page)
    assert "Real content." in parsed.cleaned_text
    assert "# Body" in parsed.cleaned_text
    assert "Language A" not in parsed.cleaned_text
    assert "Footer junk" not in parsed.cleaned_text


def test_multiple_articles_keep_whole_body() -> None:
    # Several <article> tags mean a teaser index page: every card is real
    # content, so no anchoring happens.
    page = """
    <html><head><title>Index</title></head><body>
      <article>Card one</article>
      <article>Card two</article>
    </body></html>
    """
    parsed = web_page_to_markdown(page)
    assert "Card one" in parsed.cleaned_text
    assert "Card two" in parsed.cleaned_text


def test_image_inside_link_keeps_both_urls() -> None:
    # Wikipedia pattern: a thumbnail <img> wrapped by a <a> to the File:
    # page. Both the binary URL and the page URL must survive.
    page = (
        "<html><head><title>Media</title></head><body>"
        '<a href="https://en.wikipedia.org/wiki/File:cat.jpg">'
        '<img src="https://upload.wikimedia.org/catthumb.jpg" alt="a cat">'
        "</a>"
        '<a href="https://en.wikipedia.org/wiki/File:meow.wav" title="Play audio">'
        "meow clip</a>"
        "</body></html>"
    )
    parsed = web_page_to_markdown(page)
    assert "[![a cat](https://upload.wikimedia.org/catthumb.jpg)]" in (
        parsed.cleaned_text
    )
    assert "](https://en.wikipedia.org/wiki/File:cat.jpg)" in parsed.cleaned_text
    assert (
        '[meow clip](https://en.wikipedia.org/wiki/File:meow.wav "Play audio")'
        in parsed.cleaned_text
    )


def test_url_reconstruction() -> None:
    page = (
        "<html><head><title>Urls</title></head><body>"
        "<p>"
        '<a href="https://example.com/doc?utm_source=x&utm_campaign=y&id=7">doc</a>'
        '<a href="//cdn.example.com/paper.pdf">paper</a>'
        "</p>"
        '<img src="//img.example.com/t.jpg?utm_content=z&w=250" alt="thumb">'
        '<img src="/relative/local.png" alt="local">'
        "</body></html>"
    )
    md = web_page_to_markdown(page).cleaned_text
    # utm_* dropped, legitimate params kept
    assert "[doc](https://example.com/doc?id=7)" in md
    # scheme-relative anchored to https
    assert "[paper](https://cdn.example.com/paper.pdf)" in md
    assert "![thumb](https://img.example.com/t.jpg?w=250)" in md
    # relative media without a base URL is dropped (http(s)-only invariant)
    assert "/relative/local.png" not in md


def test_parens_in_urls_percent_encoded() -> None:
    # Wikipedia "..._Cat_(film)" pattern: legal in URLs and balanced, but
    # naive parsers and LLM URL extraction truncate at the first ")". The
    # percent-encoded form is equivalent for the server and unambiguous
    # for every consumer.
    page = (
        "<html><head><title>Parens</title></head><body>"
        '<a href="https://en.wikipedia.org/wiki/Siamese_Cat_(film)"'
        ' title="Siamese Cat (film)">the film</a>'
        '<a href="https://en.wikipedia.org/wiki/Thai_cat">Thai cat</a>'
        "</body></html>"
    )
    md = web_page_to_markdown(page).cleaned_text
    assert "](https://en.wikipedia.org/wiki/Siamese_Cat_%28film%29" in md
    assert "](https://en.wikipedia.org/wiki/Siamese_Cat_(film" not in md
    # The link title keeps its literal parentheses (quoted, unambiguous);
    # paren-free URLs pass through byte-identical.
    assert ' "Siamese Cat (film)")' in md
    assert "[Thai cat](https://en.wikipedia.org/wiki/Thai_cat)" in md


def test_relative_urls_resolve_against_base_url() -> None:
    page = (
        "<html><head><title>Rel</title></head><body>"
        "<p>"
        '<a href="/wiki/Thailand">Thailand</a>'
        '<a href="#History">History</a>'
        '<a href="/w/index.php?title=X&action=edit">edit</a>'
        "</p>"
        '<img src="/img/local.png" alt="local">'
        "</body></html>"
    )
    md = web_page_to_markdown(
        page, base_url="https://en.wikipedia.org/wiki/Siamese_cat"
    ).cleaned_text
    # Root-relative references become fetchable absolute URLs.
    assert "[Thailand](https://en.wikipedia.org/wiki/Thailand)" in md
    assert "![local](https://en.wikipedia.org/img/local.png)" in md
    # Fragment-only links stay text (no self-links for the table of
    # contents); query-string relatives resolve fully.
    assert "](#History)" not in md and "[History](" not in md
    assert "[edit](https://en.wikipedia.org/w/index.php?title=X&action=edit)" in md


def test_invalid_links_and_media_degrade() -> None:
    # The http(s) invariant: unresolvable links degrade to their text,
    # unresolvable media is dropped (data:-URI bloat included).
    page = (
        "<html><head><title>Invalid</title></head><body>"
        '<p><a href="mailto:cat@example.com">mail us</a></p>'
        '<p><a href="tel:+15550100">call us</a></p>'
        '<img src="data:image/png;base64,AAAA" alt="inline">'
        '<img src="/no-base/rel.png" alt="orphan">'
        "</body></html>"
    )
    md = web_page_to_markdown(page).cleaned_text
    assert "mail us" in md and "mailto:" not in md
    assert "call us" in md and "tel:" not in md
    assert "data:image" not in md
    # Relative media without a base URL cannot be reconstructed.
    assert "orphan" not in md and "/no-base/rel.png" not in md


def test_alt_falls_back_to_anchor_title() -> None:
    page = (
        "<html><head><title>Alt</title></head><body>"
        '<a href="https://en.wikipedia.org/wiki/File:meow.wav" title="Play audio">'
        '<img src="https://upload.wikimedia.org/meow.jpg"></a>'
        '<a href="https://en.wikipedia.org/wiki/File:quiet.jpg">'
        '<img src="https://upload.wikimedia.org/quiet.jpg"></a>'
        '<img src="https://upload.wikimedia.org/lone.jpg" alt="lone cat">'
        "</body></html>"
    )
    md = web_page_to_markdown(page).cleaned_text
    # Anchor title fills the empty alt...
    assert "[![Play audio](https://upload.wikimedia.org/meow.jpg)]" in md
    # ...stays empty when the anchor has none...
    assert "[![](https://upload.wikimedia.org/quiet.jpg)]" in md
    # ...and an explicit alt always wins.
    assert "![lone cat](https://upload.wikimedia.org/lone.jpg)" in md


def test_page_level_href_dedupe() -> None:
    page = (
        "<html><head><title>Dedupe</title></head><body>"
        '<h2><a href="https://example.com/t/1">Topic one</a></h2>'
        '<a href="https://example.com/t/1"></a>'
        '<a href="https://example.com/t/1">Topic one again</a>'
        '<a href="https://example.com/t/2">Topic two</a>'
        # Text-less anchor first: it does not claim the destination, so
        # the informative link that follows survives (author-name case).
        '<a href="https://example.com/u/1"></a>'
        '<a href="https://example.com/u/1">Author</a>'
        "</body></html>"
    )
    md = web_page_to_markdown(page).cleaned_text
    # First informative occurrence keeps the link; later labeled
    # duplicates and empty duplicates of a claimed destination drop.
    assert "[Topic one](https://example.com/t/1)" in md
    assert md.count("](https://example.com/t/1)") == 1
    assert "Topic one again" not in md
    assert "[Topic two](https://example.com/t/2)" in md
    # Empty-render anchors are dropped entirely (a bare unexplained URL
    # is noise) AND the informative link keeps its name.
    assert "[](https://example.com/u/1)" not in md
    assert "[Author](https://example.com/u/1)" in md


def test_image_src_dedupe() -> None:
    page = (
        "<html><head><title>Imgs</title></head><body>"
        '<img src="https://cdn.example.com/hero.png" alt="hero">'
        '<img src="https://cdn.example.com/hero.png" alt="hero again">'
        '<img src="https://cdn.example.com/other.png" alt="other">'
        "</body></html>"
    )
    md = web_page_to_markdown(page).cleaned_text
    assert "![hero](https://cdn.example.com/hero.png)" in md
    assert "![other](https://cdn.example.com/other.png)" in md
    assert md.count("hero.png") == 1


def test_avatar_anchor_does_not_claim() -> None:
    # An image-only anchor (avatar link) must not claim the destination:
    # the informative named link that follows keeps its name.
    page = (
        "<html><head><title>Avatar</title></head><body>"
        '<a href="https://example.com/u/7">'
        '<img src="https://cdn.example.com/avatar.jpg" alt="avatar"></a>'
        '<a href="https://example.com/u/7">Querist</a>'
        "</body></html>"
    )
    md = web_page_to_markdown(page).cleaned_text
    assert "[![avatar](https://cdn.example.com/avatar.jpg)]" in md
    assert "[Querist](https://example.com/u/7)" in md


def test_code_language_from_pre_class() -> None:
    page = (
        "<html><head><title>Code</title></head><body>"
        '<pre class="language-ts">const x: number = 1;</pre>'
        "</body></html>"
    )
    parsed = web_page_to_markdown(page)
    assert "```ts\nconst x: number = 1;\n```" in parsed.cleaned_text


def test_text_path_unchanged() -> None:
    # The legacy text extraction keeps its own shape (flat text, title split
    # out); the markdown path must not affect it.
    parsed = web_html_cleanup(PAGE)
    assert parsed.title == "Test Page"
    assert "## Section" not in (parsed.cleaned_text or "")
    assert "the doc" in (parsed.cleaned_text or "")


def test_icon_only_pagination_anchor_keeps_destination() -> None:
    # IPS forums render the last-page button as an icon-only anchor whose
    # <i> the icon-glyph stripper removes; rel/data-page rescue the URL.
    page = (
        "<html><head><title>Thread</title></head><body>"
        '<ul class="ipsPagination">'
        '<li><a href="https://example.com/t/1">1</a></li>'
        '<li class="ipsPagination_last"><a href="https://example.com/t/765" '
        'rel="last" data-page="765" title="Последняя страница">'
        '<i class="fa fa-angle-double-right"></i></a></li>'
        "</ul></body></html>"
    )
    parsed = web_page_to_markdown(page, base_url="https://example.com/t/1")
    text = parsed.cleaned_text or ""
    assert '[page 765](https://example.com/t/765 "Последняя страница")' in text
    # The icon glyph name itself must not leak.
    assert "fa-angle-double-right" not in text
    assert "angle-double-right" not in text


def test_icon_only_anchor_without_rel_still_dropped() -> None:
    # The pagination exception must not resurrect other icon-link noise.
    page = (
        "<html><head><title>Page</title></head><body>"
        '<a href="https://example.com/u/7" title="User 7">'
        '<i class="fa fa-user"></i></a>'
        "</body></html>"
    )
    parsed = web_page_to_markdown(page)
    assert "example.com/u/7" not in (parsed.cleaned_text or "")


def test_pagination_dedup_body_anchor_wins_over_footer() -> None:
    # Pipeline dedup in execution order: the body's numbered anchor claims
    # page 2, so the head <link rel="next"> must not repeat it; the last
    # page appears only via the footer because the body dropped it.
    page = (
        "<html><head><title>T</title>"
        '<link rel="next" href="https://example.com/t/2">'
        '<link rel="last" href="https://example.com/t/9">'
        "</head><body>"
        '<ul class="ipsPagination">'
        '<li><a href="https://example.com/t/1">1</a></li>'
        '<li><a href="https://example.com/t/2">2</a></li>'
        "</ul></body></html>"
    )
    parsed = web_page_to_markdown(page)
    text = parsed.cleaned_text or ""
    # page 2 exactly once (the body anchor, not the footer)
    assert text.count("example.com/t/2") == 1
    # page 9 supplied by the footer
    assert "Pagination: [last](https://example.com/t/9)" in text
    assert "example.com/t/2)]" not in text.split("Pagination:")[1]


def test_pagination_footer_supplies_both_when_body_has_nothing() -> None:
    page = (
        "<html><head><title>T</title>"
        '<link rel="next" href="https://example.com/t/2">'
        '<link rel="last" href="https://example.com/t/9">'
        "</head><body><p>Just a post.</p></body></html>"
    )
    parsed = web_page_to_markdown(page)
    text = parsed.cleaned_text or ""
    assert (
        "Pagination: [next](https://example.com/t/2) · [last](https://example.com/t/9)"
        in text
    )


def test_pagination_dedup_ignores_fragments() -> None:
    # Forum nav anchors carry "#comments" fragments while head metadata
    # points at the bare page URL — same page, footer must not repeat it.
    page = (
        "<html><head><title>T</title>"
        '<link rel="next" href="https://example.com/t/2/">'
        "</head><body>"
        '<ul class="ipsPagination">'
        '<li><a href="https://example.com/t/2/#comments">2</a></li>'
        "</ul></body></html>"
    )
    parsed = web_page_to_markdown(page)
    text = parsed.cleaned_text or ""
    assert "Pagination:" not in text
    assert text.count("example.com/t/2") == 1


def test_text_pagination_anchor_wins_over_icon_copy() -> None:
    # First informative occurrence claims; a later icon-only copy of the
    # same destination drops through the seen-hrefs dedup.
    page = (
        "<html><head><title>T</title></head><body>"
        '<ul class="ipsPagination">'
        '<li><a href="https://example.com/t/765" rel="last">Последняя</a></li>'
        "</ul>"
        '<ul class="ipsPagination">'
        '<li><a href="https://example.com/t/765" rel="last" data-page="765">'
        '<i class="fa fa-angle-double-right"></i></a></li>'
        "</ul></body></html>"
    )
    parsed = web_page_to_markdown(page)
    text = parsed.cleaned_text or ""
    assert "[Последняя](https://example.com/t/765)" in text
    assert "page 765" not in text
