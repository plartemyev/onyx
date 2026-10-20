"""Tests for the generic web-parser cleanup rules and table reconstruction.

Covers: icon-font glyph dropping, private-use-area stripping, whitespace
boundaries on dropped inline elements, repeated-sibling card grids rebuilt
as pipe tables, and GFM conversion of real <table> markup.
"""

import re

import onyx.file_processing.html_to_markdown as html_to_markdown
from onyx.file_processing.html_to_markdown import web_page_to_markdown
from onyx.file_processing.html_utils import strip_pua_chars, web_html_cleanup

CARDS_HTML = """
<html><head><title>List</title></head><body>
<main>
<h2>Latest</h2>
<ul class="cards">
  <li class="card"><div class="card__title"><h2><a href="https://x.test/news/1">First headline is long enough here</a></h2></div>
    <div class="card__info"><h5><a href="https://x.test/u/amy">amy</a></h5><span>2 hours</span></div>
    <div class="card__stats"><span class="card__stat-comments"><i class="material-icons md-18">message</i>119</span><span class="card__stat-votes"><i class="material-icons md-18">add_box</i>17</span></div></li>
  <li class="card"><div class="card__title"><h2><a href="https://x.test/news/2">Second headline also long enough</a></h2></div>
    <div class="card__info"><h5><a href="https://x.test/u/bob">bob</a></h5><span>5 hours</span></div>
    <div class="card__stats"><span class="card__stat-comments"><i class="material-icons md-18">message</i>31</span><span class="card__stat-votes"><i class="material-icons md-18">add_box</i>9</span></div></li>
  <li class="card"><div class="card__title"><h2><a href="https://x.test/news/3">Third headline long enough too</a></h2></div>
    <div class="card__info"><h5><a href="https://x.test/u/sue">sue</a></h5><span>1 day</span></div>
    <div class="card__stats"><span class="card__stat-comments"><i class="material-icons md-18">message</i>7</span><span class="card__stat-votes"><i class="material-icons md-18">add_box</i>2</span></div></li>
</ul>
</main>
</body></html>
"""


def test_icon_ligature_names_dropped_and_counts_separated() -> None:
    parsed = web_page_to_markdown(CARDS_HTML, base_url="https://x.test/")
    md = parsed.cleaned_text
    # Ligature names never reach the text, with or without ignore-class env.
    assert "message" not in md
    assert "add_box" not in md
    # The whitespace boundary keeps adjacent counts distinct.
    assert "119" in md
    assert "17" in md
    assert "11917" not in md


def test_icon_drop_applies_to_legacy_text_path() -> None:
    html = "<p>A<i class='material-icons'>search</i>B</p>"
    parsed = web_html_cleanup(html)
    assert "search" not in parsed.cleaned_text
    assert "A" in parsed.cleaned_text and "B" in parsed.cleaned_text


def test_custom_icon_font_and_pua_dropped() -> None:
    html = (
        "<p>Keep <i class='pantip-icons pt-ic-topic_type_talking'></i>words"
        "<i class='material-icons'>\ue313</i>end</p>"
    )
    parsed = web_page_to_markdown(html)
    assert "pantip-icons" not in parsed.cleaned_text
    assert "\ue313" not in parsed.cleaned_text
    assert "Keep" in parsed.cleaned_text and "words" in parsed.cleaned_text


def test_strips_pua_chars_helper() -> None:
    assert strip_pua_chars("a\ue313b\uf8ffc") == "abc"


def test_emphasis_text_survives_icon_rule() -> None:
    # Real <em> content is not an icon: longer than a ligature name.
    html = "<p>say <em>this is emphasized words</em> now</p>"
    parsed = web_page_to_markdown(html)
    assert "emphasized" in parsed.cleaned_text


def test_repeated_siblings_become_pipe_table() -> None:
    md = web_page_to_markdown(CARDS_HTML, base_url="https://x.test/").cleaned_text
    header = "| Title |"
    assert header in md
    assert "|---|" in md
    # Title cell keeps its link; cells align across rows.
    assert "[First headline is long enough here](https://x.test/news/1)" in md
    assert "| amy |" in md
    assert "| 119 |" in md
    # Two items only: below the group minimum, no table is forced.
    two = CARDS_HTML.replace(
        """  <li class="card"><div class="card__title"><h2><a href="https://x.test/news/3">Third headline long enough too</a></h2></div>
    <div class="card__info"><h5><a href="https://x.test/u/sue">sue</a></h5><span>1 day</span></div>
    <div class="card__stats"><span class="card__stat-comments"><i class="material-icons md-18">message</i>7</span><span class="card__stat-votes"><i class="material-icons md-18">add_box</i>2</span></div></li>
""",
        "",
    )
    md_two = web_page_to_markdown(two, base_url="https://x.test/").cleaned_text
    assert "|---|" not in md_two
    assert "Third headline" not in md_two


def test_title_shape_follows_dominant_destination() -> None:
    # A card whose longest link is a tag: the title column must still pick
    # the destination shape that provides titles across the group.
    html = CARDS_HTML.replace(
        "Third headline long enough too",
        "หัวข้อยาวๆ ที่ทดสอบการเลือกลิงก์หัวข้อหลักจากรูปทรงลิงก์",
    )
    md = web_page_to_markdown(html, base_url="https://x.test/").cleaned_text
    assert "](https://x.test/news/3)" in md


def test_real_table_converts_to_gfm() -> None:
    html = """
    <table><thead><tr><th>Key</th><th>Val|ue</th></tr></thead>
    <tbody><tr><td>a</td><td>1</td></tr><tr><td>b</td><td>2</td></tr></tbody></table>
    """
    md = web_page_to_markdown(html).cleaned_text
    assert "| Key | Val\\|ue |" in md
    assert "|---|---|" in md
    assert "| a | 1 |" in md
    assert "| b | 2 |" in md


def test_table_converter_ignores_single_row() -> None:
    soup = html_to_markdown.bs4.BeautifulSoup(
        "<table><tr><td>x</td></tr></table>", "lxml"
    )
    converter = html_to_markdown.WebContentMarkdownConverter()
    text = converter.convert_soup(soup)
    # One row: no separator, stays linearized instead of a fake table.
    assert "|---|" not in text


def test_action_controls_dropped_real_links_kept() -> None:
    html = (
        "<p>Post text here</p>"
        "<a class='reply' href='javascript:void(0)'>Reply</a>"
        "<button type='button'>Share</button>"
        "<a href='https://x.test/item'>Real content link</a>"
    )
    md = web_page_to_markdown(html).cleaned_text
    assert "Reply" not in md
    assert "Share" not in md
    assert "Post text here" in md
    assert "[Real content link](https://x.test/item)" in md


def test_repeated_boilerplate_dropped_linked_text_kept() -> None:
    # 12 unlinked copies: boilerplate. 12 linked copies: real content
    # (listing rows repeat tag names as links). Numbers never dropped.
    label = "<span class='btn' onclick='go()'>Give points</span>"
    linked = "".join(
        f"<a href='https://x.test/tag/t{i}'>Widget</a> " for i in range(12)
    )
    html = (
        "<p>intro</p>" + label * 12 + linked + "<p>outro</p>" + "<span>0</span> " * 12
    )
    md = web_page_to_markdown(html).cleaned_text
    assert "Give points" not in md
    assert md.count("Widget") == 12
    # URL digits ("t0", "t10") must not count as the standalone zeros.
    assert len(re.findall(r"\b0\b", md)) == 12


def test_repeated_short_content_below_threshold_kept() -> None:
    html = "<p>agreed</p>" * 5
    md = web_page_to_markdown(html).cleaned_text
    assert md.count("agreed") == 5


def test_empty_render_anchors_dropped() -> None:
    # The avatar image was deduped away / label dropped: nothing renders.
    html = (
        "<p>body text</p>"
        "<a href='https://x.test/profile/1'><span class='dropped'></span></a>"
        "<a href='https://x.test/profile/2'></a>"
    )
    md = web_page_to_markdown(html).cleaned_text
    assert "](https://x.test/profile/" not in md
    assert "body text" in md


def test_image_only_anchor_kept() -> None:
    html = "<a href='https://x.test/full.png'><img src='https://x.test/small.jpg'></a>"
    md = web_page_to_markdown(html).cleaned_text
    assert "](https://x.test/full.png)" in md
    assert "![](https://x.test/small.jpg)" in md


def test_sequential_markers_promoted_with_random_ids() -> None:
    # Random, unordered, unique numbers: still block markers.
    html = "".join(
        f"<div class='post'><span class='num' id='p{n}'>Post {n}</span>"
        f"<p>body {n}</p></div>"
        for n in (8123, 945, 12, 77, 4)
    )
    md = web_page_to_markdown(html).cleaned_text
    assert "### Post 8123 (p8123)" in md
    assert "### Post 12 (p12)" in md
    assert md.count("### ") == 5


def test_duplicate_marker_numbers_not_promoted() -> None:
    html = "".join(f"<span class='num'>Post {n}</span>" for n in (3, 3, 3, 3, 3, 3))
    md = web_page_to_markdown(html).cleaned_text
    assert "###" not in md


def test_markers_inside_table_cells_not_promoted() -> None:
    rows = "".join(f"<tr><td>Part {n}</td><td>desc {n}</td></tr>" for n in range(1, 6))
    md = web_page_to_markdown(f"<table>{rows}</table>").cleaned_text
    assert "### Part" not in md
    assert "| Part 1 |" in md


def test_prose_block_run_gets_separators() -> None:
    articles = "".join(
        "<article class='entry'>"
        f"<h2><a href='https://x.test/p/{i}'>Title {i} here</a></h2>"
        f"<p>{'lorem ipsum dolor sit amet ' * 5}</p>"
        "</article>"
        for i in range(4)
    )
    md = web_page_to_markdown(f"<div class='feed'>{articles}</div>").cleaned_text
    # One rule between each pair of consecutive block members.
    assert md.count("---") >= 3


def test_link_heavy_listing_unchanged() -> None:
    # Link-dense card rows are a listing, not a block run: no separators,
    # and the table pass handles them before this pass even looks.
    cards = "".join(
        "<div class='card'>"
        f"<a href='https://x.test/t/{i}'>Title number {i} about things</a>"
        "<a href='https://x.test/tag/a'>taga</a>"
        "<a href='https://x.test/tag/b'>tagb</a>"
        "<span>2 hours</span>"
        "</div>"
        for i in range(10)
    )
    md = web_page_to_markdown(f"<div class='feed'>{cards}</div>").cleaned_text
    # the table's |---| separator is fine; a standalone rule is not
    assert not re.search(r"(?m)^---\s*$", md)
    assert "| Title |" in md


def test_marker_headed_blocks_do_not_double_separate() -> None:
    posts = "".join(
        f"<div class='post'><span class='num' id='p{n}'>Post {n}</span>"
        f"<p>{'comment body text goes here and it is long enough ' * 3}</p></div>"
        for n in (1, 2, 3, 4, 5)
    )
    md = web_page_to_markdown(f"<div class='thread'>{posts}</div>").cleaned_text
    assert md.count("### Post") == 5
    # headings already delimit the blocks: no rules between them
    assert not re.search(r"(?m)^---\s*$", md)


def test_no_rules_inside_marker_headed_posts() -> None:
    # Real-page regression: a post body with several sibling paragraphs
    # (a numbered list) must never become a block run of its own.
    paragraphs = "".join(
        f"<div class='line'>{'point text with enough prose to qualify ' * 2} {i}</div>"
        for i in range(4)
    )
    posts = "".join(
        f"<div class='post'><span class='num' id='p{n}'>Post {n}</span>"
        f"<div class='body'>{paragraphs}</div></div>"
        for n in (1, 2, 3, 4, 5)
    )
    md = web_page_to_markdown(f"<div class='thread'>{posts}</div>").cleaned_text
    assert md.count("### Post") == 5
    assert not re.search(r"(?m)^---\s*$", md)


def test_consecutive_rules_collapsed() -> None:
    md = "alpha\n\n---\n\n---\n\nbeta\n\n| a | b |\n|---|---|\n"
    collapsed = html_to_markdown._collapse_rules(md)
    # doubled rules become one; the table separator is untouched
    assert collapsed == "alpha\n\n---\n\nbeta\n\n| a | b |\n|---|---|\n"
