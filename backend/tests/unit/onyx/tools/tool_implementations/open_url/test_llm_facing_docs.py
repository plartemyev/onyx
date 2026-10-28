"""Unit tests for the open_url LLM-facing surface: the per-call URL cap note
and the document-number title prefix (see numbered_title)."""

import json

from onyx.configs.constants import DocumentSource
from onyx.context.search.models import InferenceChunk, InferenceSection
from onyx.tools.models import OPEN_URL_MAX_URLS
from onyx.tools.tool_implementations.open_url.open_url_tool import (
    _convert_sections_to_llm_string_with_citations,
    _over_limit_note,
)
from onyx.tools.tool_implementations.utils import numbered_title


class TestOverLimitNote:
    def test_note_states_cap_and_next_step(self) -> None:
        note = _over_limit_note(received=12, max_urls=6)
        assert "12" in note
        assert "6" in note
        assert "Nothing was fetched" in note
        assert "Split the URLs" in note

    def test_note_is_llm_facing_only(self) -> None:
        # The note must stay a short instruction, not a page dump.
        assert len(_over_limit_note(received=50, max_urls=OPEN_URL_MAX_URLS)) < 300


def _make_chunk(
    document_id: str,
    semantic_identifier: str | None = None,
    chunk_id: int = 0,
) -> InferenceChunk:
    return InferenceChunk(
        document_id=document_id,
        chunk_id=chunk_id,
        content=f"content-{document_id}",
        source_type=DocumentSource.MOCK_CONNECTOR,
        semantic_identifier=semantic_identifier or f"sem-{document_id}",
        title=document_id,
        boost=1,
        score=0.5,
        hidden=False,
        metadata={},
        match_highlights=[],
        doc_summary="",
        chunk_context="",
        updated_at=None,
        image_file_id=None,
        source_links={0: f"https://example.com/{document_id}"},
        section_continuation=False,
        blurb=f"blurb-{document_id}",
        file_id=None,
        image_urls=None,
    )


def _make_section(chunk: InferenceChunk) -> InferenceSection:
    return InferenceSection(
        center_chunk=chunk,
        chunks=[chunk],
        combined_content=chunk.content,
    )


class TestDocumentNumberTitlePrefix:
    def test_title_carries_citation_number(self) -> None:
        sections = [
            _make_section(_make_chunk("doc-a", semantic_identifier="Левада: Рейтинги")),
            _make_section(_make_chunk("doc-b", semantic_identifier="Forum thread")),
        ]
        docs_str, citation_mapping = _convert_sections_to_llm_string_with_citations(
            sections=sections,
            existing_citation_mapping={},
            citation_start=1,
        )
        payload = json.loads(docs_str)
        titles = [r["title"] for r in payload["results"]]
        assert titles == ["[document 1] Левада: Рейтинги", "[document 2] Forum thread"]
        assert citation_mapping == {1: "doc-a", 2: "doc-b"}

    def test_existing_citation_number_is_reused_in_prefix(self) -> None:
        section = _make_section(_make_chunk("doc-a", semantic_identifier="Thread"))
        docs_str, citation_mapping = _convert_sections_to_llm_string_with_citations(
            sections=[section],
            existing_citation_mapping={"doc-a": 114},
            citation_start=200,
        )
        payload = json.loads(docs_str)
        assert payload["results"][0]["title"] == "[document 114] Thread"
        assert citation_mapping == {114: "doc-a"}


def test_prefix_handles_missing_title() -> None:
    # The shared helper degrades to the bare number; open_url's run() fills
    # semantic_identifier from source_links before this point, so this only
    # guards the helper's contract.
    assert numbered_title(1, None) == "[document 1]"
    assert numbered_title(2, "") == "[document 2]"
