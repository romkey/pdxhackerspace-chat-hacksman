from __future__ import annotations

from app.llm import _build_system_prompt
from app.models import ContextChunk


def test_build_system_prompt_enriches_calibre_chunk_metadata() -> None:
    prompt = _build_system_prompt(
        "Base prompt",
        [
            ContextChunk(
                collection="calibre_books",
                score=0.88,
                text="In the beginning...",
                metadata={
                    "title": "The Bible",
                    "authors": ["Various Authors"],
                    "chapter_title": "Genesis",
                    "chunk_type": "content",
                    "source_url": "http://calibre/download/42/epub",
                },
            )
        ],
    )
    assert 'Book: "The Bible" by Various Authors' in prompt
    assert "Chapter: Genesis" in prompt
    assert "Type: content" in prompt
    assert "Source: http://calibre/download/42/epub" in prompt
    assert "collection=calibre_books" in prompt


def test_build_system_prompt_keeps_default_format_for_non_calibre_chunks() -> None:
    prompt = _build_system_prompt(
        "Base prompt",
        [
            ContextChunk(
                collection="wiki",
                score=0.91,
                text="Safety glasses are required.",
                metadata={},
            )
        ],
    )
    assert "collection=wiki score=0.9100" in prompt
    assert "Safety glasses are required." in prompt
